"""Conditional rate-path comparators: positive basis and matched DCA.

One shared MLP maps the cutoff history, the latest-interval rates and the
static attributes to a full Gaussian over 12 fast-basis or 9 DCA log
parameters. Latent draws give nonnegative rate paths; reports are drawn from
the accumulated Gaussian report bridge, which pins the observed history.
This is a conditional predictive law, not a Bayesian latent posterior.

Training minimizes, at coefficient one each, equal-cell/equal-phase scaled
increment CRPS and the equal-origin joint energy score on measured,
context-unobserved positive-day reports (normalized by sqrt of the scored
dimension), both estimated as U-statistics of iid draws. One epoch visits every
training well at every origin once. Context/mask, latent, report and
initialization random streams are separate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dca_rates import DCARate
from .history_features import last_interval_rate_features
from .metrics import fit_increment_scales
from .rates import PositiveBasisRate
from .torch_observation import sample_accumulated_reports


INTERPRETATION = "conditional_mixing_law_not_bayesian_posterior"


def _u_scores(target_reports, report_samples, context_observed, days, origin_index,
              increment_scales, target_weights):
    """Per-context iid U-CRPS and U-energy; input volumes are phase-normalized.

    increment_scales are native cell scales divided by the same phase scales.
    target_weights=N_train/n_observed_cell_phase are frozen during fitting.
    Every future interval subtracts the SAME sampled origin endpoint.
    """
    count = report_samples.shape[1]
    if count < 2:
        raise ValueError("Unbiased proper-score training requires at least two iid draws")
    future = days > days[origin_index]
    target_mask = torch.isfinite(target_reports[:, future]) & torch.isfinite(target_reports[:, origin_index, None])
    target_increment = target_reports[:, future] - target_reports[:, origin_index, None]
    target_increment = torch.where(target_mask, target_increment, torch.zeros_like(target_increment))
    sample_increment = report_samples[:, :, future] - report_samples[:, :, origin_index, None]
    standardized = sample_increment / increment_scales
    actual = target_increment / increment_scales
    pair_left, pair_right = torch.triu_indices(count, count, offset=1, device=report_samples.device)
    marginal = (standardized - actual[:, None]).abs().mean(dim=1)
    marginal -= (standardized[:, pair_left] - standardized[:, pair_right]).abs().sum(dim=1) / (count * (count - 1))
    marginal = (marginal * target_mask * target_weights).mean(dim=(-2, -1))
    unobserved = torch.isfinite(target_reports) & ~context_observed & (days > 0)[None, :, None]
    safe_target = torch.where(unobserved, target_reports, torch.zeros_like(target_reports))
    residual = torch.where(unobserved[:, None], report_samples - safe_target[:, None],
                           torch.zeros_like(report_samples)).flatten(start_dim=2)
    # vector_norm has a finite zero subgradient; do not sqrt a squared zero
    # distance and then mask an undefined derivative afterwards.
    energy = torch.linalg.vector_norm(residual, dim=-1).mean(dim=1)
    pairs = residual[:, pair_left] - residual[:, pair_right]
    energy -= torch.linalg.vector_norm(pairs, dim=-1).sum(dim=1) / (count * (count - 1))
    dimension = unobserved.sum(dim=(-2, -1))
    energy /= dimension.clamp_min(1).to(report_samples.dtype).sqrt()
    return marginal, energy, dimension


class _ContextNetwork(nn.Module):
    """MLP giving a latent mean and a lower-triangular factor with bounded entries."""

    def __init__(self, input_size, width, initial, min_std, max_std, initial_std, max_offdiag):
        super().__init__()
        self.dimension = len(initial)
        self.min_std, self.max_std, self.max_offdiag = min_std, max_std, max_offdiag
        self.hidden = nn.Sequential(nn.Linear(input_size, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.mean = nn.Linear(width, self.dimension)
        indices = torch.tril_indices(self.dimension, self.dimension)
        self.register_buffer("lower_rows", indices[0])
        self.register_buffer("lower_columns", indices[1])
        self.cholesky = nn.Linear(width, indices.shape[1])
        nn.init.zeros_(self.mean.weight)
        with torch.no_grad():
            self.mean.bias.copy_(initial)
        nn.init.zeros_(self.cholesky.weight)
        nn.init.zeros_(self.cholesky.bias)
        fraction = (initial_std - min_std) / (max_std - min_std)
        with torch.no_grad():
            self.cholesky.bias[indices[0] == indices[1]] = math.log(fraction / (1 - fraction))

    def forward(self, context):
        hidden = self.hidden(context)
        raw = self.cholesky(hidden)
        diagonal = self.lower_rows == self.lower_columns
        entries = self.max_offdiag * torch.tanh(raw)
        standard_deviations = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(raw)
        entries = torch.where(diagonal[None], standard_deviations, entries)
        lower = raw.new_zeros((len(raw), self.dimension, self.dimension))
        lower[:, self.lower_rows, self.lower_columns] = entries
        return self.mean(hidden), lower


class ConditionalRateForecaster(nn.Module):
    """History-conditioned rate law with an exact Gaussian report bridge.

    ``covariates`` have shape [well, feature] and must already be encoded on
    training data by the caller (finite, no further scaling). Include origin
    zero in ``fit(origins=...)`` for attribute-only forecasts; the only observed
    report then is the deterministic accounting zero.

    ``basis_dictionary``, ``history_rate_features`` and ``covariate_preprocessing``
    have one supported value each; they remain arguments because the saved
    configurations record them.
    """

    def __init__(self, rate_family="basis", *, basis_dictionary="fast", seed=42,
                 epochs=100, hidden_width=128, batch_size=256, train_samples=16,
                 learning_rate=1e-3, history_dropout=0.1, phase_dropout=0.1,
                 initial_latent_std=0.25, min_latent_std=0.025, max_latent_std=1.0,
                 max_offdiag=0.2, noise_floor=1e-4, initial_noise_abs=0.02,
                 initial_noise_rel=0.08, noise_regularization=0.0,
                 gradient_clip=10.0, log_every=25, device="cpu",
                 history_rate_features="last_interval", covariate_preprocessing="external"):
        super().__init__()
        if rate_family not in {"basis", "dca"} or basis_dictionary != "fast":
            raise ValueError("rate_family must be basis (fast dictionary) or dca")
        if history_rate_features != "last_interval" or covariate_preprocessing != "external":
            raise ValueError("Only last_interval history features and external covariates are supported")
        if min(epochs, hidden_width, batch_size) < 1 or train_samples < 2:
            raise ValueError("Positive epoch/width/batch settings and at least two iid samples are required")
        if not 0 < min_latent_std < initial_latent_std < max_latent_std or max_offdiag <= 0:
            raise ValueError("Require 0 < min_std < initial_std < max_std and positive off-diagonal bound")
        if (learning_rate <= 0 or gradient_clip <= 0 or log_every < 0 or noise_floor <= 0
                or min(initial_noise_abs, initial_noise_rel) <= noise_floor or noise_regularization < 0):
            raise ValueError("Invalid optimizer or noise parameters")
        if not 0 <= history_dropout < 1 or not 0 <= phase_dropout < 1:
            raise ValueError("Masking probabilities must be in [0,1)")
        self.n_covariates_ = self.covariate_feature_count_ = 0
        self.configuration = dict(rate_family=rate_family, basis_dictionary=basis_dictionary, seed=int(seed),
            epochs=int(epochs), hidden_width=int(hidden_width), batch_size=int(batch_size), train_samples=int(train_samples),
            learning_rate=float(learning_rate), history_dropout=float(history_dropout), phase_dropout=float(phase_dropout),
            initial_latent_std=float(initial_latent_std), min_latent_std=float(min_latent_std), max_latent_std=float(max_latent_std),
            max_offdiag=float(max_offdiag), noise_floor=float(noise_floor), initial_noise_abs=float(initial_noise_abs),
            initial_noise_rel=float(initial_noise_rel), noise_regularization=float(noise_regularization),
            gradient_clip=float(gradient_clip), log_every=int(log_every), device=str(device),
            history_rate_features=history_rate_features, covariate_preprocessing=covariate_preprocessing)
        for key, value in self.configuration.items():
            if key != "device":
                setattr(self, key, value)
        self.rate_model = PositiveBasisRate() if rate_family == "basis" else DCARate()
        self.latent_size = self.rate_model.latent_size
        self.network = None
        self.register_buffer("report_days", torch.empty(0, dtype=torch.float64, device=device))
        self.register_buffer("volume_scale", torch.empty(0, dtype=torch.float64, device=device))
        self.register_buffer("increment_scales", torch.empty(0, dtype=torch.float64, device=device))
        inverse = lambda value: math.log(math.expm1(value - noise_floor))
        self.noise_abs_raw = nn.Parameter(torch.full((3,), inverse(initial_noise_abs), dtype=torch.float32, device=device))
        self.noise_rel_raw = nn.Parameter(torch.full((3,), inverse(initial_noise_rel), dtype=torch.float32, device=device))

    @property
    def device(self):
        return self.noise_abs_raw.device

    @property
    def noise_abs(self):
        return torch.nn.functional.softplus(self.noise_abs_raw) + self.noise_floor

    @property
    def noise_rel(self):
        return torch.nn.functional.softplus(self.noise_rel_raw) + self.noise_floor

    def _initialize(self, days, phase_scales, increment_scales):
        self.report_days = torch.as_tensor(days, dtype=torch.float64, device=self.device)
        self.volume_scale = torch.as_tensor(phase_scales, dtype=torch.float64, device=self.device)
        self.increment_scales = torch.as_tensor(increment_scales, dtype=torch.float64, device=self.device)
        cuda_devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            initial = self.rate_model.initial_latent(dtype=torch.float32)
            self.network = _ContextNetwork(2 * len(days) * 3 + 2 * len(days) + 1 + 6 + self.covariate_feature_count_,
                                           self.hidden_width, initial, self.min_latent_std, self.max_latent_std,
                                           self.initial_latent_std, self.max_offdiag).to(self.device)
        digest = hashlib.sha256()
        for name, parameter in self.named_parameters():
            values = parameter.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(values.dtype), list(values.shape)], separators=(",", ":")).encode())
            digest.update(values.numpy().tobytes())
        self.initial_parameter_sha256_ = digest.hexdigest()

    @staticmethod
    def _context(reports, observed, days, origin):
        values = torch.asinh(torch.where(observed, reports, torch.zeros_like(reports)))
        grid = (days / days[-1]).expand(len(reports), -1)
        normalized_origin = torch.full((len(reports), 1), float(origin) / float(days[-1]),
                                       dtype=reports.dtype, device=reports.device)
        return torch.cat((values.flatten(start_dim=1), observed.to(reports.dtype).flatten(start_dim=1),
                          normalized_origin, grid, grid-normalized_origin), dim=-1)

    def _encoder_context(self, reports, observed, days, origin, *, native_reports, covariates=None):
        """Append static attributes and latest-interval rates (from native float64 reports)."""
        context = self._context(reports, observed, days, origin)
        if covariates is not None:
            context = torch.cat((context, covariates), dim=-1)
        features = last_interval_rate_features(native_reports, observed, self.report_days,
            origin, self.volume_scale, dtype=context.dtype, zero_control=False)
        return torch.cat((context, features), dim=-1)

    def _covariates(self, covariates, n, *, fit):
        if covariates is None:
            if fit:
                self.n_covariates_ = self.covariate_feature_count_ = 0
            elif self.n_covariates_:
                raise ValueError("This fitted model requires cutoff-available covariates")
            return np.empty((n, 0), dtype=np.float32)
        values = np.asarray(covariates, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != n or not np.isfinite(values).all():
            raise ValueError("covariates must be a finite [well, feature] matrix")
        if fit:
            self.n_covariates_ = self.covariate_feature_count_ = values.shape[1]
        if values.shape[1] != self.n_covariates_:
            raise ValueError("Covariate dimension differs from training")
        if (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError("Covariate features cannot be represented in float32")
        return values.astype(np.float32)

    def fit(self, train_reports, report_days, *, origins=None, covariates=None):
        reports = np.asarray(train_reports, dtype=np.float64)
        days = np.asarray(report_days, dtype=np.float64)
        if (reports.ndim != 3 or reports.shape[-1] != 3 or not all(reports.shape)
                or days.shape != (reports.shape[1],) or len(days) < 3
                or not np.isfinite(days).all() or days[0] != 0 or (np.diff(days) <= 0).any()
                or np.isinf(reports).any()):
            raise ValueError("Require finite increasing canonical days starting at zero and reports [well,time,3], NaN allowed")
        if np.any(np.isfinite(reports[:, 0]) & (reports[:, 0] != 0)):
            raise ValueError("Day zero is the deterministic zero accounting origin")
        reports = reports.copy()
        reports[:, 0] = 0.0
        self.origins_ = np.asarray(days[1:-1] if origins is None else origins, dtype=float)
        if (self.origins_.ndim != 1 or not len(self.origins_) or not np.isin(self.origins_, days[:-1]).all()
                or len(np.unique(self.origins_)) != len(self.origins_)):
            raise ValueError("Origins must be distinct report days with a future endpoint")
        self.cells_ = [(float(s), float(t)) for s in self.origins_ for t in days if t > s]
        indices = {float(day): index for index, day in enumerate(days)}
        with np.errstate(over="raise", invalid="ignore"):
            increments = np.stack([reports[:, indices[t]]-reports[:, indices[s]] for s, t in self.cells_], axis=1)
        scales, scale_details = fit_increment_scales(increments, return_details=True)
        counts = np.isfinite(increments).sum(axis=0)
        if (counts == 0).any():
            raise ValueError("Every frozen cell/phase requires at least one measured training increment")
        phase_scales, phase_sources = [], []
        for phase in range(3):
            positive = reports[:, -1, phase]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            source = "positive_terminal_train_report_median"
            if not len(positive):
                positive = reports[:, :, phase]
                positive = positive[np.isfinite(positive) & (positive > 0)]
                source = "pooled_positive_train_report_median"
            phase_scales.append(float(np.median(positive)) if len(positive) else 1.0)
            phase_sources.append(source if len(positive) else "one_native_unit_fallback")
        phase_scales = np.asarray(phase_scales)
        if not np.all(np.isfinite(phase_scales) & (phase_scales > 0)):
            raise ValueError("Phase scales must be finite and positive")
        covariate_features = self._covariates(covariates, len(reports), fit=True)
        self._initialize(days, phase_scales, scales)
        self.scale_provenance_ = {"fit_split": "train", "phase_scales": phase_scales.tolist(),
                                  "phase_scale_sources": phase_sources, "increment_scale_details": scale_details,
                                  "cells_origin_endpoint_days": self.cells_}
        normalized = reports / phase_scales
        if not np.isfinite(normalized[np.isfinite(reports)]).all():
            raise ValueError("Training normalization exceeded floating-point support")
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        y, times = tensor(normalized), tensor(days)
        covariates_tensor = tensor(covariate_features) if self.covariate_feature_count_ else None
        native_reports = torch.as_tensor(reports, dtype=torch.float64, device=self.device)
        cell_scales, count_weights = tensor(scales / phase_scales), tensor(len(reports) / counts)
        if (bool(torch.isinf(y).any()) or not bool(torch.isfinite(cell_scales).all())
                or bool((cell_scales <= 0).any())):
            raise ValueError("Training inputs cannot be represented in float32")
        measured = torch.isfinite(y)
        context_rng = np.random.default_rng(self.seed + 101)
        mask_rng = np.random.default_rng(self.seed + 1009)
        latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 2017)
        report_rng = torch.Generator(device=self.device).manual_seed(self.seed + 3001)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        history = []
        total_updates = 0
        context_hashes = []
        per_origin_count = np.array([int((days > s).sum()) for s in self.origins_])
        mean_count = float(per_origin_count.mean())
        self.train()
        for epoch in range(self.epochs):
            batches = []
            for oi in range(len(self.origins_)):
                order = context_rng.permutation(len(reports))
                batches.extend((oi, order[start:start+self.batch_size]) for start in range(0, len(reports), self.batch_size))
            totals = np.zeros(4)
            clipped, largest_gradient = 0, 0.0
            digest = hashlib.sha256()
            for bi in context_rng.permutation(len(batches)):
                oi, rows = batches[bi]
                origin = self.origins_[oi]
                selected = torch.as_tensor(rows, device=self.device)
                actual = y[selected]
                observed = measured[selected] & (times <= origin)[None, :, None]
                keep_report = mask_rng.random((len(rows), len(days), 3)) >= self.history_dropout
                keep_phase = mask_rng.random((len(rows), 1, 3)) >= self.phase_dropout
                observed &= torch.as_tensor(keep_report & keep_phase, device=self.device)
                observed[:, 0] = True
                digest.update(np.asarray([oi], dtype=np.int64).tobytes())
                digest.update(rows.astype(np.int64).tobytes())
                digest.update(observed.detach().cpu().numpy().tobytes())
                context = self._encoder_context(actual, observed, times, origin,
                    native_reports=native_reports[selected],
                    covariates=None if covariates_tensor is None else covariates_tensor[selected])
                mean, lower = self.network(context)
                epsilon = torch.randn((len(rows), self.train_samples, self.latent_size),
                                      generator=latent_rng, dtype=torch.float32, device=self.device)
                z = mean[:, None] + torch.einsum("nij,nsj->nsi", lower, epsilon)
                cumulative = self.rate_model.cumulative(z, times)
                context_reports = torch.where(observed, actual, torch.full_like(actual, float("nan")))
                draws = sample_accumulated_reports(cumulative, context_reports, times, self.noise_abs,
                    self.noise_rel, cutoff=float(origin), generator=report_rng)
                cell = [i for i, (s, _) in enumerate(self.cells_) if s == origin]
                marginal, energy, dimension = _u_scores(actual, draws, observed, times, indices[float(origin)],
                                                        cell_scales[cell], count_weights[cell])
                marginal = marginal * (per_origin_count[oi] / mean_count)
                penalty = 0.5 * self.noise_regularization * (
                    torch.log(self.noise_abs / self.initial_noise_abs).square().sum()
                    + torch.log(self.noise_rel / self.initial_noise_rel).square().sum())
                loss = (marginal + energy).mean() + penalty
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite conditional proper-score objective at epoch {epoch+1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                norm = float(gradient)
                clipped += norm > self.gradient_clip
                largest_gradient = max(largest_gradient, norm)
                totals += np.array([float(loss.detach()), float(marginal.mean().detach()),
                                    float(energy.mean().detach()), float(penalty.detach())]) * len(rows)
                total_updates += 1
            totals /= len(reports) * len(self.origins_)
            entry = dict(epoch=epoch+1, loss=float(totals[0]), macro_increment_crps=float(totals[1]),
                         joint_energy=float(totals[2]), noise_penalty=float(totals[3]),
                         max_gradient_norm=largest_gradient, clipped_updates=int(clipped))
            history.append(entry)
            context_hashes.append(digest.hexdigest())
            if self.log_every and ((epoch+1) % self.log_every == 0 or epoch+1 == self.epochs):
                print(json.dumps({"event": "conditional_rate_epoch", "family": self.rate_family,
                                  "seed": self.seed, **entry}), flush=True)
        self.eval()
        self.fit_diagnostics_ = {
            "interpretation": INTERPRETATION, "fit_split": "train", "training_wells": len(reports),
            "origins_days": self.origins_.tolist(), "cells_origin_endpoint_days": self.cells_,
            "increment_scales_native": scales.tolist(), "increment_observation_counts": counts.tolist(),
            "phase_scales_native": phase_scales.tolist(), "phase_order": ["oil", "gas", "water"],
            "epoch_semantics": "every_training_well_at_every_origin_once",
            "contexts_per_epoch": len(reports)*len(self.origins_), "optimizer_updates": total_updates,
            "total_training_contexts": len(reports)*len(self.origins_)*self.epochs,
            "context_mask_sha256_by_epoch": context_hashes,
            "history_rate_features": self.history_rate_features,
            "initial_parameter_sha256": self.initial_parameter_sha256_,
            "rng_seeds": {"network": self.seed, "context_order": self.seed+101, "masks": self.seed+1009,
                          "latent": self.seed+2017, "report": self.seed+3001},
            "objective": "macro_scaled_increment_crps_u + mean_origin_report_energy_u/sqrt(d)",
            "score_coefficients": [1.0, 1.0], "train_samples_iid": self.train_samples,
            "uniform_origin_marginal_weights": (per_origin_count/mean_count).tolist(),
            "missing_target_weighting": "inverse frozen training observation fraction per cell/phase",
            "noise_abs_scaled": self.noise_abs.detach().cpu().tolist(),
            "noise_rel": self.noise_rel.detach().cpu().tolist(), "optimization": history,
            "configuration": self.configuration,
            "covariate_preprocessing": self.covariate_preprocessing,
            "covariate_count": self.n_covariates_, "covariate_feature_count": self.covariate_feature_count_,
        }
        return self

    def _history(self, history, origin):
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the conditional rate model before forecasting")
        reports = np.asarray(history, dtype=float)
        days = self.report_days.cpu().numpy()
        if reports.ndim != 3 or not len(reports) or reports.shape[1:] != (len(days), 3):
            raise ValueError("History must have shape [nonempty well,report day,3]")
        if origin not in self.origins_:
            raise ValueError("Forecast origin was not included in training")
        reports = reports.copy()
        reports[:, days > origin] = np.nan  # Before any value validation or feature construction.
        if np.isinf(reports).any():
            raise ValueError("Available history must be finite or NaN, never infinite")
        if np.any(np.isfinite(reports[:, 0]) & (reports[:, 0] != 0)):
            raise ValueError("Day zero is deterministic zero")
        reports[:, 0] = 0.0
        return reports

    def conditional_parameters(self, history, origin, covariates=None):
        """Gaussian mean and lower factor of the latent log parameters given the cutoff history."""
        reports = self._history(history, origin)
        covariate_features = self._covariates(covariates, len(reports), fit=False)
        normalized = reports / self.volume_scale.cpu().numpy()
        if not np.isfinite(normalized[np.isfinite(reports)]).all():
            raise ValueError("History normalization is not representable")
        values = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        if bool(torch.isinf(values).any()):
            raise ValueError("History is not representable in float32")
        times = self.report_days.to(torch.float32)
        native_reports = torch.as_tensor(reports, dtype=torch.float64, device=self.device)
        covariates_tensor = (torch.as_tensor(covariate_features, dtype=torch.float32, device=self.device)
                             if self.covariate_feature_count_ else None)
        with torch.no_grad():
            context = self._encoder_context(values, torch.isfinite(values), times, origin,
                                            native_reports=native_reports, covariates=covariates_tensor)
            mean, lower = self.network(context)
        if not bool(torch.isfinite(mean).all() & torch.isfinite(lower).all()):
            raise FloatingPointError("Nonfinite conditional Gaussian parameters")
        return mean.double(), lower.double()

    def evaluate_trajectories(self, latent_parameters, query_days):
        """Latent cumulative volume, rate and total volume of fixed draws, in native units."""
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the model before querying trajectories")
        z = (latent_parameters.to(dtype=torch.float64, device=self.device) if isinstance(latent_parameters, torch.Tensor)
             else torch.tensor(latent_parameters, dtype=torch.float64, device=self.device))
        days = torch.as_tensor(query_days, dtype=torch.float64, device=self.device)
        if (z.shape[-1] != self.latent_size or days.ndim != 1 or not bool(torch.isfinite(days).all())
                or bool((days < 0).any()) or not bool(torch.isfinite(z).all())):
            raise ValueError("Expected finite latent parameters and finite nonnegative query days")
        with torch.no_grad():
            result = {"latent_cumulative": self.rate_model.cumulative(z, days)*self.volume_scale,
                      "latent_rate": self.rate_model.rate(z, days)*self.volume_scale,
                      "latent_total_volume": self.rate_model.total_volume(z)*self.volume_scale}
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError("Latent trajectory query exceeded floating-point support; no clipping")
        return {key: value.cpu().numpy() for key, value in result.items()}

    def predict_distribution(self, history, origin, n_samples, rng, covariates=None):
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        reports = self._history(history, origin)
        mean, lower = self.conditional_parameters(reports, origin, covariates)
        epsilon = torch.as_tensor(rng.standard_normal((len(reports), n_samples, self.latent_size)),
                                  dtype=torch.float64, device=self.device)
        z = mean[:, None] + torch.einsum("nij,nsj->nsi", lower, epsilon)
        paths = self.evaluate_trajectories(z, self.report_days)
        q = torch.as_tensor(paths["latent_cumulative"], dtype=torch.float64, device=self.device) / self.volume_scale
        normalized = torch.as_tensor(reports, dtype=torch.float64, device=self.device) / self.volume_scale
        normals = torch.as_tensor(rng.standard_normal(q.shape), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            predicted = sample_accumulated_reports(q, normalized, self.report_days, self.noise_abs.double(),
                self.noise_rel.double(), cutoff=float(origin), standard_normals=normals) * self.volume_scale
        report = predicted.cpu().numpy()
        # Restore original native float64 observations, avoiding scaling roundoff.
        report = np.where(np.isfinite(reports)[:, None], reports[:, None], report)
        if not np.isfinite(report).all():
            raise FloatingPointError("Native report conversion exceeded floating-point support; no clipping")
        return {**paths, "latent_parameters": z.cpu().numpy(), "report_cumulative": report,
                "interpretation": INTERPRETATION, "phase_order": ["oil", "gas", "water"]}

    def predict_samples(self, history, origin, n_samples, rng, covariates=None):
        return self.predict_distribution(history, origin, n_samples, rng, covariates)["report_cumulative"]

    def save(self, path):
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the model before saving")
        torch.save({"schema_version": 1, "configuration": self.configuration,
                    "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
                    "origins": self.origins_.tolist(), "cells": self.cells_,
                    "fit_diagnostics": self.fit_diagnostics_, "scale_provenance": self.scale_provenance_,
                    "covariate_metadata": {"n_covariates": self.n_covariates_,
                                           "feature_count": self.covariate_feature_count_,
                                           "center": [], "scale": []}}, Path(path))

    @classmethod
    def load(cls, path, *, device="cpu"):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] != 1:
            raise ValueError("Unsupported conditional rate checkpoint")
        model = cls(**dict(checkpoint["configuration"], device=device))
        metadata = checkpoint["covariate_metadata"]
        n, width = metadata["n_covariates"], metadata["feature_count"]
        if (isinstance(n, bool) or not isinstance(n, int) or n < 0 or width != n
                or len(metadata["center"]) or len(metadata["scale"])):
            raise ValueError("Invalid conditional rate checkpoint covariate metadata")
        model.n_covariates_ = model.covariate_feature_count_ = n
        state = checkpoint["state_dict"]
        if state["network.hidden.0.weight"].shape[1] != 8 * len(state["report_days"]) + 1 + 6 + n:
            raise ValueError("Conditional rate checkpoint network and covariate dimensions disagree")
        model._initialize(state["report_days"], state["volume_scale"], state["increment_scales"])
        model.load_state_dict(state)
        model.origins_ = np.asarray(checkpoint["origins"])
        model.cells_ = checkpoint["cells"]
        model.fit_diagnostics_ = checkpoint["fit_diagnostics"]
        model.scale_provenance_ = checkpoint["scale_provenance"]
        model.eval()
        return model
