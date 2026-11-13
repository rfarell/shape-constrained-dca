"""Signed joint report baseline matched to the conditional-rate scoring protocol.

The network parameterizes Z | H ~ N(mu(H), F(H) F(H)' + diag(v(H))) and
Y = training_phase_scale * sinh(Z). Unobserved past and future reports share
one Gaussian draw; known reports and day zero are fixed. This is already a
conditional predictive law: no second Gaussian conditioning or latent-rate
interpretation applies. Finite Gaussian parameters give signed support and
finite moments of every order; numerical overflow is an error, never clipping.

The imported context and iid U-score functions are exactly those used by
ConditionalRateForecaster. Training scores phase-normalized ORIGINAL reports,
not transformed reports: macro scaled increment CRPS plus equal-origin joint
report energy / sqrt(d), each with coefficient one. Hidden origins are
subtracted from the same sampled report trajectory. Repeated origins are
conditional tasks, not independent observations in a generative likelihood.
The context/mask schedule, native training scales, Adam settings and epoch
semantics match the conditional-rate protocol. The predictive distribution
family differs, so this comparison alone cannot isolate a benefit of physics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .conditional_rates import ConditionalRateForecaster, _u_scores
from .metrics import fit_increment_scales
from .neural import _JointReportNetwork


INTERPRETATION = "conditional_joint_report_law_no_latent_rate"
_context = ConditionalRateForecaster._context


def _draw_reports(location, diagonal_variance, factor, actual, observed,
                  factor_normals, diagonal_normals):
    """Reparameterized iid joint draws in original phase-normalized units.

    Gaussian arrays flatten [time,phase]; standard normals have shape
    [well,sample,rank] and [well,sample,time*phase]. Mask known Gaussian
    coordinates before sinh, then restore their reports exactly. No Schur
    conditioning is applied to this directly conditional distribution.
    """
    n, dimension = location.shape
    count = factor_normals.shape[1]
    if (actual.shape != observed.shape or actual.shape != (n, dimension // 3, 3)
            or diagonal_variance.shape != location.shape or factor.shape[:2] != location.shape
            or factor_normals.shape != (n, count, factor.shape[-1])
            or diagonal_normals.shape != (n, count, dimension)):
        raise ValueError("Joint Gaussian draw shapes do not match report coordinates")
    if (not all(bool(torch.isfinite(value).all()) for value in
                (location, diagonal_variance, factor, factor_normals, diagonal_normals))
            or bool((diagonal_variance <= 0).any())):
        raise FloatingPointError("Conditional Gaussian parameters and normals must be finite with positive variance")
    transformed = (location[:, None] + torch.einsum("ndr,nsr->nsd", factor, factor_normals)
                   + diagonal_variance.sqrt()[:, None] * diagonal_normals)
    transformed = torch.where(observed.flatten(start_dim=1)[:, None],
                              torch.zeros_like(transformed), transformed)
    generated = torch.sinh(transformed).reshape(n, count, *actual.shape[1:])
    draws = torch.where(observed[:, None], actual[:, None], generated)
    if not bool(torch.isfinite(draws).all()):
        raise FloatingPointError("Joint report inverse transform exceeded floating-point support; no clipping")
    return draws


class ConditionalReportGaussianForecaster(nn.Module):
    """Report-only Gaussian baseline with the conditional-rate proper objective.

    One epoch visits every training well at every origin once. Independent
    context-order and mask RNGs match the rate models regardless of output
    dimensions. A shared factor normal and coordinate normals form each iid
    ensemble member. Covariance bounds apply smoothly to parameters, never
    sampled reports; the mean is unbounded. Volumes use oil BBL, gas MCF,
    water BBL, with phase order recorded at the array boundary.
    """

    def __init__(self, *, rank=4, hidden_width=128, epochs=60, batch_size=256,
                 train_samples=16, learning_rate=1e-3, seed=42, device="cpu",
                 history_dropout=0.1, phase_dropout=0.1, min_std=0.025,
                 max_std=3.0, max_factor=1.0, gradient_clip=10.0, log_every=5):
        super().__init__()
        if min(rank, hidden_width, epochs, batch_size) < 1 or train_samples < 2:
            raise ValueError("Positive dimensions and at least two iid training samples are required")
        if (not np.isfinite([learning_rate, min_std, max_std, max_factor, gradient_clip]).all()
                or learning_rate <= 0 or gradient_clip <= 0 or log_every < 0
                or not 0 < min_std < max_std or max_factor <= 0):
            raise ValueError("Invalid optimizer or Gaussian covariance bounds")
        if not 0 <= history_dropout < 1 or not 0 <= phase_dropout < 1:
            raise ValueError("Masking probabilities must be in [0,1)")
        self.configuration = dict(rank=int(rank), hidden_width=int(hidden_width), epochs=int(epochs),
            batch_size=int(batch_size), train_samples=int(train_samples), learning_rate=float(learning_rate),
            seed=int(seed), device=str(device), history_dropout=float(history_dropout),
            phase_dropout=float(phase_dropout), min_std=float(min_std), max_std=float(max_std),
            max_factor=float(max_factor), gradient_clip=float(gradient_clip), log_every=int(log_every))
        for key, value in self.configuration.items():
            if key != "device":
                setattr(self, key, value)
        self.network = None
        for name in ("report_days", "volume_scale", "increment_scales"):
            self.register_buffer(name, torch.empty(0, dtype=torch.float64, device=device))

    @property
    def device(self):
        return self.report_days.device

    def _initialize(self, days, phase_scales, increment_scales, mean_initial):
        self.report_days = torch.as_tensor(days, dtype=torch.float64, device=self.device)
        self.volume_scale = torch.as_tensor(phase_scales, dtype=torch.float64, device=self.device)
        self.increment_scales = torch.as_tensor(increment_scales, dtype=torch.float64, device=self.device)
        cuda_devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            self.network = _JointReportNetwork(2*len(days)*3 + 2*len(days) + 1, len(days)*3,
                self.hidden_width, self.rank, mean_initial, self.min_std, self.max_std,
                self.max_factor).to(self.device)

    def fit(self, train_reports, report_days, *, origins=None):
        reports = np.asarray(train_reports, dtype=np.float64)
        days = np.asarray(report_days, dtype=np.float64)
        if (reports.ndim != 3 or reports.shape[-1] != 3 or not all(reports.shape)
                or days.shape != (reports.shape[1],) or len(days) < 3
                or not np.isfinite(days).all() or days[0] != 0 or (np.diff(days) <= 0).any()
                or np.isinf(reports).any()):
            raise ValueError("Require increasing finite days starting at zero and reports [well,time,3], NaN allowed")
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
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = reports / phase_scales
        if not np.isfinite(normalized[np.isfinite(reports)]).all():
            raise ValueError("Training normalization exceeded floating-point support")
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        y, times = tensor(normalized), tensor(days)
        cell_scales, count_weights = tensor(scales / phase_scales), tensor(len(reports) / counts)
        if (bool(torch.isinf(y).any()) or not bool(torch.isfinite(cell_scales).all())
                or bool((cell_scales <= 0).any()) or not bool(torch.isfinite(times).all())
                or bool((torch.diff(times) <= 0).any()) or not bool(torch.isfinite(count_weights).all())):
            raise ValueError("Training inputs cannot be represented in float32")
        # Only training targets initialize this unconditional starting location;
        # subsequent contexts contain masked cutoff-available reports exclusively.
        transformed = np.arcsinh(normalized)
        finite = np.isfinite(transformed)
        mean_initial = np.where(finite, transformed, 0).sum(axis=0) / finite.sum(axis=0).clip(min=1)
        self._initialize(days, phase_scales, scales, mean_initial.ravel())
        self.scale_provenance_ = {"fit_split": "train", "phase_scales": phase_scales.tolist(),
            "phase_scale_sources": phase_sources, "increment_scale_details": scale_details,
            "cells_origin_endpoint_days": self.cells_}
        measured = torch.isfinite(y)
        context_rng = np.random.default_rng(self.seed + 101)
        mask_rng = np.random.default_rng(self.seed + 1009)
        factor_rng = torch.Generator(device=self.device).manual_seed(self.seed + 2017)
        diagonal_rng = torch.Generator(device=self.device).manual_seed(self.seed + 3001)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        history, context_hashes, total_updates = [], [], 0
        per_origin_count = np.array([int((days > s).sum()) for s in self.origins_])
        mean_count = float(per_origin_count.mean())
        self.train()
        for epoch in range(self.epochs):
            batches = []
            for oi in range(len(self.origins_)):
                order = context_rng.permutation(len(reports))
                batches.extend((oi, order[start:start+self.batch_size]) for start in range(0, len(reports), self.batch_size))
            totals = np.zeros(3)
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
                parameters = self.network(_context(actual, observed, times, origin))
                factor_normal = torch.randn((len(rows), self.train_samples, self.rank),
                    generator=factor_rng, dtype=torch.float32, device=self.device)
                diagonal_normal = torch.randn((len(rows), self.train_samples, len(days)*3),
                    generator=diagonal_rng, dtype=torch.float32, device=self.device)
                draws = _draw_reports(*parameters, actual, observed, factor_normal, diagonal_normal)
                cell = [i for i, (s, _) in enumerate(self.cells_) if s == origin]
                marginal, energy, _ = _u_scores(actual, draws, observed, times, indices[float(origin)],
                                              cell_scales[cell], count_weights[cell])
                marginal = marginal * (per_origin_count[oi] / mean_count)
                loss = (marginal + energy).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite conditional report proper-score objective at epoch {epoch+1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                norm = float(gradient)
                clipped += norm > self.gradient_clip
                largest_gradient = max(largest_gradient, norm)
                totals += np.array([float(loss.detach()), float(marginal.mean().detach()),
                                    float(energy.mean().detach())]) * len(rows)
                total_updates += 1
            totals /= len(reports) * len(self.origins_)
            entry = dict(epoch=epoch+1, loss=float(totals[0]), macro_increment_crps=float(totals[1]),
                joint_energy=float(totals[2]), noise_penalty=0.0,
                max_gradient_norm=largest_gradient, clipped_updates=int(clipped))
            history.append(entry)
            context_hashes.append(digest.hexdigest())
            if self.log_every and ((epoch+1) % self.log_every == 0 or epoch+1 == self.epochs):
                print(json.dumps({"event": "conditional_report_epoch", "seed": self.seed, **entry}), flush=True)
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
            "rng_seeds": {"network": self.seed, "context_order": self.seed+101, "masks": self.seed+1009,
                          "factor": self.seed+2017, "diagonal": self.seed+3001},
            "objective": "macro_scaled_increment_crps_u + mean_origin_report_energy_u/sqrt(d)",
            "score_coefficients": [1.0, 1.0], "train_samples_iid": self.train_samples,
            "uniform_origin_marginal_weights": (per_origin_count/mean_count).tolist(),
            "missing_target_weighting": "inverse frozen training observation fraction per cell/phase",
            "optimizer": "Adam", "weight_decay": 0.0, "optimization": history,
            "initial_location": "training_endpoint_mean_asinh_phase_normalized_report",
            "covariance_structure": "low_rank_plus_diagonal_in_asinh_reports",
            "configuration": self.configuration,
        }
        self.training_diagnostics_ = self.fit_diagnostics_
        return self

    def _history(self, history, origin):
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the conditional report model before forecasting")
        reports = np.asarray(history, dtype=np.float64)
        days = self.report_days.cpu().numpy()
        if reports.ndim != 3 or not len(reports) or reports.shape[1:] != (len(days), 3):
            raise ValueError("History must have shape [nonempty well,report day,3]")
        if origin not in self.origins_:
            raise ValueError("Forecast origin was not included in training")
        reports = reports.copy()
        reports[:, days > origin] = np.nan
        if np.isinf(reports).any():
            raise ValueError("Available history must be finite or NaN, never infinite")
        if np.any(np.isfinite(reports[:, 0]) & (reports[:, 0] != 0)):
            raise ValueError("Day zero is deterministic zero")
        reports[:, 0] = 0.0
        return reports

    def conditional_parameters(self, history, origin):
        reports = self._history(history, origin)
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = reports / self.volume_scale.cpu().numpy()
        if not np.isfinite(normalized[np.isfinite(reports)]).all():
            raise ValueError("History normalization is not representable")
        values = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        if bool(torch.isinf(values).any()):
            raise ValueError("History is not representable in float32")
        with torch.no_grad():
            parameters = self.network(_context(values, torch.isfinite(values), self.report_days.float(), origin))
        if not all(bool(torch.isfinite(value).all()) for value in parameters):
            raise FloatingPointError("Nonfinite conditional Gaussian parameters")
        return tuple(value.double() for value in parameters)

    def predict_distribution(self, history, origin, n_samples, rng):
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        reports = self._history(history, origin)
        parameters = self.conditional_parameters(reports, origin)
        actual = torch.as_tensor(reports, dtype=torch.float64, device=self.device) / self.volume_scale
        observed = torch.isfinite(actual)
        factor_normal = torch.as_tensor(rng.standard_normal((len(reports), n_samples, self.rank)),
                                        dtype=torch.float64, device=self.device)
        diagonal_normal = torch.as_tensor(rng.standard_normal((len(reports), n_samples, reports.shape[1]*3)),
                                          dtype=torch.float64, device=self.device)
        with torch.no_grad():
            draws = _draw_reports(*parameters, actual, observed, factor_normal, diagonal_normal)
            native = (draws*self.volume_scale).cpu().numpy()
        # Float64 original reports are authoritative, including signed corrections.
        native = np.where(np.isfinite(reports)[:, None], reports[:, None], native)
        if not np.isfinite(native).all():
            raise FloatingPointError("Report native-unit conversion exceeded floating-point support; no clipping")
        self.last_conditional_diagnostics_ = {
            "interpretation": INTERPRETATION, "phase_order": ["oil", "gas", "water"],
            "history_observed_counts_by_phase": (np.isfinite(reports[:, 1:])).sum(axis=1).tolist(),
            "unknown_reports_sampled_jointly": True, "observed_reports_restored_exactly": True,
            "cross_cutoff_bayesian_consistency": False, "latent_production_interpretation": False}
        return {"report_cumulative": native, "interpretation": INTERPRETATION,
                "phase_order": ["oil", "gas", "water"]}

    def predict_samples(self, history, origin, n_samples, rng):
        return self.predict_distribution(history, origin, n_samples, rng)["report_cumulative"]

    def save(self, path):
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the model before saving")
        torch.save({"schema_version": 1, "model_kind": "conditional_report", "configuration": self.configuration,
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
            "origins": self.origins_.tolist(), "cells": self.cells_, "fit_diagnostics": self.fit_diagnostics_,
            "scale_provenance": self.scale_provenance_}, Path(path))

    @classmethod
    def load(cls, path, *, device="cpu"):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] != 1 or checkpoint.get("model_kind") != "conditional_report":
            raise ValueError("Unsupported conditional report checkpoint")
        model = cls(**dict(checkpoint["configuration"], device=device))
        state = checkpoint["state_dict"]
        model._initialize(state["report_days"], state["volume_scale"], state["increment_scales"],
                          np.zeros(len(state["report_days"])*3))
        model.load_state_dict(state)
        model.origins_ = np.asarray(checkpoint["origins"])
        model.cells_ = checkpoint["cells"]
        model.fit_diagnostics_ = checkpoint["fit_diagnostics"]
        model.training_diagnostics_ = model.fit_diagnostics_
        model.scale_provenance_ = checkpoint["scale_provenance"]
        model.eval()
        return model
