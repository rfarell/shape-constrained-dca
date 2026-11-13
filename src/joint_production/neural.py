"""Contextual joint Gaussian *report* comparator.

This is a conditional predictive distribution over the canonical cumulative
reports, not a latent positive-rate model or a Bayesian parameter posterior.
A shared neural network maps cutoff-available endpoints, their phase masks,
the origin and the time grid to a low-rank-plus-diagonal Gaussian distribution
of ``asinh(report / training_phase_scale)``. Joint normal draws followed by
``sinh`` give signed reports with finite moments of every order. Every derived
increment must use the same report draw. Different origins define separate
conditional predictions; sequential Bayesian consistency is not claimed.

Phase order is oil/gas/water.

Training averages conditional log scores over well/origin contexts. Each
unobserved but measured endpoint enters a context's marginal likelihood once;
overlapping increments are never independent training targets. Random history
and whole-phase masks teach joint reconstruction of missing past reports as
well as forecasting. Repeated origins are conditional training tasks, not
independent observations in a full-data generative likelihood. All observed
past reports are fixed when sampling. Unobserved past and future reports are
sampled together. Day zero is a deterministic zero accounting origin, excluded
from the likelihood and history masking. Timestamps remain explicit. Static covariates are
supplied by the caller, already encoded on training data.
"""

from __future__ import annotations

import math
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .history_features import last_interval_rate_features


def _masked_gaussian_nll(
    target: torch.Tensor,
    location: torch.Tensor,
    diagonal_variance: torch.Tensor,
    factor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Exact marginal NLL, Woodbury implementation; absent coordinates vanish.

    ``target`` must be finite (missing values are filled before this function).
    Output is one NLL per context, zero for contexts with no targets.
    """
    # The small covariance algebra needs double precision near the variance
    # floor: rank-deficient factors can otherwise corrupt factor gradients.
    # Gradients still flow to the network's float32 parameters through casts.
    if target.dtype in (torch.float16, torch.bfloat16, torch.float32):
        target = target.to(torch.float64)
        location = location.to(torch.float64)
        diagonal_variance = diagonal_variance.to(torch.float64)
        factor = factor.to(torch.float64)
    weight = mask.to(target.dtype) / diagonal_variance
    residual = target - location
    weighted_factor = factor * weight.unsqueeze(-1)
    core = factor.transpose(-2, -1) @ weighted_factor
    identity = torch.eye(factor.shape[-1], dtype=target.dtype, device=target.device)
    cholesky = torch.linalg.cholesky(core + identity)
    projection = (weighted_factor.transpose(-2, -1) @ residual.unsqueeze(-1))
    solved = torch.cholesky_solve(projection, cholesky)
    # Equivalent to Woodbury subtraction but avoids cancellation when small
    # diagonal noise accompanies residuals aligned with the shared factors.
    posterior_residual = residual - (factor @ solved).squeeze(-1)
    quadratic = (posterior_residual.square() * weight).sum(-1) + solved.square().sum((-2, -1))
    logdet = (torch.log(diagonal_variance) * mask).sum(-1)
    logdet = logdet + 2 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
    dimension = mask.sum(-1).to(target.dtype)
    return 0.5 * (quadratic + logdet + dimension * math.log(2 * math.pi))


class _JointReportNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, rank: int,
                 mean_initial: np.ndarray, min_std: float, max_std: float,
                 max_factor: float):
        super().__init__()
        self.rank = rank
        self.output_dim = output_dim
        self.min_std, self.max_std, self.max_factor = min_std, max_std, max_factor
        self.hidden = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.mean = nn.Linear(width, output_dim)
        self.std = nn.Linear(width, output_dim)
        self.factor = nn.Linear(width, output_dim * rank)
        nn.init.normal_(self.mean.weight, std=0.01)
        with torch.no_grad():
            self.mean.bias.copy_(torch.as_tensor(mean_initial, dtype=torch.float32))
        nn.init.normal_(self.std.weight, std=0.005)
        initial_std = min(max(0.4, min_std + 0.01 * (max_std - min_std)),
                          max_std - 0.01 * (max_std - min_std))
        std_fraction = (initial_std - min_std) / (max_std - min_std)
        nn.init.constant_(self.std.bias, math.log(std_fraction / (1 - std_fraction)))
        # Nonzero factor weights matter: exactly zero factors have zero gradient.
        nn.init.normal_(self.factor.weight, std=0.01)
        nn.init.zeros_(self.factor.bias)

    def forward(self, context: torch.Tensor):
        hidden = self.hidden(context)
        location = self.mean(hidden)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(self.std(hidden))
        factor = self.max_factor * torch.tanh(self.factor(hidden)).reshape(
            -1, self.output_dim, self.rank)
        return location, std.square(), factor


class ContextualGaussianNeuralBaseline:
    """Conditional joint Gaussian in asinh reports.

    ``fit`` consumes only a training split. Its ``origins`` are report-day
    values. ``predict_samples`` returns ``[well, sample, endpoint, phase]`` in
    native report units, including joint draws for missing historical phases.
    The diagonal standard deviation and factor entries are smoothly bounded
    in transformed units for stable optimization; these bounds are part of
    the model family. There is no sample clipping or positivity projection.

    Covariates are an already preprocessed finite matrix, fitted on training
    data only; feature availability at the forecast cutoff is the caller's
    responsibility. The context also carries the latest-interval report rates.
    ``history_rate_features`` and ``covariate_preprocessing`` have one supported
    value each and are kept because saved configurations record them.

    ``fit_diagnostics_`` records per-epoch native-report NLL (including the
    asinh Jacobian), train-only scales, masking, optimization and device.
    No validation or test observations are accepted by the fitting API.
    """

    def __init__(self, *, rank: int = 4, hidden_width: int = 128,
                 epochs: int = 150, batch_size: int = 1024,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 seed: int = 42, device: str = "cpu", history_dropout: float = 0.1,
                 phase_dropout: float = 0.1, min_std: float = 0.025,
                 max_std: float = 3.0, max_factor: float = 1.0,
                 gradient_clip: float = 10.0, log_every: int = 25,
                 history_rate_features: str = "last_interval",
                 covariate_preprocessing: str = "external"):
        if history_rate_features != "last_interval" or covariate_preprocessing != "external":
            raise ValueError("Only last_interval history features and external covariates are supported")
        if min(rank, hidden_width, epochs, batch_size) < 1:
            raise ValueError("rank, hidden_width, epochs and batch_size must be positive")
        if learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= history_dropout < 1 or not 0 <= phase_dropout < 1:
            raise ValueError("Masking probabilities must lie in [0, 1)")
        if not 0 < min_std < max_std or max_factor <= 0:
            raise ValueError("Require 0 < min_std < max_std and max_factor > 0")
        self.rank, self.hidden_width = int(rank), int(hidden_width)
        self.epochs, self.batch_size = int(epochs), int(batch_size)
        self.learning_rate, self.weight_decay = float(learning_rate), float(weight_decay)
        self.seed, self.device = int(seed), str(device)
        self.history_dropout, self.phase_dropout = float(history_dropout), float(phase_dropout)
        self.min_std, self.max_std, self.max_factor = min_std, max_std, max_factor
        self.gradient_clip = float(gradient_clip)
        if log_every < 0:
            raise ValueError("log_every must be nonnegative; zero disables progress output")
        self.log_every = int(log_every)
        self.history_rate_features = history_rate_features
        self.covariate_preprocessing = covariate_preprocessing

    @staticmethod
    def _reports(reports: np.ndarray) -> np.ndarray:
        reports = np.asarray(reports, dtype=np.float64)
        if reports.ndim != 3 or reports.shape[-1] != 3 or not len(reports):
            raise ValueError("reports must have shape [nonempty well, endpoint, 3]")
        if np.isinf(reports).any():
            raise ValueError("Reports may contain NaN for missingness, never infinity")
        return reports

    def _covariates(self, covariates: np.ndarray | None, n: int, *, fit: bool) -> np.ndarray:
        if covariates is None:
            if fit:
                self.n_covariates_ = 0
            elif self.n_covariates_:
                raise ValueError("This fitted model requires cutoff-available covariates")
            return np.empty((n, 0), dtype=np.float32)
        covariates = np.asarray(covariates, dtype=np.float64)
        if (covariates.ndim != 2 or covariates.shape[0] != n
                or not np.isfinite(covariates).all()):
            raise ValueError("Covariates must have shape [well, feature] and be finite")
        if (np.abs(covariates) > np.finfo(np.float32).max).any():
            raise ValueError("Covariates must be representable as finite float32")
        if fit:
            self.n_covariates_ = covariates.shape[1]
        elif covariates.shape[1] != self.n_covariates_:
            raise ValueError("Covariate dimension differs from training")
        return covariates.astype(np.float32)

    def fit(self, train_reports: np.ndarray, report_days: np.ndarray,
            origins: list[float] | None = None, covariates: np.ndarray | None = None):
        reports = self._reports(train_reports)
        days = np.asarray(report_days, dtype=np.float64)
        if (days.ndim != 1 or len(days) != reports.shape[1] or len(days) < 2
                or not np.isfinite(days).all() or (np.diff(days) <= 0).any()
                or days[0] < 0):
            raise ValueError("report_days must be nonnegative, finite, increasing, and match reports")
        self.report_days_ = days.copy()
        reports = reports.copy()
        zero_days = days == 0
        if np.any(np.isfinite(reports[:, zero_days]) & (reports[:, zero_days] != 0)):
            raise ValueError("Day-zero reports must equal the deterministic zero accounting origin")
        reports[:, zero_days] = 0.0
        self.origins_ = np.asarray(days[1:-1] if origins is None else origins, dtype=float)
        if (self.origins_.ndim != 1 or not len(self.origins_)
                or not np.isin(self.origins_, days[:-1]).all()
                or len(np.unique(self.origins_)) != len(self.origins_)):
            raise ValueError("origins must be distinct canonical report days with a future endpoint")
        self.phase_scales_ = np.array([
            np.median(np.abs(reports[:, :, k][np.isfinite(reports[:, :, k]) & (reports[:, :, k] != 0)]))
            if np.any(np.isfinite(reports[:, :, k]) & (reports[:, :, k] != 0)) else 1.0
            for k in range(3)])
        valid = np.isfinite(reports)
        transformed = np.arcsinh(np.where(valid, reports, 0) / self.phase_scales_)
        if not np.isfinite(transformed).all():
            raise ValueError("Reports cannot be represented after phase scaling")
        covariate_features = self._covariates(covariates, len(reports), fit=True)
        dimension = len(days) * 3
        flat_target = transformed.reshape(len(reports), dimension)
        flat_valid = valid.reshape(len(reports), dimension)
        counts = flat_valid.sum(0)
        if (counts == 0).any():
            raise ValueError("Every endpoint/phase requires at least one measured training report")
        target_initial = np.sum(flat_target, axis=0) / counts
        # d(asinh(y/s))/dy = 1/sqrt(s**2 + y**2), evaluated stably.
        log_jacobian_inverse = np.log(np.hypot(np.where(valid, reports, 0), self.phase_scales_))
        self.torch_device_ = torch.device(self.device)
        if self.torch_device_.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        context_dim = 2 * dimension + 2 * len(days) + 1 + covariate_features.shape[1] + 6
        cuda_devices = ([self.torch_device_.index or torch.cuda.current_device()]
                        if self.torch_device_.type == "cuda" else [])
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            self.network_ = _JointReportNetwork(
                context_dim, dimension, self.hidden_width, self.rank, target_initial,
                self.min_std, self.max_std, self.max_factor).to(self.torch_device_)
        initial_digest = hashlib.sha256()
        for name, parameter in self.network_.named_parameters():
            value = parameter.detach().cpu().contiguous()
            initial_digest.update(name.encode())
            initial_digest.update(str(value.dtype).encode())
            initial_digest.update(str(tuple(value.shape)).encode())
            initial_digest.update(value.numpy().tobytes())
        tensor = lambda array: torch.as_tensor(array, dtype=torch.float32, device=self.torch_device_)
        y = tensor(flat_target)
        native = torch.as_tensor(reports, dtype=torch.float64, device=self.torch_device_)
        native_origins = torch.as_tensor(self.origins_, dtype=torch.float64, device=self.torch_device_)
        measured = torch.as_tensor(flat_valid, dtype=torch.bool, device=self.torch_device_)
        jacobian = tensor(log_jacobian_inverse.reshape(len(reports), dimension))
        covariates_tensor = tensor(covariate_features)
        normalized_days = tensor(days / days[-1])
        origin_values = tensor(self.origins_ / days[-1])
        allowed = torch.as_tensor(
            np.repeat(days[None, :] <= self.origins_[:, None], 3, axis=1),
            device=self.torch_device_, dtype=torch.bool)
        deterministic_zero = torch.as_tensor(np.repeat(zero_days, 3),
                                              device=self.torch_device_, dtype=torch.bool)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        generator = np.random.default_rng(self.seed)
        contexts_count = len(reports) * len(self.origins_)
        losses: list[float] = []
        gradient_norms: list[float] = []
        context_hashes: list[str] = []
        total_updates = 0
        self.network_.train()
        for epoch in range(self.epochs):
            context_digest = hashlib.sha256()
            order = generator.permutation(contexts_count)
            total_loss, active_count, largest_gradient = 0.0, 0, 0.0
            for start in range(0, contexts_count, self.batch_size):
                indices = order[start:start + self.batch_size]
                row = torch.as_tensor(indices // len(self.origins_), device=self.torch_device_)
                origin = torch.as_tensor(indices % len(self.origins_), device=self.torch_device_)
                observed = measured[row] & allowed[origin]
                if self.history_dropout or self.phase_dropout:
                    keep_report = generator.random((len(indices), dimension)) >= self.history_dropout
                    keep_phase = np.tile(generator.random((len(indices), 3)) >= self.phase_dropout,
                                         (1, len(days)))
                    observed = observed & torch.as_tensor(keep_report & keep_phase, device=self.torch_device_)
                observed = observed | deterministic_zero
                context_digest.update(indices.astype(np.int64).tobytes())
                context_digest.update(observed.detach().cpu().numpy().tobytes())
                target_mask = measured[row] & ~observed
                active = target_mask.any(-1)
                if not bool(active.any()):
                    continue
                context = self._torch_context(y[row], observed, origin_values[origin],
                                              normalized_days, covariates_tensor[row])
                context = self._append_rate_features(context, native[row],
                    observed.reshape(-1, len(days), 3), native_origins[origin])
                location, diagonal_variance, factor = self.network_(context)
                nll = _masked_gaussian_nll(y[row], location, diagonal_variance, factor, target_mask)
                nll = nll + (jacobian[row] * target_mask).sum(-1)
                loss = nll[active].mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite neural training NLL at epoch {epoch + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.network_.parameters(), self.gradient_clip,
                                                               error_if_nonfinite=True)
                optimizer.step()
                total_updates += 1
                count = int(active.sum())
                total_loss += float(loss.detach()) * count
                active_count += count
                largest_gradient = max(largest_gradient, float(gradient_norm))
            if not active_count:
                raise ValueError("No measured, unobserved endpoints are available for neural training")
            losses.append(total_loss / active_count)
            gradient_norms.append(largest_gradient)
            context_hashes.append(context_digest.hexdigest())
            if self.log_every and ((epoch + 1) % self.log_every == 0 or epoch + 1 == self.epochs):
                print(json.dumps({"event": "neural_epoch", "epoch": epoch + 1,
                                  "epochs": self.epochs, "seed": self.seed,
                                  "conditional_report_nll": losses[-1],
                                  "max_gradient_norm": largest_gradient}), flush=True)
        self.network_.eval()
        self.fit_diagnostics_: dict[str, Any] = {
            "model": "contextual_joint_asinh_gaussian_report_baseline",
            "phase_order": ["oil", "gas", "water"], "fit_split": "train",
            "training_wells": len(reports), "contexts_per_epoch": contexts_count,
            "origins_days": self.origins_.tolist(), "phase_scales": self.phase_scales_.tolist(),
            "seed": self.seed, "device": str(self.torch_device_), "epochs": self.epochs,
            "rank": self.rank, "hidden_width": self.hidden_width, "batch_size": self.batch_size,
            "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
            "history_dropout": self.history_dropout, "phase_dropout": self.phase_dropout,
            "min_transformed_std": self.min_std, "max_transformed_std": self.max_std,
            "max_transformed_factor": self.max_factor,
            "network_dtype": "float32", "covariance_algebra_dtype": "float64",
            "conditional_report_nll_by_epoch": losses,
            "max_gradient_norm_before_clipping_by_epoch": gradient_norms,
            "nll_definition": "mean native-report joint conditional NLL per nonempty context",
            "missing_history": "jointly sampled; trained with constructed history/phase masks",
            "day_zero": "deterministic zero; excluded from masks and residual likelihood",
            "covariate_count": self.n_covariates_,
            "history_rate_features": self.history_rate_features,
            "initial_parameter_sha256": initial_digest.hexdigest(),
            "context_mask_sha256_by_epoch": context_hashes,
            "optimizer_updates": total_updates,
            "total_training_contexts": contexts_count * self.epochs,
            "covariate_preprocessing": self.covariate_preprocessing,
            "covariate_feature_count": self.n_covariates_,
        }
        return self

    def save(self, path: str | Path) -> None:
        """Write a portable CPU tensor checkpoint with training preprocessing."""
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the model before saving")
        names = ("rank", "hidden_width", "epochs", "batch_size", "learning_rate",
                 "weight_decay", "seed", "device", "history_dropout", "phase_dropout",
                 "min_std", "max_std", "max_factor", "gradient_clip", "log_every",
                 "history_rate_features", "covariate_preprocessing")
        metadata = {
            "report_days": self.report_days_.tolist(), "origins": self.origins_.tolist(),
            "phase_scales": self.phase_scales_.tolist(), "n_covariates": self.n_covariates_,
        }
        torch.save({"format_version": 1,
                    "configuration": {name: getattr(self, name) for name in names},
                    "metadata": metadata, "fit_diagnostics": self.fit_diagnostics_,
                    "state_dict": {name: value.detach().cpu() for name, value in self.network_.state_dict().items()}},
                   Path(path))

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu"):
        """Restore a saved conditional law; default inference uses the CPU."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError("Unsupported neural checkpoint format")
        configuration = dict(checkpoint["configuration"])
        configuration["device"] = device
        model = cls(**configuration)
        metadata = checkpoint["metadata"]
        model.report_days_ = np.array(metadata["report_days"], dtype=float)
        model.origins_ = np.array(metadata["origins"], dtype=float)
        model.phase_scales_ = np.array(metadata["phase_scales"], dtype=float)
        model.n_covariates_ = metadata["n_covariates"]
        if (isinstance(model.n_covariates_, bool) or not isinstance(model.n_covariates_, int)
                or model.n_covariates_ < 0):
            raise ValueError("Invalid neural checkpoint covariate count")
        model.torch_device_ = torch.device(device)
        state = checkpoint["state_dict"]
        if state["hidden.0.weight"].shape[1] != 8 * len(model.report_days_) + 1 + model.n_covariates_ + 6:
            raise ValueError("Neural checkpoint network width disagrees with covariate metadata")
        with torch.random.fork_rng(devices=[]):
            model.network_ = _JointReportNetwork(
                state["hidden.0.weight"].shape[1], state["mean.bias"].shape[0],
                model.hidden_width, model.rank, state["mean.bias"].numpy(),
                model.min_std, model.max_std, model.max_factor)
        model.network_.load_state_dict(state)
        model.network_.to(model.torch_device_).eval()
        model.fit_diagnostics_ = checkpoint["fit_diagnostics"]
        return model

    @staticmethod
    def _torch_context(target, observed, normalized_origin, normalized_days, covariates):
        n = len(target)
        day_grid = normalized_days.expand(n, -1)
        return torch.cat((target * observed, observed.to(target.dtype),
                          normalized_origin[:, None], day_grid,
                          day_grid - normalized_origin[:, None], covariates), dim=-1)

    def _append_rate_features(self, context, native_reports, observed, cutoff):
        """Append deterministic report-rate summaries after actual history masking."""
        extra = last_interval_rate_features(native_reports, observed,
            torch.as_tensor(self.report_days_, dtype=torch.float64, device=context.device),
            cutoff, torch.as_tensor(self.phase_scales_, dtype=torch.float64, device=context.device),
            dtype=context.dtype, zero_control=False)
        return torch.cat((context, extra), dim=-1)

    def predict_transformed_parameters(self, masked_history: np.ndarray, origin: float,
                                       covariates: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Return network parameters for unobserved coordinates; history is fixed.

        Parameters have flattened time-major, then oil/gas/water coordinates.
        The covariance is ``diag(variance) + F F.T``. Parameters at observed
        coordinates are unused, not conditional variance.
        """
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the model before forecasting")
        reports = np.asarray(masked_history, dtype=np.float64)
        if reports.ndim != 3 or not len(reports) or reports.shape[1:] != (len(self.report_days_), 3):
            raise ValueError("Prediction endpoint grid differs from training")
        if origin not in self.origins_:
            raise ValueError("Prediction origin was not included in training")
        reports = reports.copy()
        reports[:, self.report_days_ > origin] = np.nan
        reports = self._reports(reports)
        zero_days = self.report_days_ == 0
        if np.any(np.isfinite(reports[:, zero_days]) & (reports[:, zero_days] != 0)):
            raise ValueError("Day-zero reports must equal the deterministic zero accounting origin")
        reports[:, zero_days] = 0.0
        observed = np.isfinite(reports) & (self.report_days_[None, :, None] <= origin)
        values = np.arcsinh(np.where(observed, reports, 0) / self.phase_scales_)
        if not np.isfinite(values).all():
            raise ValueError("History cannot be represented after phase scaling")
        covariate_features = self._covariates(covariates, len(reports), fit=False)
        tensor = lambda array: torch.as_tensor(array, dtype=torch.float32, device=self.torch_device_)
        outputs = {"location": [], "diagonal_variance": [], "factor": []}
        with torch.no_grad():
            for start in range(0, len(reports), self.batch_size):
                stop = start + self.batch_size
                count = len(reports[start:stop])
                context = self._torch_context(
                    tensor(values[start:stop].reshape(count, -1)),
                    tensor(observed[start:stop].reshape(count, -1)),
                    tensor(np.full(count, origin / self.report_days_[-1])),
                    tensor(self.report_days_ / self.report_days_[-1]),
                    tensor(covariate_features[start:stop]))
                context = self._append_rate_features(context,
                    torch.as_tensor(reports[start:stop], dtype=torch.float64, device=self.torch_device_),
                    torch.as_tensor(observed[start:stop], dtype=torch.bool, device=self.torch_device_),
                    origin)
                parameters = self.network_(context)
                for name, parameter in zip(outputs, parameters):
                    outputs[name].append(parameter.cpu().numpy().astype(np.float64))
        result = {name: np.concatenate(value) for name, value in outputs.items()}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Network produced nonfinite transformed Gaussian parameters")
        result["observed_mask"] = observed.reshape(len(reports), -1)
        return result

    def predict_samples(self, masked_history: np.ndarray, origin: float, n_samples: int,
                        rng: np.random.Generator, covariates: np.ndarray | None = None) -> np.ndarray:
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        parameters = self.predict_transformed_parameters(masked_history, origin, covariates)
        n, dimension = parameters["diagonal_variance"].shape
        output = np.empty((n, n_samples, len(self.report_days_), 3), dtype=np.float64)
        reports = np.asarray(masked_history, dtype=np.float64).copy()
        reports[:, self.report_days_ == 0] = 0.0
        # Chunking bounds temporary memory; the caller owns the supplied RNG.
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            count = stop - start
            shared = rng.standard_normal((count, n_samples, self.rank))
            independent = rng.standard_normal((count, n_samples, dimension))
            location = parameters["location"][start:stop, None, :]
            transformed = location + np.einsum(
                "nsr,ndr->nsd", shared, parameters["factor"][start:stop])
            transformed += np.sqrt(parameters["diagonal_variance"][start:stop, None, :]) * independent
            with np.errstate(over="raise", invalid="raise"):
                try:
                    samples = np.sinh(transformed.reshape(count, n_samples, len(self.report_days_), 3))
                    samples *= self.phase_scales_
                except FloatingPointError as error:
                    raise FloatingPointError("Gaussian report draw exceeds float64; no clipping was applied") from error
            observed = parameters["observed_mask"][start:stop].reshape(count, len(self.report_days_), 3)
            output[start:stop] = np.where(observed[:, None, :, :], reports[start:stop, None, :, :], samples)
        return output
