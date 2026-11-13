"""Population DCA with an exact posterior under a finite empirical prior.

Array phase order is always oil, gas, water. Times are days, amplitudes are
native phase volume/day, and integrals are BBL, MCF, BBL respectively.

This intentionally small empirical-Bayes benchmark fits smooth latent curves
to training endpoints and freezes a seeded, smoothed parameter library. The
library is the *declared finite prior*, not an importance proposal for an
unavailable continuous prior. Categorical conditioning is exact for that
prior; finite-library resolution and residual-based dispersion estimation are
limitations, reported in the diagnostics. Residual dispersion combines report
error and curve misspecification and does not identify meter accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PHASE_ORDER = ("oil", "gas", "water")


def _times(values: Any) -> np.ndarray:
    result = np.atleast_1d(np.asarray(values, dtype=float))
    if result.ndim != 1 or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("Times must be a finite, nonnegative scalar or vector.")
    return result


def _integrated_shape(t: np.ndarray, tau: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Stable integral of (1 + t/tau)**(-p), including p=1."""
    exponent = 1.0 - p
    logarithm = np.log1p(t / tau)
    near_one = np.abs(exponent) < 1e-7
    divisor = np.where(near_one, 1.0, exponent)
    quotient = np.expm1(exponent * logarithm) / divisor
    return tau * np.where(near_one, logarithm, quotient)


@dataclass(frozen=True)
class DCATrajectorySamples:
    """A fixed sample of latent functions, independently of any query grid.

    ``parameters`` has shape [..., 3 phases, 3 parameters], with parameter
    order amplitude, tau, p. A normal forecast uses prefix [well, sample].
    After ``tail_start`` the rate follows a derivative-matched exponential.
    This finite-tail assumption is imposed on all phases; post-terminal water
    and EUR are conditional extrapolations, not validated reserves.
    """

    parameters: np.ndarray
    tail_start: float = 1080.0

    def __post_init__(self) -> None:
        params = np.array(self.parameters, dtype=float, copy=True)
        if params.ndim < 2 or params.shape[-2:] != (3, 3):
            raise ValueError("Parameters must end in [3 phases, A/tau/p].")
        if not np.all(np.isfinite(params)) or np.any(params <= 0):
            raise ValueError("All DCA parameters must be finite and positive.")
        if not np.isfinite(self.tail_start) or self.tail_start <= 0:
            raise ValueError("tail_start must be finite and positive.")
        params.flags.writeable = False
        object.__setattr__(self, "parameters", params)

    def rates(self, times: Any) -> np.ndarray:
        """Return [..., time, phase] nonnegative continuous-time rates."""
        t = _times(times)[:, None]
        amplitude = self.parameters[..., 0][..., None, :]
        tau = self.parameters[..., 1][..., None, :]
        p = self.parameters[..., 2][..., None, :]
        decay = p / (tau + self.tail_start)
        return amplitude * np.exp(
            -p * np.log1p(np.minimum(t, self.tail_start) / tau)
            - decay * np.maximum(t - self.tail_start, 0.0)
        )

    def cumulative(self, times: Any) -> np.ndarray:
        """Analytical integrals from zero, shape [..., time, phase]."""
        t = _times(times)[:, None]
        amplitude = self.parameters[..., 0][..., None, :]
        tau = self.parameters[..., 1][..., None, :]
        p = self.parameters[..., 2][..., None, :]
        integral = _integrated_shape(np.minimum(t, self.tail_start), tau, p)
        terminal_rate = np.exp(-p * np.log1p(self.tail_start / tau))
        decay = p / (tau + self.tail_start)
        tail = terminal_rate * (-np.expm1(-decay * np.maximum(t - self.tail_start, 0.0))) / decay
        return amplitude * (integral + tail)

    def interval(self, start: Any, end: Any) -> np.ndarray:
        """Integrate matched scalar/vector endpoints without resampling."""
        a, b = np.broadcast_arrays(_times(start), _times(end))
        if np.any(b < a):
            raise ValueError("Interval ends must not precede starts.")
        return self.cumulative(b) - self.cumulative(a)

    def ultimate(self) -> np.ndarray:
        """Finite [..., phase] integrals under the imposed terminal regime."""
        amplitude, tau, p = np.moveaxis(self.parameters, -1, 0)
        pre = _integrated_shape(np.asarray(self.tail_start), tau, p)
        post = np.exp(-p * np.log1p(self.tail_start / tau)) * (tau + self.tail_start) / p
        return amplitude * (pre + post)


class HierarchicalDCABaseline:
    """Joint triphase empirical-Bayes DCA conditioned on all observed history.

    Each curve is q(t)=A*(1+t/tau)**(-p) through day 1080. Training phase
    curves use a fixed shape grid and positive amplitudes; complete training
    wells supply a joint library across all nine log parameters. Joint
    Gaussian jitter expands that library, and clipping defines bounded prior
    support. Wells missing an entire phase inform that phase's fitting/noise
    estimates when available, but are excluded from the joint parameter prior.

    Positive-day reports follow independent Gaussian endpoint errors with
    sigma(t,Q)^2 = absolute_scale[k]^2*t/tail_start + relative_scale[k]^2*Q^2.
    Signed, negative, and decreasing reports are supported. Endpoint errors
    are sampled once and shared by every derived increment. Day zero is a
    bookkeeping convention excluded from the likelihood. There is no claim
    that raw/conditioned reported paths themselves are monotone.

    Noise parameters are trained by a bounded residual likelihood search,
    alternating with grid/iteratively reweighted curve fitting. This is a
    plug-in population estimate; weight/hyperparameter uncertainty is omitted.
    ``covariates`` are currently rejected when nonempty because no input has
    yet been verified as available at forecast time.
    """

    joint = True

    def __init__(
        self,
        n_particles: int = 4096,
        seed: int = 42,
        jitter: float = 0.15,
        tail_start: float = 1080.0,
        tau_grid: tuple[float, ...] = (15.0, 45.0, 120.0, 360.0, 1080.0, 3000.0),
        p_grid: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0, 1.4, 2.0),
        posterior_batch_size: int = 64,
    ) -> None:
        if n_particles < 2 or posterior_batch_size < 1:
            raise ValueError("At least two particles and a positive batch size are required.")
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError("jitter must be finite and nonnegative.")
        if not np.isfinite(tail_start) or tail_start <= 0:
            raise ValueError("tail_start must be finite and positive.")
        if not tau_grid or not p_grid or any(not np.all(np.isfinite(grid)) or np.any(np.asarray(grid) <= 0) for grid in (tau_grid, p_grid)):
            raise ValueError("DCA shape grids must contain finite positive values.")
        self.n_particles = int(n_particles)
        self.seed = int(seed)
        self.jitter = float(jitter)
        self.tail_start = float(tail_start)
        self.tau_grid = tuple(float(v) for v in tau_grid)
        self.p_grid = tuple(float(v) for v in p_grid)
        self.posterior_batch_size = int(posterior_batch_size)

    @staticmethod
    def _check_covariates(covariates: Any) -> None:
        if covariates is not None and np.asarray(covariates).size:
            raise ValueError("This baseline currently uses history only; covariates require an explicit availability contract.")

    @staticmethod
    def _check_reports(reports: Any, n_times: int) -> np.ndarray:
        result = np.asarray(reports, dtype=float)
        if result.ndim != 3 or result.shape[1:] != (n_times, 3):
            raise ValueError("Reports must have shape [well, report_day, oil/gas/water].")
        if np.any(np.isinf(result)):
            raise ValueError("Missing reports must be NaN; infinite reports are unsupported.")
        return result

    def _check_day_zero(self, reports: np.ndarray) -> None:
        zero_day = reports[:, self.report_days_ == 0]
        if np.any(np.isfinite(zero_day) & (zero_day != 0)):
            raise ValueError("Day zero is deterministic zero under this model; nonzero day-zero reports are outside its support.")

    def _fit_curves(self, reports: np.ndarray, shape: np.ndarray, phase: int, initial: bool) -> tuple[np.ndarray, np.ndarray]:
        """Grid/profile fit, retaining genuinely observed positive-day endpoints."""
        mask = np.isfinite(reports) & (self.report_days_[None, :] > 0)
        y = np.where(mask, reports, 0.0)
        phase_scale = self.phase_scales_[phase]
        time_scale = np.sqrt(self.report_days_ / self.tail_start)
        weights = mask / np.maximum(phase_scale * time_scale, phase_scale * 1e-4)[None, :] ** 2
        numerator = np.einsum("nt,gt->ng", y * weights, shape)
        denominator = np.einsum("nt,gt->ng", weights, shape**2)
        amplitude_floor = phase_scale / self.tail_start * 1e-7
        amplitude = np.maximum(numerator / np.maximum(denominator, 1e-30), amplitude_floor)
        if not initial:
            # Each shape's amplitude receives three weighted least-squares
            # updates. Final choice is ranked by the actual observation NLL.
            for _ in range(3):
                mean = amplitude[:, :, None] * shape[None, :, :]
                variance = self.absolute_scale_[phase] ** 2 * time_scale[None, None, :] ** 2 + (self.relative_scale_[phase] * mean) ** 2
                weights_3d = mask[:, None, :] / np.maximum(variance, 1e-24)
                numerator = np.sum(weights_3d * y[:, None, :] * shape[None, :, :], axis=-1)
                denominator = np.sum(weights_3d * shape[None, :, :] ** 2, axis=-1)
                amplitude = np.maximum(numerator / np.maximum(denominator, 1e-30), amplitude_floor)
        mean = amplitude[:, :, None] * shape[None, :, :]
        residual = y[:, None, :] - mean
        if initial:
            loss = np.sum(weights[:, None, :] * residual**2, axis=-1)
        else:
            variance = self.absolute_scale_[phase] ** 2 * time_scale[None, None, :] ** 2 + (self.relative_scale_[phase] * mean) ** 2
            variance = np.maximum(variance, 1e-24)
            loss = np.sum(mask[:, None, :] * (residual**2 / variance + np.log(variance)), axis=-1)
        best = np.argmin(loss, axis=1)
        chosen_amplitude = amplitude[np.arange(len(reports)), best]
        params = np.column_stack((chosen_amplitude, self.shape_grid_[best]))
        params[~mask.any(axis=1)] = np.nan
        return params, chosen_amplitude[:, None] * shape[best]

    def _fit_dispersion(self, reports: np.ndarray, fitted: np.ndarray) -> None:
        """Bounded plug-in Gaussian residual scale estimation, training only."""
        absolute_fractions = np.geomspace(0.002, 0.5, 14)
        relative_values = np.geomspace(0.005, 0.8, 15)
        t_scale = self.report_days_ / self.tail_start
        for phase in range(3):
            mask = np.isfinite(reports[:, :, phase]) & (self.report_days_[None, :] > 0)
            y = reports[:, :, phase][mask]
            mean = fitted[:, :, phase][mask]
            times = np.broadcast_to(t_scale[None, :], mask.shape)[mask]
            residual2 = (y - mean) ** 2
            absolute_values = absolute_fractions * self.phase_scales_[phase]
            best: tuple[float, float, float] | None = None
            for absolute in absolute_values:
                variance = absolute**2 * times[:, None] + (mean[:, None] * relative_values[None, :]) ** 2
                nll = np.mean(residual2[:, None] / variance + np.log(variance), axis=0)
                choice = int(np.argmin(nll))
                option = (float(nll[choice]), float(absolute), float(relative_values[choice]))
                if best is None or option[0] < best[0]:
                    best = option
            assert best is not None
            self.absolute_scale_[phase], self.relative_scale_[phase] = best[1:]

    def fit(self, train_reports: Any, report_days: Any, covariates: Any = None) -> HierarchicalDCABaseline:
        self._check_covariates(covariates)
        self.report_days_ = _times(report_days)
        if len(self.report_days_) < 2 or np.any(np.diff(self.report_days_) <= 0):
            raise ValueError("Report days must be strictly increasing and contain at least two times.")
        if self.report_days_[-1] > self.tail_start:
            raise ValueError("Training endpoints must not exceed the prespecified terminal switch.")
        reports = self._check_reports(train_reports, len(self.report_days_))
        self._check_day_zero(reports)
        positive_days = self.report_days_ > 0
        available = np.isfinite(reports) & positive_days[None, :, None]
        if len(reports) < 2 or np.any(available.sum(axis=(0, 1)) < 2):
            raise ValueError("At least two wells and two observed endpoints per phase are required.")
        self.phase_scales_ = np.array([
            max(float(np.median(np.abs(reports[:, :, k][available[:, :, k]]))), 1.0)
            for k in range(3)
        ])
        self.shape_grid_ = np.array([(tau, p) for tau in self.tau_grid for p in self.p_grid])
        shape = _integrated_shape(self.report_days_[None, :], self.shape_grid_[:, :1], self.shape_grid_[:, 1:])
        self.absolute_scale_ = 0.05 * self.phase_scales_
        self.relative_scale_ = np.full(3, 0.1)
        params = np.empty((len(reports), 3, 3))
        fitted = np.empty_like(reports)
        for iteration in range(3):
            for phase in range(3):
                params[:, phase], fitted[:, :, phase] = self._fit_curves(reports[:, :, phase], shape, phase, initial=iteration == 0)
            self._fit_dispersion(reports, fitted)
        complete = np.isfinite(params).all(axis=(1, 2))
        if complete.sum() < 2:
            raise ValueError("A joint empirical prior needs at least two training wells with an observed positive-day report in every phase.")
        self.train_parameters_ = params
        log_params = np.log(params[complete]).reshape((-1, 9))
        covariance = np.atleast_2d(np.cov(log_params, rowvar=False))
        covariance = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance))
        covariance += np.eye(9) * 1e-8
        rng = np.random.default_rng(self.seed)
        # The same training well supplies all phase parameters of a component.
        base = rng.choice(len(log_params), size=self.n_particles, replace=True)
        jitter = rng.multivariate_normal(np.zeros(9), covariance * self.jitter**2, size=self.n_particles)
        prior = np.exp(np.clip(log_params[base] + jitter, -50.0, 50.0)).reshape((-1, 3, 3))
        amplitude_floor = self.phase_scales_ / self.tail_start * 1e-7
        amplitude_ceiling = np.nanmax(params[:, :, 0], axis=0) * 5.0
        prior[:, :, 0] = np.clip(prior[:, :, 0], amplitude_floor, amplitude_ceiling)
        prior[:, :, 1] = np.clip(prior[:, :, 1], min(self.tau_grid) / 5, max(self.tau_grid) * 5)
        prior[:, :, 2] = np.clip(prior[:, :, 2], min(self.p_grid) / 5, max(self.p_grid) * 2)
        self.prior_parameters_ = prior
        self.prior_cumulative_ = DCATrajectorySamples(prior, self.tail_start).cumulative(self.report_days_)
        self.prior_sigma_ = self.observation_sigma(self.prior_cumulative_, self.report_days_)
        self.fit_diagnostics_ = {
            "model": type(self).__name__,
            "phase_order": list(PHASE_ORDER),
            "training_wells": len(reports),
            "joint_prior_training_wells": int(complete.sum()),
            "wells_excluded_from_joint_prior_missing_entire_phase": int((~complete).sum()),
            "observed_positive_day_endpoints": available.sum(axis=(0, 1)).tolist(),
            "prior_particles": self.n_particles,
            "prior_seed": self.seed,
            "log_parameter_jitter": self.jitter,
            "absolute_noise_scale_native": self.absolute_scale_.tolist(),
            "relative_noise_scale": self.relative_scale_.tolist(),
            "phase_training_scales_native": self.phase_scales_.tolist(),
            "terminal_switch_days": self.tail_start,
            "tau_grid_days": list(self.tau_grid),
            "p_grid": list(self.p_grid),
            "posterior": "exact finite categorical prior, joint phases" if self.joint else "exact factorized phase marginals of finite prior",
            "fit": "three alternating grid/IRLS curve fits and bounded residual Gaussian dispersion searches",
            "observation_law": "independent signed Gaussian cumulative endpoints; abs^2*t/terminal + rel^2*latent_cumulative^2 variance",
            "dispersion_limitation": "training fit residuals combine reporting error and decline misspecification; plug-in estimation omits parameter uncertainty",
            "tail_assumption": "all phases switch to derivative-matched exponential at terminal day; unvalidated conditional extrapolation",
        }
        self.last_posterior_diagnostics_: dict[str, Any] = {}
        return self

    def observation_sigma(self, latent_cumulative: Any, times: Any) -> np.ndarray:
        """Native-unit standard deviation of the fitted endpoint law."""
        mean = np.asarray(latent_cumulative, dtype=float)
        t = _times(times)
        if mean.shape[-2:] != (len(t), 3):
            raise ValueError("Latent cumulatives must end in [time, 3 phases].")
        return np.sqrt(self.absolute_scale_**2 * (t[:, None] / self.tail_start) + (self.relative_scale_ * mean) ** 2)

    def _history(self, history_reports: Any, origin: float) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "prior_parameters_"):
            raise RuntimeError("Fit the DCA population prior before prediction.")
        if not np.isfinite(origin) or origin < 0:
            raise ValueError("origin must be finite and nonnegative.")
        history = self._check_reports(history_reports, len(self.report_days_))
        self._check_day_zero(history)
        mask = np.isfinite(history) & (self.report_days_[None, :, None] <= origin) & (self.report_days_[None, :, None] > 0)
        return history, mask

    def _weights(self, history: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Accumulate each endpoint once. Do not construct triangular increments.
        log_weights = np.zeros((len(history), self.n_particles, 3))
        for j, day in enumerate(self.report_days_):
            if day == 0:
                continue
            observed = np.where(mask[:, j, :], history[:, j, :], 0.0)
            sigma = self.prior_sigma_[None, :, j, :]
            difference = (observed[:, None, :] - self.prior_cumulative_[None, :, j, :]) / sigma
            # The normalizing term matters: variance depends on the particle.
            terms = -0.5 * difference**2 - np.log(sigma)
            log_weights += np.where(mask[:, None, j, :], terms, 0.0)
        if self.joint:
            log_weights = log_weights.sum(axis=-1)
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def posterior_weights(self, history_reports: Any, origin: float) -> np.ndarray:
        """Exact normalized finite-prior weights, useful for posterior audits.

        Joint shape is [well, particle]; the phase-factorized ablation adds
        a last phase axis. This method materializes the full weight matrix.
        """
        history, mask = self._history(history_reports, origin)
        chunks = [self._weights(history[start:start + self.posterior_batch_size], mask[start:start + self.posterior_batch_size]) for start in range(0, len(history), self.posterior_batch_size)]
        if not chunks:
            return np.empty((0, self.n_particles) + (() if self.joint else (3,)))
        return np.concatenate(chunks, axis=0)

    def sample_latent(self, history_reports: Any, origin: float, n_samples: int, rng: np.random.Generator | None = None, covariates: Any = None) -> DCATrajectorySamples:
        """Condition on cutoff-available reports and sample fixed latent paths."""
        self._check_covariates(covariates)
        history, mask = self._history(history_reports, origin)
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        rng = np.random.default_rng(self.seed) if rng is None else rng
        parameters = np.empty((len(history), n_samples, 3, 3))
        ess_chunks, maximum_chunks = [], []
        for start in range(0, len(history), self.posterior_batch_size):
            stop = min(start + self.posterior_batch_size, len(history))
            weights = self._weights(history[start:stop], mask[start:stop])
            ess_chunks.append(1.0 / np.sum(weights**2, axis=1))
            maximum_chunks.append(np.max(weights, axis=1))
            for local, well in enumerate(range(start, stop)):
                if self.joint:
                    indices = rng.choice(self.n_particles, size=n_samples, p=weights[local])
                    parameters[well] = self.prior_parameters_[indices]
                else:
                    for phase in range(3):
                        indices = rng.choice(self.n_particles, size=n_samples, p=weights[local, :, phase])
                        parameters[well, :, phase] = self.prior_parameters_[indices, phase]
        ess = np.concatenate(ess_chunks) if ess_chunks else np.empty(0)
        maximum = np.concatenate(maximum_chunks) if maximum_chunks else np.empty(0)
        self.last_posterior_diagnostics_ = {
            "origin_days": float(origin),
            "effective_sample_size": ess,
            "maximum_particle_weight": maximum,
            "history_endpoint_count": mask.sum(axis=(1, 2)),
            "ess_median": float(np.median(ess)) if ess.size else None,
            "ess_min": float(np.min(ess)) if ess.size else None,
            "fraction_ess_below_10": float(np.mean(ess < 10)) if ess.size else None,
            "n_particles": self.n_particles,
            "meaning": "categorical posterior diversity, not a Monte Carlo error guarantee for a continuous prior",
        }
        return DCATrajectorySamples(parameters, self.tail_start)

    def sample_reports(self, latent: DCATrajectorySamples, rng: np.random.Generator, history_reports: Any | None = None, origin: float | None = None) -> np.ndarray:
        """Sample the fitted report law once per endpoint on fixed latent paths.

        If a history is supplied, its observed cutoff-available entries are
        fixed exactly in every sample. Missing historical reports remain
        posterior-predictive draws. No latent curve is anchored to noisy data.
        """
        means = latent.cumulative(self.report_days_)
        reports = means + self.observation_sigma(means, self.report_days_) * rng.normal(size=means.shape)
        if history_reports is not None:
            if origin is None:
                raise ValueError("origin is required when conditioning report output on history.")
            history, _ = self._history(history_reports, origin)
            if means.ndim != 4 or len(means) != len(history):
                raise ValueError("Latent samples and history must have matching well dimensions.")
            fixed = np.isfinite(history) & (self.report_days_[None, :, None] <= origin)
            reports = np.where(fixed[:, None, :, :], history[:, None, :, :], reports)
        return reports

    def predict_samples(self, history_reports: Any, origin: float, n_samples: int, rng: np.random.Generator | None = None, covariates: Any = None) -> np.ndarray:
        """Return [well, sample, report_day, phase] conditional reported paths."""
        rng = np.random.default_rng(self.seed) if rng is None else rng
        latent = self.sample_latent(history_reports, origin, n_samples, rng, covariates)
        return self.sample_reports(latent, rng, history_reports, origin)


class ConditionalDCABaseline(HierarchicalDCABaseline):
    """Phase-factorized conditional DCA ablation.

    Each phase has the fitted joint library's marginal prior and conditions
    only on its own reports. This deliberately removes cross-phase updating
    and dependence, so it is an information/dependence ablation rather than
    the fully informed main benchmark.
    """

    joint = False
