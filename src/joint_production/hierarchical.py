"""Hierarchical continuous-rate models with local Gaussian variational inference.

The population law is z ~ N(mu, LL^T), with phase-major rate parameters: twelve
for envelope/basis families and nine for the joint decline-curve comparator.
Observed cumulative endpoints use one explicitly selected reporting law in
training-frozen phase volume units. The default is independent Student-t(df=5,
location=Q(z,t), scale=sigma_abs + sigma_rel * Q(z,t)). Matched Gaussian
alternatives have V(t,Q)=sigma_abs**2*t/360 + sigma_rel**2*Q**2, with either
independent endpoint errors or accumulated errors with covariance V(min(t,u)).
Changing the law requires refitting and fresh history-conditioned inference.
Negative/decreasing reports are allowed; latent production remains nonnegative.
Day zero is a deterministic accounting origin and supplies no likelihood term.

Training is penalized empirical Bayes: a full-covariance Gaussian q_i is fitted
for each training well, alongside population and reporting parameters. The
penalties are stated in ``configuration`` and ``training_diagnostics_``. New
wells receive fresh full-covariance Gaussian variational optimization under the
frozen generative law. This is an approximate posterior, not an exact sampler;
ELBO diagnostics alone do not establish posterior accuracy or convergence.

This initial implementation uses production history only. Attribute vintages
are unresolved in the pilot data, so covariates are not silently treated as
available. Phase order at every array boundary is oil, gas, water.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal, StudentT, kl_divergence

from .rates import EnvelopeRate, PositiveBasisRate
from .observation import gaussian_endpoint_log_likelihood, sample_gaussian_reports


PHASE_ORDER = ("oil", "gas", "water")
NATIVE_VOLUME_UNITS = ("BBL", "MCF", "BBL")
OBSERVATION_LAWS = ("student_t_independent", "gaussian_independent", "gaussian_accumulated")


def likelihood_mask(reports: Tensor, report_days: Tensor, cutoff: float | None = None) -> Tensor:
    """Finite observed endpoints after day zero, optionally available by cutoff.

    Apply this mask before using report values. In particular, future entries
    supplied by a caller never enter initialization, likelihood, or diagnostics.
    """
    eligible = report_days > 0
    if cutoff is not None:
        eligible = eligible & (report_days <= cutoff)
    return torch.isfinite(reports) & eligible[None, :, None]


def endpoint_log_likelihood(
    reports: Tensor,
    cumulative: Tensor,
    noise_abs: Tensor,
    noise_rel: Tensor,
    mask: Tensor,
) -> Tensor:
    """Sum the actual endpoint log likelihood once per well.

    ``reports`` and ``mask`` have shape [N,T,3], and ``cumulative`` has shape
    [...,N,T,3]. The result has shape [...,N]. Student-t scale is not its SD.
    """
    safe_reports = torch.where(mask, reports, torch.zeros_like(reports))
    distribution = StudentT(5.0, cumulative, noise_abs + noise_rel * cumulative)
    log_prob = distribution.log_prob(safe_reports)
    return torch.where(mask, log_prob, torch.zeros_like(log_prob)).sum(dim=(-2, -1))


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _scale_tril(raw: Tensor, floor: float) -> Tensor:
    diagonal = torch.nn.functional.softplus(torch.diagonal(raw, dim1=-2, dim2=-1)) + floor
    return torch.tril(raw, diagonal=-1) + torch.diag_embed(diagonal)


def _raw_tril(cholesky: Tensor, floor: float) -> Tensor:
    diagonal = torch.diagonal(cholesky, dim1=-2, dim2=-1)
    if bool(torch.any(diagonal <= floor)):
        raise ValueError("Cholesky diagonal must exceed the configured positive floor")
    return torch.tril(cholesky, diagonal=-1) + torch.diag_embed(_inverse_softplus(diagonal - floor))


@dataclass
class VariationalPosterior:
    """Well-specific q(z); tensors are detached, on the model device."""

    mean: Tensor
    scale_tril: Tensor
    diagnostics: dict[str, Any]


class HierarchicalForecaster(nn.Module):
    """History-conditioned population model of complete continuous trajectories.

    All noise floors are explicit model restrictions in scaled volume units.
    Cholesky floors ensure positive diagonals, not an eigenvalue lower bound.
    The objective averages per-well ELBOs, subtracts ``prior_regularization``
    times KL(population || N(initial_latent, reference_prior_sd^2 I)), and
    subtracts ``noise_regularization`` times a squared log-scale penalty around
    the initial noise scales. These regularizers help identify a finite-data
    empirical-Bayes fit; their effect must be evaluated experimentally.
    """

    def __init__(
        self,
        rate_family: str = "envelope",
        *,
        basis_dictionary: str = "original",
        observation_law: str = "student_t_independent",
        seed: int = 42,
        epochs: int = 300,
        inference_steps: int = 200,
        batch_size: int = 512,
        learning_rate: float = 0.02,
        inference_learning_rate: float = 0.025,
        train_mc_samples: int = 4,
        inference_mc_samples: int = 16,
        prior_regularization: float = 0.01,
        noise_regularization: float = 0.01,
        reference_prior_sd: float = 1.5,
        cholesky_floor: float = 1e-3,
        noise_floor: float = 1e-4,
        initial_noise_abs: float = 0.02,
        initial_noise_rel: float = 0.08,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        if rate_family not in {"envelope", "basis", "dca"}:
            raise ValueError("rate_family must be 'envelope', 'basis', or 'dca'")
        if observation_law not in OBSERVATION_LAWS:
            raise ValueError(f"observation_law must be one of {OBSERVATION_LAWS}")
        if basis_dictionary not in {"original", "fast"}:
            raise ValueError("basis_dictionary must be 'original' or 'fast'")
        if rate_family != "basis" and basis_dictionary != "original":
            raise ValueError("A nondefault basis_dictionary applies only to rate_family='basis'")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if min(epochs, inference_steps) < 0 or min(batch_size, train_mc_samples, inference_mc_samples) < 1:
            raise ValueError("Steps must be nonnegative and batch/sample counts positive")
        if min(learning_rate, inference_learning_rate, reference_prior_sd, cholesky_floor, noise_floor) <= 0:
            raise ValueError("Learning rates, reference SD, and positive floors must be positive")
        if min(initial_noise_abs, initial_noise_rel) <= noise_floor:
            raise ValueError("Initial noise scales must exceed the explicit noise floor")
        if min(prior_regularization, noise_regularization) < 0:
            raise ValueError("Regularization weights must be nonnegative")
        self.configuration = dict(
            rate_family=rate_family, observation_law=observation_law, seed=seed, epochs=epochs, inference_steps=inference_steps,
            batch_size=batch_size, learning_rate=learning_rate,
            inference_learning_rate=inference_learning_rate, train_mc_samples=train_mc_samples,
            inference_mc_samples=inference_mc_samples, prior_regularization=prior_regularization,
            noise_regularization=noise_regularization, reference_prior_sd=reference_prior_sd,
            cholesky_floor=cholesky_floor, noise_floor=noise_floor,
            initial_noise_abs=initial_noise_abs, initial_noise_rel=initial_noise_rel,
            device=device, dtype=dtype,
        )
        if rate_family == "basis":
            self.configuration["basis_dictionary"] = basis_dictionary
        for key, value in self.configuration.items():
            if key not in {"device", "dtype"}:
                setattr(self, key, value)
        if rate_family == "envelope":
            self.rate_model = EnvelopeRate()
        elif rate_family == "basis":
            self.rate_model = PositiveBasisRate(preset=basis_dictionary)
        else:
            from .dca_rates import DCARate
            self.rate_model = DCARate()
        self.latent_size = self.rate_model.latent_size
        self.parameters_per_phase = self.rate_model.parameters_per_phase
        self.amplitude_indices = self.rate_model.amplitude_indices
        if self.latent_size != len(PHASE_ORDER) * self.parameters_per_phase:
            raise ValueError("Rate-family metadata does not define equal-size phase-major parameter blocks")
        torch_dtype = getattr(torch, dtype)
        initial = self.rate_model.initial_latent(dtype=torch_dtype, device=device)
        self.register_buffer("reference_mean", initial)
        self.register_buffer("volume_scale", torch.empty(0, dtype=torch_dtype, device=device))
        self.register_buffer("report_days", torch.empty(0, dtype=torch_dtype, device=device))
        self.prior_mean = nn.Parameter(initial.clone())
        self.prior_raw_tril = nn.Parameter(_raw_tril(torch.eye(self.latent_size, dtype=torch_dtype, device=device), cholesky_floor))
        self.noise_abs_raw = nn.Parameter(_inverse_softplus(torch.full((3,), initial_noise_abs - noise_floor, dtype=torch_dtype, device=device)))
        self.noise_rel_raw = nn.Parameter(_inverse_softplus(torch.full((3,), initial_noise_rel - noise_floor, dtype=torch_dtype, device=device)))
        self.training_diagnostics_: dict[str, Any] = {}
        self.last_diagnostics_: dict[str, Any] = {}
        self.scale_provenance_: list[str] = []

    @property
    def device(self) -> torch.device:
        return self.prior_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.prior_mean.dtype

    @property
    def prior_scale_tril(self) -> Tensor:
        return _scale_tril(self.prior_raw_tril, self.cholesky_floor)

    @property
    def noise_abs(self) -> Tensor:
        return torch.nn.functional.softplus(self.noise_abs_raw) + self.noise_floor

    @property
    def noise_rel(self) -> Tensor:
        return torch.nn.functional.softplus(self.noise_rel_raw) + self.noise_floor

    def _tensor(self, value: Any) -> Tensor:
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _generator(self, offset: int = 0) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.seed) + offset)
        return generator

    def _check_reports(self, reports: Any, *, cutoff: float | None = None) -> Tensor:
        values = self._tensor(reports)
        if values.ndim != 3 or values.shape[1:] != (len(self.report_days), 3):
            raise ValueError("Reports must have shape [N,len(report_days),3] in oil/gas/water order")
        if values.shape[0] == 0:
            raise ValueError("At least one well is required")
        available = torch.ones_like(self.report_days, dtype=torch.bool)
        if cutoff is not None:
            if not np.isfinite(cutoff) or cutoff < 0:
                raise ValueError("Forecast origin must be a finite nonnegative day")
            available = self.report_days <= cutoff
        values = torch.where(available[None, :, None], values, torch.full_like(values, float("nan")))
        if bool(torch.isinf(values).any()):
            raise ValueError("Available reports may be finite or NaN, never infinite")
        day_zero = self.report_days == 0
        bad_zero = torch.isfinite(values[:, day_zero]) & (values[:, day_zero] != 0)
        if bool(bad_zero.any()):
            raise ValueError("This observation law requires deterministic day-zero reports equal to zero")
        return values

    def _fit_scale(self, reports: Tensor) -> None:
        """Median positive terminal report, phase-only fallback, then one native unit."""
        scales: list[Tensor] = []
        self.scale_provenance_ = []
        for phase, unit in enumerate(NATIVE_VOLUME_UNITS):
            terminal = reports[:, -1, phase]
            positive = terminal[torch.isfinite(terminal) & (terminal > 0)]
            source = f"positive train cumulative at day {float(self.report_days[-1]):g}; {unit}"
            if not positive.numel():
                pooled = reports[:, :, phase]
                positive = pooled[torch.isfinite(pooled) & (pooled > 0)]
                source = f"fallback: positive train cumulative endpoints in phase; {unit}"
            if positive.numel():
                scales.append(torch.quantile(positive, 0.5))
            else:
                scales.append(self._tensor(1.0))
                source = f"fallback: 1 native {unit}; no positive train reports"
            self.scale_provenance_.append(source)
        self.volume_scale = torch.stack(scales)

    def _initialize_locations(self, scaled_reports: Tensor, mask: Tensor, *, training: bool) -> Tensor:
        """Amplitude warm start using the latest permitted positive endpoint only."""
        baseline = self.reference_mean if training else self.prior_mean.detach()
        locations = baseline.expand(len(scaled_reports), -1).clone()
        with torch.no_grad():
            expected = self.rate_model.cumulative(baseline, self.report_days)
            for phase in range(3):
                positive = mask[:, :, phase] & (scaled_reports[:, :, phase] > 0)
                indices = torch.arange(len(self.report_days), device=self.device)[None, :].expand_as(positive)
                latest = torch.where(positive, indices, -1).max(dim=1).values
                has_positive = latest >= 0
                selected = latest[has_positive]
                numerator = scaled_reports[has_positive, selected, phase]
                denominator = expected[selected, phase]
                shift = torch.log(numerator / denominator)
                first = phase * self.parameters_per_phase
                last = first + self.parameters_per_phase
                for index in self.amplitude_indices:
                    if first <= index < last:
                        locations[has_positive, index] += shift
        return locations

    def _draw(self, mean: Tensor, cholesky: Tensor, count: int, generator: torch.Generator) -> Tensor:
        epsilon = torch.randn((count, *mean.shape), generator=generator, dtype=self.dtype, device=self.device)
        return mean[None] + torch.einsum("nij,mnj->mni", cholesky, epsilon)

    def _elbo(
        self, mean: Tensor, cholesky: Tensor, reports: Tensor, mask: Tensor,
        prior: MultivariateNormal, noise_abs: Tensor, noise_rel: Tensor,
        samples: int, generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent = self._draw(mean, cholesky, samples, generator)
        cumulative = self.rate_model.cumulative(latent, self.report_days)
        log_likelihood = self._report_log_likelihood(reports, cumulative, noise_abs, noise_rel, mask).mean(dim=0)
        divergence = kl_divergence(MultivariateNormal(mean, scale_tril=cholesky), prior)
        return log_likelihood - divergence, log_likelihood, divergence

    def _report_log_likelihood(self, reports: Tensor, cumulative: Tensor, noise_abs: Tensor, noise_rel: Tensor, mask: Tensor) -> Tensor:
        if self.observation_law == "student_t_independent":
            return endpoint_log_likelihood(reports, cumulative, noise_abs, noise_rel, mask)
        return gaussian_endpoint_log_likelihood(
            reports, cumulative, noise_abs, noise_rel, mask, self.report_days,
            law=self.observation_law.removeprefix("gaussian_"),
        )

    def _report_law_description(self) -> dict[str, Any]:
        if self.observation_law == "student_t_independent":
            return {"name": self.observation_law, "distribution": "Student-t", "df": 5,
                    "scale": "a + r*Q; Student-t scale is not standard deviation",
                    "endpoint_errors": "independent conditional on the joint latent trajectory"}
        return {"name": self.observation_law, "distribution": "Gaussian", "variance": "V(t,Q)=a^2*t/360 + r^2*Q^2",
                "endpoint_covariance": "diagonal V" if self.observation_law == "gaussian_independent" else "V(min(t,u),Q(min(t,u)))",
                "phase_errors": "independent conditional on the joint latent trajectory"}

    def _penalty(self, prior: MultivariateNormal) -> Tensor:
        reference = MultivariateNormal(
            self.reference_mean,
            scale_tril=torch.eye(self.latent_size, dtype=self.dtype, device=self.device) * self.reference_prior_sd,
        )
        population_penalty = self.prior_regularization * kl_divergence(prior, reference)
        noise_penalty = 0.5 * self.noise_regularization * (
            torch.log(self.noise_abs / self.initial_noise_abs).square().sum()
            + torch.log(self.noise_rel / self.initial_noise_rel).square().sum()
        )
        return population_penalty + noise_penalty

    def fit(self, train_reports: Any, report_days: Any) -> "HierarchicalForecaster":
        """Fit population/noise parameters and each training well's q using endpoints."""
        if self.observation_law not in OBSERVATION_LAWS:
            raise ValueError(f"observation_law must be one of {OBSERVATION_LAWS}")
        self.configuration["observation_law"] = self.observation_law
        days = self._tensor(report_days)
        if days.ndim != 1 or len(days) < 2 or not bool(torch.isfinite(days).all()):
            raise ValueError("report_days must contain at least two finite ordered days")
        if bool((days < 0).any()) or bool((days[1:] <= days[:-1]).any()):
            raise ValueError("report_days must be nonnegative and strictly increasing")
        self.report_days = days
        reports = self._check_reports(train_reports)
        self._fit_scale(reports)
        reports = reports / self.volume_scale
        mask = likelihood_mask(reports, self.report_days)
        if not bool(mask.any()):
            raise ValueError("Training requires observed positive-time reports")
        mean = nn.Parameter(self._initialize_locations(reports, mask, training=True))
        with torch.no_grad():
            self.prior_mean.copy_(mean.mean(dim=0))
            centered = mean - self.prior_mean
            covariance = centered.T @ centered / max(len(reports) - 1, 1)
            covariance += 0.25 * torch.eye(self.latent_size, dtype=self.dtype, device=self.device)
            self.prior_raw_tril.copy_(_raw_tril(torch.linalg.cholesky(covariance), self.cholesky_floor))
        initial_cholesky = 0.25 * torch.eye(self.latent_size, dtype=self.dtype, device=self.device)
        raw = nn.Parameter(_raw_tril(initial_cholesky.expand(len(reports), -1, -1).clone(), self.cholesky_floor))
        parameters = [mean, raw, *self.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        generator = self._generator()
        trace: list[dict[str, float | int]] = []
        self.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(reports), generator=generator, device=self.device)
            totals = np.zeros(4, dtype=float)
            for start in range(0, len(reports), self.batch_size):
                indices = order[start:start + self.batch_size]
                prior = MultivariateNormal(self.prior_mean, scale_tril=self.prior_scale_tril)
                elbo, log_likelihood, divergence = self._elbo(
                    mean[indices], _scale_tril(raw[indices], self.cholesky_floor), reports[indices], mask[indices],
                    prior, self.noise_abs, self.noise_rel, self.train_mc_samples, generator,
                )
                penalty = self._penalty(prior)
                loss = -elbo.mean() + penalty
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite training objective at epoch {epoch}; no law-changing clipping applied")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0, error_if_nonfinite=True)
                optimizer.step()
                totals += len(indices) * np.array([float(elbo.mean().detach()), float(log_likelihood.mean().detach()), float(divergence.mean().detach()), float(penalty.detach())])
            averaged = totals / len(reports)
            trace.append(dict(epoch=epoch + 1, elbo=float(averaged[0]), expected_log_likelihood=float(averaged[1]), kl=float(averaged[2]), penalty=float(averaged[3])))
            if (epoch + 1) % 25 == 0 or epoch + 1 == self.epochs:
                print(json.dumps({"event": "hierarchical_training", "rate_family": self.rate_family, "observation_law": self.observation_law, **trace[-1]}), flush=True)
        self.eval()
        with torch.no_grad():
            cholesky = self.prior_scale_tril
            covariance = cholesky @ cholesky.T
            self.training_diagnostics_ = {
                "method": "penalized empirical Bayes with full-covariance Gaussian variational posteriors",
                "posterior_accuracy_verified": False,
                "phase_order": list(PHASE_ORDER), "n_wells": len(reports),
                "latent_size": self.latent_size, "parameters_per_phase": self.parameters_per_phase,
                "n_observed_endpoints": int(mask.sum()), "epochs": self.epochs,
                "train_mc_samples": self.train_mc_samples, "trace": trace,
                "volume_scale": self.volume_scale.cpu().tolist(), "scale_provenance": self.scale_provenance_,
                "noise_abs_scaled": self.noise_abs.cpu().tolist(), "noise_rel": self.noise_rel.cpu().tolist(),
                "observation_law": self.observation_law, "report_law": self._report_law_description(),
                "population_covariance_eigenvalues": torch.linalg.eigvalsh(covariance).cpu().tolist(),
                "population_mean": self.prior_mean.cpu().tolist(),
                "training_q_min_cholesky_diagonal": float(torch.diagonal(_scale_tril(raw, self.cholesky_floor), dim1=-2, dim2=-1).min()),
                "regularization": {key: self.configuration[key] for key in ("prior_regularization", "noise_regularization", "reference_prior_sd", "cholesky_floor", "noise_floor")},
            }
            if self.observation_law == "student_t_independent":
                self.training_diagnostics_.update(student_t_df=5, student_t_scale_is_standard_deviation=False)
        return self

    def _require_fitted(self) -> None:
        if self.volume_scale.numel() != 3 or self.report_days.numel() < 2:
            raise RuntimeError("Call fit before posterior inference or sampling")
        fitted_law = self.training_diagnostics_.get("observation_law", "student_t_independent")
        if fitted_law != self.observation_law:
            raise RuntimeError("Changing the observation law requires refitting and fresh posterior inference")

    def infer_posterior(self, history_reports: Any, origin: float) -> VariationalPosterior:
        """Optimize q using only finite reports through origin; population is frozen."""
        self._require_fitted()
        reports = self._check_reports(history_reports, cutoff=origin) / self.volume_scale
        mask = likelihood_mask(reports, self.report_days, cutoff=origin)
        observed = mask.any(dim=(-2, -1))
        prior_mean = self.prior_mean.detach()
        prior_cholesky = self.prior_scale_tril.detach()
        prior = MultivariateNormal(prior_mean, scale_tril=prior_cholesky)
        noise_abs, noise_rel = self.noise_abs.detach(), self.noise_rel.detach()
        mean = nn.Parameter(self._initialize_locations(reports, mask, training=False))
        # The prior itself is the exact answer for an empty history. Observed
        # wells start narrower to reduce stochastic likelihood-gradient variance.
        initial_cholesky = prior_cholesky.expand(len(reports), -1, -1).clone()
        initial_cholesky[observed] = 0.25 * torch.eye(self.latent_size, dtype=self.dtype, device=self.device)
        raw = nn.Parameter(_raw_tril(initial_cholesky, self.cholesky_floor))
        optimizer = torch.optim.Adam([mean, raw], lr=self.inference_learning_rate)
        generator = self._generator(1000003)
        evaluation_samples = max(64, self.inference_mc_samples)
        with torch.no_grad():
            initial_elbo = self._elbo(mean, _scale_tril(raw, self.cholesky_floor), reports, mask, prior, noise_abs, noise_rel, evaluation_samples, self._generator(2000003))[0]
        trace: list[dict[str, float | int]] = []
        if bool(observed.any()):
            for step in range(self.inference_steps):
                elbo, _, _ = self._elbo(mean, _scale_tril(raw, self.cholesky_floor), reports, mask, prior, noise_abs, noise_rel, self.inference_mc_samples, generator)
                loss = -elbo[observed].mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite inference objective at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([mean, raw], max_norm=10.0, error_if_nonfinite=True)
                optimizer.step()
                if step == 0 or (step + 1) % 25 == 0 or step + 1 == self.inference_steps:
                    trace.append({"step": step + 1, "stochastic_mean_elbo": -float(loss.detach())})
        with torch.no_grad():
            final_mean = mean.detach().clone()
            final_cholesky = _scale_tril(raw, self.cholesky_floor).detach()
            final_mean[~observed] = prior_mean
            final_cholesky[~observed] = prior_cholesky
            final_elbo = self._elbo(final_mean, final_cholesky, reports, mask, prior, noise_abs, noise_rel, evaluation_samples, self._generator(2000003))[0]
            diagnostics = {
                "method": "local full-covariance Gaussian ELBO optimization; population/noise frozen",
                "posterior_accuracy_verified": False, "origin": float(origin),
                "observation_law": self.observation_law,
                "latent_size": self.latent_size,
                "steps": self.inference_steps if bool(observed.any()) else 0,
                "mc_samples": self.inference_mc_samples, "diagnostic_mc_samples": evaluation_samples,
                "n_wells": len(reports), "n_empty_histories": int((~observed).sum()),
                "n_observed_endpoints": int(mask.sum()),
                "initial_mean_elbo": float(initial_elbo.mean()), "final_mean_elbo": float(final_elbo.mean()),
                "fraction_wells_elbo_improved": float((final_elbo >= initial_elbo).to(self.dtype).mean()),
                "minimum_cholesky_diagonal": float(torch.diagonal(final_cholesky, dim1=-2, dim2=-1).min()),
                "mean_logdet_covariance": float((2 * torch.log(torch.diagonal(final_cholesky, dim1=-2, dim2=-1)).sum(dim=-1)).mean()),
                "trace": trace,
            }
        self.last_diagnostics_ = diagnostics
        return VariationalPosterior(final_mean, final_cholesky, diagnostics)

    def sample_latents(self, posterior: VariationalPosterior, n_samples: int, rng: np.random.Generator) -> Tensor:
        """Draw [N,S,latent_size] parameters; retain these for all time queries."""
        self._require_fitted()
        if posterior.diagnostics.get("observation_law", "student_t_independent") != self.observation_law:
            raise ValueError("Posterior was inferred under a different observation law; reinfer under the fitted law")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        epsilon = self._tensor(rng.standard_normal((len(posterior.mean), n_samples, self.latent_size)))
        return posterior.mean[:, None] + torch.einsum("nij,nsj->nsi", posterior.scale_tril, epsilon)

    def evaluate_trajectories(self, latent_parameters: Any, query_days: Any) -> dict[str, np.ndarray]:
        """Evaluate the same latent draws at arbitrary nonnegative days, in native units.

        No new random quantities are sampled here. Rates are volume per day;
        finite total/tail production is conditional on the rate-family's
        operating/extrapolation assumptions, not a validated reserve estimate.
        """
        self._require_fitted()
        latent, days = self._tensor(latent_parameters), self._tensor(query_days)
        if latent.shape[-1] != self.latent_size or days.ndim != 1 or not bool(torch.isfinite(days).all()) or bool((days < 0).any()):
            raise ValueError(f"Expected latent [...,{self.latent_size}] and a finite nonnegative day vector")
        with torch.no_grad():
            cumulative = self.rate_model.cumulative(latent, days) * self.volume_scale
            rates = self.rate_model.rate(latent, days) * self.volume_scale
            totals = self.rate_model.total_volume(latent) * self.volume_scale
        return {"latent_cumulative": cumulative.cpu().numpy(), "latent_rate": rates.cpu().numpy(), "latent_total_volume": totals.cpu().numpy()}

    def predict_distribution(self, history_reports: Any, origin: float, n_samples: int, rng: np.random.Generator) -> dict[str, Any]:
        """Paired latent and reported cumulative trajectories on the reporting grid.

        The report sampler uses the same law as fitting and posterior inference.
        Accumulated Gaussian errors use conditional bridges through observed
        history. Observed historical reports are repeated exactly; missing past
        endpoints are posterior predictions, including missing cutoff phases.
        Differences of shared endpoints retain the selected error dependence.
        """
        posterior = self.infer_posterior(history_reports, origin)
        latents = self.sample_latents(posterior, n_samples, rng)
        paths = self.evaluate_trajectories(latents, self.report_days)
        cumulative = paths["latent_cumulative"]
        scale = self.volume_scale.detach().cpu().numpy()
        abs_noise = self.noise_abs.detach().cpu().numpy() * scale
        rel_noise = self.noise_rel.detach().cpu().numpy()
        days = self.report_days.detach().cpu().numpy()
        # Inference may use float32 internally, but conditioning repeats the
        # actual supplied reports, not their rounded optimization representation.
        self._check_reports(history_reports, cutoff=origin)
        source_history = history_reports.detach().cpu().numpy() if isinstance(history_reports, Tensor) else history_reports
        history = np.asarray(source_history, dtype=np.float64)
        if self.observation_law == "student_t_independent":
            report = cumulative + (abs_noise + rel_noise * cumulative) * rng.standard_t(5.0, size=cumulative.shape)
            report[:, :, days == 0] = 0.0
            observed = np.isfinite(history) & (days[None, :, None] <= origin)
            report = np.where(observed[:, None], history[:, None], report)
        else:
            report = sample_gaussian_reports(cumulative, history, days, abs_noise, rel_noise,
                                             cutoff=origin, rng=rng, law=self.observation_law.removeprefix("gaussian_"))
        return {
            **paths, "report_cumulative": report,
            "latent_parameters": latents.detach().cpu().numpy(),
            "posterior_diagnostics": posterior.diagnostics,
            "observation_law": self.observation_law,
            "phase_order": PHASE_ORDER,
        }

    def predict_samples(self, history_reports: Any, origin: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return reported cumulative samples [N,S,T,3], oil/gas/water order."""
        return self.predict_distribution(history_reports, origin, n_samples, rng)["report_cumulative"]

    def save(self, path: str | Path) -> None:
        """Persist configuration, frozen scales, fitted parameters, and diagnostics."""
        self._require_fitted()
        checkpoint = {
            "schema_version": 1, "configuration": self.configuration,
            "state_dict": self.state_dict(), "training_diagnostics": self.training_diagnostics_,
            "scale_provenance": self.scale_provenance_,
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "HierarchicalForecaster":
        """Load a checkpoint on an explicitly chosen runtime device."""
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        if checkpoint["schema_version"] != 1:
            raise ValueError("Unsupported hierarchical checkpoint schema")
        configuration = dict(checkpoint["configuration"], device=device)
        model = cls(**configuration)
        model.volume_scale = checkpoint["state_dict"]["volume_scale"].clone()
        model.report_days = checkpoint["state_dict"]["report_days"].clone()
        model.load_state_dict(checkpoint["state_dict"])
        model.training_diagnostics_ = checkpoint["training_diagnostics"]
        model.scale_provenance_ = checkpoint["scale_provenance"]
        model.eval()
        return model
