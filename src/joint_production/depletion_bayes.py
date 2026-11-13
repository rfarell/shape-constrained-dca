"""Physics-structured Bayesian joint rate model with depletion-indexed ratios.

Phase order is oil, gas, water at every array boundary. Internally volumes are
normalized by frozen TRAIN phase scales S_k (BBL, MCF, BBL); time is in days.

Latent oil deliverability, in normalized volume per day::

    q_o(t) = q_p * (rho0 + (1 - rho0) * t / t_p),          0 <= t < t_p
    q_o(t) = sum_j w_j lam_j exp(-lam_j (t - t_p)),        t >= t_p

with w_j = exp(a_j) > 0 on a fixed log-spaced decay grid lam_j, q_p = sum_j
w_j lam_j, t_p = PEAK_MAX_DAYS * sigmoid(b_peak) and rho0 = sigmoid(b_ramp).
The rate is continuous, nondecreasing on [0, t_p] and completely monotone
after t_p: a positive mixture of exponentials, which is the constant-drawdown
single-phase depletion solution class. The slowest grid mode fixes the
terminal decline, an explicit extrapolation assumption, and EUR_o =
q_p t_p (1 + rho0) / 2 + sum_j w_j is finite.

Gas and water are indexed by the oil depletion fraction F(t) = Q_o(t)/EUR_o::

    Q_k(t) = EUR_o * Gamma_k(F(t)),   Gamma_k'(f) = R_k(f) = sum_m c_km B_m(f)

where B_m are degree-K Bernstein polynomials and c_km = exp(.) > 0. Hence
q_k = R_k(F) q_o, the normalized ratio R_k lies between min_m c_km and max_m
c_km, cumulatives are nondecreasing, and EUR_k = EUR_o * mean_m c_km is
finite. F is strictly increasing in time, so this is a reparameterization that
imposes no sign or monotonicity on GOR or WOR. Native GOR is R_g * S_g / S_o.

The well parameters z (LATENT_DIM) have an attribute-conditioned Gaussian
population prior. Under the Brownian law, reports follow the accumulated
Gaussian law Y = Q + B(V), V = a^2 t/360 + r^2 Q^2 per phase, independent
across phases given z; the interval-OU laws add a persistent deviation
(``interval_ou_covariance``). The well posterior given permitted
reports is approximated by prior x Gaussian pseudo-likelihood from a history
encoder, so empty history returns the prior exactly and posterior precision
is never below prior precision. Prediction reweights the approximation by
importance sampling against the exact unnormalized posterior. Population
parameters are empirical-Bayes point estimates; their uncertainty is omitted.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from joint_production.history_features import last_interval_rate_features

PHASES = ("oil", "gas", "water")
DECAY_TIMES_DAYS = tuple(10.0 * 365.0 ** (j / 7) for j in range(8))
PEAK_MAX_DAYS = 180.0
RATIO_DEGREE = 3
N_MODES = len(DECAY_TIMES_DAYS)
N_RATIO = RATIO_DEGREE + 1
FLOWBACK_BASE = N_MODES + 2 + 2 * N_RATIO
RATE_DIM = FLOWBACK_BASE + 2
LATENT_DIM = RATE_DIM + 3
MODES, PEAK, RAMP = slice(0, N_MODES), N_MODES, N_MODES + 1
GAS = slice(N_MODES + 2, N_MODES + 2 + N_RATIO)
WATER = slice(N_MODES + 2 + N_RATIO, FLOWBACK_BASE)
FLOWBACK_VOLUME, FLOWBACK_TIME = FLOWBACK_BASE, FLOWBACK_BASE + 1
FLOWBACK_TAU_DAYS = (5.0, 120.0)
NOISE = slice(RATE_DIM, LATENT_DIM)
LOG_2PI = math.log(2 * math.pi)


def _powers(value: Tensor, degree: int) -> Tensor:
    """[..., degree+1] powers by repeated products (finite gradients at zero)."""
    out = [torch.ones_like(value)]
    for _ in range(degree):
        out.append(out[-1] * value)
    return torch.stack(out, dim=-1)


def bernstein(f: Tensor, degree: int) -> Tensor:
    """Bernstein basis [..., degree+1] on [0, 1]."""
    binom = f.new_tensor([math.comb(degree, k) for k in range(degree + 1)])
    up = _powers(f, degree)
    down = _powers(1 - f, degree).flip(-1)
    return binom * up * down


def _exprel(y: Tensor) -> Tensor:
    """(e^y - 1) / y, stable near y = 0."""
    small = y.abs() < 1e-6
    safe = torch.where(small, torch.ones_like(y), y)
    return torch.where(small, 1 + y / 2, torch.expm1(safe) / safe)


def _ramp_moment(y: Tensor) -> Tensor:
    """h(y) = (1 - e^-y (1 + y)) / y^2 = int_0^1 s e^(-y s) ds, stable near y = 0."""
    small = y < 0.1
    safe = torch.where(small, torch.ones_like(y), y)
    series = 0.5 - y / 3 + y ** 2 / 8 - y ** 3 / 30 + y ** 4 / 144 - y ** 5 / 840
    return torch.where(small, series, (-torch.expm1(-safe) - safe * torch.exp(-safe)) / safe ** 2)


class DepletionRate:
    """Closed-form rates and cumulatives for the structured joint law.

    ``ratio_index='depletion'`` (the model): gas/water-to-oil ratios are Bernstein
    polynomials in the oil depletion fraction F = Q_o/EUR_o. ``ratio_index='time'``
    (ablation): the same ratios in u = 1 - exp(-t/T) with a fixed population time
    scale T (days). Then R(u) is a sum of exponentials sum_j a_j e^(-j t/T), so the
    phase cumulative int R(u) q_o dt stays closed form and finite.
    """

    def __init__(self, decay_times=DECAY_TIMES_DAYS, peak_max=PEAK_MAX_DAYS, ratio_degree=RATIO_DEGREE,
                 ratio_index: str = "depletion", ratio_time_scale: float = 730.0):
        self.decay_times = tuple(float(v) for v in decay_times)
        self.peak_max = float(peak_max)
        self.degree = int(ratio_degree)
        if len(self.decay_times) != N_MODES or self.degree != RATIO_DEGREE:
            raise ValueError("Latent layout is fixed by module constants")
        if ratio_index not in {"depletion", "time"}:
            raise ValueError("ratio_index must be depletion or time")
        self.ratio_index, self.ratio_time_scale = ratio_index, float(ratio_time_scale)
        k = self.degree
        # Bernstein-in-u to power-in-v (v = 1 - u): C(k,m) (1-v)^m v^(k-m) = sum_l C(k,m) C(m,l) (-1)^l v^(l+k-m).
        convert = np.zeros((k + 1, k + 1))
        for m in range(k + 1):
            for l in range(m + 1):
                convert[m, l + k - m] += math.comb(k, m) * math.comb(m, l) * (-1) ** l
        self._to_power = torch.as_tensor(convert)

    def unpack(self, z: Tensor) -> dict[str, Tensor]:
        if z.shape[-1] not in (RATE_DIM, LATENT_DIM):
            raise ValueError(f"z must end with dimension {RATE_DIM} or {LATENT_DIM}")
        lam = z.new_tensor([1.0 / tau for tau in self.decay_times])
        weights = z[..., MODES].exp()
        peak = (self.peak_max * torch.sigmoid(z[..., PEAK])).clamp_min(1e-4)
        rho0 = torch.sigmoid(z[..., RAMP])
        q_peak = (weights * lam).sum(-1)
        eur = q_peak * peak * (1 + rho0) / 2 + weights.sum(-1)
        low, high = FLOWBACK_TAU_DAYS
        return dict(lam=lam, weights=weights, peak=peak, rho0=rho0, q_peak=q_peak, eur_oil=eur,
                    gas=z[..., GAS].exp(), water=z[..., WATER].exp(),
                    flowback=z[..., FLOWBACK_VOLUME].exp(),
                    flowback_tau=low + (high - low) * torch.sigmoid(z[..., FLOWBACK_TIME]))

    @staticmethod
    def _times(t, like: Tensor) -> Tensor:
        return torch.as_tensor(t, dtype=like.dtype, device=like.device)

    def oil_cumulative(self, p: dict[str, Tensor], t) -> Tensor:
        t = self._times(t, p["weights"])
        peak, rho0, q_peak = p["peak"][..., None], p["rho0"][..., None], p["q_peak"][..., None]
        ramp_time = torch.minimum(t, peak)
        ramp = q_peak * ramp_time * (rho0 + (1 - rho0) * torch.clamp(t / peak, max=1.0) / 2)
        after = (t - peak).clamp_min(0)
        decline = (p["weights"][..., None, :] * -torch.expm1(-p["lam"] * after[..., None])).sum(-1)
        return ramp + decline

    def oil_rate(self, p: dict[str, Tensor], t) -> Tensor:
        t = self._times(t, p["weights"])
        peak, rho0, q_peak = p["peak"][..., None], p["rho0"][..., None], p["q_peak"][..., None]
        ramp = q_peak * (rho0 + (1 - rho0) * torch.clamp(t / peak, max=1.0))
        after = (t - peak).clamp_min(0)
        decline = (p["weights"][..., None, :] * p["lam"] * torch.exp(-p["lam"] * after[..., None])).sum(-1)
        return torch.where(t < peak, ramp, decline)

    def _ratio_cumulative(self, coefficients: Tensor, fraction: Tensor) -> Tensor:
        """Gamma(F) = int_0^F sum_m c_m B_m(f) df, via degree-(K+1) Bernstein."""
        partial = torch.cat((torch.zeros_like(coefficients[..., :1]), coefficients.cumsum(-1)), dim=-1)
        basis = bernstein(fraction, self.degree + 1)
        return (basis * partial[..., None, :]).sum(-1) / (self.degree + 1)

    def _ratio(self, coefficients: Tensor, fraction: Tensor) -> Tensor:
        return (bernstein(fraction, self.degree) * coefficients[..., None, :]).sum(-1)

    def depletion_fraction(self, p: dict[str, Tensor], t) -> Tensor:
        return (self.oil_cumulative(p, t) / p["eur_oil"][..., None]).clamp(0.0, 1.0)

    def ratio_argument(self, p: dict[str, Tensor], t) -> Tensor:
        """Argument of the ratio polynomials [..., T]: depletion fraction or u = 1 - exp(-t/T)."""
        if self.ratio_index == "depletion":
            return self.depletion_fraction(p, t)
        t = self._times(t, p["weights"])
        return (-torch.expm1(-t / self.ratio_time_scale)).expand(p["weights"].shape[:-1] + t.shape)

    def _discounted_oil(self, p: dict[str, Tensor], t) -> Tensor:
        """G_j(t) = int_0^t q_o(s) exp(-j s/T) ds [..., T, K+1] (t = inf allowed)."""
        like = p["weights"]
        t = self._times(t, like)
        kappa = torch.arange(self.degree + 1, dtype=like.dtype, device=like.device) / self.ratio_time_scale
        peak, rho0, q_peak = p["peak"][..., None], p["rho0"][..., None], p["q_peak"][..., None]
        x = torch.minimum(t, peak)[..., None]  # [..., T, 1]
        alpha, beta = (q_peak * rho0)[..., None], (q_peak * (1 - rho0) / peak)[..., None]
        y = kappa * x
        ramp = alpha * x * _exprel(-y) + beta * x.square() * _ramp_moment(y)
        after = (t - peak).clamp_min(0).clamp(max=1e12)[..., None]  # [..., T, 1]; t = inf gives the total
        scaled = p["weights"][..., None, :] * p["lam"]  # [..., 1, modes]
        # One discount rate at a time keeps memory at the size of the oil cumulative itself.
        decline = torch.stack([(scaled * -torch.expm1(-(p["lam"] + k) * after) / (p["lam"] + k)).sum(-1)
                               for k in kappa.unbind()], dim=-1)
        return ramp + torch.exp(-kappa * peak[..., None]) * decline

    def _time_ratio_cumulative(self, p: dict[str, Tensor], coefficients: Tensor, t) -> Tensor:
        power = coefficients @ self._to_power.to(coefficients)  # [..., K+1]
        return (self._discounted_oil(p, t) * power[..., None, :]).sum(-1)

    def cumulative(self, z: Tensor, t) -> Tensor:
        """Normalized cumulative volumes [..., T, 3]."""
        p = self.unpack(z)
        oil = self.oil_cumulative(p, t)
        if self.ratio_index == "time":
            gas = self._time_ratio_cumulative(p, p["gas"], t)
            water = self._time_ratio_cumulative(p, p["water"], t) + self._flowback_cumulative(p, t)
            return torch.stack((oil, gas, water), dim=-1)
        fraction = (oil / p["eur_oil"][..., None]).clamp(0.0, 1.0)
        eur = p["eur_oil"][..., None]
        gas = eur * self._ratio_cumulative(p["gas"], fraction)
        water = eur * self._ratio_cumulative(p["water"], fraction) + self._flowback_cumulative(p, t)
        return torch.stack((oil, gas, water), dim=-1)

    def _flowback_cumulative(self, p: dict[str, Tensor], t) -> Tensor:
        t = self._times(t, p["weights"])
        return p["flowback"][..., None] * -torch.expm1(-t / p["flowback_tau"][..., None])

    def _flowback_rate(self, p: dict[str, Tensor], t) -> Tensor:
        t = self._times(t, p["weights"])
        tau = p["flowback_tau"][..., None]
        return p["flowback"][..., None] / tau * torch.exp(-t / tau)

    def rate(self, z: Tensor, t) -> Tensor:
        """Normalized rates [..., T, 3] (volume/day)."""
        p = self.unpack(z)
        oil = self.oil_rate(p, t)
        fraction = self.ratio_argument(p, t)
        return torch.stack((oil, self._ratio(p["gas"], fraction) * oil,
                            self._ratio(p["water"], fraction) * oil + self._flowback_rate(p, t)), dim=-1)

    def total_volume(self, z: Tensor) -> Tensor:
        """Finite normalized EUR [..., 3]."""
        p = self.unpack(z)
        eur = p["eur_oil"]
        if self.ratio_index == "time":
            inf = torch.full((1,), math.inf, dtype=eur.dtype, device=eur.device)
            gas = self._time_ratio_cumulative(p, p["gas"], inf)[..., 0]
            water = self._time_ratio_cumulative(p, p["water"], inf)[..., 0] + p["flowback"]
            return torch.stack((eur, gas, water), dim=-1)
        return torch.stack((eur, eur * p["gas"].mean(-1), eur * p["water"].mean(-1) + p["flowback"]), dim=-1)


def sample_bridge_reports(cumulative: Tensor, history: Tensor, days: Tensor, noise_abs: Tensor, noise_rel: Tensor,
                          *, cutoff: float, generator=None) -> Tensor:
    """Accumulated-Gaussian conditional reports [N,S,T,3] with per-draw noise [N,S,3].

    Same law and bridge construction as ``torch_observation.sample_accumulated_reports``:
    known reports at or before the cutoff are repeated exactly; each missing
    endpoint is drawn conditional on the previous sampled error and the next
    known error, or by a Brownian increment after the last known endpoint.
    """
    q = cumulative
    n, s, t, _ = q.shape
    a, r = noise_abs, noise_rel
    if a.shape != (n, s, 3) or r.shape != (n, s, 3):
        raise ValueError("Per-draw noise must have shape [N,S,3]")
    available = torch.where((days <= cutoff)[None, :, None], history, torch.full_like(history, float("nan")))
    known = torch.isfinite(available) & (days > 0)[None, :, None]
    safe_history = torch.where(known, available, torch.zeros_like(available))
    normals = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator)
    indices = torch.arange(t, device=q.device)[None, :, None].expand(n, t, 3)
    right_indices = torch.where(known, indices, t).flip(1).cummin(1).values.flip(1)
    previous_q = torch.zeros_like(q[:, :, 0])
    previous_error = torch.zeros_like(previous_q)
    previous_day = days.new_zeros(())
    outputs = []
    for j in range(t):
        current_q = q[:, :, j]
        right_index = right_indices[:, j]
        has_right = (right_index < t)[:, None]
        safe_index = right_index.clamp_max(t - 1)
        right_q = q.gather(2, safe_index[:, None, None].expand(n, s, 1, 3)).squeeze(2)
        right_q = torch.where(has_right, right_q, current_q)
        right_day = torch.where(has_right, days[safe_index][:, None], days[j])
        right_report = safe_history.gather(1, safe_index[:, None]).squeeze(1)[:, None]
        first = (a.square() * ((days[j] - previous_day) / 360)
                 + (r * (current_q - previous_q)) * (r * current_q + r * previous_q)).clamp_min(0)
        last = (a.square() * ((right_day - days[j]) / 360)
                + (r * (right_q - current_q)) * (r * right_q + r * current_q)).clamp_min(0)
        whole = torch.where(has_right & (days[j] > 0), first + last, torch.ones_like(first))
        fraction = first / whole
        mean = torch.where(has_right, previous_error + fraction * (right_report - right_q - previous_error), previous_error)
        variance = torch.where(has_right, first * (last / whole), first)
        random_endpoint = (~known[:, j, None]) & (days[j] > 0)
        safe_variance = torch.where(random_endpoint, variance, torch.ones_like(variance))
        error = mean + safe_variance.sqrt() * normals[:, :, j]
        report = torch.where(known[:, j, None], safe_history[:, j, None], current_q + error)
        report = torch.where(days[j] == 0, torch.zeros_like(report), report)
        outputs.append(report)
        previous_q, previous_day = current_q, days[j]
        previous_error = report - current_q
    result = torch.stack(outputs, dim=2)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Report sampler produced a numerical failure")
    return result


OU_LAWS = ("ou_interval", "lognormal_ou")  # laws with a persistent operational-deviation term
LOGNORMAL = "lognormal_ou"


def interval_ou_kernel(days: Tensor, tau: Tensor) -> Tensor:
    """Mean OU correlation between report intervals [3, T', T'] for time constants tau [3].

    K_kl = (L_k L_l)^-1 int_{I_k} int_{I_l} exp(-|s - s'|/tau) ds ds' for the successive
    intervals I_k = (t_{k-1}, t_k] between positive report days (t_0 = 0).
    """
    edges = torch.cat((days.new_zeros(1), days[days > 0]))
    a, b = edges[:-1], edges[1:]
    length = b - a
    t = tau[:, None, None]
    # Disjoint I_k before I_l: tau^2 (e^-(c-b)/tau - e^-(d-b)/tau - e^-(c-a)/tau + e^-(d-a)/tau).
    def g(x):
        return torch.exp(-x.clamp_min(0) / t)
    ak, bk, al, bl = a[:, None], b[:, None], a[None, :], b[None, :]
    cross = t.square() * (g(al - bk) - g(bl - bk) - g(al - ak) + g(bl - ak))
    lk = length[None, :, None].expand(len(tau), -1, -1)
    same = 2 * t * lk - 2 * t.square() * -torch.expm1(-lk / t)
    upper = torch.triu(torch.ones_like(cross[0]), diagonal=1).bool()
    eye = torch.eye(len(length), dtype=days.dtype, device=days.device).bool()
    raw = torch.where(upper, cross, torch.where(eye, same.expand_as(cross), cross.transpose(-1, -2)))
    return raw / (length[:, None] * length[None, :])


def interval_ou_covariance(q_days: Tensor, days: Tensor, sigma: Tensor, tau: Tensor) -> Tensor:
    """Persistent-deviation covariance [..., 3, T', T'] of cumulative report discrepancies.

    The relative deviation u(t) of production from deliverability is a stationary OU
    process (SD sigma, time constant tau); within each report interval it is weighted
    by the interval-average rate, so the interval discrepancy int_I q u ~ dQ_I * mean_I(u)
    and Cov = sigma^2 dQ_k dQ_l K_kl. q_days is [..., T, 3] at all report days (day 0 first).
    """
    dq = torch.diff(q_days, dim=-2).clamp_min(0).transpose(-1, -2)  # [..., 3, T']
    kernel = interval_ou_kernel(days, tau)  # [3, T', T']
    increments = sigma[..., None, None].square() * dq[..., :, None] * dq[..., None, :] * kernel
    return torch.cumsum(torch.cumsum(increments, dim=-1), dim=-2)


# ----- positive (log-normal) report law on report intervals ---------------------------------
#
# Reported volume over report interval k (between successive positive report days, day 0
# counting as zero) is y_k = dQ_k exp(e_k) with e ~ N(-diag(C)/2, C): positive, mean dQ_k.
# C = sigma^2 Kbar (persistent OU deviation, as in ``interval_ou_covariance``) + diag(log(1 +
# dV_k / dQ_k^2)), where dV_k is the interval increment of the accumulated variance clock (so
# the independent part keeps the clock's absolute/level/rate structure as a relative variance).
# A reported non-positive interval volume is an atom with TRAIN frequency zero_prob[k, phase].
# The law is defined on the report intervals; a report following a missing endpoint only sets
# the level from which later intervals accumulate (the merged interval is not scored).

def interval_log_moments(dq: Tensor, dv: Tensor, days: Tensor, sigma: Tensor, tau: Tensor,
                         inflation: float = 1.0) -> tuple[Tensor, Tensor]:
    """Mean [..., 3, T'] and covariance [..., 3, T', T'] of log reported interval volumes."""
    dq = dq.clamp_min(1e-12)
    independent = torch.log1p(dv.clamp_min(0) / dq.square())
    cov = sigma[..., None, None].square() * interval_ou_kernel(days, tau) + torch.diag_embed(independent)
    cov = cov * inflation ** 2
    return dq.log() - 0.5 * torch.diagonal(cov, dim1=-2, dim2=-1), cov


def observed_intervals(reports: Tensor, observed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Reported interval volumes, known and positive masks [N, 3, T'] from reports/observed [N, T, 3]."""
    values = torch.where(observed, reports, torch.zeros_like(reports))
    start = torch.cat((torch.zeros_like(values[:, :1]), values[:, 1:-1]), 1)
    start_ok = torch.cat((torch.ones_like(observed[:, :1]), observed[:, 1:-1]), 1)
    known = observed[:, 1:] & start_ok
    volume = torch.where(known, values[:, 1:] - start, torch.zeros_like(start))
    return volume.transpose(1, 2), known.transpose(1, 2), (known & (volume > 0)).transpose(1, 2)


def lognormal_log_likelihood(volume: Tensor, known: Tensor, positive: Tensor, mean: Tensor, cov: Tensor,
                             zero_prob: Tensor, *, fast: bool = False) -> Tensor:
    """Exact log likelihood [N, S] of known report intervals (log-normal volumes plus atoms)."""
    obs = positive[:, None]  # [N,1,3,T']
    x = torch.where(positive, volume.clamp_min(1e-30).log(), torch.zeros_like(volume))[:, None]
    out_dtype = cov.dtype
    resid = torch.where(obs, x - mean, torch.zeros_like(mean))
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    if fast:
        cov, resid, eye = cov.float() + 1e-8 * eye.float(), resid.float(), eye.float()
    pair = obs[..., :, None] & obs[..., None, :]
    masked = torch.where(pair, cov, eye.expand_as(cov))
    chol = small_cholesky(masked) if fast else torch.linalg.cholesky(masked)
    solved = small_solve_lower(chol, resid) if fast else torch.linalg.solve_triangular(
        chol, resid[..., None], upper=False).squeeze(-1)
    log_det = torch.where(obs, torch.diagonal(chol, dim1=-2, dim2=-1).log(), torch.zeros_like(resid))
    count = obs.to(resid.dtype).sum(-1)
    gauss = (-0.5 * solved.square().sum(-1) - log_det.sum(-1) - 0.5 * count * LOG_2PI).to(out_dtype)
    jacobian = -(x * obs).sum(-1).to(out_dtype)  # density of y = exp(x)
    p0 = zero_prob.T.to(out_dtype)  # [3, T']
    atoms = torch.where(known & ~positive, p0.log(), torch.where(positive, torch.log1p(-p0), torch.zeros_like(p0)))
    return (gauss + jacobian).sum(-1) + atoms.sum((-1, -2))[:, None]


def sample_log_intervals(mean: Tensor, cov: Tensor, x: Tensor, obs: Tensor, draws: int, generator=None,
                         small: bool = False) -> Tensor:
    """Log interval volumes [R, N, S, 3, T'] given the known positive ones (Matheron's rule).

    mean [N,S,3,T'], cov [N,S,3,T',T'], x/obs [N,3,T']. Differentiable in mean and cov.
    """
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    chol_full = small_cholesky(cov + 1e-8 * eye) if small else torch.linalg.cholesky(cov + 1e-12 * eye)
    eps = torch.randn((draws, *mean.shape), dtype=mean.dtype, device=mean.device, generator=generator)
    e = (chol_full @ eps[..., None]).squeeze(-1)
    o = obs[:, None]
    pair = o[..., :, None] & o[..., None, :]
    c_oo = torch.where(pair, cov, eye.expand_as(cov))
    c_xo = torch.where(o[..., None, :], cov, torch.zeros_like(cov))
    gap = torch.where(o, (x[:, None] - mean) - e, torch.zeros_like(e))
    if small:
        chol = small_cholesky(c_oo)
        alpha = small_solve_upper_t(chol, small_solve_lower(chol, gap))
    else:
        alpha = torch.cholesky_solve(gap[..., None], torch.linalg.cholesky(c_oo)).squeeze(-1)
    return mean + e + (c_xo @ alpha[..., None]).squeeze(-1)


def interval_levels(log_volumes: Tensor, reports: Tensor, observed: Tensor, zero_prob: Tensor,
                    generator=None) -> tuple[Tensor, Tensor]:
    """Reported cumulative levels [R,N,S,3,T'] after the last observed endpoint, and that mask [N,3,T'].

    Levels accumulate sampled interval volumes (with atoms at zero) from the last observed
    report (zero if none); reports/observed [N,T,3] at all report days.
    """
    t = reports.shape[1]
    index = torch.arange(1, t, device=reports.device)
    last = torch.where(observed[:, 1:], index[None, :, None], torch.zeros_like(index)[None, :, None]).max(1).values
    base = torch.where(last > 0, reports.gather(1, last.clamp_min(1)[:, None, :]).squeeze(1), torch.zeros_like(last, dtype=reports.dtype))
    after = index[None, None, :] > last[..., None]  # [N,3,T']
    volume = log_volumes.exp()
    if zero_prob is not None:
        u = torch.rand(volume.shape, dtype=volume.dtype, device=volume.device, generator=generator)
        volume = torch.where(u < zero_prob.T.to(volume.dtype), torch.zeros_like(volume), volume)
    added = torch.cumsum(torch.where(after[:, None], volume, torch.zeros_like(volume)), dim=-1)
    return base[:, None, :, None].to(added.dtype) + added, after


def weighted_sample_crps(points: Tensor, weights: Tensor, target: Tensor) -> Tensor:
    """CRPS of a weighted sample (weights sum to one on the last axis) at target [...]."""
    weights = weights.expand_as(points)
    first = (weights * (points - target[..., None]).abs()).sum(-1)
    order = torch.argsort(points, dim=-1)
    xs, ws = points.gather(-1, order), weights.gather(-1, order)
    below = torch.cumsum(ws, -1) - ws
    return first - (ws * xs * (2 * below + ws - 1)).sum(-1)


def small_cholesky(a: Tensor) -> Tensor:
    """Unrolled Cholesky for small matrices [..., n, n] using elementwise ops only.

    For n ~ 7 and very large batches this is far faster on GPUs than batched
    LAPACK-style kernels, and it differentiates through ordinary autograd.
    """
    n = a.shape[-1]
    lower = [[None] * n for _ in range(n)]
    for j in range(n):
        acc = a[..., j, j]
        for k in range(j):
            acc = acc - lower[j][k].square()
        diag = acc.clamp_min(1e-20).sqrt()
        lower[j][j] = diag
        for i in range(j + 1, n):
            acc = a[..., i, j]
            for k in range(j):
                acc = acc - lower[i][k] * lower[j][k]
            lower[i][j] = acc / diag
    zero = torch.zeros_like(a[..., 0, 0])
    return torch.stack([torch.stack([lower[i][j] if j <= i else zero for j in range(n)], -1) for i in range(n)], -2)


def small_solve_lower(lower: Tensor, b: Tensor) -> Tensor:
    """Solve L y = b for b [..., n] (forward substitution)."""
    n = lower.shape[-1]
    out = []
    for i in range(n):
        acc = b[..., i]
        for k in range(i):
            acc = acc - lower[..., i, k] * out[k]
        out.append(acc / lower[..., i, i])
    return torch.stack(out, -1)


def small_solve_upper_t(lower: Tensor, y: Tensor) -> Tensor:
    """Solve L^T x = y for y [..., n] (backward substitution)."""
    n = lower.shape[-1]
    out = [None] * n
    for i in reversed(range(n)):
        acc = y[..., i]
        for k in range(i + 1, n):
            acc = acc - lower[..., k, i] * out[k]
        out[i] = acc / lower[..., i, i]
    return torch.stack(out, -1)


def gaussian_discrepancy_log_likelihood(reports: Tensor, cumulative: Tensor, covariance: Tensor, mask: Tensor,
                                        days: Tensor, *, fast: bool = False) -> Tensor:
    """Exact masked multivariate-normal log likelihood [N,S] of positive-day reports.

    reports/mask [N,T,3]; cumulative [N,S,T,3]; covariance [N,S,3,T',T'] over the
    positive days. Unobserved endpoints are marginalized exactly by restricting
    the covariance (implemented with identity rows and zero residuals).
    """
    positive = days > 0
    obs = mask[:, positive].transpose(-1, -2)[:, None]  # [N,1,3,T']
    resid = (torch.where(mask, reports, torch.zeros_like(reports))[:, None] - cumulative)[:, :, positive]
    resid = torch.where(obs, resid.transpose(-1, -2), torch.zeros_like(resid.transpose(-1, -2)))  # [N,S,3,T']
    pair = obs[..., :, None] & obs[..., None, :]
    out_dtype = covariance.dtype
    if fast:  # float32 7x7 algebra with jitter: fast on low-FP64 GPUs, ~1e-4 nat error
        covariance = covariance.float() + 1e-8 * torch.eye(obs.shape[-1], device=covariance.device)
        resid = resid.float()
    eye = torch.eye(obs.shape[-1], dtype=covariance.dtype, device=covariance.device)
    cov = torch.where(pair, covariance, eye.expand_as(covariance))
    if fast:
        chol = small_cholesky(cov)
        solved = small_solve_lower(chol, resid)
    else:
        chol = torch.linalg.cholesky(cov)
        solved = torch.linalg.solve_triangular(chol, resid[..., None], upper=False).squeeze(-1)
    log_det = torch.where(obs, torch.diagonal(chol, dim1=-2, dim2=-1).log(), torch.zeros_like(resid))
    count = obs.to(covariance.dtype).sum(-1)
    return (-0.5 * solved.square().sum(-1) - log_det.sum(-1) - 0.5 * count * LOG_2PI).sum(-1).to(out_dtype)


def sample_gaussian_discrepancy_reports(cumulative: Tensor, history: Tensor, days: Tensor, covariance: Tensor, *,
                                        cutoff: float, generator=None) -> Tensor:
    """Reports [N,S,T,3] conditional on finite history at or before ``cutoff``.

    Exact Gaussian conditioning by Matheron's rule: draw e ~ N(0, C), then
    e + C[:,O] C[O,O]^-1 (d_O - e_O). Known reports are repeated exactly; hidden
    past endpoints and the future are sampled jointly.
    """
    n, s, t, _ = cumulative.shape
    positive = days > 0
    available = torch.where((days <= cutoff)[None, :, None], history, torch.full_like(history, float("nan")))
    known = (torch.isfinite(available) & positive[None, :, None])[:, positive].transpose(-1, -2)[:, None]  # [N,1,3,T']
    q = cumulative[:, :, positive].transpose(-1, -2)  # [N,S,3,T']
    target = torch.where(known, available[:, positive].transpose(-1, -2)[:, None] - q, torch.zeros_like(q))
    chol_full = torch.linalg.cholesky(covariance + 1e-12 * torch.eye(q.shape[-1], dtype=q.dtype, device=q.device))
    eps = torch.randn((*q.shape, 1), dtype=q.dtype, device=q.device, generator=generator)
    e = (chol_full @ eps).squeeze(-1)
    pair = known[..., :, None] & known[..., None, :]
    eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    c_oo = torch.where(pair, covariance, eye.expand_as(covariance))
    c_xo = torch.where(known[..., None, :], covariance, torch.zeros_like(covariance))
    gap = torch.where(known, target - e, torch.zeros_like(e))
    correction = (c_xo @ torch.cholesky_solve(gap[..., None], torch.linalg.cholesky(c_oo))).squeeze(-1)
    e = e + correction
    sampled = (q + e).transpose(-1, -2)  # [N,S,T',3]
    result = torch.zeros_like(cumulative)
    result[:, :, positive] = sampled
    known_full = torch.isfinite(available) & positive[None, :, None]
    result = torch.where(known_full[:, None], torch.where(known_full, available, torch.zeros_like(available))[:, None],
                         result)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Gaussian discrepancy sampler produced a numerical failure")
    return result


def conditional_report_moments(covariance: Tensor, cumulative: Tensor, reports: Tensor, known: Tensor,
                               days: Tensor, small: bool = False) -> tuple[Tensor, Tensor]:
    """Gaussian conditional mean/variance [N,S,3,T'] of positive-day reports given known ones."""
    positive = days > 0
    q = cumulative[:, :, positive].transpose(-1, -2)
    known_p = known[:, positive].transpose(-1, -2)[:, None]
    resid = torch.where(known_p, reports[:, positive].transpose(-1, -2)[:, None] - q, torch.zeros_like(q))
    pair = known_p[..., :, None] & known_p[..., None, :]
    eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    c_oo = torch.where(pair, covariance, eye.expand_as(covariance))
    c_xo = torch.where(known_p[..., None, :], covariance, torch.zeros_like(covariance))
    if small:
        chol = small_cholesky(c_oo)
        alpha = small_solve_upper_t(chol, small_solve_lower(chol, resid))
        mean = q + (c_xo @ alpha[..., None]).squeeze(-1)
        half = torch.stack([small_solve_lower(chol, c_xo[..., i, :]) for i in range(c_xo.shape[-2])], -2)
        var = torch.diagonal(covariance, dim1=-2, dim2=-1) - half.square().sum(-1)
    else:
        chol = torch.linalg.cholesky(c_oo)
        mean = q + (c_xo @ torch.cholesky_solve(resid[..., None], chol)).squeeze(-1)
        gain = torch.cholesky_solve(c_xo.transpose(-1, -2), chol)
        var = torch.diagonal(covariance, dim1=-2, dim2=-1) - (c_xo * gain.transpose(-1, -2)).sum(-1)
    return mean, var.clamp_min(1e-14)


def brownian_future_moments(cumulative: Tensor, a: Tensor, r: Tensor, reports: Tensor, known: Tensor,
                            days: Tensor, clock: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Closed-form conditional moments [N,S,3,T'] under the accumulated Brownian law.

    The discrepancy is Markov in its variance clock, so a report after the last known
    endpoint l has mean Q(t) + (Y_l - Q_l) and variance V(t) - V(l) (l = day 0 if none).
    Valid for endpoints after the last known one, which covers every forecast target.
    """
    positive = days > 0
    q = cumulative[:, :, positive].transpose(-1, -2)  # [N,S,3,T']
    t = days[positive]
    if clock is None:
        v = a[..., None].square() * (t / 360) + (r[..., None] * q).square()
    else:
        v = clock[:, :, positive].transpose(-1, -2)
    known_p = known[:, positive].transpose(-1, -2)  # [N,3,T']
    index = torch.arange(len(t), device=q.device)
    last = torch.where(known_p, index, torch.full_like(index, -1)).max(-1).values  # [N,3]
    has = (last >= 0)[:, None]
    safe = last.clamp_min(0)[:, None, :, None].expand(-1, q.shape[1], -1, 1)
    q_last = q.gather(-1, safe).squeeze(-1)
    v_last = torch.where(has, v.gather(-1, safe).squeeze(-1), torch.zeros_like(q_last))
    y_last = reports[:, positive].transpose(-1, -2).gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)[:, None]
    d_last = torch.where(has, y_last - q_last, torch.zeros_like(q_last))
    return q + d_last[..., None], (v - v_last[..., None]).clamp_min(1e-14)


def _abs_normal_mean(m: Tensor, var: Tensor) -> Tensor:
    sd = var.sqrt()
    u = m / sd
    normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
    return sd * 2 * normal.log_prob(u).exp() + m * (2 * normal.cdf(u) - 1)


def gaussian_mixture_crps(weights: Tensor, mean: Tensor, var: Tensor, target: Tensor) -> Tensor:
    """Closed-form CRPS of sum_i w_i N(mean_i, var_i) at target; mixture axis is 1.

    weights [N,M]; mean/var [N,M,...]; target [N,...] -> [N,...].
    """
    w = weights.reshape(*weights.shape, *([1] * (mean.ndim - 2)))
    first = (w * _abs_normal_mean(target[:, None] - mean, var)).sum(1)
    pair_w = w[:, :, None] * w[:, None, :]
    second = (pair_w * _abs_normal_mean(mean[:, :, None] - mean[:, None, :], var[:, :, None] + var[:, None, :])).sum((1, 2))
    return first - 0.5 * second


CLOCK_BIN_DAYS = 15.0


def clock_log_likelihood(reports: Tensor, cumulative: Tensor, clock: Tensor, mask: Tensor, days: Tensor) -> Tensor:
    """Exact masked Brownian log likelihood [N,S] for a general increasing variance clock.

    clock [N,S,T,3] is V at the report days (V(0)=0); successive OBSERVED endpoints
    factor the law with innovation variance V(t_j) - V(t_prev).
    """
    n, s, t, _ = cumulative.shape
    safe_reports = torch.where(mask, reports, torch.zeros_like(reports))
    prev_err = cumulative.new_zeros((n, s, 3))
    prev_v = cumulative.new_zeros((n, s, 3))
    total = cumulative.new_zeros((n, s, 3))
    for j in range(t):
        if float(days[j]) <= 0:
            continue
        err = safe_reports[:, None, j] - cumulative[:, :, j]
        include = mask[:, None, j]
        var = clock[:, :, j] - prev_v
        safe_var = torch.where(include, var.clamp_min(1e-30), torch.ones_like(var))
        innovation = torch.where(include, err - prev_err, torch.zeros_like(err))
        log_p = -0.5 * (LOG_2PI + safe_var.log() + innovation.square() / safe_var)
        total = total + torch.where(include, log_p, torch.zeros_like(log_p))
        prev_err = torch.where(include, err, prev_err)
        prev_v = torch.where(include, clock[:, :, j], prev_v)
    return total.sum(-1)


def sample_clock_reports(cumulative: Tensor, history: Tensor, days: Tensor, clock: Tensor, *, cutoff: float,
                         generator=None) -> Tensor:
    """Brownian-bridge conditional reports [N,S,T,3] for a general variance clock [N,S,T,3]."""
    q = cumulative
    n, s, t, _ = q.shape
    available = torch.where((days <= cutoff)[None, :, None], history, torch.full_like(history, float("nan")))
    known = torch.isfinite(available) & (days > 0)[None, :, None]
    safe_history = torch.where(known, available, torch.zeros_like(available))
    normals = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator)
    indices = torch.arange(t, device=q.device)[None, :, None].expand(n, t, 3)
    right_indices = torch.where(known, indices, t).flip(1).cummin(1).values.flip(1)
    previous_v = torch.zeros_like(q[:, :, 0])
    previous_error = torch.zeros_like(previous_v)
    outputs = []
    for j in range(t):
        right_index = right_indices[:, j]
        has_right = (right_index < t)[:, None]
        safe_index = right_index.clamp_max(t - 1)
        right_q = q.gather(2, safe_index[:, None, None].expand(n, s, 1, 3)).squeeze(2)
        right_v = clock.gather(2, safe_index[:, None, None].expand(n, s, 1, 3)).squeeze(2)
        right_report = safe_history.gather(1, safe_index[:, None]).squeeze(1)[:, None]
        current_v = clock[:, :, j]
        first = (current_v - previous_v).clamp_min(0)
        last = torch.where(has_right, (right_v - current_v).clamp_min(0), torch.zeros_like(first))
        whole = torch.where(has_right & (days[j] > 0), first + last, torch.ones_like(first)).clamp_min(1e-30)
        fraction = first / whole
        mean = torch.where(has_right, previous_error + fraction * (right_report - right_q - previous_error), previous_error)
        variance = torch.where(has_right, first * (last / whole), first)
        random_endpoint = (~known[:, j, None]) & (days[j] > 0)
        safe_variance = torch.where(random_endpoint, variance, torch.ones_like(variance))
        error = mean + safe_variance.sqrt() * normals[:, :, j]
        report = torch.where(known[:, j, None], safe_history[:, j, None], q[:, :, j] + error)
        report = torch.where(days[j] == 0, torch.zeros_like(report), report)
        outputs.append(report)
        previous_v = current_v
        previous_error = report - q[:, :, j]
    result = torch.stack(outputs, dim=2)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Clock report sampler produced a numerical failure")
    return result


def _mlp(inputs: int, width: int, outputs: int, linear: bool = False) -> nn.Sequential:
    if linear:  # ablation: attribute-linear prior (same output layer indexing)
        return nn.Sequential(nn.Linear(inputs, outputs))
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, outputs))


def gaussian_log_prob_tril(z: Tensor, mean: Tensor, scale_tril: Tensor) -> Tensor:
    """log N(z; mean, LL^T) for z [..., S, D], mean [..., D], L [..., D, D]."""
    diff = (z - mean[..., None, :]).transpose(-1, -2)
    solved = torch.linalg.solve_triangular(scale_tril, diff, upper=False)
    return (-0.5 * solved.square().sum(-2) - torch.diagonal(scale_tril, dim1=-2, dim2=-1).log().sum(-1)[..., None]
            - 0.5 * z.shape[-1] * LOG_2PI)


def gaussian_log_prob_precision(z: Tensor, mean: Tensor, precision_tril: Tensor) -> Tensor:
    """log N(z; mean, (LL^T)^-1) for precision Cholesky L."""
    diff = (z - mean[..., None, :])
    projected = diff @ precision_tril
    return (-0.5 * projected.square().sum(-1) + torch.diagonal(precision_tril, dim1=-2, dim2=-1).log().sum(-1)[..., None]
            - 0.5 * z.shape[-1] * LOG_2PI)


class DepletionBayesModel(nn.Module):
    """Attribute-conditioned prior, history pseudo-likelihood encoder and report law.

    The population prior is a Gaussian whose mean and scales depend on the
    (clipped) attributes. The encoder returns a Gaussian pseudo-likelihood
    exp(h'z - z'Lz/2) of the permitted history; multiplying the prior by it gives
    the Gaussian posterior approximation. Empty history gives h = 0, L = 0 and
    therefore exactly the prior. Prior and posterior tensors carry a singleton
    component axis [N,1,...], which the importance-sampling code treats as a
    one-component mixture.
    """

    def __init__(self, covariate_dim: int, report_days, phase_scales, *, width: int = 128, rank: int = 4,
                 initial_prior_std: float = 0.7,
                 initial_noise_abs: float = 0.02, initial_noise_rel: float = 0.08, noise_floor: float = 1e-4,
                 noise_mode: str = "well", covariate_clip: float = 4.0, prior_width: int | None = None,
                 component_seed: int = 0, discrepancy: str = "brownian",
                 initial_ou_sigma: float = 0.1, initial_ou_tau: float = 180.0,
                 rate_clock: bool = False, initial_rate_noise: float = 0.1, ratio_index: str = "depletion",
                 ratio_time_scale: float = 730.0, prior_linear: bool = False,
                 decay_max_days: float = DECAY_TIMES_DAYS[-1], random_prior_scale: float = 0.0):
        super().__init__()
        if discrepancy not in {"brownian", *OU_LAWS}:
            raise ValueError("discrepancy must be brownian, ou_interval or lognormal_ou")
        self.discrepancy = discrepancy
        if noise_mode not in {"global", "well"}:
            raise ValueError("noise_mode must be global or well")
        self.noise_mode = noise_mode
        self.covariate_clip = float(covariate_clip)
        # Decay-time grid: 8 log-spaced times from 10 d to decay_max_days (default 3,650 d; sensitivity option).
        grid = tuple(10.0 * (float(decay_max_days) / 10.0) ** (j / (N_MODES - 1)) for j in range(N_MODES))
        self.rate = DepletionRate(decay_times=grid, ratio_index=ratio_index, ratio_time_scale=ratio_time_scale)
        self.latent_dim = LATENT_DIM
        self.covariate_dim, self.width, self.rank = int(covariate_dim), int(width), int(rank)
        self.noise_floor = float(noise_floor)
        self.register_buffer("report_days", torch.as_tensor(np.asarray(report_days, dtype=np.float64)).float())
        self.register_buffer("phase_scales", torch.as_tensor(np.asarray(phase_scales, dtype=np.float64)).float())
        positive_days = int((np.asarray(report_days, dtype=np.float64) > 0).sum())
        self.register_buffer("zero_prob", torch.full((positive_days, 3), 1e-3))  # set from TRAIN in fit
        d = self.latent_dim
        inputs = max(self.covariate_dim, 1)
        prior_width = int(prior_width or width)
        self.prior_mean = _mlp(inputs, prior_width, d, prior_linear)
        initial = torch.zeros(d)
        initial[MODES] = math.log(0.15)
        initial[PEAK] = math.log(30.0 / (PEAK_MAX_DAYS - 30.0))
        initial[FLOWBACK_VOLUME] = math.log(0.05)
        with torch.no_grad():
            self.prior_mean[-1].weight.mul_(0.1)
            self.prior_mean[-1].bias.copy_(initial)
        # Randomized prior function (Osband et al. 2018): a fixed, untrained random network added to the
        # attribute -> prior-mean map, different for every ensemble member (seeded by component_seed).
        self.random_prior_scale = float(random_prior_scale)
        if self.random_prior_scale > 0:
            state = torch.random.get_rng_state()
            torch.manual_seed(int(component_seed) + 7919)
            self.random_prior = _mlp(inputs, prior_width, d, prior_linear)
            torch.random.set_rng_state(state)
            for param in self.random_prior.parameters():
                param.requires_grad_(False)
        self.prior_tril_raw = nn.Parameter(torch.zeros(1, d, d))
        diag0 = torch.full((d,), math.log(math.expm1(initial_prior_std)))
        diag0[NOISE] = math.log(math.expm1(0.5))
        self.prior_diag_raw = nn.Parameter(diag0.expand(1, d).clone())
        self.prior_log_scale = _mlp(inputs, prior_width, d, prior_linear)
        with torch.no_grad():
            self.prior_log_scale[-1].weight.zero_()
            self.prior_log_scale[-1].bias.zero_()
        self.context_dim = 2 * 21 + 1 + 6 + self.covariate_dim
        self.encoder = _mlp(self.context_dim, width, 2 * d + d * self.rank)
        with torch.no_grad():
            self.encoder[-1].weight.mul_(0.01)
            self.encoder[-1].bias.zero_()
            self.encoder[-1].bias[d:2 * d].fill_(-4.0)
        self.log_noise_abs = nn.Parameter(torch.full((3,), math.log(initial_noise_abs)))
        self.log_noise_rel = nn.Parameter(torch.full((3,), math.log(initial_noise_rel)))
        # Optional rate-based variance clock: V += g^2 * int q(u)^2 du (relative daily-rate fluctuations).
        self.rate_clock = bool(rate_clock)
        self.log_noise_rate = nn.Parameter(torch.full((3,), math.log(initial_rate_noise)), requires_grad=self.rate_clock)
        if discrepancy in OU_LAWS:
            self.log_ou_sigma = nn.Parameter(torch.full((3,), math.log(initial_ou_sigma)))
            self.log_ou_tau = nn.Parameter(torch.full((3,), math.log(initial_ou_tau)))

    # ----- law components -------------------------------------------------
    def noise(self) -> tuple[Tensor, Tensor]:
        return self.noise_floor + self.log_noise_abs.exp(), self.noise_floor + self.log_noise_rel.exp()

    def noise_for(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Report-noise a, r for each draw [..., 3] (normalized a; r dimensionless).

        With ``noise_mode='well'`` both are multiplied by exp(z_noise), a latent
        per-well, per-phase reporting-reliability scale with a population prior.
        """
        a, r = (v.to(z.dtype) for v in self.noise())
        if self.noise_mode == "global":
            shape = (*z.shape[:-1], 3)
            return a.expand(shape), r.expand(shape)
        multiplier = z[..., NOISE].exp()
        return a * multiplier, r * multiplier

    def clock(self, z: Tensor) -> Tensor:
        """Brownian variance clock V [..., T, 3] at the report days (normalized units).

        V(t) = a^2 t/360 + r^2 Q(t)^2 [+ g^2 * sum over 15-day bins of 2*(dQ_bin)^2, i.e. a
        30-day-correlation approximation of int q(u)^2 du], each scaled by the per-well
        reliability multiplier. Additive and increasing, so the law stays Markov.
        """
        days = self.report_days.to(z.dtype)
        q = self.rate.cumulative(z, days)
        a, r = self.noise_for(z)
        v = a.square()[..., None, :] * (days[:, None] / 360) + (r[..., None, :] * q).square()
        if self.rate_clock:
            g = self.log_noise_rate.exp().to(z.dtype)
            if self.noise_mode == "well":
                g = g * z[..., NOISE].exp()
            else:
                g = g.expand((*z.shape[:-1], 3))
            edges = torch.arange(0.0, float(days[-1]) + CLOCK_BIN_DAYS / 2, CLOCK_BIN_DAYS, dtype=z.dtype, device=z.device)
            dq = torch.diff(self.rate.cumulative(z, edges), dim=-2)
            term = torch.cumsum(dq.square() * (30.0 / CLOCK_BIN_DAYS), dim=-2)
            term = torch.cat((torch.zeros_like(term[..., :1, :]), term), dim=-2)
            index = torch.round(days / CLOCK_BIN_DAYS).long()
            v = v + g.square()[..., None, :] * term[..., index, :]
        return v

    def _clip(self, covariates: Tensor) -> Tensor:
        """Standardized attributes are clipped so atypical inputs cannot extrapolate the prior."""
        return covariates.clamp(-self.covariate_clip, self.covariate_clip) if self.covariate_clip > 0 else covariates

    def prior(self, covariates: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Float64 prior: log weights [N,1] (zero), means [N,1,D], scale_trils [N,1,D,D]."""
        covariates = self._clip(covariates)
        x = covariates if self.covariate_dim else covariates.new_zeros((len(covariates), 1))
        n, d = len(x), self.latent_dim
        mean = self.prior_mean(x).double().reshape(n, 1, d)
        if self.random_prior_scale > 0:
            mean = mean + self.random_prior_scale * self.random_prior(x).double().reshape(n, 1, d)
        diag = nn.functional.softplus(self.prior_diag_raw.double()) + 1e-3
        base = torch.tril(self.prior_tril_raw.double(), -1) + torch.diag_embed(diag)
        log_scale = math.log(3.0) * torch.tanh(self.prior_log_scale(x).double().reshape(n, 1, d) / math.log(3.0))
        return mean.new_zeros((n, 1)), mean, log_scale.exp()[..., :, None] * base

    def context(self, reports_native: Tensor, observed: Tensor, origin: Tensor, covariates: Tensor) -> Tensor:
        """Encoder input from permitted reports only; float64 native reports required."""
        days = self.report_days.to(reports_native.dtype)
        scales = self.phase_scales.to(reports_native.dtype)
        normalized = torch.where(observed, reports_native / scales, torch.zeros_like(reports_native))
        values = torch.asinh(normalized[:, 1:]).float().flatten(1)
        mask = observed[:, 1:].float().flatten(1)
        rates = last_interval_rate_features(reports_native, observed | (days == 0)[None, :, None],
                                            days, origin, scales, dtype=torch.float32)
        return torch.cat((values, mask, (origin.float() / 1080.0)[:, None], rates, self._clip(covariates.float())), dim=-1)

    def posterior(self, covariates: Tensor, context: Tensor, has_data: Tensor) -> dict[str, Tensor]:
        """Mixture prior and conjugate Gaussian-mixture posterior approximation (float64).

        Keys: log_pi, mean_p, tril_p (prior, scale trils); log_w, mean_q, tril_q
        (posterior component log weights, means and PRECISION Cholesky factors).
        """
        log_pi, mean_p, tril_p = self.prior(covariates)
        d = self.latent_dim
        eye = torch.eye(d, dtype=tril_p.dtype, device=tril_p.device)
        inv_tril_p = torch.linalg.solve_triangular(tril_p, eye.expand_as(tril_p), upper=False)
        precision_p = inv_tril_p.transpose(-1, -2) @ inv_tril_p
        out = self.encoder(context).double()
        shift = out[:, :d]
        diag = nn.functional.softplus(out[:, d:2 * d])
        factor = out[:, 2 * d:].reshape(-1, d, self.rank)
        data_precision = torch.diag_embed(diag) + factor @ factor.transpose(-1, -2)
        gate = has_data.to(mean_p.dtype)
        data_precision = (data_precision * gate[:, None, None])[:, None]
        shift = (shift * gate[:, None])[:, None]
        precision_q = precision_p + data_precision
        precision_q = 0.5 * (precision_q + precision_q.transpose(-1, -2))
        tril_q = torch.linalg.cholesky(precision_q)
        prior_natural = (precision_p @ mean_p[..., None]).squeeze(-1)
        natural = prior_natural + shift
        mean_q = torch.cholesky_solve(natural[..., None], tril_q).squeeze(-1)
        log_norm = (-torch.diagonal(tril_p, dim1=-2, dim2=-1).log().sum(-1)
                    - torch.diagonal(tril_q, dim1=-2, dim2=-1).log().sum(-1)
                    - 0.5 * (mean_p * prior_natural).sum(-1) + 0.5 * (natural * mean_q).sum(-1))
        log_w = torch.log_softmax(log_pi + log_norm, dim=-1)
        return dict(log_pi=log_pi, mean_p=mean_p, tril_p=tril_p, log_w=log_w, mean_q=mean_q, tril_q=tril_q)

    @staticmethod
    def log_prior_mix(z: Tensor, post: dict[str, Tensor]) -> Tensor:
        """log p(z|x) for z [N,S,D] under the mixture prior -> [N,S]."""
        terms = gaussian_log_prob_tril(z[:, None], post["mean_p"], post["tril_p"])
        return torch.logsumexp(post["log_pi"][..., None] + terms, dim=1)

    @staticmethod
    def log_q_mix(z: Tensor, post: dict[str, Tensor]) -> Tensor:
        terms = gaussian_log_prob_precision(z[:, None], post["mean_q"], post["tril_q"])
        return torch.logsumexp(post["log_w"][..., None] + terms, dim=1)

    @staticmethod
    def sample_posterior(mean_q: Tensor, tril_q: Tensor, count: int, generator=None, scale: float = 1.0) -> Tensor:
        eps = torch.randn((*mean_q.shape[:-1], mean_q.shape[-1], count), dtype=mean_q.dtype,
                          device=mean_q.device, generator=generator)
        offsets = torch.linalg.solve_triangular(tril_q.transpose(-1, -2), eps, upper=True)
        return mean_q[..., None, :] + scale * offsets.transpose(-1, -2)

    def stratified_terms(self, post: dict[str, Tensor], per_component: int, reports_normalized: Tensor,
                         observed: Tensor, generator=None) -> tuple[Tensor, Tensor]:
        """Stratified reparameterized ELBO and importance-weighted bound [N].

        Draw ``per_component`` samples from every posterior component; the ELBO
        weights component expectations by w_k, and the bound estimates p(H|x)
        as sum_k w_k mean_s p(H,z_ks)/q(z_ks).
        """
        z = self.sample_posterior(post["mean_q"], post["tril_q"], per_component, generator)  # [N,K,S,D]
        n, k, s, d = z.shape
        flat = z.reshape(n, k * s, d)
        log_lik = self.log_likelihood(flat, reports_normalized, observed)
        log_f = (log_lik + self.log_prior_mix(flat, post) - self.log_q_mix(flat, post)).reshape(n, k, s)
        weights = post["log_w"].exp()
        elbo = (weights * log_f.mean(-1)).sum(-1)
        bound = torch.logsumexp(post["log_w"] + torch.logsumexp(log_f, dim=-1) - math.log(s), dim=-1)
        return elbo, bound

    def predictive_crps(self, post: dict[str, Tensor], per_component: int, reports_normalized: Tensor,
                        observed: Tensor, future: Tensor, future_mask: Tensor, cell_scale: Tensor,
                        generator=None, well_weight: Tensor | None = None) -> Tensor:
        """Mean scaled CRPS of the importance-weighted Bayesian posterior predictive.

        Proposals are stratified draws from every component of q; self-normalized
        weights target prior(z|x) p(H|z). Each draw contributes the exact Gaussian
        conditional law of future reports given z and the permitted history, so the
        predictive is a weighted Gaussian mixture with closed-form CRPS. future and
        future_mask are [N,T,3] (normalized reports after the cutoff); cell_scale
        [N,T,3] holds normalized increment scales for (cutoff, endpoint).
        """
        z = self.sample_posterior(post["mean_q"], post["tril_q"], per_component, generator)
        n, k, s, d = z.shape
        z = z.reshape(n, k * s, d)
        components = gaussian_log_prob_precision(z[:, None], post["mean_q"], post["tril_q"])
        log_prop = torch.logsumexp(components, dim=1) - math.log(k)
        log_w = self.log_prior_mix(z, post) + self.log_likelihood(z, reports_normalized, observed) - log_prop
        weights = torch.softmax(log_w, dim=-1)
        q = self.cumulative(z)
        days = self.report_days.to(z.dtype)
        if self.discrepancy == LOGNORMAL:
            # Sample-based CRPS (2 reparameterized report paths per posterior draw).
            volume, known, positive = observed_intervals(reports_normalized, observed)
            mean, cov = self.log_interval_moments(z)
            x = torch.where(positive, volume.clamp_min(1e-30).log(), torch.zeros_like(volume))
            logs = sample_log_intervals(mean.float(), cov.float(), x.float(), positive, 2, generator, small=True)
            levels, after = interval_levels(logs, reports_normalized.float(), observed, self.zero_prob, generator)
            r = levels.shape[0]
            points = levels.permute(1, 3, 4, 2, 0).reshape(*levels.shape[1:2], *levels.shape[3:], -1)  # [N,3,T',S*R]
            point_w = (weights.float()[:, None, None, :, None] / r).expand(-1, 1, 1, -1, r).reshape(weights.shape[0], 1, 1, -1)
            positive_days = days > 0
            target = future[:, positive_days].transpose(-1, -2).float()
            mask = future_mask[:, positive_days].transpose(-1, -2)
            crps = weighted_sample_crps(points, point_w, torch.where(mask, target, torch.zeros_like(target))).to(z.dtype)
            scale = cell_scale[:, positive_days].transpose(-1, -2).to(z.dtype)
            scored = torch.where(mask, crps / scale, torch.zeros_like(crps))
            if well_weight is not None:
                w = well_weight.to(scored.dtype)[:, None, None]
                return (scored * w).sum() / (mask * w).sum().clamp_min(1e-12)
            return scored.sum() / mask.sum().clamp_min(1)
        if self.discrepancy in OU_LAWS:
            # Small 7x7 conditioning in float32 (fast on low-FP64 GPUs; jitter keeps it stable).
            cov = self.covariance(z)
            cov = cov + 1e-8 * torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
            mean, var = conditional_report_moments(cov.float(), q.float(), reports_normalized.float(), observed,
                                                   days.float(), small=True)
            mean, var = mean.to(z.dtype), var.to(z.dtype)
        else:
            mean, var = brownian_future_moments(q, None, None, reports_normalized.to(z.dtype), observed, days,
                                                clock=self.clock(z))
        positive = days > 0
        target = future[:, positive].transpose(-1, -2).to(z.dtype)
        mask = future_mask[:, positive].transpose(-1, -2)
        # Pairwise mixture CRPS in float32: ample precision for a score, and far faster on
        # GPUs with low float64 throughput. Likelihood and importance weights stay float64.
        crps = gaussian_mixture_crps(weights.float(), mean.float(), var.float(),
                                     torch.where(mask, target, torch.zeros_like(target)).float()).to(z.dtype)
        scale = cell_scale[:, positive].transpose(-1, -2).to(z.dtype)
        scored = torch.where(mask, crps / scale, torch.zeros_like(crps))
        if well_weight is not None:
            w = well_weight.to(scored.dtype)[:, None, None]
            return (scored * w).sum() / (mask * w).sum().clamp_min(1e-12)
        return scored.sum() / mask.sum().clamp_min(1)

    def cumulative(self, z: Tensor) -> Tensor:
        return self.rate.cumulative(z, self.report_days.to(z.dtype))

    def ou_parameters(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Per-draw OU deviation SD [...,3] (well multiplier applied) and time constants [3].

        For the Brownian law the persistent SD is zero, so ``covariance`` returns V(min(s,t)).
        """
        if self.discrepancy not in OU_LAWS:
            return z.new_zeros((*z.shape[:-1], 3)), z.new_ones(3)
        sigma = self.log_ou_sigma.exp().to(z.dtype)
        if self.noise_mode == "well":
            sigma = sigma * z[..., NOISE].exp()
        else:
            sigma = sigma.expand((*z.shape[:-1], 3))
        return sigma, self.log_ou_tau.exp().to(z.dtype)

    def covariance(self, z: Tensor) -> Tensor:
        """Report-discrepancy covariance [..., 3, T', T'] at positive report days.

        Brownian part V(min(s, t)) from ``clock``; the interval OU law adds the persistent term.
        """
        days = self.report_days.to(z.dtype)
        positive = days > 0
        v = self.clock(z)[..., positive, :].transpose(-1, -2)
        index = torch.arange(int(positive.sum()), device=z.device)
        v_min = v[..., torch.minimum(index[:, None], index[None, :])]
        if self.discrepancy == "brownian":
            return v_min
        if self.discrepancy != "ou_interval":
            raise ValueError("The log-normal report law has no Gaussian report covariance")
        sigma, tau = self.ou_parameters(z)
        return interval_ou_covariance(self.rate.cumulative(z, days), days, sigma, tau) + v_min

    def log_interval_moments(self, z: Tensor, inflation: float = 1.0) -> tuple[Tensor, Tensor]:
        """Log-normal report-law moments [..., 3, T'] / [..., 3, T', T'] for draws z."""
        days = self.report_days.to(z.dtype)
        dq = torch.diff(self.cumulative(z), dim=-2).transpose(-1, -2)
        dv = torch.diff(self.clock(z), dim=-2).transpose(-1, -2)
        sigma, tau = self.ou_parameters(z)
        return interval_log_moments(dq, dv, days, sigma, tau, inflation)

    def log_likelihood(self, z: Tensor, reports_normalized: Tensor, observed: Tensor) -> Tensor:
        if self.discrepancy == LOGNORMAL:
            volume, known, positive = observed_intervals(reports_normalized.to(z.dtype), observed)
            mean, cov = self.log_interval_moments(z)
            return lognormal_log_likelihood(volume, known, positive, mean, cov, self.zero_prob, fast=self.training)
        q = self.cumulative(z)
        days = self.report_days.to(z.dtype)
        if self.discrepancy in OU_LAWS:
            return gaussian_discrepancy_log_likelihood(reports_normalized.to(z.dtype), q, self.covariance(z),
                                                       observed, days, fast=self.training)
        return clock_log_likelihood(reports_normalized.to(z.dtype), q, self.clock(z), observed, days)


# ----- fitting and prediction ------------------------------------------------

def phase_volume_scales(train_reports: np.ndarray) -> np.ndarray:
    """Median positive terminal TRAIN report per phase, with fallbacks."""
    scales = []
    for phase in range(3):
        terminal = train_reports[:, -1, phase]
        positive = terminal[np.isfinite(terminal) & (terminal > 0)]
        if not positive.size:
            values = train_reports[:, 1:, phase]
            positive = values[np.isfinite(values) & (values > 0)]
        scales.append(float(np.median(positive)) if positive.size else 1.0)
    return np.asarray(scales)


class DepletionBayesForecaster:
    """Fit over random permitted histories (encoder by ELBO; law by importance-weighted bound or
    predictive CRPS); predict by importance sampling."""

    def __init__(self, *, seed: int = 42, epochs: int = 200, batch_size: int = 512, learning_rate: float = 2e-3,
                 train_samples: int = 8, width: int = 128, rank: int = 4,
                 history_dropout: float = 0.1, phase_dropout: float = 0.1, proposals: int = 2048,
                 defensive_weight: float = 0.25, defensive_scale: float = 2.0, device: str = "cpu",
                 log_every: int = 25, weight_decay: float = 0.0,
                 noise_mode: str = "well", covariate_clip: float = 4.0, objective: str = "iwae_theta",
                 noise_warmup: float = 0.25, initial_noise_abs: float = 0.01, initial_noise_rel: float = 0.03,
                 adaptive_rounds: int = 2, prior_width: int = 64, prior_weight_decay: float = 1e-3,
                 discrepancy: str = "brownian", predictive_samples: int = 32,
                 report_inflation: float = 1.0, rate_clock: bool = False,
                 prior_inflation: float = 1.0, bootstrap: bool = False, ratio_index: str = "depletion",
                 ratio_time_scale: float = 730.0, prior_linear: bool = False,
                 decay_max_days: float = DECAY_TIMES_DAYS[-1], random_prior_scale: float = 0.0,
                 feature_keep: float = 1.0, feature_mask: list | None = None):
        if objective not in {"iwae_theta", "predictive"}:
            raise ValueError("objective must be iwae_theta or predictive")
        self.config = dict(seed=seed, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                           train_samples=train_samples, width=width, rank=rank,
                           history_dropout=history_dropout, phase_dropout=phase_dropout, proposals=proposals,
                           defensive_weight=defensive_weight, defensive_scale=defensive_scale, device=device,
                           log_every=log_every, weight_decay=weight_decay,
                           noise_mode=noise_mode, covariate_clip=covariate_clip, objective=objective,
                           noise_warmup=noise_warmup, initial_noise_abs=initial_noise_abs,
                           initial_noise_rel=initial_noise_rel, adaptive_rounds=adaptive_rounds,
                           prior_width=prior_width, prior_weight_decay=prior_weight_decay,
                           discrepancy=discrepancy, predictive_samples=predictive_samples,
                           report_inflation=report_inflation, rate_clock=rate_clock,
                           prior_inflation=prior_inflation, bootstrap=bootstrap, ratio_index=ratio_index,
                           ratio_time_scale=ratio_time_scale, prior_linear=prior_linear,
                           decay_max_days=decay_max_days, random_prior_scale=random_prior_scale,
                           feature_keep=feature_keep, feature_mask=feature_mask)
        self.device = torch.device(device)
        self.model: DepletionBayesModel | None = None
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_diagnostics_: dict[str, Any] = {}

    # ---- helpers
    def _mask_features(self, x: np.ndarray) -> np.ndarray:
        mask = self.config.get("feature_mask")
        return x if mask is None or not x.shape[1] else x * np.asarray(mask, dtype=x.dtype)[None, :]

    @staticmethod
    def _model_kwargs(cfg) -> dict[str, Any]:
        names = ("width", "rank", "prior_width", "discrepancy", "rate_clock", "ratio_index", "ratio_time_scale",
                 "prior_linear", "decay_max_days", "random_prior_scale", "noise_mode", "covariate_clip",
                 "initial_noise_abs", "initial_noise_rel")
        return dict({name: cfg[name] for name in names}, component_seed=cfg["seed"])

    def _observed(self, reports: Tensor, cutoff: Tensor) -> Tensor:
        days = self.model.report_days.to(reports.dtype)
        return torch.isfinite(reports) & (days <= cutoff[:, None])[..., None] & (days > 0)[None, :, None]

    def _batch_terms(self, reports_native: Tensor, observed: Tensor, cutoff: Tensor, covariates: Tensor):
        model = self.model
        safe_native = torch.where(observed, reports_native, torch.zeros_like(reports_native))
        context = model.context(safe_native, observed, cutoff, covariates)
        has_data = observed.any(dim=(1, 2))
        normalized = (safe_native / model.phase_scales.to(safe_native.dtype)).float()
        return context, has_data, normalized

    def fit(self, train_reports: np.ndarray, report_days, *, covariates: np.ndarray | None = None,
            cutoffs=None) -> "DepletionBayesForecaster":
        cfg = self.config
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        reports = np.asarray(train_reports, dtype=np.float64)
        days = np.asarray(report_days, dtype=np.float64)
        if reports.ndim != 3 or reports.shape[1:] != (len(days), 3) or days[0] != 0:
            raise ValueError("train_reports must be [N,T,3] with a day-zero origin")
        n = len(reports)
        x = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, dtype=np.float64)
        if float(cfg.get("feature_keep", 1.0)) < 1.0 and x.shape[1] and cfg.get("feature_mask") is None:
            # Feature bagging: each member keeps a random subset of the attributes (value and presence
            # columns together; dropped attributes read as missing). The mask is stored for prediction.
            half = x.shape[1] // 2
            keep = np.random.default_rng(cfg["seed"] + 31337).choice(half, size=max(1, round(cfg["feature_keep"] * half)), replace=False)
            mask = np.zeros(half)
            mask[keep] = 1.0
            cfg["feature_mask"] = np.concatenate((mask, mask)).tolist()
        x = self._mask_features(x)
        scales = phase_volume_scales(reports)
        self.model = DepletionBayesModel(x.shape[1], days, scales, **self._model_kwargs(cfg)).to(self.device)
        model = self.model
        model.train()
        with torch.no_grad():  # TRAIN frequency of reported non-positive interval volumes (log-normal law atoms)
            native = torch.as_tensor(np.asarray(reports, dtype=np.float64))
            seen = torch.isfinite(native) & (model.report_days.double().cpu() > 0)[None, :, None]
            _, known_iv, positive_iv = observed_intervals(torch.nan_to_num(native), seen)
            freq = ((known_iv & ~positive_iv).sum(0) / known_iv.sum(0).clamp_min(1)).T
            model.zero_prob.copy_(freq.clamp(1e-4, 0.5).to(model.zero_prob))
        if cutoffs is None:
            cutoffs = days[:-1] if cfg["objective"] == "predictive" else days[1:]
        cutoff_choices = np.asarray(cutoffs, dtype=np.float64)
        # Normalized increment scales for every (cutoff, endpoint, phase), TRAIN only.
        cell_table = np.ones((len(days), len(days), 3))
        for k in range(3):
            phase_values = []
            for i in range(len(days)):
                for j in range(i + 1, len(days)):
                    inc = np.abs(reports[:, j, k] - reports[:, i, k]) / scales[k]
                    inc = inc[np.isfinite(inc)]
                    med = float(np.median(inc)) if inc.size else 0.0
                    cell_table[i, j, k] = med
                    phase_values.append(med)
            fallback = float(np.median([v for v in phase_values if v > 0])) if any(v > 0 for v in phase_values) else 1.0
            cell_table[..., k] = np.where(cell_table[..., k] > 0, cell_table[..., k], fallback)
        cell_table_t = torch.as_tensor(cell_table, device=self.device)
        day_index = {float(v): i for i, v in enumerate(days)}
        y = torch.as_tensor(reports, device=self.device)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        prior_nets = [v for name, v in model.named_parameters()
                      if name.startswith(("prior_mean.", "prior_log_scale.")) and v.ndim == 2]
        prior_ids = {id(v) for v in prior_nets}
        others = [v for v in model.parameters() if id(v) not in prior_ids]
        optimizer = torch.optim.AdamW([dict(params=others, weight_decay=cfg["weight_decay"]),
                                       dict(params=prior_nets, weight_decay=cfg.get("prior_weight_decay", 0.0))],
                                      lr=cfg["learning_rate"])
        steps_per_epoch = math.ceil(n / cfg["batch_size"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["learning_rate"],
                                                        total_steps=cfg["epochs"] * steps_per_epoch, pct_start=0.1)
        generator = torch.Generator(device=self.device).manual_seed(cfg["seed"] + 1)
        phi = list(model.encoder.parameters())
        phi_ids = {id(v) for v in phi}
        theta = [v for v in model.parameters() if id(v) not in phi_ids and v.requires_grad]
        noise_params = [model.log_noise_abs, model.log_noise_rel] + ([model.log_noise_rate] if model.rate_clock else []) + (
            [model.log_ou_sigma, model.log_ou_tau] if model.discrepancy in OU_LAWS else [])
        warmup_epochs = int(round(cfg.get("noise_warmup", 0.0) * cfg["epochs"]))
        # Bayesian bootstrap over TRAIN wells (Dirichlet weights, mean one): each fit then draws
        # population parameters from an approximate sampling distribution, so an equal mixture of
        # fits integrates over population-parameter uncertainty.
        if cfg.get("bootstrap", False):
            boot_np = np.random.default_rng(cfg["seed"] + 7919).exponential(1.0, n)
            boot = torch.as_tensor(boot_np / boot_np.mean(), dtype=torch.float64, device=self.device)
        else:
            boot = torch.ones(n, dtype=torch.float64, device=self.device)
        history = []
        for epoch in range(cfg["epochs"]):
            order = rng.permutation(n)
            cut = rng.choice(cutoff_choices, size=n)
            keep_point = rng.random((n, len(days), 1)) >= cfg["history_dropout"]
            keep_phase = rng.random((n, 1, 3)) >= cfg["phase_dropout"]
            keep = torch.as_tensor(keep_point & keep_phase, device=self.device)
            totals = np.zeros(3)
            for start in range(0, n, cfg["batch_size"]):
                idx = torch.as_tensor(order[start:start + cfg["batch_size"]], device=self.device)
                cutoff = torch.as_tensor(cut[order[start:start + cfg["batch_size"]]], device=self.device)
                batch = y[idx]
                observed = self._observed(batch, cutoff) & keep[idx]
                context, has_data, normalized = self._batch_terms(batch, observed, cutoff, xt[idx])
                post = model.posterior(xt[idx], context, has_data)
                per_component = max(2, cfg["train_samples"])
                elbo, bound = model.stratified_terms(post, per_component, normalized, observed, generator)
                bw = boot[idx]
                optimizer.zero_grad(set_to_none=True)
                if cfg["objective"] == "predictive":
                    # Empirical Bayes by prediction: population parameters minimize the CRPS of
                    # the importance-weighted posterior predictive of later reports; the encoder
                    # (proposal) follows the ELBO. Each well's forecast remains Bayes under theta.
                    future_mask = torch.isfinite(batch) & (model.report_days.to(batch.dtype) > cutoff[:, None])[..., None]
                    future = torch.where(future_mask, batch / model.phase_scales.to(batch.dtype), torch.zeros_like(batch))
                    rows = torch.as_tensor([day_index[float(c)] for c in cut[order[start:start + cfg["batch_size"]]]],
                                           device=self.device)
                    cell_scale = cell_table_t[rows]
                    per_pred = max(2, cfg["predictive_samples"])
                    crps = model.predictive_crps(post, per_pred, normalized, observed, future, future_mask,
                                                 cell_scale, generator, well_weight=bw)
                    theta_grads = torch.autograd.grad(crps, theta, retain_graph=True, allow_unused=True)
                    phi_grads = torch.autograd.grad(-(elbo * bw).sum() / bw.sum(), phi, allow_unused=True)
                    for param, grad in zip(theta + phi, theta_grads + phi_grads):
                        param.grad = grad
                    bound = bound.detach() * 0 + crps.detach()  # logged as mean_bound column
                else:
                    # Population (prior and report-law) parameters: importance-weighted
                    # bound, which a loose amortized posterior biases less. Encoder: ELBO.
                    theta_grads = torch.autograd.grad(-(bound * bw).sum() / bw.sum(), theta, retain_graph=True,
                                                      allow_unused=True)
                    phi_grads = torch.autograd.grad(-(elbo * bw).sum() / bw.sum(), phi, allow_unused=True)
                    for param, grad in zip(theta + phi, theta_grads + phi_grads):
                        param.grad = grad
                if epoch < warmup_epochs:
                    for param in noise_params:
                        param.grad = None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                scheduler.step()
                totals += np.array([float(elbo.detach().sum()), float(bound.detach().sum()), float(observed.sum())])
            record = dict(epoch=epoch + 1, mean_elbo=totals[0] / n, mean_bound=totals[1] / n,
                          observed_points=int(totals[2]))
            history.append(record)
            if cfg["log_every"] and ((epoch + 1) % cfg["log_every"] == 0 or epoch == 0):
                a, r = model.noise()
                extra = {}
                if model.discrepancy in OU_LAWS:
                    extra = dict(ou_sigma=model.log_ou_sigma.exp().detach().cpu().numpy().round(4).tolist(),
                                 ou_tau=model.log_ou_tau.exp().detach().cpu().numpy().round(1).tolist())
                print(dict(record, noise_abs=a.detach().cpu().numpy().round(4).tolist(),
                           noise_rel=r.detach().cpu().numpy().round(4).tolist(),
                           noise_rate=model.log_noise_rate.exp().detach().cpu().numpy().round(4).tolist(),
                           **extra), flush=True)
        model.eval()  # exact float64 likelihood for prediction-time importance weights
        a, r = model.noise()
        if model.discrepancy in OU_LAWS:
            history.append(dict(ou_sigma=model.log_ou_sigma.exp().detach().cpu().tolist(),
                                ou_tau_days=model.log_ou_tau.exp().detach().cpu().tolist()))
        self.fit_diagnostics_ = dict(training_wells=n, history=history, phase_scales=scales.tolist(),
                                     noise_abs_normalized=a.detach().cpu().tolist(),
                                     noise_rel=r.detach().cpu().tolist(), configuration=dict(cfg),
                                     latent_dim=model.latent_dim, decay_times_days=list(DECAY_TIMES_DAYS),
                                     peak_max_days=PEAK_MAX_DAYS, ratio_degree=RATIO_DEGREE)
        return self

    @torch.no_grad()
    def posterior_draws(self, history: np.ndarray, origin: float, n_samples: int, rng: np.random.Generator,
                        covariates: np.ndarray | None = None, *, dtype=torch.float64) -> dict[str, Any]:
        """Posterior draws of z [N,S,D] for one origin by adaptive importance sampling.

        Round 0 proposes from every component of the amortized mixture q and a
        widened copy (equal allocation across components). Each later round
        moment-matches a Gaussian to the weighted draws and proposes from a
        deterministic mixture of q, that Gaussian and a widened copy. The full
        proposal density enters the weights, so every round targets the exact
        unnormalized posterior p(z|x) p(H|z). Rows without positive-day
        history return iid draws from the prior.
        """
        model = self.model
        cfg = self.config
        reports = torch.as_tensor(np.asarray(history, dtype=np.float64), device=self.device)
        n, d = len(reports), self.model.latent_dim
        x = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, dtype=np.float64)
        x = self._mask_features(x)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        cutoff = torch.full((n,), float(origin), dtype=torch.float64, device=self.device)
        observed = self._observed(reports, cutoff)
        context, has_data, normalized = self._batch_terms(reports, observed, cutoff, xt)
        post = model.posterior(xt, context, has_data)
        inflation = float(cfg.get("prior_inflation", 1.0))
        if inflation != 1.0:
            # Calibration knob: widen the prior covariance (prior tempering) in the target and prior draws.
            post = dict(post, tril_p=post["tril_p"] * math.sqrt(inflation))
        eye = torch.eye(d, dtype=dtype, device=self.device)
        cov_q = torch.cholesky_inverse(post["tril_q"])
        chol_q = torch.linalg.cholesky(0.5 * (cov_q + cov_q.transpose(-1, -2)))  # [N,K,D,D]
        generator = torch.Generator(device=self.device).manual_seed(int(rng.integers(2**62)))

        def draw(mean, chol, count):
            eps = torch.randn((*mean.shape[:-1], count, d), dtype=dtype, device=self.device, generator=generator)
            return mean[..., None, :] + eps @ chol.transpose(-1, -2)

        def as_mixture(mean, chol):
            return (mean, chol) if mean.ndim == 3 else (mean[:, None], chol[:, None])

        def propose(components, total):
            """components: (weight, mean [N,(K,)D], chol [N,(K,)D,D]); equal split within a mixture."""
            weight_sum = sum(c[0] for c in components)
            parts, terms = [], []
            for weight, mean, chol in components:
                mean, chol = as_mixture(mean, chol)
                sub = mean.shape[1]
                per = max(1, int(round(total * weight / weight_sum / sub)))
                parts.append(draw(mean, chol, per).reshape(n, sub * per, d))
            z = torch.cat(parts, dim=1)
            for weight, mean, chol in components:
                mean, chol = as_mixture(mean, chol)
                sub = mean.shape[1]
                log_sub = torch.logsumexp(gaussian_log_prob_tril(z[:, None], mean, chol), dim=1) - math.log(sub)
                terms.append(math.log(weight / weight_sum) + log_sub)
            return z, torch.logsumexp(torch.stack(terms), dim=0)

        def log_target(z):
            return model.log_prior_mix(z, post) + model.log_likelihood(z, normalized, observed)

        wide = cfg["defensive_scale"]
        mean_q = post["mean_q"]
        components = [(1 - cfg["defensive_weight"], mean_q, chol_q), (cfg["defensive_weight"], mean_q, wide * chol_q)]
        z, log_prop = propose(components, cfg["proposals"])
        log_w = log_target(z) - log_prop
        for _ in range(int(cfg.get("adaptive_rounds", 2))):
            w = torch.softmax(log_w, dim=-1)
            mean = (w[..., None] * z).sum(1)
            centered = z - mean[:, None, :]
            cov = (w[..., None, None] * centered[..., :, None] * centered[..., None, :]).sum(1)
            cov = 0.5 * (cov + cov.transpose(-1, -2)) * 1.2 + 1e-6 * eye
            chol = torch.linalg.cholesky(cov)
            components = [(0.2, mean_q, chol_q), (0.5, mean, chol), (0.3, mean, wide * chol)]
            z, log_prop = propose(components, cfg["proposals"])
            log_w = log_target(z) - log_prop
        m = z.shape[1]
        weights = torch.softmax(log_w, dim=-1)
        ess = 1.0 / weights.square().sum(-1)
        u = (torch.as_tensor(rng.random(n), dtype=dtype, device=self.device)[:, None]
             + torch.arange(n_samples, dtype=dtype, device=self.device)[None, :]) / n_samples
        cdf = weights.cumsum(-1)
        cdf[:, -1] = 1.0
        index = torch.searchsorted(cdf, u).clamp_max(m - 1)
        draws = z.gather(1, index[..., None].expand(-1, -1, d))
        prior_rows = ~has_data
        if bool(prior_rows.any()):
            labels = torch.multinomial(post["log_pi"][prior_rows].exp(), n_samples, replacement=True, generator=generator)
            rows = torch.nonzero(prior_rows).squeeze(-1)
            mean_sel = post["mean_p"][rows[:, None], labels]
            chol_sel = post["tril_p"][rows[:, None], labels]
            eps = torch.randn((len(rows), n_samples, d, 1), dtype=dtype, device=self.device, generator=generator)
            draws[prior_rows] = mean_sel + (chol_sel @ eps).squeeze(-1)
            ess = torch.where(prior_rows, torch.full_like(ess, float(m)), ess)
        unique = torch.tensor([len(torch.unique(row)) for row in index], device=self.device, dtype=dtype)
        unique = torch.where(prior_rows, torch.full_like(unique, float(n_samples)), unique)
        return dict(z=draws, ess=ess, max_weight=torch.where(prior_rows, torch.full_like(ess, 1 / m), weights.max(-1).values),
                    has_data=has_data, unique_draws=unique)

    @torch.no_grad()
    def predict_distribution(self, history: np.ndarray, origin: float, n_samples: int, rng: np.random.Generator,
                             covariates: np.ndarray | None = None, chunk: int | None = None) -> dict[str, Any]:
        model = self.model
        history = np.asarray(history, dtype=np.float64)
        if chunk is None and os.environ.get("JPF_PREDICT_CHUNK"):
            chunk = int(os.environ["JPF_PREDICT_CHUNK"])  # memory-bounded forecasts when fits share one GPU
        if chunk is None:
            # Bound the largest per-chunk tensor (wells x proposals x 3 x 8 x 8, doubled for the interval OU law).
            per_well = self.config["proposals"] * 3 * 8 * 8 * (2 if model.discrepancy == "ou_interval" else 1)
            chunk = int(max(1, min(256, 1.2e8 // per_well)))
        outputs = {"report_cumulative": [], "latent_cumulative": [], "latent_parameters": []}
        ess, max_weight, unique = [], [], []
        for start in range(0, len(history), chunk):
            part = history[start:start + chunk]
            cov = None if covariates is None else np.asarray(covariates)[start:start + chunk]
            drawn = self.posterior_draws(part, origin, n_samples, rng, cov)
            z = drawn["z"]
            scales = model.phase_scales.to(torch.float64)
            latent = model.rate.cumulative(z, model.report_days.to(torch.float64)) * scales
            monotone = torch.cummax(latent, dim=2).values
            if bool(((monotone - latent) > 1e-9 * (1 + latent.abs())).any()):
                raise ValueError("Latent cumulative violated monotonicity beyond rounding")
            latent = monotone
            hist = torch.as_tensor(part, device=self.device)
            generator = torch.Generator(device=self.device).manual_seed(int(rng.integers(2**62)))
            days64 = model.report_days.to(torch.float64)
            # Report inflation: one scalar or one per phase (oil, gas, water), fitted on the calibration split.
            inflation = torch.as_tensor(self.config.get("report_inflation", 1.0), dtype=torch.float64,
                                        device=self.device).expand(3).clone()
            if model.discrepancy == LOGNORMAL:
                hist_norm = hist / scales
                obs = torch.isfinite(hist) & (days64 <= float(origin))[None, :, None] & (days64 > 0)[None, :, None]
                safe = torch.where(obs, hist_norm, torch.zeros_like(hist_norm))
                volume, known, positive = observed_intervals(safe, obs)
                mean, cov = model.log_interval_moments(z, inflation[:, None, None])
                x = torch.where(positive, volume.clamp_min(1e-30).log(), torch.zeros_like(volume))
                logs = sample_log_intervals(mean, cov, x, positive, 1, generator)
                levels, after = interval_levels(logs, safe, obs, model.zero_prob, generator)
                sampled = (levels[0] * scales[:, None]).transpose(-1, -2)  # [N,S,T',3] native
                reports = torch.zeros_like(latent)
                after_days = after.transpose(-1, -2)[:, None]  # [N,1,T',3]
                # Known endpoints are repeated; hidden endpoints before the last report (absent in the
                # curated data) are not sampled under this law and are filled with the latent path.
                past = torch.where(obs[:, None], hist[:, None], latent)[:, :, 1:]
                reports[:, :, 1:] = torch.where(after_days, sampled, past)
            elif model.discrepancy in OU_LAWS:
                native_scale = scales[:, None, None]
                cov = model.covariance(z) * native_scale * native_scale.transpose(-1, -2) * inflation[:, None, None] ** 2
                reports = sample_gaussian_discrepancy_reports(latent, hist, days64, cov, cutoff=float(origin),
                                                              generator=generator)
            else:
                clock = model.clock(z) * (scales * inflation).square()
                reports = sample_clock_reports(latent, hist, days64, clock, cutoff=float(origin), generator=generator)
            outputs["report_cumulative"].append(reports.cpu().numpy())
            outputs["latent_cumulative"].append(latent.cpu().numpy())
            outputs["latent_parameters"].append(z.cpu().numpy())
            ess.append(drawn["ess"].cpu().numpy())
            max_weight.append(drawn["max_weight"].cpu().numpy())
            unique.append(drawn["unique_draws"].cpu().numpy())
        result = {key: np.concatenate(value) for key, value in outputs.items()}
        ess, max_weight, unique = np.concatenate(ess), np.concatenate(max_weight), np.concatenate(unique)
        self.last_diagnostics_ = dict(origin=float(origin), ess_median=float(np.median(ess)),
                                      ess_p10=float(np.quantile(ess, 0.1)), max_weight_p90=float(np.quantile(max_weight, 0.9)),
                                      unique_draws_median=float(np.median(unique)),
                                      unique_draws_p10=float(np.quantile(unique, 0.1)))
        result.update(ess=ess, max_weight=max_weight, unique_draws=unique)
        return result

    def predict_samples(self, history, origin, n_samples, rng, covariates=None):
        return self.predict_distribution(history, origin, n_samples, rng, covariates)["report_cumulative"]

    def save(self, path) -> None:
        torch.save(dict(config=self.config, state=self.model.state_dict(),
                        covariate_dim=self.model.covariate_dim,
                        report_days=self.model.report_days.cpu().numpy(),
                        phase_scales=self.model.phase_scales.cpu().numpy(),
                        fit_diagnostics=self.fit_diagnostics_), path)

    # Options of the development code that every released checkpoint holds at these values.
    FIXED_OPTIONS = dict(prior_scale="contextual", importance=True, components=1, predictive_bound_weight=0.0,
                         likelihood_temperature=1.0, learn_temperature=False, oil_penalty=False)

    @classmethod
    def load(cls, path, *, device: str = "cpu") -> "DepletionBayesForecaster":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = dict(payload["config"], device=device)
        state = dict(payload["state"])
        for name, value in cls.FIXED_OPTIONS.items():
            if config.pop(name, value) != value:
                raise ValueError(f"Checkpoint uses the unsupported option {name}")
        if float(state.pop("log_eta", 0.0)) != 0.0:
            raise ValueError("Checkpoint uses a learned likelihood temperature")
        forecaster = cls(**config)
        forecaster.model = DepletionBayesModel(payload["covariate_dim"], payload["report_days"], payload["phase_scales"],
                                               **cls._model_kwargs(config)).to(device)
        forecaster.model.load_state_dict(state)
        forecaster.model.eval()
        forecaster.fit_diagnostics_ = payload.get("fit_diagnostics", {})
        return forecaster
