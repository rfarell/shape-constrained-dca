"""Per-well decline-curve benchmarks: deterministic and probabilistic Arps, Duong, SEDM, scaling.

Industry practice fits each phase of each well independently to that well's own
production history. Here every phase uses the same family, in phase-normalized
volume units and days, fitted to the permitted cumulative reports:

  arps   modified hyperbolic, q = qi (1 + b Di t)^(-1/b), switching to exponential
         decline at the terminal nominal rate D_min (8%/year effective);
         parameters log qi, log Di, b = 0.05 + 1.9 sigmoid(.)
  sedm   stretched exponential (Valko 2009), q = qi exp(-(t/tau)^n);
         parameters log qi, log tau, n = 0.05 + 0.95 sigmoid(.)
  duong  Duong (2011), Q = (q1/a) exp(a/(1-m) (tm^(1-m) - 1)), tm in months;
         parameters log q1, log a, m = 1.01 + 0.59 sigmoid(.)
  scaling  Patzek, Male & Marder (2013) slab scaling curve, Q = M m(t/tau) with
         m(x) = 1 - (8/pi^2) sum_k exp(-(2k+1)^2 pi^2 x/4)/(2k+1)^2 (linear flow,
         then exponential decline; finite EUR = M); parameters log M, log tau; the
         third coordinate is inert (common layout, no likelihood effect)

Report errors are independent across successive observed increments, with
variance (sigma_r dQ)^2 + sigma_a^2 dt/360 per phase, fitted on TRAIN.

``mode='bayes'`` is probabilistic DCA in the style of Gong et al. (2014): a
Gaussian parameter prior from the fitted parameters of the k most similar TRAIN
wells (standardized attributes), updated with the well's permitted reports by
adaptive importance sampling; forecasts add report noise. ``mode='map'`` is the
deterministic per-well fit (posterior mode under the same analog prior, which
regularizes sparse early histories); it issues identical draws (a P50 point
forecast). Phases are independent, as in standard per-well DCA.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

FAMILIES = ("arps", "sedm", "duong", "scaling")
_SLAB_SWITCH = 0.1  # below this dimensionless time the short-time expansion is used
D_MIN = -math.log(1 - 0.08) / 365.25
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def _exprel(x: Tensor) -> Tensor:
    small = x.abs() < 1e-6
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 + x / 2, torch.expm1(safe) / safe)


def dca_cumulative(family: str, theta: Tensor, t: Tensor) -> Tensor:
    """Cumulative Q(t) [..., T] for parameters theta [..., 3] (normalized units, days)."""
    t = t.to(theta.dtype)
    p0, p1, p2 = theta[..., 0:1], theta[..., 1:2], theta[..., 2:3]
    if family == "arps":
        qi, di = p0.exp(), p1.exp()
        b = 0.05 + 1.9 * torch.sigmoid(p2)
        t_star = ((di / D_MIN - 1) / (b * di)).clamp_min(0)

        def hyperbolic(tt):
            log1p = torch.log1p(b * di * tt)
            return qi * log1p / (b * di) * _exprel((b - 1) / b * log1p)

        tt = torch.minimum(t, t_star)
        q_star = qi * torch.exp(-torch.log1p(b * di * t_star) / b)
        tail = q_star / D_MIN * -torch.expm1(-D_MIN * (t - t_star).clamp_min(0))
        return hyperbolic(tt) + tail
    if family == "sedm":
        qi, tau = p0.exp(), p1.exp()
        n = 0.05 + 0.95 * torch.sigmoid(p2)
        # int_0^{t/tau} exp(-v^n) dv = (1/n) int_0^X w^(1/n-1) e^-w dw with X = (t/tau)^n; the
        # gamma-type integrand is negligible beyond w = 200, and w = X s^2 removes the
        # endpoint singularity so Gauss-Legendre converges rapidly for any tau.
        x = torch.as_tensor(_GL_X, dtype=theta.dtype, device=theta.device)
        gw = torch.as_tensor(_GL_W, dtype=theta.dtype, device=theta.device)
        upper = (t / tau).clamp_min(1e-300)
        big_x = torch.exp(n * torch.log(upper)).clamp(max=200.0)[..., None]
        sq = (x + 1) / 2
        w = big_x * sq.square()
        power = (1 / n - 1)[..., None]
        integrand = 2 * big_x * sq * torch.exp(power * torch.log(w.clamp_min(1e-300)) - w)
        integral = (integrand * gw / 2).sum(-1)
        value = qi * tau / n * torch.where(t > 0, integral, torch.zeros_like(integral))
        return torch.cummax(value, dim=-1).values if bool((torch.diff(t) >= 0).all()) else value
    if family == "duong":
        q1, a = p0.exp(), p1.exp()
        m = 1.01 + 0.59 * torch.sigmoid(p2)
        def duong(tm):
            exponent = a / (1 - m) * (tm.pow(1 - m) - 1)
            return q1 / a * torch.exp(exponent.clamp(max=60.0))

        start = duong(torch.full_like(t[:1], 1e-6))  # Duong's limit at t -> 0 need not vanish
        value = duong((t / 30.4375).clamp_min(1e-6)) - start
        return torch.where(t > 0, value, torch.zeros_like(value))
    if family == "scaling":
        x = t / p1.exp()
        small = x.clamp(1e-12, _SLAB_SWITCH)
        root = small.sqrt()
        # Short time: m = 2 sqrt(x/pi) (1 - 2 sqrt(pi) ierfc(1/sqrt x) + ...), ierfc(z) = e^-z^2/sqrt(pi) - z erfc(z).
        ierfc = torch.exp(-1 / small) / math.sqrt(math.pi) - torch.erfc(1 / root) / root
        early = 2 * root / math.sqrt(math.pi) * (1 - 2 * math.sqrt(math.pi) * ierfc)
        odd = torch.arange(1, 22, 2, dtype=theta.dtype, device=theta.device)
        late_x = x.clamp_min(_SLAB_SWITCH)[..., None]
        series = (torch.exp(-odd.square() * math.pi ** 2 * late_x / 4) / odd.square()).sum(-1)
        late = 1 - 8 / math.pi ** 2 * series
        value = p0.exp() * torch.where(x < _SLAB_SWITCH, early, late)
        return torch.where(t > 0, value, torch.zeros_like(value))
    raise ValueError(f"family must be one of {FAMILIES}")


def increment_log_likelihood(theta: Tensor, reports: Tensor, mask: Tensor, days: Tensor, family: str,
                             sigma_r: Tensor, sigma_a: Tensor) -> Tensor:
    """log p(successive observed increments | theta) [..., S] for one phase.

    reports/mask [N,T]; theta [N,S,3]; sigma scalars (normalized units).
    """
    q = dca_cumulative(family, theta, days)  # [N,S,T]
    n, s, t = q.shape
    prev_q = q.new_zeros((n, s))
    prev_y = q.new_zeros((n, 1))
    prev_t = q.new_zeros((n, 1))
    total = q.new_zeros((n, s))
    for j in range(t):
        if float(days[j]) <= 0:
            continue
        include = mask[:, j:j + 1]
        dq = q[:, :, j] - prev_q
        dy = torch.where(include, reports[:, j:j + 1], torch.zeros_like(prev_y)) - prev_y
        var = (sigma_r * dq).square() + sigma_a.square() * (days[j] - prev_t) / 360
        var = torch.where(include, var.clamp_min(1e-12), torch.ones_like(var))
        lp = -0.5 * (math.log(2 * math.pi) + var.log() + (dy - dq).square() / var)
        total = total + torch.where(include, lp, torch.zeros_like(lp))
        prev_q = torch.where(include, q[:, :, j], prev_q)
        prev_y = torch.where(include, reports[:, j:j + 1], prev_y)
        prev_t = torch.where(include, days[j].expand_as(prev_t), prev_t)
    return total


def _initial(family: str) -> list[float]:
    return {"arps": [math.log(0.01), math.log(0.01), 0.0],
            "sedm": [math.log(0.01), math.log(200.0), 0.0],
            "duong": [math.log(0.05), math.log(1.0), 0.0],
            "scaling": [math.log(2.0), math.log(2000.0), 0.0]}[family]


class PerWellDCAForecaster:
    """Per-well DCA benchmark with analog-informed priors (see module docstring)."""

    def __init__(self, *, family: str = "arps", mode: str = "bayes", analogs: int = 200, proposals: int = 4096,
                 rounds: int = 2, fit_steps: int = 1500, seed: int = 42, device: str = "cpu", threads: int = 8):
        if family not in FAMILIES or mode not in {"bayes", "map"}:
            raise ValueError("Unsupported family or mode")
        self.config = dict(family=family, mode=mode, analogs=analogs, proposals=proposals, rounds=rounds,
                           fit_steps=fit_steps, seed=seed, device=device, threads=threads)
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(int(threads))
        self.fit_diagnostics_: dict[str, Any] = {}
        self.last_diagnostics_: dict[str, Any] = {}

    # ----- fitting
    def _map_fit(self, reports: Tensor, mask: Tensor, prior_mean: Tensor, prior_prec: Tensor, sigma_r, sigma_a,
                 steps: int) -> Tensor:
        """Batched posterior-mode fit for one phase: theta [N,3]."""
        family = self.config["family"]
        days = self.days
        theta = prior_mean.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=0.05)
        for _ in range(steps):
            optimizer.zero_grad()
            ll = increment_log_likelihood(theta[:, None], reports, mask, days, family, sigma_r, sigma_a)[:, 0]
            diff = (theta - prior_mean)[..., None]
            lp = -0.5 * (diff.transpose(-1, -2) @ prior_prec @ diff).squeeze(-1).squeeze(-1)
            (-(ll + lp).sum()).backward()
            optimizer.step()
        return theta.detach()

    def fit(self, train_reports: np.ndarray, report_days, *, covariates: np.ndarray, origins=None):
        torch.manual_seed(self.config["seed"])
        reports = np.asarray(train_reports, dtype=np.float64)
        days = np.asarray(report_days, dtype=np.float64)
        scales = []
        for k in range(3):
            terminal = reports[:, -1, k]
            positive = terminal[np.isfinite(terminal) & (terminal > 0)]
            scales.append(float(np.median(positive)) if positive.size else 1.0)
        self.scales = np.asarray(scales)
        self.days = torch.as_tensor(days, device=self.device)
        x = np.asarray(covariates, dtype=np.float64)
        self.train_x = np.clip(x, -4, 4)
        y = torch.as_tensor(reports / self.scales, device=self.device)
        observed = torch.isfinite(y) & (self.days > 0)[None, :, None]
        y = torch.nan_to_num(y)
        family = self.config["family"]
        init = torch.as_tensor(_initial(family), dtype=torch.float64, device=self.device)
        weak = torch.eye(3, dtype=torch.float64, device=self.device) / 9.0
        self.train_theta = torch.zeros((len(reports), 3, 3), dtype=torch.float64, device=self.device)
        self.sigma_r = torch.full((3,), 0.08, dtype=torch.float64, device=self.device)
        self.sigma_a = torch.full((3,), 0.02, dtype=torch.float64, device=self.device)
        for k in range(3):
            prior_mean = init.expand(len(reports), 3).clone()
            theta = self._map_fit(y[:, :, k], observed[:, :, k], prior_mean, weak, self.sigma_r[k], self.sigma_a[k],
                                  self.config["fit_steps"])
            # Report-noise scales maximizing the pooled TRAIN likelihood at the fitted curves.
            log_s = torch.log(torch.stack((self.sigma_r[k], self.sigma_a[k]))).clone().requires_grad_(True)
            opt = torch.optim.Adam([log_s], lr=0.05)
            for _ in range(300):
                opt.zero_grad()
                ll = increment_log_likelihood(theta[:, None], y[:, :, k], observed[:, :, k], self.days, family,
                                              log_s[0].exp(), log_s[1].exp())
                (-ll.sum()).backward()
                opt.step()
            self.sigma_r[k], self.sigma_a[k] = log_s.detach().exp()
            self.train_theta[:, k] = self._map_fit(y[:, :, k], observed[:, :, k], theta, weak, self.sigma_r[k],
                                                   self.sigma_a[k], 300)
        self.fit_diagnostics_ = dict(family=family, mode=self.config["mode"], training_wells=len(reports),
                                     sigma_r=self.sigma_r.tolist(), sigma_a=self.sigma_a.tolist(),
                                     phase_scales=self.scales.tolist(), configuration=dict(self.config))
        return self

    def _analog_prior(self, covariates: np.ndarray, phase: int) -> tuple[Tensor, Tensor]:
        """Gaussian prior [N,3], [N,3,3] from the k nearest TRAIN wells' fitted parameters."""
        x = np.clip(np.asarray(covariates, dtype=np.float64), -4, 4)
        k = min(self.config["analogs"], len(self.train_x))
        theta = self.train_theta[:, phase].cpu().numpy()
        global_cov = np.cov(theta.T) + 1e-6 * np.eye(3)
        means, covs = [], []
        for start in range(0, len(x), 256):
            d2 = ((x[start:start + 256, None, :] - self.train_x[None]) ** 2).sum(-1)
            nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
            sample = theta[nearest]  # [n,k,3]
            mean = sample.mean(1)
            centered = sample - mean[:, None]
            cov = centered.transpose(0, 2, 1) @ centered / (k - 1)
            covs.append(0.8 * cov + 0.2 * global_cov + 1e-4 * np.eye(3))
            means.append(mean)
        return (torch.as_tensor(np.concatenate(means), device=self.device),
                torch.as_tensor(np.concatenate(covs), device=self.device))

    # ----- prediction
    @torch.no_grad()
    def _posterior(self, y, observed, mean, cov, phase, count, generator):
        """Adaptive importance sampling of theta [N,count,3] for one phase."""
        family = self.config["family"]
        m = self.config["proposals"]
        n = len(y)
        prior = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        has = observed.any(-1)

        def draw(mu, chol, size):
            eps = torch.randn((n, size, 3), dtype=mu.dtype, device=mu.device, generator=generator)
            return mu[:, None] + eps @ chol.transpose(-1, -2)

        chol_p = torch.linalg.cholesky(cov)
        z = draw(mean, chol_p, m)
        log_prop = prior.log_prob(z.transpose(0, 1)).T
        log_w = increment_log_likelihood(z, y, observed, self.days, family, self.sigma_r[phase], self.sigma_a[phase])
        for _ in range(self.config["rounds"]):
            w = torch.softmax(log_w, -1)
            mu = (w[..., None] * z).sum(1)
            c = z - mu[:, None]
            cv = (w[..., None, None] * c[..., :, None] * c[..., None, :]).sum(1) * 1.3 + 1e-6 * torch.eye(3, dtype=z.dtype, device=z.device)
            chol = torch.linalg.cholesky(0.5 * (cv + cv.transpose(-1, -2)))
            half = m // 2
            z = torch.cat((draw(mu, chol, half), draw(mean, chol_p, m - half)), 1)
            local = torch.distributions.MultivariateNormal(mu, scale_tril=chol)
            log_prop = torch.logaddexp(math.log(0.5) + local.log_prob(z.transpose(0, 1)).T,
                                       math.log(0.5) + prior.log_prob(z.transpose(0, 1)).T)
            log_w = (prior.log_prob(z.transpose(0, 1)).T - log_prop
                     + increment_log_likelihood(z, y, observed, self.days, family, self.sigma_r[phase], self.sigma_a[phase]))
        log_w = torch.where(has[:, None], log_w, torch.zeros_like(log_w))
        weights = torch.softmax(log_w, -1)
        index = torch.multinomial(weights, count, replacement=True, generator=generator)
        ess = 1 / weights.square().sum(-1)
        return z.gather(1, index[..., None].expand(-1, -1, 3)), ess

    @torch.no_grad()
    def predict_samples(self, history, origin, n_samples, rng, covariates=None):
        family = self.config["family"]
        hist = np.asarray(history, dtype=np.float64)
        n = len(hist)
        days = self.days
        cutoff = float(origin)
        out = np.empty((n, n_samples, len(days), 3))
        generator = torch.Generator(device=self.device).manual_seed(int(rng.integers(2**62)))
        ess_all = []
        for k in range(3):
            y = torch.as_tensor(hist[:, :, k] / self.scales[k], device=self.device)
            observed = torch.isfinite(y) & (days > 0)[None] & (days <= cutoff)[None]
            y = torch.nan_to_num(y)
            mean, cov = self._analog_prior(covariates, k)
            if self.config["mode"] == "map":
                prec = torch.linalg.inv(cov)
                with torch.enable_grad():
                    theta = self._map_fit(y, observed, mean, prec, self.sigma_r[k], self.sigma_a[k], 600)
                draws = theta[:, None].expand(n, n_samples, 3)
            else:
                # Bounded memory: SEDM quadrature is proposals x times x 48 nodes per well.
                chunk = 32 if family == "sedm" else 256
                parts, ess_parts = [], []
                for start in range(0, n, chunk):
                    sl = slice(start, start + chunk)
                    d_, e_ = self._posterior(y[sl], observed[sl], mean[sl], cov[sl], k, n_samples, generator)
                    parts.append(d_)
                    ess_parts.append(e_)
                draws = torch.cat(parts)
                ess_all.append(torch.cat(ess_parts).cpu().numpy())
            q = dca_cumulative(family, draws, days)  # [N,S,T]
            # Anchor at the last observed report; add independent increment noise afterwards (bayes only).
            index = torch.arange(len(days), device=self.device)
            last = torch.where(observed, index, torch.full_like(index, -1)).max(-1).values  # [N]
            anchor_y = torch.where(last >= 0, y.gather(1, last.clamp_min(0)[:, None]).squeeze(1), torch.zeros(n, dtype=y.dtype, device=y.device))
            anchor_q = torch.where((last >= 0)[:, None], q.gather(2, last.clamp_min(0)[:, None, None].expand(n, n_samples, 1)).squeeze(-1),
                                   torch.zeros_like(q[:, :, 0]))
            path = anchor_y[:, None, None] + q - anchor_q[..., None]
            if self.config["mode"] == "bayes":
                dq = torch.diff(q, dim=-1, prepend=torch.zeros_like(q[..., :1]))
                dt = torch.diff(days, prepend=days[:1] * 0)
                var = (self.sigma_r[k] * dq).square() + self.sigma_a[k].square() * dt / 360
                after = (index[None, None] > last[:, None, None])
                noise = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator) * var.sqrt()
                path = path + torch.cumsum(torch.where(after, noise, torch.zeros_like(noise)), dim=-1)
            known = observed[:, None].expand_as(path)
            path = torch.where(known, y[:, None].expand_as(path), path)
            path[:, :, days == 0] = 0.0
            out[..., k] = path.cpu().numpy() * self.scales[k]
        self.last_diagnostics_ = dict(origin=cutoff, ess_median=[float(np.median(e)) for e in ess_all])
        return out
