"""Differentiable conventional DCA for the shared probabilistic hierarchy.

Phase-major latents are log A, log tau, log p for oil, gas, water. A is in
the caller's phase-normalized volume/day, tau and query times are in days,
and p is dimensionless. There is no internal time conversion. Multiplying
cumulative volumes, rates, or totals by the caller's native phase-volume
scale recovers BBL, MCF, BBL and their per-day rates.

The mathematical law matches ``dca.DCATrajectorySamples``. This module adds
no observation model or inference approximation: those belong to the common
hierarchy. Queries are deterministic functions of a fixed latent sample.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _exprel(x: Tensor) -> Tensor:
    """Differentiable expm1(x)/x with its correct value/derivatives at zero.

    The polynomial avoids cancellation in autograd's quotient derivative.
    Both branches receive safe arguments, because an invalid unused branch
    can still contaminate gradients through ``torch.where``.
    """
    small = x.abs() < 0.05
    series_x = torch.where(small, x, torch.zeros_like(x))
    series = 1 + series_x * (1 / 2 + series_x * (1 / 6 + series_x * (1 / 24 + series_x * (1 / 120 + series_x * (1 / 720 + series_x / 5040)))))
    ordinary_x = torch.where(small, torch.ones_like(x), x)
    ordinary = torch.expm1(ordinary_x) / ordinary_x
    return torch.where(small, series, ordinary)


class DCARate(nn.Module):
    """Modified hyperbolic decline with a derivative-matched finite tail.

    Through T=1080 days, q(t)=A*(1+t/tau)**(-p). Subsequently,
    q(t)=q(T)*exp[-p*(t-T)/(tau+T)]. Positive exponentiated parameters
    guarantee nonnegative rates and coherent analytical integrals. The
    finite terminal regime is an operating/extrapolation assumption, also
    for water; total volumes are not independently validated reserves.

    No latent clipping changes the modeled law. Very extreme log parameters
    or outputs can exceed floating-point precision, as with other families.
    """

    latent_dimension = 9
    latent_size = 9
    parameters_per_phase = 3
    amplitude_indices = (0, 3, 6)
    phase_order = ("oil", "gas", "water")
    time_scale_days = 1.0

    def __init__(self, tail_start_days: float = 1080.0):
        super().__init__()
        if not math.isfinite(tail_start_days) or tail_start_days <= 0:
            raise ValueError("tail_start_days must be finite and positive")
        self.tail_start_days = float(tail_start_days)

    @staticmethod
    def _latent_parameters(z: Tensor) -> Tensor:
        if z.ndim < 1 or z.shape[-1] != 9 or not z.is_floating_point():
            raise ValueError("z must be a floating tensor with last dimension 9")
        return z.reshape(*z.shape[:-1], 3, 3)

    def _parameters_and_times(self, z: Tensor, times: Tensor | Sequence[float]) -> tuple[Tensor, Tensor, Tensor]:
        parameters = self._latent_parameters(z)
        times = torch.as_tensor(times, device=z.device, dtype=z.dtype)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if bool(torch.any(torch.isnan(times) | (times < 0))):
            raise ValueError("times must be nonnegative, finite or positive infinity")
        infinite = torch.isposinf(times)
        finite_times = torch.where(infinite, torch.zeros_like(times), times)
        return parameters, finite_times[:, None], infinite

    @staticmethod
    def _unit_integral(time: Tensor, tau: Tensor, p: Tensor) -> Tensor:
        logarithm = torch.log1p(time / tau)
        exponent = 1 - p
        x = exponent * logarithm
        near_zero = x.abs() < 0.05
        series_logarithm = torch.where(near_zero, logarithm, torch.zeros_like(logarithm))
        series_x = torch.where(near_zero, x, torch.zeros_like(x))
        series = tau * series_logarithm * _exprel(series_x)
        safe_exponent = torch.where(near_zero, torch.ones_like(exponent), exponent)
        ordinary_x = torch.where(near_zero, torch.zeros_like(x), x)
        # Away from p=1, do not multiply by log1p and divide it back out.
        # When expm1 has saturated at -1, that cancellation can make successive
        # float32 cumulatives differ downward by one ULP. The direct formula
        # contains only a monotone numerator and time-constant scale/divisor.
        ordinary = tau * torch.expm1(ordinary_x) / safe_exponent
        return torch.where(near_zero, series, ordinary)

    def initial_latent(self, *, device=None, dtype=None) -> Tensor:
        """Starting shape scaled to Q(tail_start_days)=1 per phase."""
        tau, p = 120.0, 0.8
        logarithm = math.log1p(self.tail_start_days / tau)
        unit_volume = tau * math.expm1((1 - p) * logarithm) / (1 - p)
        phase = (-math.log(unit_volume), math.log(tau), math.log(p))
        return torch.tensor(phase * 3, device=device, dtype=dtype)

    def forward(self, z: Tensor, times: Tensor | Sequence[float]) -> Tensor:
        return self.cumulative(z, times)

    def cumulative(self, z: Tensor, times: Tensor | Sequence[float]) -> Tensor:
        """Return [..., time, phase] integrated normalized volumes."""
        parameters, times, infinite = self._parameters_and_times(z, times)
        log_a = parameters[..., 0].unsqueeze(-2)
        tau = parameters[..., 1].exp().unsqueeze(-2)
        p = parameters[..., 2].exp().unsqueeze(-2)
        # At equality the pre-terminal branch supplies the entire derivative;
        # combining two clamp derivatives would double-count the boundary.
        before = times <= self.tail_start_days
        pre_time = torch.where(before, times, torch.full_like(times, self.tail_start_days))
        tail_time = torch.where(before, torch.zeros_like(times), times - self.tail_start_days)
        pre = log_a.exp() * self._unit_integral(pre_time, tau, p)
        log_terminal_rate = log_a - p * torch.log1p(self.tail_start_days / tau)
        decay = p / (tau + self.tail_start_days)
        tail = log_terminal_rate.exp() * tail_time * _exprel(-decay * tail_time)
        volume = pre + tail
        if not bool(torch.any(infinite)):
            # A representable finite-time forecast need not have a total that
            # fits this dtype. Do not let an unused overflowing total poison
            # otherwise valid endpoint gradients through torch.where.
            return volume
        return torch.where(infinite[:, None], self.total_volume(z).unsqueeze(-2), volume)

    def rate(self, z: Tensor, times: Tensor | Sequence[float]) -> Tensor:
        """Return [..., time, phase] rates in normalized volume/day."""
        parameters, times, infinite = self._parameters_and_times(z, times)
        log_a = parameters[..., 0].unsqueeze(-2)
        tau = parameters[..., 1].exp().unsqueeze(-2)
        p = parameters[..., 2].exp().unsqueeze(-2)
        before = times <= self.tail_start_days
        pre_time = torch.where(before, times, torch.full_like(times, self.tail_start_days))
        tail_time = torch.where(before, torch.zeros_like(times), times - self.tail_start_days)
        log_rate = log_a - p * torch.log1p(pre_time / tau) - p * tail_time / (tau + self.tail_start_days)
        rate = log_rate.exp()
        return torch.where(infinite[:, None], torch.zeros_like(rate), rate)

    def total_volume(self, z: Tensor) -> Tensor:
        """Return [..., phase] lifetime integrals under the terminal assumption."""
        parameters = self._latent_parameters(z)
        log_a, log_tau, log_p = parameters.unbind(dim=-1)
        tau, p = log_tau.exp(), log_p.exp()
        pre = log_a.exp() * self._unit_integral(z.new_tensor(self.tail_start_days), tau, p)
        log_terminal_rate = log_a - p * torch.log1p(self.tail_start_days / tau)
        log_inverse_decay = torch.logaddexp(log_tau, torch.full_like(log_tau, math.log(self.tail_start_days))) - log_p
        return pre + (log_terminal_rate + log_inverse_decay).exp()
