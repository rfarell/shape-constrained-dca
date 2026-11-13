"""Positive-basis rate family with analytic integrals (the positive-basis comparator).

Latents are phase-major (oil, gas, water), four parameters per phase. Volumes
are in the caller's phase-normalized units; query times and returned rates
use days and normalized volume/day. No random draws occur when querying a
trajectory. These finite tails describe continued operation under the model,
not a reservoir/material-balance guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class PositiveBasisRate(nn.Module):
    """Four fixed Erlang densities per phase, with positive volume weights.

    The (integer shape, scale in days) pairs are (1,30), (1,180), (2,360), (2,720).
    Each density integrates to one on [0,infinity); exp(z) gives its volume
    contribution. No monotone-decline constraint is imposed.
    """

    latent_size = 12
    parameters_per_phase = 4
    phase_order = ("oil", "gas", "water")
    time_scale_days = 360.0
    amplitude_indices = tuple(range(12))
    shapes = (1, 1, 2, 2)
    scales_days = (30., 180., 360., 720.)

    @staticmethod
    def _parameters_and_times(z: Tensor, times: Tensor | Sequence[float]):
        if z.shape[-1] != 12 or not z.is_floating_point():
            raise ValueError("z must be a floating tensor with last dimension 12")
        times = torch.as_tensor(times, device=z.device, dtype=z.dtype)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if bool(torch.any(torch.isnan(times) | (times < 0))):
            raise ValueError("times must be nonnegative, finite or positive infinity")
        infinite = torch.isposinf(times)
        # Avoid inf*0 and invalid gradients in branches later discarded by where.
        finite_times = torch.where(infinite, torch.zeros_like(times), times)
        return z.reshape(*z.shape[:-1], 3, 4), finite_times / 360.0, infinite

    def initial_latent(self, *, device=None, dtype=None) -> Tensor:
        # Equal contributions normalized so Q(1080)=1 in each phase.
        prototype = torch.zeros(12, device=device, dtype=dtype)
        mass = self.cumulative(prototype, [1080.0])[0, 0]
        return torch.full_like(prototype, -float(mass.log()))

    def cumulative(self, z: Tensor, times: Tensor | Sequence[float]) -> Tensor:
        p, u, infinite = self._parameters_and_times(z, times)
        tau = z.new_tensor([scale / self.time_scale_days for scale in self.scales_days])
        shape = z.new_tensor(self.shapes)
        x = u[:, None] / tau
        # Gamma CDF's library backward has a 0*log(0) singularity at zero.
        # Its convergent lower-tail series gives the correct boundary derivative
        # and avoids cancellation: exp(-x)*x**n/n! * sum_j x**j/(n+1)_j.
        term = torch.ones_like(x)
        series = term
        small_x = x.clamp_max(.1)
        for j in range(1, 9):
            term = term * small_x / (shape + j)
            series = series + term
        small_cdf = small_x.pow(shape) * torch.exp(-small_x) * series / z.new_tensor([math.factorial(n) for n in self.shapes])
        ordinary_cdf = torch.special.gammainc(shape, x.clamp_min(.1))
        basis = torch.where(x < .1, small_cdf, ordinary_cdf)
        basis = torch.where(infinite[:, None], torch.ones_like(basis), basis)
        return torch.einsum("...pk,tk->...tp", p.exp(), basis)

    def rate(self, z: Tensor, times: Tensor | Sequence[float]) -> Tensor:
        p, u, infinite = self._parameters_and_times(z, times)
        tau = z.new_tensor([scale / self.time_scale_days for scale in self.scales_days])
        x = u[:, None] / tau
        polynomial = torch.stack(tuple(torch.ones_like(u) if n == 1 else x[:, index].pow(n - 1) / math.factorial(n - 1)
                                       for index, n in enumerate(self.shapes)), dim=-1)
        basis = polynomial * torch.exp(-x) / tau / 360.0
        basis = torch.where(infinite[:, None], torch.zeros_like(basis), basis)
        return torch.einsum("...pk,tk->...tp", p.exp(), basis)

    def total_volume(self, z: Tensor) -> Tensor:
        if z.shape[-1] != 12:
            raise ValueError("z must have last dimension 12")
        return z.reshape(*z.shape[:-1], 3, 4).exp().sum(-1)
