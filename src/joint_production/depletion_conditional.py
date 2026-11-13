"""Score-trained control: the depletion rate family inside the conditional trainer.

This isolates the rate representation from the inference method. The trainer,
encoder, accumulated-Gaussian report bridge, sampled CRPS/energy objective,
budget and inputs are exactly those of ``ConditionalRateForecaster``; only the
rate family changes to the physics-structured depletion law of
``depletion_bayes`` (ramp then completely monotone oil, depletion-indexed
Bernstein gas/water ratios). It is a direct conditional predictive law, not a
Bayesian posterior.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .conditional_rates import ConditionalRateForecaster
from .depletion_bayes import FLOWBACK_VOLUME, MODES, PEAK, PEAK_MAX_DAYS, RATE_DIM, DepletionRate


class DepletionRateModule(nn.Module):
    """Rate-family interface expected by ConditionalRateForecaster."""

    latent_size = RATE_DIM
    phase_order = ("oil", "gas", "water")

    def __init__(self):
        super().__init__()
        self.law = DepletionRate()

    def initial_latent(self, *, device=None, dtype=None) -> Tensor:
        z = torch.zeros(RATE_DIM, device=device, dtype=dtype)
        z[MODES] = math.log(0.15)
        z[PEAK] = math.log(30.0 / (PEAK_MAX_DAYS - 30.0))
        z[FLOWBACK_VOLUME] = math.log(0.05)
        return z

    def forward(self, z: Tensor, times) -> Tensor:
        return self.cumulative(z, times)

    def cumulative(self, z: Tensor, times) -> Tensor:
        return self.law.cumulative(z, times)

    def rate(self, z: Tensor, times) -> Tensor:
        return self.law.rate(z, times)

    def total_volume(self, z: Tensor) -> Tensor:
        return self.law.total_volume(z)


class DepletionConditionalRateForecaster(ConditionalRateForecaster):
    """ConditionalRateForecaster with the depletion rate family swapped in."""

    def __init__(self, **kwargs):
        kwargs.pop("rate_family", None)
        super().__init__(rate_family="dca", **kwargs)
        self.rate_model = DepletionRateModule()
        self.latent_size = self.rate_model.latent_size
        self.configuration["rate_family"] = "depletion"
        self.rate_family = "depletion"
