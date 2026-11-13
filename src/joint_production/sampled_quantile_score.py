"""Differentiable marginal scores for a finite bank of predictive samples.

These helpers do not aggregate cells or phases, set scales, or change a model's
sampling law. Empirical WIS scores five interpolated sample quantiles. Its
finite-bank expectation is generally different from the population-quantile
score; there is no unbiasedness claim. The U-CRPS estimator is unbiased for
population CRPS when the bank is iid and the relevant first moments exist.
All nonfinite targets are unscored, matching the existing metric convention.
Callers must validate upstream differencing/scaling against arithmetic overflow
before invoking these helpers; they cannot distinguish overflow from missingness.
"""
from __future__ import annotations

import torch
from torch import Tensor


WIS_QUANTILE_LEVELS = (.025, .1, .5, .9, .975)


def _masked_inputs(targets: Tensor, samples: Tensor, target_mask: Tensor | None,
                   minimum_samples: int) -> tuple[Tensor, Tensor, Tensor]:
    if (not isinstance(targets, Tensor) or not isinstance(samples, Tensor)
            or targets.dtype not in (torch.float32, torch.float64)
            or samples.dtype != targets.dtype or samples.device != targets.device):
        raise ValueError("Targets and samples must share a float32/float64 dtype and device")
    if (targets.ndim < 1 or samples.ndim != targets.ndim + 1
            or not all(targets.shape)
            or samples.shape[:1] + samples.shape[2:] != targets.shape
            or samples.shape[1] < minimum_samples):
        raise ValueError(
            f"Require targets [N,...] and samples [N,M,...] with nonempty axes and M>={minimum_samples}"
        )
    valid = torch.isfinite(targets)
    if target_mask is not None:
        if (not isinstance(target_mask, Tensor) or target_mask.dtype != torch.bool
                or target_mask.device != targets.device or target_mask.shape != targets.shape):
            raise ValueError("target_mask must be a boolean tensor with the targets' shape and device")
        valid = valid & target_mask
    # Missing or explicitly hidden targets and their samples are unscored.
    # Remove them before value validation, quantiles, differences, or sorting.
    clean_targets = torch.where(valid, targets, torch.zeros_like(targets))
    clean_samples = torch.where(valid.unsqueeze(1), samples, torch.zeros_like(samples))
    if not bool(torch.isfinite(clean_samples).all()):
        raise ValueError("Every active target requires finite predictive samples")
    return clean_targets, clean_samples, valid


def _finite_score(score: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("Sample score exceeded floating-point support")
    return torch.where(valid, score, torch.zeros_like(score)), valid


def empirical_wis(targets: Tensor, samples: Tensor, *,
                  target_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return per-cell empirical 80%/95% WIS and its boolean validity mask.

    Shapes are targets=[N,...], samples=[N,M,...], M>=1; both tensors must
    share a float32/float64 dtype/device. Signed values are retained. Validity
    is finite(targets), intersected with an optional same-shape boolean mask.
    Inactive targets and samples are sanitized before arithmetic and return
    zero scores and zero gradients. Active samples must all be finite.

    Quantile levels are .025, .1, .5, .9, .975, using torch.quantile with
    interpolation='linear' along M. The sum of their five pinball losses is
    divided by 2.5, matching metrics.weighted_interval_score. Scores have the
    input's volume units. The caller supplies any scaling and weighted
    aggregation; the returned mask distinguishes missing cells from zero loss.

    Gradients are piecewise derivatives of the finite-bank statistic. At order
    ties or quantile/target equality PyTorch chooses a subgradient; individual
    tied samples need not receive equal gradients. An iid predictive bank is
    the intended interpretation, but this empirical quantile score is not an
    unbiased estimator of the population-quantile WIS.
    """
    targets, samples, valid = _masked_inputs(targets, samples, target_mask, 1)
    levels = samples.new_tensor(WIS_QUANTILE_LEVELS)
    quantiles = torch.quantile(samples, levels, dim=1, interpolation="linear")
    residual = targets.unsqueeze(0) - quantiles
    probability = levels.reshape((len(WIS_QUANTILE_LEVELS),) + (1,) * targets.ndim)
    pinball = torch.maximum(probability * residual, (probability - 1) * residual)
    return _finite_score(pinball.sum(dim=0) / 2.5, valid)


def sampled_u_crps(targets: Tensor, samples: Tensor, *,
                   target_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return per-cell iid U-CRPS and validity, excluding all self pairs.

    Input units, shapes, masks, and gradient conventions match empirical_wis,
    with M>=2 required. The estimator is mean_i |X_i-y| minus
    sum_{i<j}|X_i-X_j|/[M(M-1)], as in conditional_rates._u_scores.

    Sorting costs O(M log M) and uses O(M) working storage per cell. For sorted
    x, the unordered pair sum equals sum_{i=1}^{M-1} i(M-i)(x[i]-x[i-1]).
    Weighted adjacent gaps avoid a pairwise tensor and cancellation from a
    large common sample level. No sample, target, or score is clipped. The
    finite score and its derivatives agree with the pairwise expression away
    from order ties; tied-sample subgradients can differ from pairwise abs(0).
    """
    targets, samples, valid = _masked_inputs(targets, samples, target_mask, 2)
    count = samples.shape[1]
    ordered = samples.sort(dim=1).values
    gap = ordered[:, 1:] - ordered[:, :-1]
    index = torch.arange(1, count, dtype=samples.dtype, device=samples.device)
    # Normalize factors before multiplying, avoiding large integer products.
    weight = (index / count) * ((count - index) / (count - 1))
    weight = weight.reshape((1, count - 1) + (1,) * (targets.ndim - 1))
    dispersion = (gap * weight).sum(dim=1)
    score = (samples - targets.unsqueeze(1)).abs().mean(dim=1) - dispersion
    return _finite_score(score, valid)
