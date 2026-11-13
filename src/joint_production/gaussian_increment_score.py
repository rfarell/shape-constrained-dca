"""Integrate accumulated Gaussian report noise out of marginal U-CRPS.

This helper changes neither the predictive law nor the joint energy score.
It is the report-noise conditional expectation of the iid marginal estimator
in ``conditional_rates._u_scores``, conditional on the same latent base draws.
Its reduced marginal noise does not guarantee reduced variance of a combined
marginal-plus-energy gradient when both original terms share report noise.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import Tensor


class _GaussianAbsoluteMoment(torch.autograd.Function):
    """Analytic first derivatives avoid overflow in ratio autodifferentiation."""

    @staticmethod
    def forward(ctx, difference, variance):
        positive = variance > 0
        sigma = torch.where(positive, variance, torch.ones_like(variance)).sqrt()
        ratio = difference / sigma
        density = torch.exp(-.5 * ratio.square()) / math.sqrt(2 * math.pi)
        signed_probability = torch.erf(ratio / math.sqrt(2))
        result = 2 * sigma * density + difference * signed_probability
        ctx.save_for_backward(difference, variance)
        return torch.where(positive, result, difference.abs())

    @staticmethod
    def backward(ctx, output_gradient):
        difference, variance = ctx.saved_tensors
        positive = variance > 0
        sigma = torch.where(positive, variance, torch.ones_like(variance)).sqrt()
        ratio = difference / sigma
        difference_gradient = torch.where(positive, torch.erf(ratio / math.sqrt(2)), difference.sign())
        variance_gradient = torch.where(positive,
            torch.exp(-.5 * ratio.square()) / sigma / math.sqrt(2 * math.pi),
            torch.zeros_like(variance))
        return output_gradient * difference_gradient, output_gradient * variance_gradient


def _same_float_tensors(reference, others):
    if not isinstance(reference, Tensor) or reference.dtype not in (torch.float32, torch.float64):
        raise ValueError("Inputs must be float32 or float64 tensors")
    if any(not isinstance(value, Tensor) or value.dtype != reference.dtype
           or value.device != reference.device for value in others):
        raise ValueError("Inputs must share dtype and device")


def gaussian_absolute_moment(mean_difference: Tensor, variance: Tensor) -> Tensor:
    """Return ``E|Normal(mean_difference, variance)|`` with broadcasting.

    Inputs must be finite float32/float64 tensors on one device, with variance
    nonnegative. Positive variances have analytic gradients with respect to
    both arguments. At exactly zero variance the value is ``abs(difference)``;
    the implementation chooses ``sign(difference)`` and zero as the gradients
    with respect to difference and variance. In particular the variance
    derivative at difference=variance=0 is singular, so this convention is
    not a differentiability claim. The retained report model's positive
    absolute noise gives strictly positive future increment variance.

    Analytic backward expressions avoid ``0 * inf`` from differentiating
    squared standardized residuals when their Gaussian density underflows.
    No epsilon floor or support clipping is introduced.
    """
    _same_float_tensors(mean_difference, (variance,))
    if (not bool(torch.isfinite(mean_difference).all())
            or not bool(torch.isfinite(variance).all()) or bool((variance < 0).any())):
        raise ValueError("Require finite differences and finite nonnegative variances")
    try:
        difference, variance = torch.broadcast_tensors(mean_difference, variance)
    except RuntimeError as error:
        raise ValueError("Differences and variances must broadcast") from error
    result = _GaussianAbsoluteMoment.apply(difference, variance)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Gaussian absolute moment exceeded floating-point support")
    return result


def accumulated_increment_u_crps(
    target_reports: Tensor,
    latent_cumulative: Tensor,
    days: Tensor,
    origin_index: int,
    noise_abs: Tensor,
    noise_rel: Tensor,
    increment_scales: Tensor,
    target_weights: Tensor,
) -> Tensor:
    """Return one report-noise-integrated marginal U-CRPS per context.

    Shapes are targets=[N,T,3], Q=[N,M,T,3], days=[T], a=r=[3], and
    scales=weights=[F,3], where F is the number of times after the origin.
    M>=2 latent draws must be iid for the U-statistic interpretation. All
    tensors share a float32/float64 dtype/device; phase order is oil/gas/water.
    Targets, Q, a and scales use the same phase-normalized volume units;
    r is dimensionless and times are days. Scales and missing-target count
    weights are caller-supplied, frozen TRAIN quantities.

    For each latent draw, the future reported increment has mean Q(t)-Q(s)
    and variance a²(t-s)/360 + r²[Q(t)-Q(s)][Q(t)+Q(s)]. This remains true
    with any history mask confined to times <=s, including a missing origin:
    subtracting the SAME sampled origin cancels its shared past report error.
    Thus no context mask or historical report value enters this calculation.
    It applies to the accumulated law, not to the fading report-error law.

    The target mask is finite(raw Y(t)) AND finite(raw Y(s)), independent of
    artificial history dropout. Signed targets are retained; nonfinite target
    cells are sanitized before arithmetic and contribute zero. The score is
    the masked count-weighted mean over ALL F*3 cells, as in ``_u_scores``;
    it is not divided by each context's number of present targets. The external
    origin multiplicity weight and the joint energy term are not included.

    The unordered-pair correction excludes self pairs and uses denominator
    M(M-1). Variances add across independent report draws. Computation scales
    differences and standard deviations before forming variances and retains
    gradients through Q, a and r. Zero noise is supported as the deterministic
    limit with ``gaussian_absolute_moment``'s documented boundary convention.
    A positive clock increment that underflows to zero after standardization
    is rejected instead of silently replacing report noise by a point mass.
    """
    q = latent_cumulative
    _same_float_tensors(q, (target_reports, days, noise_abs, noise_rel,
                           increment_scales, target_weights))
    if q.ndim != 4 or q.shape[-1] != 3 or not all(q.shape):
        raise ValueError("latent_cumulative must have nonempty shape [N,M,T,3]")
    n, count, times, _ = q.shape
    if count < 2:
        raise ValueError("Unbiased marginal scoring requires at least two iid latent draws")
    if (isinstance(origin_index, bool) or not isinstance(origin_index, Integral)
            or not 0 <= origin_index < times - 1):
        raise ValueError("origin_index must select an endpoint with a future target")
    future_count = times - origin_index - 1
    if (target_reports.shape != (n, times, 3) or days.shape != (times,)
            or noise_abs.shape != (3,) or noise_rel.shape != (3,)
            or increment_scales.shape != (future_count, 3)
            or target_weights.shape != (future_count, 3)):
        raise ValueError("Target, time, noise, scale or weight axes differ")
    if (not bool(torch.isfinite(days).all()) or bool((days < 0).any())
            or bool((torch.diff(days) <= 0).any())):
        raise ValueError("Days must be finite, nonnegative and strictly increasing")
    if (not bool(torch.isfinite(q).all()) or bool((q < 0).any())
            or bool((torch.diff(q, dim=2) < 0).any())
            or bool((q[:, :, days == 0] != 0).any())):
        raise ValueError("Latent cumulatives must be finite, nonnegative, nondecreasing and zero at day zero")
    for value in (noise_abs, noise_rel, target_weights):
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("Noise and target weights must be finite and nonnegative")
    if not bool(torch.isfinite(increment_scales).all()) or bool((increment_scales <= 0).any()):
        raise ValueError("Increment scales must be finite and strictly positive")

    future = slice(origin_index + 1, None)
    finite_targets = torch.isfinite(target_reports)
    target_mask = finite_targets[:, future] & finite_targets[:, origin_index, None]
    safe_targets = torch.where(finite_targets, target_reports, torch.zeros_like(target_reports))
    target_increment = safe_targets[:, future] - safe_targets[:, origin_index, None]
    target_increment = torch.where(target_mask, target_increment, torch.zeros_like(target_increment))
    actual = target_increment / increment_scales
    before, after = q[:, :, origin_index, None], q[:, :, future]
    mean = (after - before) / increment_scales
    elapsed = (days[future] - days[origin_index])[:, None] / 360
    absolute_sd = noise_abs * elapsed.sqrt() / increment_scales
    relative_increment = noise_rel * mean
    relative_sum = noise_rel * (after / increment_scales) + noise_rel * (before / increment_scales)
    variance = absolute_sd.square() + relative_increment * relative_sum
    positive_variance = (noise_abs > 0) | ((noise_rel > 0) & (after > before))
    if (not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(variance).all()) or bool((variance < 0).any())
            or bool((positive_variance & (variance == 0)).any())):
        raise FloatingPointError("Standardized increment moments exceeded floating-point support")
    left, right = torch.triu_indices(count, count, offset=1, device=q.device)
    target_difference = mean - actual[:, None]
    pair_difference = mean[:, left] - mean[:, right]
    pair_variance = variance[:, left] + variance[:, right]
    if (not bool(torch.isfinite(target_difference).all())
            or not bool(torch.isfinite(pair_difference).all())
            or not bool(torch.isfinite(pair_variance).all())):
        raise FloatingPointError("Absolute-moment arguments exceeded floating-point support")
    first = _GaussianAbsoluteMoment.apply(target_difference, variance).mean(dim=1)
    pairs = _GaussianAbsoluteMoment.apply(pair_difference, pair_variance).sum(dim=1) / (count * (count - 1))
    marginal = (first - pairs) * target_mask * target_weights
    result = marginal.mean(dim=(-2, -1))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Marginal U-CRPS exceeded floating-point support")
    return result
