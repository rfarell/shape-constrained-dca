"""Exact mean-increment scores for the retained positive Erlang basis.

This is a squared-error score for the conditional mean, not an expected
squared error of individual trajectory draws. It adds no variance penalty.
The conditional latent Gaussian is a learned mixing law, not a claim of
Bayesian inference or identification of latent production from reports.
"""
from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor


def _floating(value, name, *, device=None):
    if (not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64)
            or (device is not None and value.device != device)):
        raise ValueError(f"{name} must be a float32/float64 tensor on the common device")


def _interval_masses(days: Tensor, origin_index: int) -> Tensor:
    """Integrate basis densities directly, avoiding subtraction of saturated CDFs."""
    if (days.ndim != 1 or len(days) < 2 or not bool(torch.isfinite(days).all())
            or bool((days < 0).any()) or bool((torch.diff(days) <= 0).any())):
        raise ValueError("Days must be finite, nonnegative and strictly increasing")
    if (isinstance(origin_index, bool) or not isinstance(origin_index, Integral)
            or not 0 <= origin_index < len(days)-1):
        raise ValueError("origin_index must select a time with a future endpoint")
    times = days.to(torch.float64)
    scales = times.new_tensor([30., 180., 360., 720.])
    start = times[origin_index] / scales
    elapsed = (times[origin_index+1:, None]-times[origin_index]) / scales
    exponential_mass = -torch.expm1(-elapsed)
    survival = torch.exp(-start)
    # For shape2, the interval survival difference rearranges to
    # exp(-s/theta) * [(s/theta)*(1-exp(-h)) + GammaCDF(shape2,h)].
    # Both terms are nonnegative, including h tiny or a saturated endpoint CDF.
    shape2_tail = torch.special.gammainc(torch.full_like(elapsed[:, 2:], 2.), elapsed[:, 2:])
    mass = torch.cat((survival[:2]*exponential_mass[:, :2],
                      survival[2:]*(start[2:]*exponential_mass[:, 2:]+shape2_tail)), dim=-1)
    if not bool((torch.isfinite(mass) & (mass > 0)).all()):
        raise FloatingPointError("Positive basis interval masses exceed float64 support")
    return mass


def fast_basis_increment_mean(mean: Tensor, lower: Tensor, days: Tensor,
                              origin_index: int) -> Tensor:
    """Return exact conditional mean increments [N,F,3] in float64.

    ``mean`` is [N,12] and ``lower`` is its lower Cholesky factor [N,12,12].
    Latents are phase-major oil/gas/water log coefficients, with fixed shapes
    (1,1,2,2) and native-day scales (30,180,360,720). Each coefficient mean is
    exp(mu + .5*sum_j L_ij**2), including every off-diagonal covariance entry
    that contributes to the marginal variance. Outputs use the caller's
    phase-normalized volume units and retain gradients to mean and lower.

    All supplied values are promoted to float64 without recomputing their
    normalization or reversing any previous float32 conversion. Days may
    start above zero and must be finite and increasing. The positive
    coefficient expectations, interval masses and resulting means must be
    representable in float64; overflow and mass/expectation underflow fail
    explicitly, without clipping the positive trajectory law.

    Under the accumulated Gaussian report bridge the SAME sampled-origin
    subtraction cancels its past report residual. Consequently this is also
    the reported-increment mean for any permitted history mask, including a
    hidden origin or phase. It does not require report-noise parameters.
    This identity does not extend to the fading report law.
    """
    _floating(mean, "mean")
    _floating(lower, "lower", device=mean.device)
    _floating(days, "days", device=mean.device)
    if (mean.ndim != 2 or mean.shape[1] != 12 or not len(mean)
            or lower.shape != (len(mean), 12, 12) or lower.dtype != mean.dtype):
        raise ValueError("Require matching mean[N,12] and lower[N,12,12] tensors")
    if (not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(lower).all())
            or bool((torch.triu(lower, diagonal=1) != 0).any())
            or bool((torch.diagonal(lower, dim1=-2, dim2=-1) <= 0).any())):
        raise ValueError("Require finite means and lower triangular factors with positive diagonals")
    masses = _interval_masses(days, origin_index)
    variance = lower.to(torch.float64).square().sum(dim=-1)
    coefficient_mean = torch.exp(mean.to(torch.float64)+.5*variance)
    if not bool((torch.isfinite(coefficient_mean) & (coefficient_mean > 0)).all()):
        raise FloatingPointError("Positive lognormal coefficient expectations exceed float64 support")
    result = torch.einsum("npk,fk->nfp", coefficient_mean.reshape(len(mean), 3, 4), masses)
    if not bool((torch.isfinite(result) & (result > 0)).all()):
        raise FloatingPointError("Positive mean increments exceed float64 support")
    return result


def fast_basis_increment_mean_mse(
    target_reports: Tensor,
    mean: Tensor,
    lower: Tensor,
    days: Tensor,
    origin_index: int,
    increment_scales: Tensor,
    target_weights: Tensor,
) -> Tensor:
    """Return count-weighted squared conditional-mean error per context [N].

    Targets are [N,T,3] and frozen TRAIN scales/weights are [F,3], where F is
    the number of endpoints after the origin. All volume quantities use the
    same caller-supplied phase normalization; times remain native days.
    The result averages over ALL F*3 cells, including zero contributions from
    missing targets, exactly as the existing marginal count weighting. The
    caller applies the original origin-multiplicity weight once afterward.

    The target mask is finite(raw Y_t) AND finite(raw Y_s), independently of
    artificial history dropout. A measured but hidden origin remains a valid
    target. Signed corrections are retained. Nonfinite endpoints are excluded
    before subtraction; the caller must validate any earlier normalization
    so numerical overflow cannot be mistaken for original missingness.
    Earlier reports neither form targets nor enter this helper's prediction.

    This is ((E[deltaY]-observed_deltaY)/scale)**2, rather than
    E[((deltaY-observed_deltaY)/scale)**2]. The latter also penalizes predictive
    variance. There is no report-noise input or direct noise gradient here.
    """
    _floating(mean, "mean")
    for name, value in (("target_reports", target_reports), ("days", days),
                        ("increment_scales", increment_scales), ("target_weights", target_weights)):
        _floating(value, name, device=mean.device)
    predicted = fast_basis_increment_mean(mean, lower, days, origin_index)
    future_count = len(days)-origin_index-1
    if (target_reports.shape != (len(mean), len(days), 3)
            or increment_scales.shape != (future_count, 3) or target_weights.shape != (future_count, 3)):
        raise ValueError("Target, scale or weight axes differ from the requested intervals")
    if (not bool((torch.isfinite(increment_scales) & (increment_scales > 0)).all())
            or not bool((torch.isfinite(target_weights) & (target_weights >= 0)).all())):
        raise ValueError("Require finite positive scales and finite nonnegative count weights")
    future = target_reports[:, origin_index+1:].to(torch.float64)
    anchor = target_reports[:, origin_index, None].to(torch.float64)
    mask = torch.isfinite(future) & torch.isfinite(anchor)
    zero = torch.zeros_like(future)
    actual = torch.where(mask, future, zero)-torch.where(mask, anchor, zero)
    difference = torch.where(mask, predicted, zero)-actual
    standardized = difference/increment_scales.to(torch.float64)
    cells = standardized.square()*target_weights.to(torch.float64)
    result = cells.mean(dim=(-2, -1))
    if (not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(standardized).all())
            or not bool(torch.isfinite(cells).all()) or not bool(torch.isfinite(result).all())):
        raise FloatingPointError("Conditional mean squared error exceeds float64 support")
    return result
