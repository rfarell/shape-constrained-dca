"""Finite-memory Gaussian reporting error conditional on integrated rate paths.

For each phase, V(t)=a²t/360+(rQ(t))² and
K(s,t)=exp(-log(2)|t-s|/h)V(min(s,t)), where h is a fixed positive finite
half-life in days. Equivalently, de=-lambda*e dt+sqrt(V'+2lambda V)dW,
e(0)=0. Nonnegative rates give V'>=0, hence a valid Gaussian Markov law.
The error's conditional mean decays; latent rates/integrals are unchanged.

Phases are independent conditional on the joint latent path. Reports may be
signed. Neither this assumed discrepancy law nor exact history restoration
identifies latent production or establishes calibration. The a²t term means
finite latent EUR does not imply a finite limiting reported endpoint.
"""
from __future__ import annotations

import math
import numbers

import torch
from torch import Tensor


def _half_life(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("half_life_days must be a fixed positive finite scalar")
    return float(value)


def _transition(q_left, q_right, left_day, right_day, a, r, half_life):
    """Validated callers supply nondecreasing times/cumulatives; no support clipping."""
    dt = right_day-left_day
    # Time exponents use float64 even during float32 training. This permits
    # finite half-lives outside float32's scalar range without 0/0 at dt=0.
    # Coefficients return to the path dtype before volume arithmetic.
    exponent = -math.log(2.0)*(dt.to(torch.float64)/half_life)
    rho = torch.exp(exponent).to(q_right.dtype)
    lost = (-torch.expm1(2*exponent)).to(q_right.dtype)
    left_variance = a.square()*(left_day/360)+(r*q_left).square()
    difference = a.square()*(dt/360)+(r*(q_right-q_left))*(r*q_right+r*q_left)
    return rho, difference+lost*left_variance


def fading_transition(q_left: Tensor, q_right: Tensor, left_day: Tensor,
                      right_day: Tensor, noise_abs: Tensor, noise_rel: Tensor, *,
                      half_life_days: float) -> tuple[Tensor, Tensor]:
    """Return rho and D for e(right)|e(left) ~ N(rho*e(left),D).

    All inputs are broadcast-compatible float32/float64 tensors on one device;
    scalar phase noise is allowed. This public helper validates its domain.
    Half-life and observation times are fixed model settings, not learned
    parameters. Gradients with respect to Q/a/r pass through the exact formula.
    """
    h = _half_life(half_life_days)
    values = (q_left, q_right, left_day, right_day, noise_abs, noise_rel)
    if any(not isinstance(x, Tensor) for x in values):
        raise ValueError("Transition inputs must be tensors")
    if q_right.dtype not in (torch.float32, torch.float64) or any(x.dtype != q_right.dtype or x.device != q_right.device for x in values):
        raise ValueError("Transition tensors must share float32/float64 dtype and device")
    try:
        torch.broadcast_shapes(*(x.shape for x in values))
    except RuntimeError as error:
        raise ValueError("Transition inputs are not broadcast compatible") from error
    if (any(not bool(torch.isfinite(x).all()) for x in values)
            or bool((q_left < 0).any()) or bool((q_right < q_left).any())
            or bool((left_day < 0).any()) or bool((right_day < left_day).any())
            or bool((noise_abs <= 0).any()) or bool((noise_rel <= 0).any())
            or bool(((right_day == left_day) & (q_right != q_left)).any())):
        raise ValueError("Require finite positive noise and ordered nonnegative times/cumulatives")
    rho, variance = _transition(*values, h)
    if (not bool(torch.isfinite(variance).all()) or bool((variance < 0).any())
            or bool(((right_day > left_day) & (variance <= 0)).any())):
        raise FloatingPointError("Fading transition variance exceeded numerical support")
    return rho, variance


def _check_common(cumulative, days, a, r):
    if cumulative.dtype not in (torch.float32, torch.float64):
        raise ValueError("Inputs must use float32 or float64")
    if any(x.dtype != cumulative.dtype or x.device != cumulative.device for x in (days, a, r)):
        raise ValueError("All inputs must share cumulative dtype and device")
    if (days.ndim != 1 or not len(days) or not bool(torch.isfinite(days).all())
            or bool((days < 0).any()) or bool((torch.diff(days) <= 0).any())):
        raise ValueError("Report days must be finite, nonnegative, and strictly increasing")
    if (not bool(torch.isfinite(cumulative).all()) or bool((cumulative < 0).any())
            or bool((torch.diff(cumulative, dim=-2) < 0).any())
            or bool((cumulative[..., days == 0, :] != 0).any())):
        raise ValueError("Latent cumulatives must be finite, nonnegative, nondecreasing, and zero at day zero")
    for noise in (a, r):
        if noise.shape != (3,) or not bool(torch.isfinite(noise).all()) or bool((noise <= 0).any()):
            raise ValueError("Noise a/r must be finite positive phase vectors")


def sample_fading_reports(cumulative: Tensor, history: Tensor, report_days: Tensor,
                          noise_abs: Tensor, noise_rel: Tensor, *, cutoff: float,
                          half_life_days: float, generator: torch.Generator | None = None,
                          standard_normals: Tensor | None = None) -> Tensor:
    """Sample exact conditional reports [well,draw,time,oil/gas/water].

    Signature and tensor contracts mirror sample_accumulated_reports, with an
    additional required finite half-life. Only finite reports through cutoff
    condition the law. Future garbage is ignored. Day zero is deterministic;
    known reports are restored exactly. Hidden past, hidden origin, and future
    endpoints are sampled jointly using the previous sampled error and next
    available right anchor. Supplied normals must be iid standard normal for
    stochastic sampling, or fixed values for deterministic gradient checks.
    """
    h = _half_life(half_life_days)
    q, days, a, r = cumulative, report_days, noise_abs, noise_rel
    if q.ndim != 4 or q.shape[-1] != 3 or not all(q.shape):
        raise ValueError("Cumulative must have nonempty [well,draw,time,3] shape")
    n, samples, count, _ = q.shape
    if history.shape != (n, count, 3) or days.shape != (count,):
        raise ValueError("History and days must match the cumulative well/time/phase axes")
    _check_common(q, days, a, r)
    additional = (history,) + (() if standard_normals is None else (standard_normals,))
    if any(x.dtype != q.dtype or x.device != q.device for x in additional):
        raise ValueError("History and normals must share cumulative dtype and device")
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("Cutoff must be finite and nonnegative")
    available = torch.where((days <= cutoff)[None, :, None], history, torch.full_like(history, float("nan")))
    if bool(torch.isinf(available).any()):
        raise ValueError("Available reports must be finite or NaN; infinity is invalid")
    known = torch.isfinite(available)
    if bool((known[:, days == 0] & (available[:, days == 0] != 0)).any()):
        raise ValueError("Known day-zero reports must be zero")
    known = known & (days > 0)[None, :, None]
    safe_history = torch.where(known, available, torch.zeros_like(available))
    if standard_normals is None:
        normals = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator)
    else:
        if standard_normals.shape != q.shape or not bool(torch.isfinite(standard_normals).all()):
            raise ValueError("standard_normals must be finite with cumulative shape")
        normals = standard_normals
    indices = torch.arange(count, device=q.device)[None, :, None].expand(n, count, 3)
    right_indices = torch.where(known, indices, count).flip(1).cummin(1).values.flip(1)
    previous_q = torch.zeros_like(q[:, :, 0])
    previous_error = torch.zeros_like(previous_q)
    previous_day = days.new_zeros(())
    outputs = []
    for j in range(count):
        current_q = q[:, :, j]
        right_index = right_indices[:, j]
        has_right = (right_index < count)[:, None]
        safe_index = right_index.clamp_max(count-1)
        right_q = q.gather(2, safe_index[:, None, None].expand(n, samples, 1, 3)).squeeze(2)
        right_q = torch.where(has_right, right_q, current_q)
        right_day = torch.where(has_right, days[safe_index][:, None], days[j])
        right_report = safe_history.gather(1, safe_index[:, None]).squeeze(1)[:, None]
        rho_left, first = _transition(previous_q, current_q, previous_day, days[j], a, r, h)
        rho_right, last = _transition(current_q, right_q, days[j], right_day, a, r, h)
        if (not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(last).all())
                or bool((first < 0).any()) or bool((last < 0).any())
                or (bool(days[j] > 0) and bool((first <= 0).any()))):
            raise FloatingPointError("Fading transitions must have finite positive increments")
        whole = rho_right.square()*first+last
        whole = torch.where(has_right & (days[j] > 0), whole, torch.ones_like(whole))
        if not bool(torch.isfinite(whole).all()) or bool((whole <= 0).any()):
            raise FloatingPointError("Fading bridge variance exceeded numerical support")
        mean_left = rho_left*previous_error
        bridge_mean = mean_left + rho_right*(first/whole)*(right_report-right_q-rho_right*mean_left)
        mean = torch.where(has_right, bridge_mean, mean_left)
        variance = torch.where(has_right, first*(last/whole), first)
        random_endpoint = (~known[:, j, None]) & (days[j] > 0)
        # Observed/day-zero endpoints have zero variance. Mask before sqrt so
        # unused sqrt(0) derivatives cannot poison reparameterized gradients.
        safe_variance = torch.where(random_endpoint, variance, torch.ones_like(variance))
        error = mean + safe_variance.sqrt()*normals[:, :, j]
        report = torch.where(known[:, j, None], safe_history[:, j, None], current_q+error)
        report = torch.where(days[j] == 0, torch.zeros_like(report), report)
        outputs.append(report)
        previous_q, previous_day, previous_error = current_q, days[j], report-current_q
    result = torch.stack(outputs, dim=2)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Fading report sampler exceeded numerical support; no clipping")
    return result


def fading_endpoint_log_likelihood(reports: Tensor, cumulative: Tensor, noise_abs: Tensor,
                                   noise_rel: Tensor, mask: Tensor, days: Tensor, *,
                                   half_life_days: float) -> Tensor:
    """Exact observed-endpoint density: [N,T,3], [...,N,T,3] -> [...,N].

    Missing endpoints are marginalized by transitions between successive
    OBSERVED times in each phase. Day zero contributes no density. The caller
    supplies the permitted-history mask; excluded report values are ignored.
    This utility is not an added loss for the direct conditional rate model.
    """
    h = _half_life(half_life_days)
    if (reports.ndim != 3 or reports.shape[-1] != 3 or mask.shape != reports.shape or mask.dtype != torch.bool
            or cumulative.ndim < 3 or cumulative.shape[-3:] != reports.shape or days.shape != (reports.shape[1],)):
        raise ValueError("Reports/mask must be [N,T,3], cumulative [...,N,T,3], and days [T]")
    _check_common(cumulative, days, noise_abs, noise_rel)
    if reports.dtype != cumulative.dtype or reports.device != cumulative.device or mask.device != cumulative.device:
        raise ValueError("Reports/mask must share cumulative device, and reports its dtype")
    if bool((mask & ~torch.isfinite(reports)).any()) or bool((mask[:, days == 0] & (reports[:, days == 0] != 0)).any()):
        raise ValueError("Included reports must be finite, with deterministic day-zero accounting")
    active = mask & (days > 0)[None, :, None]
    safe_reports = torch.where(active, reports, torch.zeros_like(reports))
    previous_q = torch.zeros_like(cumulative[..., 0, :])
    previous_error = torch.zeros_like(previous_q)
    previous_day = torch.zeros_like(previous_q)
    total = torch.zeros_like(previous_q)
    for j in range(len(days)):
        q = cumulative[..., j, :]
        error = safe_reports[:, j]-q
        rho, variance = _transition(previous_q, q, previous_day, days[j], noise_abs, noise_rel, h)
        included = active[:, j]
        safe_variance = torch.where(included, variance, torch.ones_like(variance))
        innovation = torch.where(included, error-rho*previous_error, torch.zeros_like(error))
        if not bool(torch.isfinite(safe_variance).all()) or bool((safe_variance <= 0).any()):
            raise FloatingPointError("Fading likelihood variance exceeded numerical support")
        log_probability = -.5*(math.log(2*math.pi)+torch.log(safe_variance)+innovation.square()/safe_variance)
        total = total+torch.where(included, log_probability, torch.zeros_like(log_probability))
        previous_q = torch.where(included, q, previous_q)
        previous_day = torch.where(included, days[j], previous_day)
        previous_error = torch.where(included, error, previous_error)
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("Fading observed-endpoint likelihood exceeded numerical support")
    return total.sum(dim=-1)
