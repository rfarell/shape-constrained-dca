"""Differentiable sampling of the accumulated Gaussian endpoint report law.

Phase order is oil, gas, water. Given a latent cumulative path Q, report errors
have covariance V(min(t,u)), where V(t) = a² t/360 + (r Q(t))². Error processes
are independent across phases conditional on Q. Reports are signed and are
never clipped. Q and a use the same caller-selected volume units; r is
dimensionless and report times are days.

Noise scales can be global phase vectors or one fixed phase vector per well.
Per-well scales apply to every sampled endpoint; they must not be recomputed
from sampled missing reports or from the set of query times.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def sample_accumulated_reports(
    cumulative: Tensor,
    history: Tensor,
    report_days: Tensor,
    noise_abs: Tensor,
    noise_rel: Tensor,
    *,
    cutoff: float,
    generator: torch.Generator | None = None,
    standard_normals: Tensor | None = None,
) -> Tensor:
    """Sample conditional reports [N,S,T,3] using differentiable bridges.

    Inputs have shapes Q=[N,S,T,3], history=[N,T,3], days=[T]. Noise a and r
    must both have shape [3] or both [N,3], with the latter fixed across all
    samples and times of each well. Mixed global/per-well shapes are rejected.
    All tensors must share a float32/float64 dtype and device. Available finite
    reports at times <= cutoff are repeated exactly; NaN denotes missingness.
    Future history values (including infinities) are ignored before validation.
    Day zero is deterministically zero, whether observed or missing.

    Each missing endpoint is sampled conditional on the previous sampled error
    and the next available error, or by a Brownian increment if no later error
    is available. Thus hidden past endpoints, a hidden forecast origin, and
    future reports all belong to one conditional Gaussian path. Clock
    differences use a factored difference of squares to retain precision.

    Optional standard_normals must have Q's shape and contain independent
    standard normal draws to sample the stated law. Entries for known reports
    and day zero are unused. Supplying them permits deterministic reparameterized
    draws and gradient checks. Otherwise generator controls torch.randn.
    Gradients pass through Q, a, and r; observed reports have zero such gradient.
    """
    q, days, a, r = cumulative, report_days, noise_abs, noise_rel
    if q.ndim != 4 or q.shape[-1] != 3 or not all(q.shape):
        raise ValueError('cumulative must have nonempty shape [N,S,T,3]')
    n, s, t, _ = q.shape
    if history.shape != (n, t, 3) or days.shape != (t,):
        raise ValueError('history must be [N,T,3] and report_days must be [T]')
    if a.shape != r.shape or a.shape not in ((3,), (n, 3)):
        raise ValueError('noise_abs and noise_rel must both have shape [3] or [N,3]')
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError('Inputs must use float32 or float64')
    inputs = (history, days, a, r) + (() if standard_normals is None else (standard_normals,))
    if any(x.dtype != q.dtype or x.device != q.device for x in inputs):
        raise ValueError('All tensors must share cumulative dtype and device')
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError('cutoff must be finite and nonnegative')
    if (not bool(torch.isfinite(days).all()) or bool((days < 0).any())
            or bool((torch.diff(days) <= 0).any())):
        raise ValueError('report_days must be finite, nonnegative, strictly increasing')
    if (not bool(torch.isfinite(q).all()) or bool((q < 0).any())
            or bool((torch.diff(q, dim=2) < 0).any())):
        raise ValueError('Latent cumulatives must be finite, nonnegative, nondecreasing')
    if bool((q[:, :, days == 0] != 0).any()):
        raise ValueError('Latent cumulative at day zero must be zero')
    for noise in (a, r):
        if not bool(torch.isfinite(noise).all()) or bool((noise <= 0).any()):
            raise ValueError('Noise a/r must be finite and strictly positive')
    if a.ndim == 2:
        a, r = a[:, None, :], r[:, None, :]
    available = torch.where((days <= cutoff)[None, :, None], history,
                            torch.full_like(history, float('nan')))
    if bool(torch.isinf(available).any()):
        raise ValueError('Available reports must be finite or NaN; infinity is invalid')
    known = torch.isfinite(available)
    if bool((known[:, days == 0] & (available[:, days == 0] != 0)).any()):
        raise ValueError('Observed day-zero reports must be zero')
    known = known & (days > 0)[None, :, None]
    safe_history = torch.where(known, available, torch.zeros_like(available))
    if standard_normals is None:
        normals = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator)
    else:
        if standard_normals.shape != q.shape or not bool(torch.isfinite(standard_normals).all()):
            raise ValueError('standard_normals must be finite with cumulative shape')
        normals = standard_normals

    # Index of the first available endpoint at or after each query, separately
    # for each well and phase; T denotes the absence of a right endpoint.
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
                 + (r * (current_q - previous_q)) * (r * current_q + r * previous_q))
        last = (a.square() * ((right_day - days[j]) / 360)
                + (r * (right_q - current_q)) * (r * right_q + r * current_q))
        if (not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(last).all())
                or bool((first < 0).any()) or bool((last < 0).any())
                or (bool(days[j] > 0) and bool((first <= 0).any()))):
            raise ValueError('Accumulated variance clock must have finite positive increments')
        # A day-zero or observed endpoint has zero conditional variance. Avoid
        # differentiating sqrt(0), including in branches discarded by where.
        whole = torch.where(has_right & (days[j] > 0), first + last, torch.ones_like(first))
        if not bool(torch.isfinite(whole).all()) or bool((whole <= 0).any()):
            raise ValueError('Accumulated bridge clock must be finite and strictly positive')
        fraction = first / whole
        mean = torch.where(has_right,
                           previous_error + fraction * (right_report - right_q - previous_error),
                           previous_error)
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
        raise ValueError('Accumulated report sampler produced a numerical failure')
    return result
