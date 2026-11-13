"""Independent reference implementation of the Gaussian endpoint report laws (tests only).

Phase order is oil/gas/water. For each phase and fixed latent trajectory,
V(t,Q) = a**2*t/360 + r**2*Q(t)**2. Independent errors have diagonal
covariance V; accumulated errors have covariance V(min(t,u),Q(min(t,u))).
The latter is Brownian motion under an increasing, trajectory-dependent clock.
Phases have independent report errors conditional on the joint latent path.
The tests check the torch sampler and the Bayesian likelihood against it.
a has cumulative-volume units and r is dimensionless.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import Tensor

LAWS = ('independent', 'accumulated')


def _check_law(law):
    if law not in LAWS:
        raise ValueError(f'law must be one of {LAWS}')


def gaussian_endpoint_log_likelihood(
    reports: Tensor, cumulative: Tensor, noise_abs: Tensor, noise_rel: Tensor,
    mask: Tensor, days: Tensor, *, law: str = 'accumulated',
) -> Tensor:
    """Sum each observed endpoint once; result has shape [...,N].

    reports/mask are [N,T,3], cumulative is [...,N,T,3], and noise a/r are
    positive [3] tensors in the SAME units as cumulative. The caller supplies
    the cutoff-available mask. Day zero is deterministic and contributes no
    density. Missing endpoints are marginalized, so successive OBSERVED endpoint
    differences, not all triangular increments, factor the accumulated law.
    """
    _check_law(law)
    if reports.ndim != 3 or reports.shape[-1] != 3 or mask.shape != reports.shape or mask.dtype != torch.bool:
        raise ValueError('reports/mask must be [N,T,3] with a boolean mask')
    if cumulative.ndim < 3 or cumulative.shape[-3:] != reports.shape:
        raise ValueError('cumulative must have shape [...,N,T,3]')
    if days.ndim != 1 or len(days) != reports.shape[1] or not len(days):
        raise ValueError('days must match the nonempty time axis')
    if not bool(torch.isfinite(days).all()) or bool((days < 0).any()) or bool((torch.diff(days) <= 0).any()):
        raise ValueError('days must be finite, nonnegative, and strictly increasing')
    for noise in (noise_abs, noise_rel):
        if noise.shape != (3,) or not bool(torch.isfinite(noise).all()) or bool((noise <= 0).any()):
            raise ValueError('Noise a/r must be finite positive tensors with shape [3]')
    if not bool(torch.isfinite(cumulative).all()) or bool((cumulative < 0).any()) or bool((torch.diff(cumulative, dim=-2) < 0).any()):
        raise ValueError('Latent cumulatives must be finite, nonnegative, and nondecreasing')
    if bool((cumulative[..., days == 0, :] != 0).any()):
        raise ValueError('Latent cumulative at day zero must be zero')
    if bool((mask & ~torch.isfinite(reports)).any()):
        raise ValueError('Masked-in reports must be finite')
    if bool((mask[:, days == 0] & (reports[:, days == 0] != 0)).any()):
        raise ValueError('Observed day-zero reports must be zero')
    active = mask & (days > 0)[None, :, None]
    safe_reports = torch.where(active, reports, torch.zeros_like(reports))
    previous_q = torch.zeros_like(cumulative[..., 0, :])
    previous_error = torch.zeros_like(previous_q)
    previous_time = torch.zeros_like(previous_q)
    total = torch.zeros_like(previous_q)
    for j in range(len(days)):
        q = cumulative[..., j, :]
        error = safe_reports[:, j, :] - q
        if law == 'independent':
            variance = noise_abs.square() * (days[j]/360) + (noise_rel*q).square()
            innovation = error
        else:
            # Difference of squares avoids subtracting two almost equal V's.
            variance = (noise_abs.square() * ((days[j]-previous_time)/360)
                        + (noise_rel*(q-previous_q)) * (noise_rel*q+noise_rel*previous_q))
            innovation = error-previous_error
        included = active[:, j, :]
        safe_variance = torch.where(included, variance, torch.ones_like(variance))
        # Mask innovations before squaring: excluded extreme values cannot
        # poison gradients through an unused numerical branch.
        innovation = torch.where(included, innovation, torch.zeros_like(innovation))
        log_probability = -.5*(math.log(2*math.pi)+torch.log(safe_variance)+innovation.square()/safe_variance)
        total = total + torch.where(included, log_probability, torch.zeros_like(log_probability))
        if law == 'accumulated':
            previous_q = torch.where(included, q, previous_q)
            previous_error = torch.where(included, error, previous_error)
            previous_time = torch.where(included, days[j], previous_time)
    if not bool(torch.isfinite(total).all()):
        raise ValueError('Gaussian endpoint likelihood produced a numerical failure')
    return total.sum(dim=-1)


def sample_gaussian_reports(
    latent_cumulative, reports, days, noise_abs, noise_rel, *, cutoff,
    rng: np.random.Generator, law: str = 'accumulated',
):
    """Draw reports [N,S,T,3], conditioning only on finite history <= cutoff.

    For accumulated errors, masked endpoints preceding an observed endpoint are
    sampled with Brownian bridges, then increments are drawn after the last
    observed endpoint. An unknown cutoff anchor is sampled jointly with future
    reports; it is never replaced by a hidden realized report. Day-zero values
    are deterministic. Observed history is restored from original float64 input,
    exactly, while latent cumulatives remain unchanged and need not equal it.
    """
    _check_law(law)
    q = np.asarray(latent_cumulative, dtype=float)
    history = np.asarray(reports, dtype=float)
    times = np.asarray(days, dtype=float)
    a, r = np.asarray(noise_abs, dtype=float), np.asarray(noise_rel, dtype=float)
    if q.ndim != 4 or q.shape[-1] != 3 or not all(q.shape) or history.shape != (q.shape[0], q.shape[2], 3):
        raise ValueError('Expected cumulative [N,S,T,3] and reports [N,T,3]')
    if times.shape != (q.shape[2],) or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError('days must be finite, nonnegative, ordered, and match T')
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError('cutoff must be finite and nonnegative')
    if not np.isfinite(q).all() or np.any(q < 0) or np.any(np.diff(q, axis=2) < 0):
        raise ValueError('Latent cumulatives must be finite, nonnegative, and nondecreasing')
    if np.any(q[:, :, times == 0] != 0):
        raise ValueError('Latent cumulative at day zero must be zero')
    for noise in (a, r):
        if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError('Noise a/r must be finite positive arrays with shape [3]')
    eligible = times <= cutoff
    if np.isinf(history[:, eligible]).any():
        raise ValueError('Available reports must be finite or NaN for missing; infinity is invalid')
    known = np.isfinite(history) & eligible[None, :, None]
    if np.any(known[:, times == 0] & (history[:, times == 0] != 0)):
        raise ValueError('Observed day-zero reports must be zero')
    known &= (times > 0)[None, :, None]
    result = np.empty_like(q)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        if law == 'independent':
            variance = a**2 * times[None, None, :, None]/360 + (r*q)**2
            result = q + np.sqrt(variance)*rng.standard_normal(q.shape)
        else:
            for well in range(q.shape[0]):
                for phase in range(3):
                    trajectory = q[well, :, :, phase]
                    def dv(left, right):
                        left_time = times[left] if left >= 0 else 0.0
                        left_q = trajectory[:, left] if left >= 0 else np.zeros(q.shape[1])
                        right_q = trajectory[:, right]
                        return (a[phase]**2*(times[right]-left_time)/360
                                + (r[phase]*(right_q-left_q))*(r[phase]*right_q+r[phase]*left_q))
                    left = -1
                    left_error = np.zeros(q.shape[1])
                    for right in np.flatnonzero(known[well, :, phase]):
                        right_error = history[well, right, phase]-trajectory[:, right]
                        # Sequential bridges condition each remaining point on
                        # the previous sampled point AND the next known endpoint.
                        for middle in range(left+1, right):
                            if times[middle] == 0:
                                result[well, :, middle, phase] = 0.0
                                left = middle
                                continue
                            whole, first = dv(left, right), dv(left, middle)
                            last = dv(middle, right)
                            fraction = first/whole
                            mean = left_error + fraction*(right_error-left_error)
                            variance = first*(last/whole)
                            left_error = mean + np.sqrt(variance)*rng.standard_normal(q.shape[1])
                            result[well, :, middle, phase] = trajectory[:, middle]+left_error
                            left = middle
                        result[well, :, right, phase] = history[well, right, phase]
                        left, left_error = right, right_error
                    for right in range(left+1, q.shape[2]):
                        if times[right] == 0:
                            result[well, :, right, phase] = 0.0
                            left = right
                            continue
                        left_error = left_error + np.sqrt(dv(left, right))*rng.standard_normal(q.shape[1])
                        result[well, :, right, phase] = trajectory[:, right]+left_error
                        left = right
        result = np.where(known[:, None], history[:, None], result)
        result[:, :, times == 0] = 0.0
    if not np.isfinite(result).all():
        raise ValueError('Gaussian report sampler produced a numerical failure')
    return result
