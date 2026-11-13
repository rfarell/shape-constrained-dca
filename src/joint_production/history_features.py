"""Deterministic features of the endpoints actually available to an encoder."""
from __future__ import annotations

import torch
from torch import Tensor


def last_interval_rate_features(
    reports: Tensor,
    observed: Tensor,
    report_days: Tensor,
    cutoff: float | Tensor,
    phase_scales: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    zero_control: bool = False,
) -> Tensor:
    """Return [N,6]: three signed interval-rate features, then availability.

    Phase order is oil, gas, water. Inputs are native cumulative reports
    [N,T,3], the actual boolean observation mask [N,T,3], report days [T],
    and frozen native-volume phase scales [3]. Cutoff is a scalar or [N]
    tensor, permitting batches with different forecast origins. All tensor
    inputs must share a device. Native reports must be float64: promoting
    already-rounded encoder inputs cannot recover precision for subtraction.
    Days and scales may use float32 or float64.

    For each phase, select the last two endpoints that survive BOTH the supplied
    mask and cutoff. Compute asinh(360 * (Y(v)-Y(u)) / ((v-u)*scale)) in float64,
    then cast to the requested encoder dtype. Retain signed and zero increments.
    This feature is a report interval average, not an instantaneous latent rate.
    Fewer than two surviving endpoints gives rate zero and availability zero.

    Unavailable values are discarded before value validation. An endpoint marked
    available must be finite. If day zero is observed, its report must be zero;
    this helper never inserts or restores day zero. Callers implementing the
    deterministic accounting-origin convention must explicitly mark it observed.
    A lone positive-time endpoint plus an observed origin therefore describes the
    average from startup. Recompute features after any constructed dropout.

    zero_control returns six zeros after identical input/support validation,
    providing the same input dimensions for a feature ablation. This function
    fits no scales and does not mutate any input. Nonfinite intermediate results
    are rejected rather than clipped or replaced.
    """
    if reports.ndim != 3 or reports.shape[-1] != 3 or not all(reports.shape):
        raise ValueError('reports must have nonempty shape [N,T,3]')
    n, t, _ = reports.shape
    if observed.shape != reports.shape or observed.dtype != torch.bool:
        raise ValueError('observed must be a boolean tensor matching reports')
    if report_days.shape != (t,) or phase_scales.shape != (3,):
        raise ValueError('report_days must be [T] and phase_scales must be [3]')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Encoder dtype must be float32 or float64')
    if not isinstance(zero_control, bool):
        raise ValueError('zero_control must be boolean')
    if reports.dtype != torch.float64:
        raise ValueError('Native reports must use float64 before any encoder rounding')
    floating = (report_days, phase_scales)
    if any(value.dtype not in (torch.float32, torch.float64) for value in floating):
        raise ValueError('Floating inputs must use float32 or float64')
    if any(value.device != reports.device for value in (observed, report_days, phase_scales)):
        raise ValueError('All tensors must share reports device')
    if isinstance(cutoff, Tensor) and cutoff.device != reports.device:
        raise ValueError('Cutoff tensor must share reports device')
    cutoffs = torch.as_tensor(cutoff, dtype=torch.float64, device=reports.device)
    if cutoffs.ndim == 0:
        cutoffs = cutoffs.expand(n)
    if cutoffs.shape != (n,) or not bool(torch.isfinite(cutoffs).all()) or bool((cutoffs < 0).any()):
        raise ValueError('cutoff must be a finite nonnegative scalar or [N] tensor')
    days, scales = report_days.double(), phase_scales.double()
    if (not bool(torch.isfinite(days).all()) or bool((days < 0).any())
            or bool((torch.diff(days) <= 0).any())):
        raise ValueError('report_days must be finite, nonnegative, strictly increasing')
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError('phase_scales must be finite and strictly positive')

    included = observed & (days[None, :, None] <= cutoffs[:, None, None])
    safe_reports = torch.where(included, reports, torch.zeros_like(reports)).double()
    if not bool(torch.isfinite(safe_reports).all()):
        raise ValueError('Available reports must be finite')
    if bool((safe_reports[:, days == 0] != 0).any()):
        raise ValueError('Observed day-zero reports must be zero')

    indices = torch.arange(t, device=reports.device)[None, :, None]
    last = torch.where(included, indices, -1).amax(dim=1)
    previous = torch.where(included & (indices < last[:, None]), indices, -1).amax(dim=1)
    available = previous >= 0
    # Safe indices and denominators also cover entirely missing phases. Mask
    # unused levels before arithmetic so they cannot cause spurious overflow.
    last_index, previous_index = last.clamp_min(0), previous.clamp_min(0)
    last_report = safe_reports.gather(1, last_index[:, None]).squeeze(1)
    previous_report = safe_reports.gather(1, previous_index[:, None]).squeeze(1)
    last_report = torch.where(available, last_report, torch.zeros_like(last_report))
    previous_report = torch.where(available, previous_report, torch.zeros_like(previous_report))
    difference = last_report - previous_report
    duration = torch.where(available, days[last_index] - days[previous_index], torch.ones_like(difference))
    normalized_rate = (difference / scales) * (360 / duration)
    if not bool(torch.isfinite(difference).all()) or not bool(torch.isfinite(normalized_rate).all()):
        raise ValueError('Interval-rate arithmetic exceeded finite floating-point support')
    features = torch.cat((torch.asinh(normalized_rate), available.to(torch.float64)), dim=1).to(dtype)
    return torch.zeros_like(features) if zero_control else features
