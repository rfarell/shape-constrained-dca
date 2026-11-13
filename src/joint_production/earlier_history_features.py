"""Signed earlier report-interval features from the actual surviving history."""
from __future__ import annotations

import torch
from torch import Tensor


REPORT_DAYS = (0, 30, 60, 90, 180, 360, 720, 1080)
EARLIER_END_DAYS = (30, 60, 90, 180, 360)
FEATURE_COUNT = 45


def earlier_interval_rate_features(
    reports: Tensor,
    observed: Tensor,
    report_days: Tensor,
    cutoff: float | Tensor,
    phase_scales: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    zero_control: bool = False,
) -> Tensor:
    """Return [N,45] without repeating each phase's latest surviving interval.

    Inputs follow ``last_interval_rate_features``: native float64 reports
    [N,8,3], actual boolean masks, canonical report days, and positive frozen
    native-volume phase scales. Cutoff may be scalar or [N], in [0,720].
    All tensor inputs share a device. Missing day zero is never restored.

    Slots are ordered by end day (30,60,90,180,360), then oil/gas/water,
    then (asinh_signed_rate, duration_days/360, available). An included end
    contributes only when it has a preceding included report in the SAME
    phase and a later included report. Thus the latest surviving pair is
    excluded even when it ends at day720, outside the output slot list.

    Differences use consecutive SURVIVING endpoints, so gaps are bridged
    with their actual durations. In native float64 arithmetic the rate is
    ``360*(Y(end)-Y(previous))/(duration*phase_scale)`` before asinh and the
    final encoder cast. Signed and zero increments remain reports, not
    inferred instantaneous latent rates. Unavailable slots are three zeros.

    Future and masked values are discarded before value validation. Included
    reports must be finite, with exact zero if day zero is included. Neither
    inputs nor random state are changed. The zero control performs the same
    validation and arithmetic, then returns 45 zeros in the requested dtype.
    """
    if (not isinstance(reports, Tensor) or reports.ndim != 3
            or reports.shape[1:] != (len(REPORT_DAYS), 3) or not len(reports)):
        raise ValueError("reports must have nonempty shape [N,8,3]")
    n = len(reports)
    if (not isinstance(observed, Tensor) or observed.shape != reports.shape
            or observed.dtype != torch.bool):
        raise ValueError("observed must be a boolean tensor matching reports")
    if (not isinstance(report_days, Tensor) or report_days.shape != (8,)
            or not isinstance(phase_scales, Tensor) or phase_scales.shape != (3,)):
        raise ValueError("report_days must be [8] and phase_scales must be [3]")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Encoder dtype must be float32 or float64")
    if not isinstance(zero_control, bool):
        raise ValueError("zero_control must be boolean")
    if reports.dtype != torch.float64:
        raise ValueError("Native reports must use float64 before subtraction")
    if any(value.dtype not in (torch.float32, torch.float64)
           for value in (report_days, phase_scales)):
        raise ValueError("Days and scales must use float32 or float64")
    if any(value.device != reports.device for value in (observed, report_days, phase_scales)):
        raise ValueError("All tensor inputs must share reports device")
    if isinstance(cutoff, Tensor) and cutoff.device != reports.device:
        raise ValueError("Cutoff tensor must share reports device")
    days, scales = report_days.double(), phase_scales.double()
    if not torch.equal(days, days.new_tensor(REPORT_DAYS)):
        raise ValueError("Require canonical report days [0,30,60,90,180,360,720,1080]")
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("phase_scales must be finite and positive")
    cutoffs = torch.as_tensor(cutoff, dtype=torch.float64, device=reports.device)
    if cutoffs.ndim == 0:
        cutoffs = cutoffs.expand(n)
    if (cutoffs.shape != (n,) or not bool(torch.isfinite(cutoffs).all())
            or bool(((cutoffs < 0) | (cutoffs > 720)).any())):
        raise ValueError("cutoff must be a finite scalar or [N] tensor in [0,720]")

    included = observed & (days[None, :, None] <= cutoffs[:, None, None])
    safe_reports = torch.where(included, reports, torch.zeros_like(reports))
    if not bool(torch.isfinite(safe_reports).all()):
        raise ValueError("Available reports must be finite")
    if bool((safe_reports[:, 0] != 0).any()):
        raise ValueError("Observed day-zero reports must be zero")
    indices = torch.arange(len(REPORT_DAYS), device=reports.device)[None, :, None]
    latest = torch.where(included, indices, -1).amax(dim=1)
    ends = torch.arange(1, len(EARLIER_END_DAYS) + 1, device=reports.device)[None, :, None]
    # Find predecessors for all five slots together: [N,end,time,phase].
    previous = torch.where(included[:, None] & (indices[:, None] < ends[:, :, None]),
                           indices[:, None], -1).amax(dim=2)
    available = included[:, 1:6] & (previous >= 0) & (ends < latest[:, None])
    previous_index = previous.clamp_min(0)
    start_value = safe_reports.gather(1, previous_index)
    # Mask before subtracting or dividing, including for excluded latest
    # intervals, so unused extreme levels cannot create spurious failures.
    start_value = torch.where(available, start_value, torch.zeros_like(start_value))
    end_value = torch.where(available, safe_reports[:, 1:6], torch.zeros_like(start_value))
    difference = end_value - start_value
    duration = torch.where(available, days[1:6][None, :, None] - days[previous_index],
                           torch.ones_like(difference))
    normalized_rate = (difference / scales) * (360 / duration)
    if not bool((torch.isfinite(difference) & torch.isfinite(normalized_rate)).all()):
        raise ValueError("Earlier interval arithmetic exceeded finite floating-point support")
    features = torch.stack((torch.asinh(normalized_rate),
        torch.where(available, duration / 360, torch.zeros_like(duration)),
        available.to(torch.float64)), dim=-1).reshape(n, FEATURE_COUNT).to(dtype)
    return torch.zeros_like(features) if zero_control else features
