"""Independent pair enumeration, gap/mask semantics, and native-unit checks."""
import math

import numpy as np
import pytest
import torch

from joint_production.earlier_history_features import (
    EARLIER_END_DAYS,
    FEATURE_COUNT,
    REPORT_DAYS,
    earlier_interval_rate_features,
)


def inputs():
    days = torch.tensor(REPORT_DAYS, dtype=torch.float64)
    reports = torch.tensor([[[0, 0, 0], [10, -5, 0], [25, -5, 0],
                             [40, -15, 5], [30, -10, 5], [80, 0, 5],
                             [100, 5, 8], [200, 10, 10]]], dtype=torch.float64)
    return reports, torch.ones_like(reports, dtype=torch.bool), days, days.new_tensor([12, 8, 7])


def loop_oracle(reports, observed, days, cutoff, scales):
    """Enumerate surviving pairs first, remove the final pair, then slot them."""
    reports, observed, days, scales = [np.asarray(x) for x in (reports, observed, days, scales)]
    cutoffs = np.broadcast_to(np.asarray(cutoff), (len(reports),))
    result = np.zeros((len(reports), 5, 3, 3))
    for well in range(len(reports)):
        for phase in range(3):
            surviving = [j for j, day in enumerate(days) if observed[well, j, phase] and day <= cutoffs[well]]
            pairs = list(zip(surviving[:-1], surviving[1:]))
            for left, right in pairs[:-1]:
                slot = EARLIER_END_DAYS.index(days[right])
                duration = days[right] - days[left]
                rate = 360 * (reports[well, right, phase] - reports[well, left, phase]) / duration / scales[phase]
                result[well, slot, phase] = [math.asinh(rate), duration / 360, 1]
    return result.reshape(len(reports), FEATURE_COUNT)


def test_mixed_cutoffs_and_masks_match_independent_surviving_pair_oracle():
    rng = np.random.default_rng(481)
    reports = np.concatenate((np.zeros((9, 1, 3)), rng.normal(0, 50, (9, 7, 3)).cumsum(axis=1)), axis=1)
    observed = rng.random((9, 8, 3)) > .35
    cutoffs = np.array([0, 30, 60, 90, 180, 360, 720, 55, 719], dtype=float)
    scales = np.array([12., 80., 7.])
    expected = loop_oracle(reports, observed, REPORT_DAYS, cutoffs, scales)
    actual = earlier_interval_rate_features(torch.tensor(reports), torch.tensor(observed),
        torch.tensor(REPORT_DAYS, dtype=torch.float64), torch.tensor(cutoffs), torch.tensor(scales), dtype=torch.float64)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-15, atol=1e-15)
    assert actual.shape == (9, 45)


def test_gaps_and_latest_pair_are_selected_independently_by_phase():
    reports, observed, days, scales = inputs()
    observed.zero_()
    observed[:, [0, 2, 4, 6], 0] = True  # oil 0->60,60->180,180->720
    observed[:, [0, 1, 3, 5], 1] = True  # gas 0->30,30->90,90->360
    observed[:, [0, 5], 2] = True        # water has only its latest pair
    actual = earlier_interval_rate_features(reports, observed, days, 720, scales, dtype=torch.float64).reshape(1, 5, 3, 3)
    expected = torch.zeros_like(actual)
    expected[0, 1, 0] = days.new_tensor([math.asinh(25 * 360 / 60 / 12), 60 / 360, 1])
    expected[0, 3, 0] = days.new_tensor([math.asinh(5 * 360 / 120 / 12), 120 / 360, 1])
    expected[0, 0, 1] = days.new_tensor([math.asinh(-5 * 360 / 30 / 8), 30 / 360, 1])
    expected[0, 2, 1] = days.new_tensor([math.asinh(-10 * 360 / 60 / 8), 60 / 360, 1])
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=1e-15)


@pytest.mark.parametrize("cutoff", [0, 30])
def test_no_earlier_interval_at_first_two_origins(cutoff):
    reports, observed, days, scales = inputs()
    reports[:, days > cutoff] = float("inf")
    result = earlier_interval_rate_features(reports, observed, days, cutoff, scales)
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)


def test_missing_day_zero_is_not_restored_and_single_pair_is_excluded():
    reports, observed, days, scales = inputs()
    observed[:, 0] = False
    reports[:, 0] = float("inf")
    result = earlier_interval_rate_features(reports, observed, days, 90, scales, dtype=torch.float64).reshape(1, 5, 3, 3)
    assert torch.equal(result[:, 0], torch.zeros_like(result[:, 0]))
    torch.testing.assert_close(result[:, 1, :, 2], torch.ones((1, 3), dtype=torch.float64))
    assert result[:, 2:].count_nonzero() == 0  # 60->90 is latest
    single_pair = earlier_interval_rate_features(reports, observed, days, 60, scales)
    assert single_pair.count_nonzero() == 0


def test_signed_zero_increments_and_flattened_phase_triplet_order():
    reports, observed, days, scales = inputs()
    result = earlier_interval_rate_features(reports, observed, days, 180, scales, dtype=torch.float64)
    assert result[0, 3] < 0  # day30 gas signed rate
    assert result[0, 6] == 0 and result[0, 8] == 1  # known zero water interval
    assert result[0, 12] == 0 and result[0, 14] == 1  # day60 gas correction is zero
    torch.testing.assert_close(result[0, [1, 4, 7]], days.new_full((3,), 30 / 360))
    assert result[:, 27:].count_nonzero() == 0  # day180 latest; day360 future


def test_hidden_latest_endpoint_reclassifies_the_preceding_pair():
    reports, observed, days, scales = inputs()
    before = earlier_interval_rate_features(reports, observed, days, 180, scales).reshape(1, 5, 3, 3)
    observed[:, 4, 0] = False
    after = earlier_interval_rate_features(reports, observed, days, 180, scales).reshape(1, 5, 3, 3)
    assert before[0, 2, 0, 2] == 1 and after[0, 2, 0].count_nonzero() == 0
    torch.testing.assert_close(before[:, :, 1:], after[:, :, 1:], rtol=0, atol=0)


def test_future_and_hidden_poison_do_not_change_features_inputs_or_rng():
    reports, observed, days, scales = inputs()
    observed[:, 1, 0] = False
    expected = earlier_interval_rate_features(reports, observed, days, 180, scales)
    reports[:, days > 180] = reports.new_tensor([float("inf"), -float("inf"), float("nan")])
    reports[:, 1, 0] = float("inf")
    snapshot, mask = reports.clone(), observed.clone()
    rng = torch.random.get_rng_state().clone()
    actual = earlier_interval_rate_features(reports, observed, days, 180, scales)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(reports, snapshot, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(observed, mask) and torch.equal(torch.random.get_rng_state(), rng)


def test_subtraction_uses_native_float64_before_encoder_rounding():
    reports, observed, days, scales = inputs()
    observed.zero_()
    observed[:, 1:4] = True
    reports[:, 1:4] = days.new_tensor([1e12, 1e12 + 1, 1e12 + 2])[None, :, None]
    result = earlier_interval_rate_features(reports, observed, days, 90, torch.ones_like(scales)).reshape(1, 5, 3, 3)
    assert torch.equal(reports[:, 1].float(), reports[:, 2].float())
    torch.testing.assert_close(result[:, 1, :, 0], torch.full((1, 3), math.asinh(12.)), rtol=0, atol=0)
    assert result[:, 1, :, 2].eq(1).all()


def test_phase_unit_conversion_preserves_all_features():
    reports, observed, days, scales = inputs()
    factors = days.new_tensor([8, 1024, .125])
    original = earlier_interval_rate_features(reports, observed, days, 720, scales, dtype=torch.float64)
    converted = earlier_interval_rate_features(reports * factors, observed, days, 720, scales * factors, dtype=torch.float64)
    torch.testing.assert_close(converted, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_control_has_identical_validation_and_requested_dtype(dtype):
    reports, observed, days, scales = inputs()
    real = earlier_interval_rate_features(reports, observed, days.float(), 720, scales.float(), dtype=dtype)
    zero = earlier_interval_rate_features(reports, observed, days.float(), 720, scales.float(), dtype=dtype, zero_control=True)
    assert zero.shape == real.shape == (1, 45) and zero.dtype == real.dtype == dtype
    assert zero.count_nonzero() == 0 and real.count_nonzero() > 0
    reports[:, 1, 0] = float("nan")
    for control in (False, True):
        with pytest.raises(ValueError, match="Available reports"):
            earlier_interval_rate_features(reports, observed, days, 720, scales, zero_control=control)


@pytest.mark.parametrize("fault", ["float32_reports", "nonboolean_mask", "noncanonical_days", "nonzero_day0", "infinite_report", "scale_zero", "scale_nan", "cutoff_negative", "cutoff_large", "cutoff_nan", "cutoff_shape", "bad_control", "bad_dtype"])
def test_invalid_support_is_rejected(fault):
    reports, observed, days, scales = inputs()
    cutoff, kwargs = 180, {}
    if fault == "float32_reports": reports = reports.float()
    elif fault == "nonboolean_mask": observed = observed.float()
    elif fault == "noncanonical_days": days[2] = 61
    elif fault == "nonzero_day0": reports[:, 0, 0] = 1
    elif fault == "infinite_report": reports[:, 1, 0] = float("inf")
    elif fault == "scale_zero": scales[0] = 0
    elif fault == "scale_nan": scales[0] = float("nan")
    elif fault == "cutoff_negative": cutoff = -1
    elif fault == "cutoff_large": cutoff = 721
    elif fault == "cutoff_nan": cutoff = float("nan")
    elif fault == "cutoff_shape": cutoff = torch.tensor([30., 60.])
    elif fault == "bad_control": kwargs["zero_control"] = 1
    else: kwargs["dtype"] = torch.float16
    with pytest.raises(ValueError):
        earlier_interval_rate_features(reports, observed, days, cutoff, scales, **kwargs)


@pytest.mark.parametrize("fault", ["difference", "scaled_rate"])
def test_extreme_finite_earlier_arithmetic_rejected_in_both_controls(fault):
    reports, observed, days, scales = inputs()
    observed[:, 0] = False
    reports[:, 1] = -1e308
    reports[:, 2] = 1e308
    if fault == "scaled_rate":
        reports[:, 1] = 0
        scales[:] = 1e-308
    for control in (False, True):
        with pytest.raises(ValueError, match="finite floating-point support"):
            earlier_interval_rate_features(reports, observed, days, 90, scales, zero_control=control)


def test_excluded_latest_pair_never_subtracts_extreme_finite_levels():
    reports, observed, days, scales = inputs()
    observed.zero_()
    observed[:, 1:3] = True
    reports[:, 1] = -1e308
    reports[:, 2] = 1e308
    for control in (False, True):
        result = earlier_interval_rate_features(reports, observed, days, 60, scales,
                                                zero_control=control)
        assert result.count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional CUDA feature parity")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_matches_cpu_without_consuming_rng(dtype):
    reports, observed, days, scales = inputs()
    observed[:, 2, 0] = False
    observed[:, 0, 1] = False
    expected = earlier_interval_rate_features(reports, observed, days, 720, scales, dtype=dtype)
    gpu = [x.cuda() for x in (reports, observed, days, scales)]
    rng = torch.cuda.get_rng_state().clone()
    actual = earlier_interval_rate_features(gpu[0], gpu[1], gpu[2], 720, gpu[3], dtype=dtype)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
    assert actual.device.type == "cuda" and actual.dtype == dtype
    assert torch.equal(torch.cuda.get_rng_state(), rng)
