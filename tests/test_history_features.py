"""Last interval features must use only the actual surviving report history."""
import pytest
import torch

from joint_production.history_features import last_interval_rate_features


def inputs():
    days = torch.tensor([0., 17., 53., 100., 365.], dtype=torch.float64)
    reports = torch.tensor([[[0., 0., 0.], [10., 20., 7.], [22., 16., 7.],
                             [52., 46., 37.], [152., 146., 137.]]], dtype=torch.float64)
    observed = torch.ones_like(reports, dtype=torch.bool)
    scales = torch.tensor([12., 8., 7.], dtype=torch.float64)
    return reports, observed, days, scales


def test_irregular_intervals_preserve_signed_and_zero_rates_and_phase_order():
    reports, observed, days, scales = inputs()
    actual = last_interval_rate_features(reports, observed, days, 60, scales, dtype=torch.float64)
    rates = torch.tensor([12., -4., 0.], dtype=torch.float64) * 360 / 36 / scales
    expected = torch.cat((torch.asinh(rates), torch.ones(3, dtype=torch.float64)))[None]
    torch.testing.assert_close(actual, expected)
    assert actual[0, 1] < 0
    assert actual[0, 2] == 0 and actual[0, 5] == 1


def test_dropout_recomputes_the_last_pair_and_preserves_missing_phase_identity():
    reports, observed, days, scales = inputs()
    observed[:, 2, 0] = False  # Oil now uses days 0 and 17.
    observed[:, 1, 1] = False  # Gas bridges day 0 to 53.
    observed[:, :, 2] = False
    actual = last_interval_rate_features(reports, observed, days, 60, scales, dtype=torch.float64)
    rates = torch.tensor([10.*360/17/12, 16.*360/53/8, 0.], dtype=torch.float64)
    torch.testing.assert_close(actual, torch.cat((torch.asinh(rates), rates.new_tensor([1, 1, 0])))[None])


def test_origin_is_never_injected_and_one_positive_report_needs_an_observed_origin():
    reports, observed, days, scales = inputs()
    observed.zero_()
    observed[:, 1] = True
    missing_origin = last_interval_rate_features(reports, observed, days, 17, scales)
    assert torch.equal(missing_origin, torch.zeros_like(missing_origin))
    observed[:, 0] = True
    with_origin = last_interval_rate_features(reports, observed, days, 17, scales)
    assert torch.equal(with_origin[:, 3:], torch.ones_like(with_origin[:, 3:]))
    assert (with_origin[:, :3] > 0).all()


def test_future_and_hidden_values_ignored_before_validation_and_inputs_unchanged():
    reports, observed, days, scales = inputs()
    observed[:, 1, 0] = False
    expected = last_interval_rate_features(reports, observed, days, 60, scales)
    changed = reports.clone()
    changed[:, days > 60] = changed.new_tensor([float('inf'), -float('inf'), float('nan')])
    changed[:, 1, 0] = float('inf')
    snapshot, mask_snapshot = changed.clone(), observed.clone()
    actual = last_interval_rate_features(changed, observed, days, 60, scales)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(changed, snapshot, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(observed, mask_snapshot)
    for cutoff in (0., 16., 17., 52., 53., 99., 100.):
        altered = reports.clone()
        altered[:, days > cutoff] = float('inf')
        torch.testing.assert_close(
            last_interval_rate_features(altered, observed, days, cutoff, scales),
            last_interval_rate_features(reports, observed, days, cutoff, scales), rtol=0, atol=0)


def test_mixed_cutoff_batch_matches_scalar_calls():
    reports, observed, days, scales = inputs()
    cutoffs = days.new_tensor([17., 53., 100.])
    actual = last_interval_rate_features(reports.expand(3, -1, -1), observed.expand(3, -1, -1),
                                        days, cutoffs, scales)
    expected = torch.cat([last_interval_rate_features(reports, observed, days, float(c), scales)
                          for c in cutoffs])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_zero_control_preserves_shape_dtype_and_validation(dtype):
    reports, observed, days, scales = inputs()
    actual = last_interval_rate_features(reports, observed, days, 100, scales, dtype=dtype, zero_control=True)
    assert actual.shape == (1, 6) and actual.dtype == dtype
    assert torch.equal(actual, torch.zeros_like(actual))
    reports[:, 1, 0] = float('inf')
    with pytest.raises(ValueError, match='Available reports'):
        last_interval_rate_features(reports, observed, days, 100, scales, zero_control=True)


def test_subtraction_preserves_native_float64_precision_before_encoder_cast():
    reports = torch.tensor([[[1e12]*3, [1e12+1]*3]], dtype=torch.float64)
    observed = torch.ones_like(reports, dtype=torch.bool)
    days = torch.tensor([1., 361.], dtype=torch.float64)
    scales = torch.ones(3, dtype=torch.float64)
    actual = last_interval_rate_features(reports, observed, days, 361, scales)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual[:, :3], torch.full((1, 3), torch.asinh(torch.tensor(1.)).item()))
    assert torch.equal(reports.float()[:, 0], reports.float()[:, 1])
    with pytest.raises(ValueError, match='Native reports must use float64'):
        last_interval_rate_features(reports.float(), observed, days, 361, scales)


@pytest.mark.parametrize('failure', ['difference', 'normalized_rate'])
def test_extreme_finite_arithmetic_failures_rejected(failure):
    reports = torch.tensor([[[-1e308]*3, [1e308]*3]], dtype=torch.float64)
    observed = torch.ones_like(reports, dtype=torch.bool)
    days = torch.tensor([1., 361.], dtype=torch.float64)
    scales = torch.ones(3, dtype=torch.float64)
    if failure == 'normalized_rate':
        reports[:, 0] = 0
        scales *= 1e-308
    assert torch.isfinite(reports).all() and torch.isfinite(scales).all()
    with pytest.raises(ValueError, match='finite floating-point support'):
        last_interval_rate_features(reports, observed, days, 361, scales)


@pytest.mark.parametrize('failure', ['available_nan', 'available_inf', 'day_zero', 'scale', 'days'])
def test_invalid_available_inputs_rejected(failure):
    reports, observed, days, scales = inputs()
    if failure == 'available_nan': reports[:, 1, 0] = float('nan')
    elif failure == 'available_inf': reports[:, 1, 0] = float('inf')
    elif failure == 'day_zero': reports[:, 0, 0] = 1
    elif failure == 'scale': scales[0] = 0
    elif failure == 'days': days[2] = days[1]
    with pytest.raises(ValueError):
        last_interval_rate_features(reports, observed, days, 60, scales)
