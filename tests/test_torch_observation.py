"""Conditional report law, cutoff isolation, and reparameterized gradients."""
import pytest
import torch

from joint_production.torch_observation import sample_accumulated_reports


def fixture_inputs(dtype=torch.float64, samples=4, include_zero=True):
    days = torch.tensor([0., 30., 60., 90., 180., 360., 720.], dtype=dtype)
    if not include_zero:
        days = days[1:]
    q = (days / 360).sqrt()[None, None, :, None].expand(2, samples, -1, 3).clone()
    history = torch.full((2, len(days), 3), float('nan'), dtype=dtype)
    history[0, days == 90, 0] = .65
    history[0, days == 180, 0] = -.12
    history[0, days == 60, 1] = .71
    history[1, days == 30, 2] = -.3
    history[1, days == 90, 2] = .3
    a = torch.tensor([.7, .8, .9], dtype=dtype)
    r = torch.tensor([.2, .3, .4], dtype=dtype)
    return q, history, days, a, r


@pytest.mark.parametrize('include_zero', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_exact_conditional_mean_and_covariance_match_dense_gaussian(dtype, include_zero):
    # The sampler is affine in its supplied normal variates. Evaluating at zero
    # and all basis vectors yields its exact mean and covariance, without Monte
    # Carlo tolerance or assumptions about its particular bridge factorization.
    q, history, days, a, r = fixture_inputs(dtype, include_zero=include_zero)
    dimensions = len(days) * 3
    q = q[:, :1].expand(-1, dimensions + 1, -1, -1)
    normals = torch.cat((torch.zeros(1, dimensions, dtype=dtype),
                         torch.eye(dimensions, dtype=dtype))).reshape(1, dimensions + 1, len(days), 3)
    actual = sample_accumulated_reports(q, history, days, a, r, cutoff=180,
                                        standard_normals=normals.expand_as(q))
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    for well in range(2):
        mean = q[well, 0].flatten().clone()
        covariance = torch.zeros(dimensions, dimensions, dtype=dtype)
        for phase in range(3):
            variance = a[phase].square() * days / 360 + (r[phase] * q[well, 0, :, phase]).square()
            phase_covariance = torch.minimum(variance[:, None], variance[None, :])
            index = torch.arange(phase, dimensions, 3)
            covariance[index[:, None], index[None, :]] = phase_covariance
        known = torch.isfinite(history[well]) & (days[:, None] <= 180) & (days[:, None] > 0)
        index = torch.where(known.flatten())[0]
        if len(index):
            gain = torch.linalg.solve(covariance[index[:, None], index[None, :]], covariance[index]).T
            mean = mean + gain @ (history[well].flatten()[index] - mean[index])
            covariance = covariance - gain @ covariance[index]
        sampled_mean = actual[well, 0].flatten()
        factor = actual[well, 1:].flatten(1) - sampled_mean
        torch.testing.assert_close(sampled_mean, mean, atol=tolerance, rtol=tolerance)
        # Full covariance verifies temporal dependence AND conditional phase
        # independence, including hidden origins and phases without any history.
        torch.testing.assert_close(factor.T @ factor, covariance, atol=tolerance, rtol=tolerance)


def test_cutoff_values_ignored_known_reports_exact_and_inputs_preserved():
    q, history, days, a, r = fixture_inputs()
    initial_q, initial_history = q.clone(), history.clone()
    normals = torch.randn(q.shape, dtype=q.dtype, generator=torch.Generator().manual_seed(42))
    baseline = sample_accumulated_reports(q, history, days, a, r, cutoff=180, standard_normals=normals)
    changed = history.clone()
    changed[:, days > 180] = torch.tensor([float('inf'), -float('inf'), 1e300], dtype=q.dtype)
    actual = sample_accumulated_reports(q, changed, days, a, r, cutoff=180, standard_normals=normals)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(q, initial_q, rtol=0, atol=0)
    torch.testing.assert_close(history, initial_history, rtol=0, atol=0, equal_nan=True)
    known = torch.isfinite(history) & (days[None, :, None] <= 180)
    torch.testing.assert_close(actual[known[:, None].expand_as(q)],
                               history[:, None].expand_as(q)[known[:, None].expand_as(q)], rtol=0, atol=0)
    assert (actual[:, :, days == 0] == 0).all()
    assert actual[0, :, 4, 1].var() > 0, 'A missing cutoff endpoint remains random'
    assert (actual < 0).any(), 'The Gaussian report law is signed, without clipping'
    changed[0, 1, 0] = float('inf')
    with pytest.raises(ValueError, match='Available reports'):
        sample_accumulated_reports(q, changed, days, a, r, cutoff=180)


def test_pathwise_gradcheck_includes_day_zero_and_observed_reports():
    q, history, days, a, r = fixture_inputs(samples=2)
    increments = torch.full((2, 2, len(days)-1, 3), -.7, dtype=q.dtype, requires_grad=True)
    a, r = a.requires_grad_(), r.requires_grad_()
    normals = torch.randn(q.shape, dtype=q.dtype, generator=torch.Generator().manual_seed(3))

    def draw(log_increments, abs_noise, rel_noise):
        cumulative = torch.cat((torch.zeros_like(log_increments[:, :, :1]),
                                log_increments.exp().cumsum(2)), dim=2)
        return sample_accumulated_reports(cumulative, history, days, abs_noise, rel_noise,
                                          cutoff=180, standard_normals=normals)

    assert torch.autograd.gradcheck(draw, (increments, a, r), fast_mode=True)
    result = draw(increments, a, r)
    result.square().sum().backward()
    for parameter in (increments, a, r):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_seeded_generator_is_reproducible():
    args = fixture_inputs()
    first = sample_accumulated_reports(*args, cutoff=0, generator=torch.Generator().manual_seed(3))
    second = sample_accumulated_reports(*args, cutoff=0, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_float32_plateau_gradients_remain_finite():
    q, history, days, a, r = fixture_inputs(torch.float32)
    q.zero_().requires_grad_()
    a.requires_grad_()
    r.requires_grad_()
    result = sample_accumulated_reports(q, history, days, a, r, cutoff=180)
    result.square().mean().backward()
    for parameter in (q, a, r):
        assert torch.isfinite(parameter.grad).all()


def test_close_large_endpoints_use_stable_clock_differences():
    dtype = torch.float64
    days = torch.tensor([0., 360., 360.001, 360.002, 360.003], dtype=dtype)
    curve = torch.zeros(5, dtype=dtype)
    curve[1] = 1e12
    for index in range(2, 5):
        curve[index] = torch.nextafter(curve[index-1], torch.tensor(float('inf'), dtype=dtype))
    q = curve[None, None, :, None].expand(1, 16, 5, 3)
    history = torch.full((1, 5, 3), float('nan'), dtype=dtype)
    history[0, 1, 0], history[0, 3, 0] = curve[1] + 2, curve[3] - 3
    a, r = torch.full((3,), .5, dtype=dtype), torch.full((3,), .1, dtype=dtype)
    normals = torch.cat((torch.zeros(1, 15, dtype=dtype), torch.eye(15, dtype=dtype))).reshape_as(q)
    result = sample_accumulated_reports(q, history, days, a, r, cutoff=float(days[3]), standard_normals=normals)

    def dv(left, right):
        return (a[0].square() * (days[right]-days[left])/360
                + (r[0]*(curve[right]-curve[left])) * (r[0]*curve[right]+r[0]*curve[left]))

    first, last = dv(1, 2), dv(2, 3)
    expected_mean = curve[2] + 2 + first/(first+last)*(-3-2)
    torch.testing.assert_close(result[0, 0, 2, 0], expected_mean, rtol=0, atol=1e-4)
    variance = (result[0, 1:, 2, 0]-result[0, 0, 2, 0]).square().sum()
    torch.testing.assert_close(variance, first * last/(first+last), rtol=1e-6, atol=0)


def test_overflowing_bridge_clock_rejected_even_when_increments_are_finite():
    # Each factored increment is finite, but their sum exceeds float64's range.
    # Dividing by infinity would otherwise silently collapse the bridge variance.
    days = torch.tensor([1., 2.], dtype=torch.float64)
    curve = torch.tensor([1e154, 1.45e154], dtype=days.dtype)
    q = curve[None, None, :, None].expand(1, 2, 2, 3)
    history = torch.full((1, 2, 3), float('nan'), dtype=days.dtype)
    history[:, 1] = curve[1]
    noise = torch.ones(3, dtype=days.dtype)
    first = curve[0].square()
    last = (curve[1]-curve[0])*(curve[1]+curve[0])
    assert torch.isfinite(first) and torch.isfinite(last)
    assert torch.isinf(first+last)
    with pytest.raises(ValueError, match='bridge clock'):
        sample_accumulated_reports(q, history, days, noise, noise, cutoff=2,
                                   standard_normals=torch.ones_like(q))


@pytest.mark.parametrize('invalid', ['negative_q', 'decreasing_q', 'nonzero_origin',
                                    'nonzero_observed_origin', 'zero_abs', 'negative_rel',
                                    'infinite_noise', 'duplicate_days', 'nan_days', 'cutoff',
                                    'normal_shape', 'infinite_normal'])
def test_invalid_inputs_rejected(invalid):
    q, history, days, a, r = fixture_inputs()
    cutoff, normals = 180, torch.zeros_like(q)
    if invalid == 'negative_q': q[:, :, 1] = -1
    elif invalid == 'decreasing_q': q[:, :, 2] = .01
    elif invalid == 'nonzero_origin': q[:, :, 0] = .01
    elif invalid == 'nonzero_observed_origin': history[:, 0] = 1
    elif invalid == 'zero_abs': a[0] = 0
    elif invalid == 'negative_rel': r[0] = -1
    elif invalid == 'infinite_noise': a[0] = float('inf')
    elif invalid == 'duplicate_days': days[2] = days[1]
    elif invalid == 'nan_days': days[2] = float('nan')
    elif invalid == 'cutoff': cutoff = float('inf')
    elif invalid == 'normal_shape': normals = normals[..., :2]
    elif invalid == 'infinite_normal': normals[0, 0, 1, 0] = float('inf')
    with pytest.raises(ValueError):
        sample_accumulated_reports(q, history, days, a, r, cutoff=cutoff, standard_normals=normals)
