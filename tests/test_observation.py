"""The masked likelihood and report sampler must describe the same joint law."""
import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal
from observation_reference import gaussian_endpoint_log_likelihood, sample_gaussian_reports


@pytest.mark.parametrize('law', ['independent', 'accumulated'])
def test_masked_likelihood_matches_dense_mvn_and_has_finite_gradients(law):
    days = torch.tensor([0., 30., 60., 180., 720.], dtype=torch.float64)
    q = (days[None, None, :, None]/360 * torch.tensor([.7, 1.1, .4])[None, None, None]).expand(2, 2, -1, -1).clone().requires_grad_()
    y = q.detach()[0].clone()+.15
    y[:, 0] = 0
    y[0, 1, 0] = -.5
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 2, 0] = False
    mask[0, 1:4, 1] = False
    mask[1, :, 2] = False
    y[~mask] = float('nan')
    a = torch.tensor([.2, .3, .4], dtype=torch.float64, requires_grad=True)
    r = torch.tensor([.1, .2, .3], dtype=torch.float64, requires_grad=True)
    actual = gaussian_endpoint_log_likelihood(y, q, a, r, mask, days, law=law)
    expected = torch.zeros_like(actual)
    for draw in range(2):
        for well in range(2):
            for phase in range(3):
                index = torch.where(mask[well, :, phase] & (days > 0))[0]
                if not len(index): continue
                variance = a[phase]**2*days[index]/360+r[phase]**2*q[draw, well, index, phase]**2
                covariance = torch.diag(variance) if law == 'independent' else torch.minimum(variance[:, None], variance[None, :])
                expected[draw, well] += MultivariateNormal(q[draw, well, index, phase], covariance_matrix=covariance).log_prob(y[well, index, phase])
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(a.grad).all() and torch.isfinite(r.grad).all()


def conditional_normal(q, days, a, r, observed, reports, law):
    variance = a*a*days/360+r*r*q*q
    covariance = np.diag(variance) if law == 'independent' else np.minimum.outer(variance, variance)
    missing = np.array([i for i in range(1, len(days)) if i not in observed])
    if len(observed):
        cross = covariance[np.ix_(missing, observed)]
        gain = np.linalg.solve(covariance[np.ix_(observed, observed)], cross.T).T
        mean = q[missing]+gain@(reports[observed]-q[observed])
        conditional_covariance = covariance[np.ix_(missing, missing)]-gain@cross.T
    else:
        mean, conditional_covariance = q[missing], covariance[np.ix_(missing, missing)]
    return missing, mean, conditional_covariance


@pytest.mark.parametrize('law', ['independent', 'accumulated'])
def test_conditional_bridge_means_covariance_and_missing_cutoff(law):
    days = np.array([0., 30., 60., 90., 180., 360., 720.])
    curve = np.sqrt(days/360)
    q = np.broadcast_to(curve[None, None, :, None], (1, 60000, len(days), 3)).copy()
    reports = np.full((1, len(days), 3), np.nan)
    reports[:, 0] = 0
    # Phase0 bridges multiple earlier missing points with two observed anchors;
    # phase1 has a missing cutoff at180; phase2 has no observed positive-time data.
    reports[0, 3, 0] = .654321012345
    reports[0, 4, 0] = -.123456789012
    reports[0, 2, 1] = .712345678901
    a, r = np.array([.7, .8, .9]), np.array([.2, .3, .4])
    draws = sample_gaussian_reports(q, reports, days, a, r, cutoff=180, rng=np.random.default_rng(321), law=law)
    for phase, observed in enumerate(([3, 4], [2], [])):
        index, mean, covariance = conditional_normal(curve, days, a[phase], r[phase], observed, reports[0, :, phase], law)
        samples = draws[0, :, :, phase][:, index]
        assert np.all(np.abs(samples.mean(axis=0)-mean) < 6*np.sqrt(np.diag(covariance)/len(samples)) + 1e-4)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), covariance, rtol=.025, atol=.01)
        for j in observed: assert np.all(draws[0, :, j, phase] == reports[0, j, phase])
    assert np.all(draws[:, :, 0] == 0)
    assert np.any(draws < 0), 'Signed Gaussian reports must not be clipped'


@pytest.mark.parametrize('law', ['independent', 'accumulated'])
def test_future_values_never_condition_sampler_and_inputs_remain_unchanged(law):
    days = np.array([0., 30., 60., 180.])
    q = np.broadcast_to(days[None, None, :, None]/360, (2, 100, 4, 3)).copy()
    history = np.full((2, 4, 3), np.nan)
    # Missing day zero is still deterministic; infinities after the cutoff
    # must be ignored, unlike infinities in available history.
    history[:, 1, 0] = .123456789123456
    changed = history.copy()
    changed[:, days > 60] = [np.inf, -np.inf, np.nan]
    before = q.copy()
    first = sample_gaussian_reports(q, history, days, np.ones(3), np.ones(3)*.1, cutoff=60, rng=np.random.default_rng(1), law=law)
    second = sample_gaussian_reports(q, changed, days, np.ones(3), np.ones(3)*.1, cutoff=60, rng=np.random.default_rng(1), law=law)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(q, before)
    assert np.all(first[:, :, 1, 0] == history[:, None, 1, 0])
    assert np.var(first[:, :, 2, 0]) > 0, 'Unknown cutoff anchor must be sampled'
    assert np.all(first[:, :, 0] == 0)
    changed[:, 2, 0] = np.inf
    with pytest.raises(ValueError, match='Available reports'):
        sample_gaussian_reports(q, changed, days, np.ones(3), np.ones(3)*.1,
                                cutoff=60, rng=np.random.default_rng(1), law=law)


def test_likelihood_ignores_unavailable_values_and_empty_history():
    days = torch.tensor([0., 30., 60.], dtype=torch.float64)
    q = (days[None, :, None]/360).expand(1, 3, 3)
    y = q.clone()
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, 1, 0] = True
    a = torch.ones(3, dtype=torch.float64)
    r = a*.1
    before = gaussian_endpoint_log_likelihood(y, q, a, r, mask, days)
    y[:, 2] = 1e300
    y[:, 1, 1:] = float('nan')
    after = gaussian_endpoint_log_likelihood(y, q, a, r, mask, days)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    zero = gaussian_endpoint_log_likelihood(y, q, a, r, torch.zeros_like(mask), days)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_stable_variance_difference_for_large_close_cumulatives():
    days = torch.tensor([0., 360., 361.], dtype=torch.float64)
    q = torch.tensor([0., 1e10, 1e10+.01], dtype=torch.float64)[None, :, None].expand(1, 3, 3)
    reports = q.clone()
    a, r = torch.ones(3, dtype=torch.float64), torch.full((3,), .1, dtype=torch.float64)
    mask = torch.ones_like(q, dtype=torch.bool)
    result = gaussian_endpoint_log_likelihood(reports, q, a, r, mask, days)
    variance1 = 1+.01*1e20
    variance2 = 1/360+.01*(q[0, 2, 0]-q[0, 1, 0])*(q[0, 2, 0]+q[0, 1, 0])
    expected = -1.5*(2*np.log(2*np.pi)+np.log(variance1)+torch.log(variance2))
    torch.testing.assert_close(result[0], expected)


def test_bridge_clock_differences_remain_stable_at_large_close_endpoints():
    base = 1e12
    curve = np.array([0., base, np.nextafter(base, np.inf), 0., 0.])
    curve[3] = np.nextafter(curve[2], np.inf)
    curve[4] = np.nextafter(curve[3], np.inf)
    days = np.array([0., 360., 360.001, 360.002, 360.003])
    q = np.broadcast_to(curve[None, None, :, None], (1, 40000, 5, 3)).copy()
    reports = np.full((1, 5, 3), np.nan)
    reports[0, 1, 0], reports[0, 3, 0] = curve[1]+2, curve[3]-3
    a, r = np.full(3, .5), np.full(3, .1)
    draws = sample_gaussian_reports(q, reports, days, a, r, cutoff=days[3], rng=np.random.default_rng(91))
    def dv(i, j):
        return a[0]**2*(days[j]-days[i])/360+(r[0]*(curve[j]-curve[i]))*(r[0]*curve[j]+r[0]*curve[i])
    first, last, whole = dv(1, 2), dv(2, 3), dv(1, 3)
    mean = 2+first/whole*(-3-2)
    variance = first*(last/whole)
    residual = draws[0, :, 2, 0]-curve[2]
    assert abs(residual.mean()-mean) < 6*np.sqrt(variance/len(residual))
    assert np.var(residual, ddof=1) == pytest.approx(variance, rel=.03)


@pytest.mark.parametrize('law', ['independent', 'accumulated'])
def test_relative_scale_product_avoids_spurious_intermediate_overflow(law):
    days = torch.tensor([0., 360., 361.], dtype=torch.float64)
    q = torch.tensor([0., 1e200, 1e200*(1+1e-12)], dtype=torch.float64)[None, :, None].expand(1, 3, 3)
    a, r = torch.ones(3, dtype=torch.float64), torch.full((3,), 1e-200, dtype=torch.float64)
    result = gaussian_endpoint_log_likelihood(q, q, a, r, torch.ones_like(q, dtype=torch.bool), days, law=law)
    assert torch.isfinite(result).all()
    draws = sample_gaussian_reports(q.numpy()[:, None], np.zeros((1, 3, 3)), days.numpy(),
                                    a.numpy(), r.numpy(), cutoff=0, rng=np.random.default_rng(3), law=law)
    assert np.isfinite(draws).all()


@pytest.mark.parametrize('field', ['nonmonotone', 'nonzero_day0', 'negative_noise'])
def test_invalid_generative_inputs_rejected(field):
    days = np.array([0., 30., 60.])
    q = np.broadcast_to(days[None, None, :, None], (1, 2, 3, 3)).copy()
    a = np.ones(3)
    if field == 'nonmonotone': q[:, :, 2] = 10
    elif field == 'nonzero_day0': q[:, :, 0] = 1
    else: a[0] = -1
    with pytest.raises(ValueError):
        sample_gaussian_reports(q, np.zeros((1, 3, 3)), days, a, np.ones(3), cutoff=30, rng=np.random.default_rng(1))
