"""Independent Gaussian-law, conditioning, scaling, and gradient score oracles."""
import math
import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import torch

from joint_production.gaussian_increment_score import (
    accumulated_increment_u_crps, gaussian_absolute_moment,
)
from joint_production.conditional_rates import _u_scores
from joint_production.torch_observation import sample_accumulated_reports


DTYPE = torch.float64


def absolute_normal_quadrature(mean, variance):
    """Integrate against a standard-normal density, including the absolute kink."""
    if variance == 0:
        return abs(mean)
    sd = math.sqrt(variance)
    kink = -mean / sd
    points = [kink] if -12 < kink < 12 else None
    value, _ = quad(lambda z: abs(mean + sd*z) * math.exp(-z*z/2) / math.sqrt(2*math.pi),
                    -12, 12, points=points, epsabs=1e-12, epsrel=1e-12)
    return value


def inputs(dtype=DTYPE, device="cpu"):
    days = torch.tensor([0., 30., 90., 180.], dtype=dtype, device=device)
    increments = torch.tensor([
        [[[.8, 1.3, .6], [1.4, .7, 1.9], [2.1, 2.8, 1.2]],
         [[1.4, .5, 1.2], [.9, 2.1, .8], [1.7, 1.5, 3.1]],
         [[.3, 2.0, .9], [2.5, 1.1, 1.4], [3.0, .6, 2.3]]],
        [[[1.3, .8, .5], [.7, 1.6, 2.1], [1.8, 2.2, .9]],
         [[.7, 1.1, 1.5], [1.9, .8, .6], [2.4, 1.7, 2.5]],
         [[1.5, .4, 1.1], [1.2, 2.3, 1.0], [.6, 1.3, 1.7]]],
    ], dtype=dtype, device=device)
    cumulative = torch.cat((torch.zeros_like(increments[:, :, :1]), increments.cumsum(2)), dim=2)
    target = torch.tensor([
        [[0., 0., 0.], [1.2, 1.7, .9], [-.4, 2.4, 2.8], [4.3, 4.8, 4.1]],
        [[0., 0., 0.], [.8, 1.1, 1.4], [2.0, .7, 2.9], [3.9, 3.2, 4.0]],
    ], dtype=dtype, device=device)
    a = torch.tensor([.7, 1.1, .4], dtype=dtype, device=device)
    r = torch.tensor([.12, .23, .31], dtype=dtype, device=device)
    scales = torch.tensor([[1.2, 3.1, 2.4], [2.7, 4.3, 3.8]], dtype=dtype, device=device)
    weights = torch.tensor([[1.5, 2., 1.], [1., 1.25, 2.5]], dtype=dtype, device=device)
    return target, cumulative, days, 1, a, r, scales, weights


def dense_gaussian_score_oracle(arguments):
    """Project full accumulated covariance matrices onto each scored increment."""
    target, q, days, origin, a, r, scales, weights = arguments
    target, q, days, a, r, scales, weights = [x.detach().cpu().numpy()
        for x in (target, q, days, a, r, scales, weights)]
    future = np.flatnonzero(days > days[origin])
    n, draws = q.shape[:2]
    result = np.zeros(n)
    for well in range(n):
        for cell, endpoint in enumerate(future):
            contrast = np.zeros(len(days)); contrast[origin] = -1.; contrast[endpoint] = 1.
            for phase in range(3):
                if not np.isfinite(target[well, [origin, endpoint], phase]).all():
                    continue
                truth = (target[well, endpoint, phase]-target[well, origin, phase]) / scales[cell, phase]
                means, variances = [], []
                for draw in range(draws):
                    clock = a[phase]**2 * days/360. + (r[phase]*q[well, draw, :, phase])**2
                    covariance = np.minimum(clock[:, None], clock[None, :])
                    means.append(contrast @ q[well, draw, :, phase] / scales[cell, phase])
                    variances.append(contrast @ covariance @ contrast / scales[cell, phase]**2)
                accuracy = sum(absolute_normal_quadrature(mu-truth, var)
                               for mu, var in zip(means, variances, strict=True)) / draws
                # Ordered off-diagonal pairs make the U normalization explicit.
                dispersion = sum(absolute_normal_quadrature(means[i]-means[j], variances[i]+variances[j])
                    for i in range(draws) for j in range(draws) if i != j) / (2*draws*(draws-1))
                result[well] += weights[cell, phase] * (accuracy-dispersion) / (len(future)*3)
    return result


def test_absolute_moment_broadcasts_and_matches_independent_quadrature():
    means = torch.tensor([-9., -.7, 0., .3, 17.], dtype=DTYPE)[:, None]
    variances = torch.tensor([0., 1e-12, .1, 4.], dtype=DTYPE)[None, :]
    actual = gaussian_absolute_moment(means, variances)
    expected = [[absolute_normal_quadrature(float(mu), float(var)) for var in variances[0]]
                for mu in means[:, 0]]
    assert actual.shape == (5, 4) and actual.dtype == DTYPE and actual.device == means.device
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize("draws", [2, 3, 7])
def test_identical_latent_draws_reduce_to_closed_single_gaussian_crps(draws):
    target, q, days, origin, a, r, scales, weights = inputs()
    q = q[:, :1].expand(-1, draws, -1, -1)
    actual = accumulated_increment_u_crps(target, q, days, origin, a, r, scales, weights)
    expected = []
    for well in range(len(target)):
        score = 0.
        for cell, endpoint in enumerate([2, 3]):
            for phase in range(3):
                curve = q[well, 0, :, phase].numpy()
                clock = float(a[phase])**2 * days.numpy()/360. + (float(r[phase])*curve)**2
                sd = math.sqrt(clock[endpoint]-clock[origin]) / float(scales[cell, phase])
                residual = float((target[well, endpoint, phase]-target[well, origin, phase])
                    -(q[well, 0, endpoint, phase]-q[well, 0, origin, phase])) / float(scales[cell, phase])
                z = residual/sd
                score += float(weights[cell, phase]) * sd * (z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))/6
        expected.append(score)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-13, atol=2e-14)


def test_mixture_increment_score_matches_dense_covariance_and_quadrature_oracle():
    arguments = inputs()
    np.testing.assert_allclose(accumulated_increment_u_crps(*arguments).numpy(),
                               dense_gaussian_score_oracle(arguments), rtol=3e-12, atol=3e-13)


def test_deterministic_limit_matches_existing_u_score_and_hand_enumeration():
    future = torch.tensor([0., 2., 4.], dtype=DTYPE).reshape(1, 3, 1, 1).expand(1, 3, 1, 3)
    q = torch.cat((torch.zeros_like(future), future), dim=2)
    target = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]], dtype=DTYPE)
    days = torch.tensor([0., 30.], dtype=DTYPE)
    scale = weights = torch.ones((1, 3), dtype=DTYPE)
    zero = torch.zeros(3, dtype=DTYPE)
    actual = accumulated_increment_u_crps(target, q, days, 0, zero, zero, scale, weights)
    expected = _u_scores(target, q, torch.zeros_like(target, dtype=torch.bool), days, 0, scale, weights)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(actual, torch.tensor([1/3], dtype=DTYPE), rtol=0, atol=1e-15)


def test_signed_missing_targets_keep_frozen_weights_and_full_cell_denominator():
    arguments = list(inputs())
    target = arguments[0]
    target[0, 1, 1] = float("nan")  # Missing gas origin removes every future gas cell.
    target[0, 3, 2] = float("nan")
    target[1, 2, 0] = float("inf")
    target[1, 3, 1] = -4.125  # Signed reported correction remains a target.
    arguments[-1][0, 2] = 0.  # Zero weight is valid; denominator still includes the cell.
    actual = accumulated_increment_u_crps(*arguments)
    np.testing.assert_allclose(actual.numpy(), dense_gaussian_score_oracle(arguments), rtol=3e-12, atol=3e-13)
    zero = torch.zeros(3, dtype=DTYPE)
    deterministic = accumulated_increment_u_crps(*arguments[:4], zero, zero, *arguments[6:])
    reference = _u_scores(target, arguments[1], torch.zeros_like(target, dtype=torch.bool),
                          arguments[2], arguments[3], *arguments[6:])[0]
    torch.testing.assert_close(deterministic, reference, rtol=0, atol=2e-15)
    changed = target.clone(); changed[:, 0] = float("nan")
    torch.testing.assert_close(accumulated_increment_u_crps(changed, *arguments[1:]), actual, rtol=0, atol=0)


def test_hidden_origin_noise_cancels_in_exact_affine_accumulated_sampler():
    days = torch.tensor([0., 30., 90., 180.], dtype=DTYPE)
    curve = torch.tensor([0., .8, 1.4, 2.8], dtype=DTYPE)[:, None] * torch.tensor([1., .7, 2.], dtype=DTYPE)
    dimension = curve.numel()
    q = curve[None, None].expand(1, dimension+1, -1, -1)
    normals = torch.cat((torch.zeros((1, dimension), dtype=DTYPE), torch.eye(dimension, dtype=DTYPE)))
    normals = normals.reshape_as(q)
    a = torch.tensor([.7, .9, .5], dtype=DTYPE); r = torch.tensor([.2, .3, .4], dtype=DTYPE)
    history = torch.full((1, 4, 3), float("nan"), dtype=DTYPE)
    history[0, 1, 0] = -.5; history[0, 2, 2] = 1.8
    sampled = sample_accumulated_reports(q, history, days, a, r, cutoff=90., standard_normals=normals)
    difference = sampled[:, :, 3]-sampled[:, :, 2]
    mean = difference[0, 0]
    factor = difference[0, 1:]-mean
    variance = factor.square().sum(0)
    expected_mean = curve[3]-curve[2]
    expected_variance = a.square()*(days[3]-days[2])/360. + r.square()*(curve[3].square()-curve[2].square())
    torch.testing.assert_close(mean, expected_mean, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(variance, expected_variance, rtol=1e-14, atol=1e-14)
    # Subtracting independently sampled marginals wrongly retains hidden-origin noise.
    independent_variance = ((sampled[0, 1:, 3]-sampled[0, 0, 3]).square()
                            +(sampled[0, 1:, 2]-sampled[0, 0, 2]).square()).sum(0)
    assert float(independent_variance[1]) > 1.5*float(variance[1])
    history[0, 1, 0] = 7.; history[0, 2, 2] = -3.
    changed = sample_accumulated_reports(q, history, days, a, r, cutoff=90., standard_normals=normals)
    torch.testing.assert_close(changed[:, :, 3]-changed[:, :, 2], difference, rtol=2e-14, atol=2e-14)
    target = torch.zeros((1, 4, 3), dtype=DTYPE); target[0, 3] = expected_mean + torch.tensor([.2, -.4, .1])
    score = accumulated_increment_u_crps(target, q[:, :3], days, 2, a, r,
                                       torch.ones((1, 3), dtype=DTYPE), torch.ones((1, 3), dtype=DTYPE))
    oracle = np.mean([absolute_normal_quadrature(float(mean[p]-target[0, 3, p]), float(variance[p]))
                      - math.sqrt(float(variance[p])/math.pi) for p in range(3)])
    assert float(score[0]) == pytest.approx(oracle, rel=2e-12, abs=2e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_variance_convention_has_finite_zero_variance_gradient(dtype):
    mean = torch.tensor([-2., 0., 3.], dtype=dtype, requires_grad=True)
    variance = torch.zeros(3, dtype=dtype, requires_grad=True)
    actual = gaussian_absolute_moment(mean, variance)
    dmean, dvariance = torch.autograd.grad(actual.sum(), (mean, variance))
    torch.testing.assert_close(actual, mean.abs(), rtol=0, atol=0)
    torch.testing.assert_close(dmean, torch.tensor([-1., 0., 1.], dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(dvariance, torch.zeros_like(variance), rtol=0, atol=0)


@pytest.mark.parametrize("mean_value,variance_value", [(-.3, .8), (.0002, 1e-8), (0., 1e-12), (.7, 1e-12)])
def test_absolute_moment_gradients_match_small_relative_finite_differences(mean_value, variance_value):
    mean = torch.tensor(mean_value, dtype=DTYPE, requires_grad=True)
    variance = torch.tensor(variance_value, dtype=DTYPE, requires_grad=True)
    gradient = torch.autograd.grad(gaussian_absolute_moment(mean, variance), (mean, variance))
    steps = (max(abs(mean_value), math.sqrt(variance_value))*1e-5, variance_value*1e-4)
    for index, step in enumerate(steps):
        plus = [mean.detach().clone(), variance.detach().clone()]
        minus = [x.clone() for x in plus]
        plus[index] += step; minus[index] -= step
        finite_difference = (gaussian_absolute_moment(*plus)-gaussian_absolute_moment(*minus))/(2*step)
        torch.testing.assert_close(gradient[index], finite_difference, rtol=3e-7, atol=2e-9)
    z = mean_value/math.sqrt(variance_value)
    expected = [2*norm.cdf(z)-1, norm.pdf(z)/math.sqrt(variance_value)]
    np.testing.assert_allclose([float(x) for x in gradient], expected, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("noise_multiplier", [1., 1e-5])
def test_score_float64_gradient_matches_finite_difference_through_latent_and_noise(noise_multiplier):
    target, q, days, origin, a, r, scales, weights = inputs()
    target = target[:1]
    log_increments = torch.diff(q[:1], dim=2).log().requires_grad_()
    a = (a*noise_multiplier).requires_grad_(); r = (r*noise_multiplier).requires_grad_()

    def score(log_delta, absolute, relative):
        delta = log_delta.exp()
        cumulative = torch.cat((torch.zeros_like(delta[:, :, :1]), delta.cumsum(2)), dim=2)
        return accumulated_increment_u_crps(target, cumulative, days, origin, absolute, relative, scales, weights)

    assert torch.autograd.gradcheck(score, (log_increments, a, r), fast_mode=True,
                                    eps=1e-7 if noise_multiplier == 1 else 1e-9,
                                    rtol=2e-4, atol=2e-6)
    gradients = torch.autograd.grad(score(log_increments, a, r).sum(), (log_increments, a, r))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert gradients[0].abs().sum() > 0


@pytest.mark.parametrize("missing", [False, True])
def test_deterministic_ties_and_entirely_missing_targets_have_finite_zero_gradients(missing):
    target, q, days, origin, a, r, scales, weights = inputs()
    q = q[:, :1].expand_as(q).clone().requires_grad_()
    target = q[:, 0].detach().clone()
    if missing:
        target[:] = float("nan")
    a = torch.zeros_like(a, requires_grad=True); r = torch.zeros_like(r, requires_grad=True)
    score = accumulated_increment_u_crps(target, q, days, origin, a, r, scales, weights)
    torch.testing.assert_close(score, torch.zeros_like(score), rtol=0, atol=0)
    for gradient in torch.autograd.grad(score.sum(), (q, a, r)):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_phase_unit_conversion_standardizes_before_squares(dtype):
    target, q, days, origin, a, r, scales, weights = inputs(dtype)
    factors = torch.tensor([1e20, 1e-20, 1e10] if dtype == torch.float32 else [1e180, 1e-180, 1e100], dtype=dtype)
    expected = accumulated_increment_u_crps(target, q, days, origin, a, r, scales, weights)
    actual = accumulated_increment_u_crps(target*factors, q*factors, days, origin, a*factors, r, scales*factors, weights)
    assert actual.dtype == dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=3e-6 if dtype == torch.float32 else 3e-13,
                               atol=3e-7 if dtype == torch.float32 else 3e-14)


@pytest.mark.parametrize("noise_source", ["absolute", "relative"])
def test_positive_report_variance_underflow_is_rejected_instead_of_made_deterministic(noise_source):
    dtype = torch.float32
    target = torch.zeros((1, 2, 3), dtype=dtype)
    q = torch.zeros((1, 2, 2, 3), dtype=dtype)
    days = torch.tensor([0., 30.], dtype=dtype)
    absolute = torch.zeros(3, dtype=dtype)
    relative = torch.zeros(3, dtype=dtype)
    scales = torch.full((1, 3), 1e20, dtype=dtype)
    weights = torch.ones((1, 3), dtype=dtype)
    if noise_source == "absolute":
        absolute.fill_(1e-4)
        true_sd = 1e-4 * math.sqrt(30/360) / 1e20
    else:
        q[:, :, 1] = 1e-4
        relative.fill_(1.)
        true_sd = 1e-4 / 1e20
    # The standard deviation and its CRPS contribution are representable;
    # squaring it loses the variance and must not change the report law.
    represented_sd = torch.tensor(true_sd, dtype=dtype)
    assert represented_sd > 0 and represented_sd.square() == 0
    assert torch.tensor((math.sqrt(2)-1)*true_sd/math.sqrt(math.pi), dtype=dtype) > 0
    with pytest.raises(FloatingPointError, match="increment moments"):
        accumulated_increment_u_crps(target, q, days, 0, absolute, relative, scales, weights)
    # An explicitly deterministic law still has the exact ordinary U score.
    zero = torch.zeros(3, dtype=dtype)
    deterministic = accumulated_increment_u_crps(target, q, days, 0, zero, zero, scales, weights)
    reference = _u_scores(target, q, torch.zeros_like(target, dtype=torch.bool),
                          days, 0, scales, weights)[0]
    torch.testing.assert_close(deterministic, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only value and gradient parity")
def test_cuda_fixed_inputs_and_gradients_match_cpu():
    outcomes = []
    for device in ("cpu", "cuda"):
        target, q, days, origin, a, r, scales, weights = inputs(device=device)
        q.requires_grad_(); a.requires_grad_(); r.requires_grad_()
        value = accumulated_increment_u_crps(target, q, days, origin, a, r, scales, weights)
        gradients = torch.autograd.grad(value.sum(), (q, a, r))
        outcomes.append([value.detach().cpu(), *[x.cpu() for x in gradients]])
    for cpu, cuda in zip(*outcomes, strict=True):
        assert torch.isfinite(cuda).all()
        torch.testing.assert_close(cpu, cuda, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("fault", ["one_draw", "no_draws", "negative_noise", "infinite_noise", "zero_scale", "negative_weight", "bad_scale_shape", "duplicate_days", "origin_last", "nonfinite_latent", "decreasing_latent", "mixed_dtype"])
def test_invalid_score_support_or_contract_is_rejected(fault):
    args = list(inputs())
    if fault == "one_draw": args[1] = args[1][:, :1]
    elif fault == "no_draws": args[1] = args[1][:, :0]
    elif fault == "negative_noise": args[4][0] = -.1
    elif fault == "infinite_noise": args[5][0] = float("inf")
    elif fault == "zero_scale": args[6][0, 0] = 0.
    elif fault == "negative_weight": args[7][0, 0] = -.1
    elif fault == "bad_scale_shape": args[6] = args[6][:1]
    elif fault == "duplicate_days": args[2][2] = args[2][1]
    elif fault == "origin_last": args[3] = len(args[2])-1
    elif fault == "nonfinite_latent": args[1][0, 0, 2, 1] = float("nan")
    elif fault == "decreasing_latent": args[1][0, 0, 2, 1] = 0.
    else: args[4] = args[4].float()
    with pytest.raises(ValueError):
        accumulated_increment_u_crps(*args)


@pytest.mark.parametrize("mean,variance", [(float("nan"), 1.), (0., -1.), (0., float("inf"))])
def test_absolute_moment_rejects_nonfinite_mean_or_invalid_variance(mean, variance):
    with pytest.raises(ValueError):
        gaussian_absolute_moment(torch.tensor(mean, dtype=DTYPE), torch.tensor(variance, dtype=DTYPE))
