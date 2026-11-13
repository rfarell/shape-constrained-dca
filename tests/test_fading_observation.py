"""Independent Gaussian oracles for fading reports; synthetic arrays only."""
from importlib import import_module
import math

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def fading():
    return import_module("joint_production.fading_observation")


def _covariance(q, days, absolute, relative, half_life):
    """Dense covariance from its defining kernel, independent of transitions."""
    count = len(days)
    result = np.zeros((count * 3, count * 3))
    attenuation = np.exp(-math.log(2) * np.abs(days[:, None] - days[None, :]) / half_life)
    earlier = np.minimum.outer(np.arange(count), np.arange(count))
    for phase in range(3):
        variance = absolute[phase] ** 2 * days / 360 + (relative[phase] * q[:, phase]) ** 2
        indices = np.arange(count) * 3 + phase
        result[np.ix_(indices, indices)] = attenuation * variance[earlier]
    return result


def _conditional_oracle(q, history, days, absolute, relative, half_life, cutoff):
    covariance = _covariance(q, days, absolute, relative, half_life)
    observed = (np.isfinite(history) & (days[:, None] > 0) & (days[:, None] <= cutoff)).ravel()
    mean = q.ravel().copy()
    if observed.any():
        cross = covariance[:, observed]
        observed_covariance = covariance[np.ix_(observed, observed)]
        residual = history.ravel()[observed] - mean[observed]
        mean += cross @ np.linalg.solve(observed_covariance, residual)
        covariance -= cross @ np.linalg.solve(observed_covariance, cross.T)
    return mean.reshape(q.shape), covariance


def _log_likelihood_oracle(reports, cumulative, days, absolute, relative, mask, half_life):
    output = np.zeros(cumulative.shape[:-2])
    for prefix in np.ndindex(cumulative.shape[:-3]):
        for well in range(len(reports)):
            q = cumulative[prefix + (well,)]
            covariance = _covariance(q, days, absolute, relative, half_life)
            observed = (mask[well] & (days[:, None] > 0)).ravel()
            if not observed.any():
                continue
            selected = covariance[np.ix_(observed, observed)]
            residual = reports[well].ravel()[observed] - q.ravel()[observed]
            sign, logdet = np.linalg.slogdet(selected)
            assert sign == 1
            output[prefix + (well,)] = -0.5 * (
                observed.sum() * math.log(2 * math.pi) + logdet
                + residual @ np.linalg.solve(selected, residual)
            )
    return output


@pytest.fixture
def report_case():
    days = np.array([0.0, 15.0, 30.0, 90.0, 180.0, 360.0])
    amplitude = np.array([[15.0, 70.0, 25.0], [20.0, 50.0, 35.0]])
    powers = np.array([0.8, 1.1, 0.5])
    q = amplitude[:, None] * (days[:, None] / 360.0) ** powers
    history = np.full(q.shape, np.nan)
    history[:, 0] = 0.0
    for well, phase, indices in ((0, 0, [1, 4]), (0, 1, [2, 3]), (0, 2, [1, 3, 4]),
                                 (1, 0, [3]), (1, 2, [2, 4])):
        for index in indices:
            history[well, index, phase] = q[well, index, phase] + (well + 1) * (phase + 1) * (index - 2.5)
    return q, history, days, np.array([0.7, 1.1, 0.5]), np.array([0.08, 0.12, 0.1])


def test_sampler_matches_dense_conditional_mean_and_covariance_by_linear_response(fading, report_case):
    q, history, days, absolute, relative = report_case
    dimension = q.shape[1] * 3
    # Zero noise yields the mean. Unit-vector perturbations reveal the linear
    # noise map; J.T @ J is its covariance for independent standard normals.
    normals = np.r_[np.zeros((1, dimension)), np.eye(dimension)].reshape(dimension + 1, len(days), 3)
    repeated_q = np.repeat(q[:, None], dimension + 1, axis=1)
    repeated_normals = np.broadcast_to(normals[None], repeated_q.shape).copy()
    output = fading.sample_fading_reports(
        torch.tensor(repeated_q), torch.tensor(history), torch.tensor(days),
        torch.tensor(absolute), torch.tensor(relative), cutoff=180.0, half_life_days=180.0,
        standard_normals=torch.tensor(repeated_normals),
    ).numpy()
    for well in range(len(q)):
        mean, covariance = _conditional_oracle(q[well], history[well], days, absolute, relative, 180.0, 180.0)
        jacobian = (output[well, 1:] - output[well, :1]).reshape(dimension, dimension)
        np.testing.assert_allclose(output[well, 0], mean, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(jacobian.T @ jacobian, covariance, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_known_reports_day_zero_and_future_poison_are_handled_before_validation(fading, report_case, dtype):
    q, history, days, absolute, relative = report_case
    future_history = history.copy()
    future_history[:, -1] = [np.inf, -np.inf, np.nan]
    tensor = lambda value: torch.tensor(value, dtype=dtype)
    cumulative = tensor(np.repeat(q[:, None], 2, axis=1))
    normals = torch.linspace(-1.0, 1.0, cumulative.numel(), dtype=dtype).reshape(cumulative.shape)
    arguments = dict(cutoff=180.0, half_life_days=180.0, standard_normals=normals)
    original = fading.sample_fading_reports(cumulative, tensor(history), tensor(days), tensor(absolute), tensor(relative), **arguments)
    poisoned = fading.sample_fading_reports(cumulative, tensor(future_history), tensor(days), tensor(absolute), tensor(relative), **arguments)
    torch.testing.assert_close(poisoned, original, rtol=0, atol=0)
    known = torch.isfinite(tensor(history))[:, None].expand_as(original)
    observed = tensor(history)[:, None].expand_as(original)
    torch.testing.assert_close(original[known], observed[known], rtol=0, atol=0)
    torch.testing.assert_close(original[:, :, 0], torch.zeros_like(original[:, :, 0]), rtol=0, atol=0)
    # Day zero is deterministic even when its input is absent.
    missing_zero = history.copy()
    missing_zero[:, 0] = np.nan
    restored = fading.sample_fading_reports(cumulative, tensor(missing_zero), tensor(days), tensor(absolute), tensor(relative), **arguments)
    torch.testing.assert_close(restored, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sampler_and_likelihood_gradients_are_finite_with_zero_known_normal_gradients(fading, report_case, dtype):
    q, history, days, absolute, relative = report_case
    cumulative = torch.tensor(np.repeat(q[:, None], 2, axis=1), dtype=dtype, requires_grad=True)
    a = torch.tensor(absolute, dtype=dtype, requires_grad=True)
    r = torch.tensor(relative, dtype=dtype, requires_grad=True)
    normals = torch.linspace(-0.8, 0.9, cumulative.numel(), dtype=dtype).reshape(cumulative.shape).requires_grad_()
    y, times = torch.tensor(history, dtype=dtype), torch.tensor(days, dtype=dtype)
    output = fading.sample_fading_reports(cumulative, y, times, a, r, cutoff=180.0, half_life_days=180.0,
                                         standard_normals=normals)
    likelihood = fading.fading_endpoint_log_likelihood(y, cumulative.permute(1, 0, 2, 3), a, r,
                                                      torch.isfinite(y), times, half_life_days=180.0)
    gradients = torch.autograd.grad(output.square().mean() + likelihood.mean(), (cumulative, a, r, normals))
    assert all(torch.isfinite(value).all() for value in gradients)
    known = torch.isfinite(y)[:, None].expand_as(normals)
    torch.testing.assert_close(gradients[-1][known], torch.zeros_like(gradients[-1][known]), rtol=0, atol=0)


def test_sampler_and_likelihood_pass_parameterized_float64_gradcheck(fading, report_case):
    q, history, days, absolute, relative = report_case
    base = torch.tensor(q[:, None])
    y, times = torch.tensor(history), torch.tensor(days)
    normals = torch.linspace(-0.8, 0.9, base.numel(), dtype=torch.float64).reshape_as(base)
    z = torch.zeros((len(q), 1, 1, 3), dtype=torch.float64, requires_grad=True)
    log_a = torch.tensor(np.log(absolute), requires_grad=True)
    log_r = torch.tensor(np.log(relative), requires_grad=True)

    def outputs(log_volume, log_absolute, log_relative):
        # Finite differences remain inside the monotone, nonnegative support,
        # and leave deterministic day zero exactly zero.
        cumulative = base * log_volume.exp()
        a, r = log_absolute.exp(), log_relative.exp()
        sampled = fading.sample_fading_reports(
            cumulative, y, times, a, r, cutoff=180.0, half_life_days=180.0,
            standard_normals=normals,
        )
        likelihood = fading.fading_endpoint_log_likelihood(
            y, cumulative.permute(1, 0, 2, 3), a, r, torch.isfinite(y), times,
            half_life_days=180.0,
        )
        return sampled, likelihood

    assert torch.autograd.gradcheck(outputs, (z, log_a, log_r), eps=1e-6,
                                    atol=1e-5, rtol=1e-4, fast_mode=True)


def test_large_finite_half_life_recovers_accumulated_sampler_and_likelihood(fading, report_case):
    from joint_production.observation import gaussian_endpoint_log_likelihood
    from joint_production.torch_observation import sample_accumulated_reports

    q, history, days, absolute, relative = report_case
    cumulative, y, times, a, r = [torch.tensor(value) for value in
                                 (q[:, None], history, days, absolute, relative)]
    normals = torch.linspace(-0.8, 0.9, cumulative.numel(), dtype=torch.float64).reshape_as(cumulative)
    kwargs = dict(cutoff=180.0, standard_normals=normals)
    expected_sample = sample_accumulated_reports(cumulative, y, times, a, r, **kwargs)
    actual_sample = fading.sample_fading_reports(cumulative, y, times, a, r,
                                                 half_life_days=1e20, **kwargs)
    torch.testing.assert_close(actual_sample, expected_sample, rtol=1e-12, atol=1e-12)
    likelihood_args = (y, cumulative.permute(1, 0, 2, 3), a, r, torch.isfinite(y), times)
    expected_likelihood = gaussian_endpoint_log_likelihood(*likelihood_args, law="accumulated")
    actual_likelihood = fading.fading_endpoint_log_likelihood(*likelihood_args, half_life_days=1e20)
    torch.testing.assert_close(actual_likelihood, expected_likelihood, rtol=1e-12, atol=1e-12)


def test_endpoint_likelihood_matches_dense_masked_gaussian_without_counting_day_zero(fading, report_case):
    q, history, days, absolute, relative = report_case
    cumulative = np.stack([q, q * np.array([1.2, 0.9, 1.1])], axis=0)
    mask = np.isfinite(history)
    expected = _log_likelihood_oracle(history, cumulative, days, absolute, relative, mask, 180.0)
    actual = fading.fading_endpoint_log_likelihood(
        torch.tensor(history), torch.tensor(cumulative), torch.tensor(absolute), torch.tensor(relative),
        torch.tensor(mask), torch.tensor(days), half_life_days=180.0,
    )
    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-11, atol=2e-11)
    mask[:, 0] = False
    no_zero = fading.fading_endpoint_log_likelihood(
        torch.tensor(history), torch.tensor(cumulative), torch.tensor(absolute), torch.tensor(relative),
        torch.tensor(mask), torch.tensor(days), half_life_days=180.0,
    )
    torch.testing.assert_close(no_zero, actual, rtol=0, atol=0)


def test_endpoint_likelihood_ignores_unobserved_values_and_inserted_unobserved_times(fading, report_case):
    q, history, days, absolute, relative = report_case
    history[:, 1] = np.nan
    mask = np.isfinite(history)
    poisoned = np.where(mask, history, np.inf)
    common = (torch.tensor(absolute), torch.tensor(relative))
    full = fading.fading_endpoint_log_likelihood(torch.tensor(poisoned), torch.tensor(q), *common,
                                                 torch.tensor(mask), torch.tensor(days), half_life_days=180.0)
    keep = np.arange(len(days)) != 1
    coarse = fading.fading_endpoint_log_likelihood(torch.tensor(poisoned[:, keep]), torch.tensor(q[:, keep]), *common,
                                                   torch.tensor(mask[:, keep]), torch.tensor(days[keep]), half_life_days=180.0)
    torch.testing.assert_close(full, coarse, rtol=2e-12, atol=2e-12)


def test_query_refinement_uses_correctly_coupled_innovations(fading):
    days = np.array([0.0, 60.0, 180.0])
    q = days[:, None] * np.array([0.05, 0.2, 0.1])
    absolute, relative = np.array([0.7, 1.1, 0.5]), np.array([0.08, 0.12, 0.1])
    covariance = _covariance(q, days, absolute, relative, 180.0)
    a = np.diag(covariance).reshape(3, 3)[1]
    c = np.diag(covariance).reshape(3, 3)[2]
    rho = math.exp(-math.log(2) * 120.0 / 180.0)
    b = c - rho**2 * a
    alpha, beta = rho * np.sqrt(a / c), np.sqrt(b / c)
    u, v = np.array([0.3, -0.7, 1.2]), np.array([-0.2, 0.8, 0.6])
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    coarse_history = np.array([[[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]])
    coarse_normals = np.array([[[[0.0, 0.0, 0.0], u]]])
    coarse = fading.sample_fading_reports(tensor(q[[0, 2]][None, None]), tensor(coarse_history), tensor(days[[0, 2]]),
        tensor(absolute), tensor(relative), cutoff=0.0, half_life_days=180.0, standard_normals=tensor(coarse_normals))
    fine_history = np.full((1, 3, 3), np.nan)
    fine_history[:, 0] = 0.0
    fine_normals = np.array([[[[0.0, 0.0, 0.0], alpha * u + beta * v, beta * u - alpha * v]]])
    fine = fading.sample_fading_reports(tensor(q[None, None]), tensor(fine_history), tensor(days), tensor(absolute),
        tensor(relative), cutoff=0.0, half_life_days=180.0, standard_normals=tensor(fine_normals))
    torch.testing.assert_close(fine[:, :, -1], coarse[:, :, -1], rtol=2e-12, atol=2e-12)
    fine_history[:, -1] = coarse[:, 0, -1].numpy()
    bridge_normals = np.array([[[[0.0, 0.0, 0.0], v, [0.0, 0.0, 0.0]]]])
    bridged = fading.sample_fading_reports(tensor(q[None, None]), tensor(fine_history), tensor(days), tensor(absolute),
        tensor(relative), cutoff=180.0, half_life_days=180.0, standard_normals=tensor(bridge_normals))
    torch.testing.assert_close(bridged, fine, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("half_life", [0.0, -1.0, np.inf, np.nan, True, False])
def test_half_life_must_be_positive_and_finite(fading, report_case, half_life):
    q, history, days, absolute, relative = report_case
    with pytest.raises(ValueError):
        fading.sample_fading_reports(torch.tensor(q[:, None]), torch.tensor(history), torch.tensor(days),
            torch.tensor(absolute), torch.tensor(relative), cutoff=180.0, half_life_days=half_life,
            standard_normals=torch.zeros((2, 1, len(days), 3), dtype=torch.float64))


def test_small_fading_transition_remains_positive_without_variance_subtraction(fading):
    left, right, half_life = 90.0, 90.0 + 1e-6, 1e15
    absolute, relative = 0.02, 0.1
    q = torch.full((3,), 1e8, dtype=torch.float64)
    a, r = torch.full_like(q, absolute), torch.full_like(q, relative)
    rho, variance = fading.fading_transition(q, q, torch.tensor(left, dtype=torch.float64),
                                            torch.tensor(right, dtype=torch.float64), a, r,
                                            half_life_days=half_life)
    dt = right - left
    variance_left = absolute**2 * left / 360.0 + (relative * 1e8)**2
    expected = absolute**2 * dt / 360.0 - math.expm1(-2 * math.log(2) * dt / half_life) * variance_left
    assert (variance > 0).all()
    np.testing.assert_allclose(variance.numpy(), expected, rtol=1e-12, atol=0)


def test_supplied_normals_do_not_consume_global_rng(fading, report_case):
    q, history, days, absolute, relative = report_case
    before = torch.random.get_rng_state().clone()
    fading.sample_fading_reports(torch.tensor(q[:, None]), torch.tensor(history), torch.tensor(days),
        torch.tensor(absolute), torch.tensor(relative), cutoff=180.0, half_life_days=180.0,
        standard_normals=torch.zeros((2, 1, len(days), 3), dtype=torch.float64))
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["negative_cumulative", "decreasing_cumulative", "nonzero_day_zero", "infinite_available_history"])
def test_invalid_latent_support_or_known_history_is_rejected(fading, report_case, fault):
    q, history, days, absolute, relative = report_case
    if fault == "negative_cumulative":
        q[0, 2, 0] = -1.0
    elif fault == "decreasing_cumulative":
        q[0, 2, 0] = q[0, 1, 0] / 2.0
    elif fault == "nonzero_day_zero":
        history[0, 0, 0] = 1.0
    else:
        history[0, 1, 0] = np.inf
    with pytest.raises(ValueError):
        fading.sample_fading_reports(torch.tensor(q[:, None]), torch.tensor(history), torch.tensor(days),
            torch.tensor(absolute), torch.tensor(relative), cutoff=180.0, half_life_days=180.0,
            standard_normals=torch.zeros((2, 1, len(days), 3), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional CUDA synthetic parity check")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_supplied_normal_replay_matches_cpu_with_finite_gradients(fading, report_case, dtype):
    q, history, days, absolute, relative = report_case
    cpu = [torch.tensor(value, dtype=dtype) for value in (q[:, None], history, days, absolute, relative)]
    normals = torch.linspace(-0.5, 0.5, q.size, dtype=dtype).reshape(2, 1, len(days), 3)
    expected = fading.sample_fading_reports(*cpu, cutoff=180.0, half_life_days=180.0, standard_normals=normals)
    gpu = [value.cuda() for value in cpu]
    for index in (0, 3, 4):
        gpu[index].requires_grad_()
    actual = fading.sample_fading_reports(*gpu, cutoff=180.0,
                                         half_life_days=180.0, standard_normals=normals.cuda())
    tolerance = 3e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual.detach().cpu(), expected, rtol=tolerance, atol=tolerance)
    cumulative, y, times, a, r = gpu
    likelihood = fading.fading_endpoint_log_likelihood(
        y, cumulative.permute(1, 0, 2, 3), a, r, torch.isfinite(y), times, half_life_days=180.0,
    )
    gradients = torch.autograd.grad(actual.square().mean() + likelihood.mean(), (cumulative, a, r))
    assert all(torch.isfinite(value).all() for value in gradients)
