"""Independent finite-bank score, masking, derivative, and storage checks."""
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pytest
import torch

from joint_production.metrics import weighted_interval_score
from joint_production.sampled_quantile_score import empirical_wis, sampled_u_crps


SCORES = (empirical_wis, sampled_u_crps)


def _inputs(shape=(2, 2, 3), draws=7, dtype=torch.float64, device="cpu"):
    target_index = torch.arange(np.prod(shape), dtype=dtype, device=device).reshape(shape)
    sample_shape = (shape[0], draws, *shape[1:])
    sample_index = torch.arange(np.prod(sample_shape), dtype=dtype, device=device).reshape(sample_shape)
    targets = 2 * torch.cos(target_index * .43) - 1.2
    samples = torch.sin(sample_index * .37) + sample_index * .11 - 2.3
    return targets, samples


def _numpy_wis(targets, samples, mask=None):
    targets, samples = np.asarray(targets, dtype=float), np.asarray(samples, dtype=float)
    valid = np.isfinite(targets) if mask is None else np.isfinite(targets) & mask
    target = np.where(valid, targets, np.nan).reshape(len(targets), -1, 1)
    forecast = samples.reshape(len(targets), samples.shape[1], -1, 1)
    return np.where(valid, weighted_interval_score(target, forecast).reshape(targets.shape), 0.)


def _numpy_u(targets, samples, mask=None):
    targets, samples = np.asarray(targets, dtype=float), np.asarray(samples, dtype=float)
    valid = np.isfinite(targets) if mask is None else np.isfinite(targets) & mask
    target = np.where(valid, targets, 0.)
    forecasts = np.where(valid[:, None], samples, 0.)
    count = samples.shape[1]
    pairwise = np.abs(forecasts[:, :, None] - forecasts[:, None, :])
    score = np.abs(forecasts - target[:, None]).mean(axis=1) - pairwise.sum(axis=(1, 2)) / (2 * count * (count - 1))
    return np.where(valid, score, 0.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("draws", [1, 2, 7, 16])
def test_empirical_wis_matches_existing_numpy_metric(dtype, draws):
    targets, samples = _inputs(draws=draws, dtype=dtype)
    actual, valid = empirical_wis(targets, samples)
    expected = _numpy_wis(targets.numpy(), samples.numpy())
    tolerance = 3e-6 if dtype == torch.float32 else 3e-14
    np.testing.assert_allclose(actual.numpy(), expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == dtype and actual.device == targets.device and actual.shape == targets.shape
    assert valid.dtype == torch.bool and valid.all()


def test_wis_known_80_95_interval_values_and_single_draw_absolute_error():
    values = np.interp(np.arange(41), [0, 1, 4, 20, 36, 39, 40], [-2, -1, 0, 1, 2, 3, 4])
    samples = torch.tensor(np.broadcast_to(values, (2, 41)), dtype=torch.float64)
    actual, _ = empirical_wis(torch.tensor([1., 4.], dtype=torch.float64), samples)
    torch.testing.assert_close(actual, torch.tensor([.12, 1.92], dtype=torch.float64), rtol=0, atol=5e-16)
    targets = torch.tensor([-3.5, 0., 2.75], dtype=torch.float64)
    samples = torch.tensor([[1.25], [-2.], [2.75]], dtype=torch.float64)
    actual, _ = empirical_wis(targets, samples)
    torch.testing.assert_close(actual, (samples[:, 0] - targets).abs(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("draws", [2, 3, 7, 16])
def test_sorted_u_crps_matches_explicit_off_diagonal_pairs(dtype, draws):
    targets, samples = _inputs(draws=draws, dtype=dtype)
    actual, valid = sampled_u_crps(targets, samples)
    expected = _numpy_u(targets.numpy(), samples.numpy())
    tolerance = 3e-6 if dtype == torch.float32 else 3e-14
    np.testing.assert_allclose(actual.numpy(), expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == dtype and actual.device == targets.device and actual.shape == targets.shape
    assert valid.dtype == torch.bool and valid.all()


def test_u_normalization_excludes_self_pairs_and_has_exact_iid_expectation():
    actual, _ = sampled_u_crps(torch.tensor([1.], dtype=torch.float64),
                               torch.tensor([[0., 2., 4.]], dtype=torch.float64))
    torch.testing.assert_close(actual, torch.tensor([1 / 3], dtype=torch.float64), rtol=0, atol=1e-15)
    # Enumerate all iid draws from the equal-mass law on {0,2}; no simulation.
    targets = torch.full((4,), .25, dtype=torch.float64)
    samples = torch.tensor([[0., 0.], [0., 2.], [2., 0.], [2., 2.]], dtype=torch.float64)
    actual, _ = sampled_u_crps(targets, samples)
    torch.testing.assert_close(actual, torch.tensor([.25, 0., 0., 1.75], dtype=torch.float64), rtol=0, atol=0)
    assert actual.mean().item() == .5  # E|X-y|=1 and E|X-X'|=1.


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("shape", [(2,), (2, 5), (2, 2, 1, 3)])
def test_generic_trailing_axes_and_noncontiguous_inputs(function, shape):
    targets, samples = _inputs(shape=shape)
    # Slicing larger backing arrays exercises noncontiguous storage, including M.
    target_backing = torch.empty((*targets.shape, 2), dtype=targets.dtype)
    sample_backing = torch.empty((*samples.shape, 2), dtype=samples.dtype)
    target_backing[..., 0], sample_backing[..., 0] = targets, samples
    targets, samples = target_backing[..., 0], sample_backing[..., 0]
    assert not targets.is_contiguous() and not samples.is_contiguous()
    result, valid = function(targets, samples)
    oracle = _numpy_wis if function is empirical_wis else _numpy_u
    np.testing.assert_allclose(result.numpy(), oracle(targets.numpy(), samples.numpy()), rtol=3e-14, atol=3e-14)
    assert result.shape == valid.shape == shape


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_values_translation_phase_units_and_sample_permutation(function, dtype):
    targets, samples = _inputs(dtype=dtype)
    units = torch.tensor([8., .125, 16.], dtype=dtype)
    shift = torch.tensor([-32., 128., -8.], dtype=dtype)
    expected, valid = function(targets, samples)
    converted, converted_valid = function(targets * units + shift, samples * units + shift)
    reflected, _ = function(-targets, -samples)
    permutation = torch.tensor([5, 2, 6, 0, 3, 1, 4])
    permuted, _ = function(targets, samples[:, permutation])
    tolerance = 3e-5 if dtype == torch.float32 else 5e-13
    torch.testing.assert_close(converted, expected * units, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(reflected, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(permuted, expected, rtol=0, atol=tolerance)
    torch.testing.assert_close(converted_valid, valid)
    assert (targets < 0).any() and (samples < 0).any()


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_poison_is_sanitized_before_arithmetic_and_has_zero_gradients(function, dtype):
    targets, samples = _inputs(dtype=dtype)
    mask = torch.ones_like(targets, dtype=torch.bool)
    targets[0, 0, 0], targets[0, 1, 1], targets[1, 0, 2] = float("nan"), float("inf"), -float("inf")
    mask[1, 1, 0] = False
    targets[1, 1, 0] = torch.finfo(dtype).max
    valid = torch.isfinite(targets) & mask
    poison = torch.tensor([float("nan"), float("inf"), -float("inf"),
                           -torch.finfo(dtype).max, 0., 1., -1.], dtype=dtype)
    for row, cell, phase in torch.nonzero(~valid).tolist():
        samples[row, :, cell, phase] = poison
    targets.requires_grad_()
    samples.requires_grad_()
    original_targets, original_samples = targets.detach().clone(), samples.detach().clone()
    result, actual_valid = function(targets, samples, target_mask=mask)
    oracle = _numpy_wis if function is empirical_wis else _numpy_u
    expected = oracle(targets.detach().numpy(), samples.detach().numpy(), mask.numpy())
    tolerance = 3e-6 if dtype == torch.float32 else 3e-14
    np.testing.assert_allclose(result.detach().numpy(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_valid, valid)
    assert torch.isfinite(result).all() and torch.all(result[~valid] == 0)
    target_gradient, sample_gradient = torch.autograd.grad(result.sum(), (targets, samples))
    assert torch.isfinite(target_gradient).all() and torch.isfinite(sample_gradient).all()
    assert torch.all(target_gradient[~valid] == 0)
    assert torch.all(sample_gradient[~valid[:, None].expand_as(samples)] == 0)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(samples, original_samples, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("function", SCORES)
def test_entirely_inactive_batch_has_zero_scores_and_gradients(function):
    targets = torch.full((2, 3), torch.finfo(torch.float64).max, requires_grad=True, dtype=torch.float64)
    samples = torch.full((2, 5, 3), -torch.finfo(torch.float64).max, requires_grad=True, dtype=torch.float64)
    result, valid = function(targets, samples, target_mask=torch.zeros_like(targets, dtype=torch.bool))
    assert not valid.any() and torch.all(result == 0)
    gradients = torch.autograd.grad(result.sum(), (targets, samples))
    assert all(torch.isfinite(gradient).all() and torch.all(gradient == 0) for gradient in gradients)


def test_u_sample_and_target_gradients_match_independent_pairwise_expression():
    targets, samples = _inputs(shape=(2, 2), draws=5)
    targets.requires_grad_()
    samples.requires_grad_()
    weights = torch.tensor([[.7, 1.3], [2.1, .4]], dtype=torch.float64)
    score, _ = sampled_u_crps(targets, samples)
    actual = torch.autograd.grad((score * weights).sum(), (targets, samples))
    accuracy = (samples - targets[:, None]).abs().mean(dim=1)
    dispersion = sum((samples[:, i] - samples[:, j]).abs() for i in range(5) for j in range(i + 1, 5)) / 20
    expected = torch.autograd.grad(((accuracy - dispersion) * weights).sum(), (targets, samples))
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("function,oracle", [(empirical_wis, _numpy_wis), (sampled_u_crps, _numpy_u)])
def test_float64_gradients_match_independent_finite_differences_away_from_ties(function, oracle):
    targets, samples = _inputs(shape=(2, 2), draws=5)
    targets.requires_grad_()
    samples.requires_grad_()
    weights = torch.tensor([[.7, 1.3], [2.1, .4]], dtype=torch.float64)
    result, _ = function(targets, samples)
    actual = torch.autograd.grad((result * weights).sum(), (targets, samples))
    arrays = [targets.detach().numpy().copy(), samples.detach().numpy().copy()]
    for argument, gradient in enumerate(actual):
        expected = np.empty_like(arrays[argument])
        for index in np.ndindex(expected.shape):
            upper, lower = [value.copy() for value in arrays], [value.copy() for value in arrays]
            upper[argument][index] += 1e-6
            lower[argument][index] -= 1e-6
            expected[index] = np.sum((oracle(*upper) - oracle(*lower)) * weights.numpy()) / 2e-6
        np.testing.assert_allclose(gradient.numpy(), expected, rtol=4e-7, atol=3e-9)


@pytest.mark.parametrize("function", SCORES)
def test_order_ties_and_target_equality_have_finite_piecewise_gradients(function):
    targets = torch.tensor([0., 1., -1.], dtype=torch.float64, requires_grad=True)
    samples = torch.tensor([[0., 0., 0., 0.], [0., 0., 1., 1.], [-1., -1., 0., 2.]],
                           dtype=torch.float64, requires_grad=True)
    result, valid = function(targets, samples)
    oracle = _numpy_wis if function is empirical_wis else _numpy_u
    np.testing.assert_allclose(result.detach().numpy(), oracle(targets.detach().numpy(), samples.detach().numpy()),
                               rtol=1e-14, atol=1e-14)
    gradients = torch.autograd.grad(result.sum(), (targets, samples))
    assert valid.all() and all(torch.isfinite(gradient).all() for gradient in gradients)
    # At nondifferentiable ties, rank allocations need not equal pairwise abs(0).
    reordered, _ = function(targets, samples[:, [3, 1, 0, 2]])
    torch.testing.assert_close(reordered, result, rtol=0, atol=1e-14)


def test_sorted_gaps_avoid_large_common_level_cancellation():
    targets = torch.tensor([3., -5.], dtype=torch.float64)
    samples = torch.tensor([[-16., -2., 0., 7., 19.], [3., -21., -4., 0., 11.]], dtype=torch.float64)
    expected, _ = sampled_u_crps(targets, samples)
    shifted, _ = sampled_u_crps(targets + 2**45, samples + 2**45)
    torch.testing.assert_close(shifted, expected, rtol=0, atol=0)
    np.testing.assert_allclose(shifted.numpy(), _numpy_u(targets.numpy(), samples.numpy()), rtol=0, atol=4e-15)


def test_u_working_tensors_remain_linear_in_sample_count():
    from torch.utils._python_dispatch import TorchDispatchMode

    targets, samples = _inputs(shape=(2, 3), draws=257)
    maximum_elements = 4 * samples.numel()
    class RejectPairTensor(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            values = result if isinstance(result, (tuple, list)) else (result,)
            for value in values:
                if isinstance(value, torch.Tensor):
                    assert value.numel() <= maximum_elements, "U-CRPS materialized quadratic sample-pair storage"
            return result
    with RejectPairTensor():
        actual, valid = sampled_u_crps(targets, samples)
    np.testing.assert_allclose(actual.numpy(), _numpy_u(targets.numpy(), samples.numpy()), rtol=2e-14, atol=2e-13)
    assert valid.all()


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_active_nonfinite_sample_is_rejected_without_changing_validity(function, bad):
    targets, samples = _inputs()
    samples[0, 2, 1, 2] = bad
    with pytest.raises(ValueError, match="finite predictive samples"):
        function(targets, samples)


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("kind", ["shape", "numeric", "numpy", "device"])
def test_mask_requires_exact_shape_boolean_tensor_and_matching_device(function, kind):
    targets, samples = _inputs()
    masks = {"shape": torch.ones((2, 1, 3), dtype=torch.bool), "numeric": torch.ones_like(targets),
             "numpy": np.ones(targets.shape, dtype=bool),
             "device": torch.ones(targets.shape, dtype=torch.bool, device="meta")}
    with pytest.raises(ValueError, match="target_mask"):
        function(targets, samples, target_mask=masks[kind])


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("kind", ["target_numpy", "sample_numpy", "dtype_mismatch", "integer", "half", "device"])
def test_input_tensor_dtype_and_device_contract(function, kind):
    targets, samples = _inputs()
    if kind == "target_numpy":
        targets = targets.numpy()
    elif kind == "sample_numpy":
        samples = samples.numpy()
    elif kind == "dtype_mismatch":
        samples = samples.float()
    elif kind == "integer":
        targets, samples = targets.long(), samples.long()
    elif kind == "half":
        targets, samples = targets.half(), samples.half()
    else:
        samples = samples.to("meta")
    with pytest.raises(ValueError, match="dtype and device"):
        function(targets, samples)


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("kind", ["scalar", "well_mismatch", "missing_axis", "empty_wells", "empty_cells", "empty_samples"])
def test_shape_axes_and_nonempty_sample_contract(function, kind):
    targets, samples = _inputs()
    if kind == "scalar":
        targets, samples = torch.tensor(1.), torch.ones(5)
    elif kind == "well_mismatch":
        samples = samples[:1]
    elif kind == "missing_axis":
        samples = samples[:, 0]
    elif kind == "empty_wells":
        targets, samples = targets[:0], samples[:0]
    elif kind == "empty_cells":
        targets, samples = targets[:, :0], samples[:, :, :0]
    else:
        samples = samples[:, :0]
    with pytest.raises(ValueError, match="targets.*samples"):
        function(targets, samples)


def test_u_crps_rejects_single_draw_even_when_target_is_missing():
    with pytest.raises(ValueError, match="M>=2"):
        sampled_u_crps(torch.tensor([float("nan")]), torch.tensor([[float("nan")]]))


@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_active_arithmetic_overflow_fails_instead_of_returning_invalid_score(function, dtype):
    maximum = torch.finfo(dtype).max
    targets = torch.tensor([maximum], dtype=dtype)
    samples = torch.full((1, 3), -maximum, dtype=dtype)
    with pytest.raises(FloatingPointError, match="floating-point support"):
        function(targets, samples)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only score and gradient parity")
@pytest.mark.parametrize("function", SCORES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_value_mask_and_gradient_parity_on_fixed_inputs(function, dtype):
    cpu_targets, cpu_samples = _inputs(dtype=dtype)
    mask = torch.ones_like(cpu_targets, dtype=torch.bool)
    mask[0, 1, 2] = False
    cpu_samples[0, :, 1, 2] = float("nan")
    arguments = [(cpu_targets, cpu_samples, mask),
                 (cpu_targets.cuda(), cpu_samples.cuda(), mask.cuda())]
    results = []
    for targets, samples, active in arguments:
        targets.requires_grad_()
        samples.requires_grad_()
        score, valid = function(targets, samples, target_mask=active)
        gradients = torch.autograd.grad(score.sum(), (targets, samples))
        results.append((score.detach().cpu(), valid.cpu(), *(value.cpu() for value in gradients)))
    tolerance = 3e-6 if dtype == torch.float32 else 3e-13
    for cpu, cuda in zip(*results, strict=True):
        torch.testing.assert_close(cpu, cuda, rtol=tolerance, atol=tolerance)
