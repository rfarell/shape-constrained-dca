"""Independent moment, Gaussian-bridge, masking and numerical score oracles."""
import itertools
import math
import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammainc
import torch

from joint_production.analytic_mean_score import (
    fast_basis_increment_mean, fast_basis_increment_mean_mse,
)


DTYPE = torch.float64
SHAPES = (1, 1, 2, 2)
TAU = (30., 180., 360., 720.)


def inputs(dtype=DTYPE, device="cpu"):
    mean = torch.linspace(-1.2, .6, 24, dtype=dtype, device=device).reshape(2, 12)
    lower = torch.diag_embed(torch.linspace(.12, .35, 24, dtype=dtype, device=device).reshape(2, 12))
    for row in range(1, 12):
        lower[:, row, 0] = .23 - row / 80
        if row > 1: lower[:, row, row - 1] = -.17
    days = torch.tensor([0., 17., 91., 199., 811.], dtype=dtype, device=device)
    target = torch.arange(30, dtype=dtype, device=device).reshape(2, 5, 3) / 11 - .7
    target[:, 0] = 0
    target[0, 3, 1] = -1.25
    scale = torch.tensor([[1.2, 3., .7], [2., 5., 1.5], [2.5, 9., 2.]], dtype=dtype, device=device)
    weight = torch.tensor([[1., 1.5, 2.], [3., .75, 1.25], [1.1, 2.2, .9]], dtype=dtype, device=device)
    return target, mean, lower, days, 1, scale, weight


def rate_integrals(days, origin):
    """Integrate native rate densities, never subtract near-equal CDFs."""
    s = float(days[origin])
    result = []
    for endpoint in days[origin + 1:]:
        width = float(endpoint) - s
        row = []
        for shape, tau in zip(SHAPES, TAU, strict=True):
            value, _ = quad(lambda fraction: ((s + width * fraction) / tau)**(shape - 1)
                * math.exp(-(s + width * fraction) / tau) / tau, 0., 1., epsabs=0., epsrel=5e-13)
            row.append(width * value)
        result.append(row)
    return np.asarray(result)


def quadrature_coefficients(mean, lower):
    """Tensor-product normal quadrature over each row's active innovations."""
    nodes, weights = np.polynomial.hermite.hermgauss(18)
    nodes, weights = math.sqrt(2.) * nodes, weights / math.sqrt(math.pi)
    result = np.empty_like(mean)
    for well in range(len(mean)):
        for row in range(12):
            coefficients = lower[well, row][lower[well, row] != 0]
            assert len(coefficients) <= 3  # Bounded three-dimensional oracle.
            value = 0.
            for indices in itertools.product(range(len(nodes)), repeat=len(coefficients)):
                value += float(np.prod(weights[list(indices)])) * math.exp(
                    float(mean[well, row] + coefficients @ nodes[list(indices)]))
            result[well, row] = value
    return result.reshape(len(mean), 3, 4)


def mean_oracle(mean, lower, days, origin):
    coefficients = quadrature_coefficients(mean, lower)
    return np.einsum("npk,fk->nfp", coefficients, rate_integrals(days, origin))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_matches_independent_lognormal_and_rate_quadrature_with_full_row_covariance(dtype):
    _, mean, lower, days, origin, _, _ = inputs(dtype)
    result = fast_basis_increment_mean(mean, lower, days, origin)
    expected = mean_oracle(mean.double().numpy(), lower.double().numpy(), days.double().numpy(), origin)
    assert result.shape == (2, 3, 3) and result.dtype == DTYPE and result.device == mean.device
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-13, atol=2e-15)
    diagonal_only = fast_basis_increment_mean(mean, torch.diag_embed(lower.diagonal(dim1=-2, dim2=-1)), days, origin)
    assert torch.all(result > diagonal_only)
    # Signs of Cholesky off-diagonals alter dependence but not row variances.
    signs = lower.clone()
    indices = torch.tril_indices(12, 12, offset=-1)
    signs[:, indices[0], indices[1]] *= -1
    torch.testing.assert_close(fast_basis_increment_mean(mean, signs, days, origin), result, rtol=0, atol=0)


def test_mixed_float_inputs_preserve_supplied_values_during_float64_promotion():
    target, mean, lower, days, origin, scale, weight = inputs(torch.float32)
    days, target, scale, weight = days.double(), target.double(), scale.double(), weight.double()
    days[2] += 2e-8; target[0, 3, 1] += 3e-9; scale[1, 2] += 7e-10; weight[2, 0] += 9e-10
    arguments = target, mean, lower, days, origin, scale, weight
    promoted = target, mean.double(), lower.double(), days, origin, scale, weight
    torch.testing.assert_close(fast_basis_increment_mean_mse(*arguments), fast_basis_increment_mean_mse(*promoted), rtol=0, atol=0)
    torch.testing.assert_close(fast_basis_increment_mean(mean, lower, days, origin),
        fast_basis_increment_mean(mean.double(), lower.double(), days, origin), rtol=0, atol=0)


def test_float32_coefficient_moments_promote_before_exponential_overflow():
    mean = torch.full((1, 12), 100., dtype=torch.float32, requires_grad=True)
    lower = (torch.eye(12, dtype=torch.float32)[None] * .25).requires_grad_()
    days = torch.tensor([0., 30., 90.], dtype=torch.float32)
    result = fast_basis_increment_mean(mean, lower, days, 0)
    expected = math.exp(100. + .5 * .25**2) * rate_integrals(days.double().numpy(), 0).sum(axis=1)
    np.testing.assert_allclose(result.detach().numpy(), np.broadcast_to(expected[None, :, None], (1, 2, 3)), rtol=3e-14)
    # Float64 output permits this moment, although float32 exp(100) cannot.
    assert result.dtype == DTYPE and torch.isfinite(result).all()
    gm, gl = torch.autograd.grad((result / 1e44).sum(), (mean, lower))
    assert torch.isfinite(gm).all() and torch.isfinite(gl).all()


def test_signed_missing_targets_use_raw_pair_mask_frozen_weights_and_full_cell_denominator():
    target, mean, lower, days, origin, scale, weight = inputs()
    target[0, origin, 1] = torch.nan
    target[0, 4, 2] = torch.inf
    target[1, 3, 0] = -torch.inf
    weight[1, 2] = 0
    prediction = mean_oracle(mean.numpy(), lower.numpy(), days.numpy(), origin)
    expected = np.zeros(2)
    for n in range(2):
        for f, t in enumerate(range(origin + 1, len(days))):
            for p in range(3):
                if np.isfinite(target[n, [origin, t], p].numpy()).all():
                    error = (prediction[n, f, p] - float(target[n, t, p] - target[n, origin, p])) / float(scale[f, p])
                    expected[n] += float(weight[f, p]) * error**2 / 9
    actual = fast_basis_increment_mean_mse(target, mean, lower, days, origin, scale, weight)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=5e-13, atol=3e-15)
    # Earlier reports are irrelevant to this conditional-mean scoring helper.
    changed = target.clone(); changed[:, :origin] = torch.inf
    torch.testing.assert_close(fast_basis_increment_mean_mse(changed, mean, lower, days, origin, scale, weight), actual, rtol=0, atol=0)
    twice = fast_basis_increment_mean_mse(target, mean, lower, days, origin, scale, 2 * weight)
    torch.testing.assert_close(twice, 2 * actual, rtol=0, atol=0)


def test_all_missing_pairs_are_sanitized_before_subtraction_and_have_zero_parameter_gradients():
    target, mean, lower, days, origin, scale, weight = inputs()
    mean.requires_grad_(); lower.requires_grad_()
    target[:, origin] = torch.tensor([torch.nan, torch.inf, -torch.inf])
    target[:, origin + 1:] = 1e308
    value = fast_basis_increment_mean_mse(target, mean, lower, days, origin, scale, weight)
    gm, gl = torch.autograd.grad(value.sum(), (mean, lower))
    torch.testing.assert_close(value, torch.zeros(2, dtype=DTYPE), rtol=0, atol=0)
    torch.testing.assert_close(gm, torch.zeros_like(gm), rtol=0, atol=0)
    torch.testing.assert_close(gl, torch.zeros_like(gl), rtol=0, atol=0)


def test_loss_is_squared_predictive_mean_error_without_predictive_variance_penalty():
    _, mean, lower, days, origin, scale, weight = inputs()
    lower = torch.diag_embed(torch.full_like(mean, .8)).requires_grad_()
    mean = mean.requires_grad_()
    prediction = fast_basis_increment_mean(mean, lower, days, origin)
    target = torch.zeros((2, len(days), 3), dtype=DTYPE)
    target[:, origin + 1:] = prediction.detach()
    value = fast_basis_increment_mean_mse(target, mean, lower, days, origin, scale, weight)
    torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
    for derivative in torch.autograd.grad(value.sum(), (mean, lower)):
        torch.testing.assert_close(derivative, torch.zeros_like(derivative), rtol=0, atol=0)
    # The expected squared error of a random trajectory is strictly positive
    # for these same means; it would impose an unwanted variance penalty.
    coefficients_variance = np.exp(2 * mean.detach().numpy() + .8**2) * np.expm1(.8**2)
    predictive_variance = np.einsum("npk,fk->nfp", coefficients_variance.reshape(2, 3, 4), rate_integrals(days.numpy(), origin)**2)
    random_draw_squared_error = (predictive_variance / scale.numpy()**2 * weight.numpy()).mean(axis=(1, 2))
    assert np.all(random_draw_squared_error > .001)


def test_hidden_origin_dense_gaussian_bridge_means_cancel_shared_report_error():
    days = np.array([0., 13., 41., 97., 401.]); origin = 3
    mean = np.linspace(-.7, .3, 12)[None]
    lower = np.eye(12)[None] * 1e-10
    lower[0, 0, 0] = .55
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    coefficients = np.exp(mean + math.sqrt(2.) * nodes[:, None] * lower[0, :, 0])
    basis = np.column_stack([gammainc(k, days / tau) for k, tau in zip(SHAPES, TAU, strict=True)])
    q = np.einsum("spk,tk->stp", coefficients.reshape(-1, 3, 4), basis)
    histories = [np.array([[np.nan, np.nan, np.nan], [-.5, np.nan, .2], [np.nan, -1., np.nan], [np.nan, 2., np.nan], [np.inf] * 3]),
        np.array([[0., 0., 0.], [3., np.nan, -2.], [np.nan, 4., np.nan], [np.nan, -.25, np.nan], [-np.inf] * 3])]
    expected = fast_basis_increment_mean(torch.tensor(mean), torch.tensor(lower), torch.tensor(days), origin).numpy()[0, 0]
    for a, r in ((np.array([.7, .4, 1.1]), np.array([.1, .3, .2])),
                 (np.array([2.1, .8, .55]), np.array([.3, .6, .1]))):
        for history in histories:
            conditional = q.copy()
            for sample in range(len(q)):
                for p in range(3):
                    known = np.flatnonzero(np.isfinite(history[:, p]) & (days > 0) & (days <= days[origin]))
                    clock = a[p]**2 * days / 360 + (r[p] * q[sample, :, p])**2
                    covariance = np.minimum.outer(clock, clock)
                    if len(known):
                        conditional[sample, :, p] += covariance[:, known] @ np.linalg.solve(
                            covariance[np.ix_(known, known)], history[known, p] - q[sample, known, p])
            increments = conditional[:, 4] - conditional[:, origin]
            np.testing.assert_allclose(increments, q[:, 4] - q[:, origin], rtol=2e-13, atol=2e-14)
            oracle = weights @ increments / math.sqrt(math.pi)
            np.testing.assert_allclose(expected, oracle, rtol=2e-13, atol=2e-14)
            # A hidden origin must be subtracted from the same conditional path.
            wrong = weights @ (conditional[:, 4] - q[:, origin]) / math.sqrt(math.pi)
            assert np.max(np.abs(wrong - expected)) > .1


def test_positive_phase_unit_conversion_preserves_scores_and_parameter_gradients():
    target, mean, lower, days, origin, scale, weight = inputs()
    mean.requires_grad_(); lower.requires_grad_()
    units = torch.tensor([.125, 17., 512.], dtype=DTYPE)
    original = fast_basis_increment_mean_mse(target, mean, lower, days, origin, scale, weight)
    converted = fast_basis_increment_mean_mse(target * units, mean + units.log().repeat_interleave(4),
        lower, days, origin, scale * units, weight)
    torch.testing.assert_close(converted, original, rtol=2e-14, atol=2e-15)
    ga = torch.autograd.grad(original.sum(), (mean, lower), retain_graph=True)
    gb = torch.autograd.grad(converted.sum(), (mean, lower))
    for a, b in zip(ga, gb, strict=True): torch.testing.assert_close(a, b, rtol=3e-14, atol=2e-15)
    torch.testing.assert_close(fast_basis_increment_mean(mean + units.log().repeat_interleave(4), lower, days, origin),
        fast_basis_increment_mean(mean, lower, days, origin) * units, rtol=2e-14, atol=2e-15)


def test_float64_gradcheck_includes_offdiagonal_cholesky_entries():
    target, mean, lower, days, origin, scale, weight = inputs()
    row, column = torch.tril_indices(12, 12)
    packed = lower[:1, row, column].clone().requires_grad_()
    mean = mean[:1].clone().requires_grad_()
    def objective(m, entries):
        triangle = entries.new_zeros((1, 12, 12))
        triangle[:, row, column] = entries
        return fast_basis_increment_mean_mse(target[:1], m, triangle, days, origin, scale, weight)
    assert torch.autograd.gradcheck(objective, (mean, packed), eps=2e-6, atol=2e-8, rtol=2e-6)
    gradient = torch.autograd.grad(objective(mean, packed).sum(), packed)[0]
    assert torch.count_nonzero(gradient[:, row != column]) > 0


@pytest.mark.parametrize("s,width,active", [(0., 1e-9, 2), (1400., 1e-8, 0), (18000., 1e-6, 2)])
def test_tiny_startup_and_late_tail_intervals_remain_positive_and_match_rate_quadrature(s, width, active):
    mean = torch.full((1, 12), -80., dtype=DTYPE)
    mean[:, active::4] = 0.
    lower = torch.eye(12, dtype=DTYPE)[None] * .1
    # A valid report grid can begin after day zero.
    days = torch.tensor([s, s + width], dtype=DTYPE)
    coefficients = np.exp(mean.numpy().reshape(1, 3, 4) + .005)
    expected = np.einsum("npk,fk->nfp", coefficients, rate_integrals(days.numpy(), 0))
    result = fast_basis_increment_mean(mean, lower, days, 0)
    assert torch.all(result > 0)
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-13, atol=0)
    if s:
        k, tau = SHAPES[active], TAU[active]
        assert gammainc(k, float(days[0]) / tau) == gammainc(k, float(days[1]) / tau) == 1.


def test_standardization_precedes_squared_error_for_extreme_finite_targets():
    _, mean, lower, _, _, _, _ = inputs()
    target = torch.zeros((2, 2, 3), dtype=DTYPE); target[:, 1] = 1e200
    value = fast_basis_increment_mean_mse(target, mean, lower, torch.tensor([0., 30.]), 0,
        torch.full((1, 3), 1e200, dtype=DTYPE), torch.ones((1, 3), dtype=DTYPE))
    torch.testing.assert_close(value, torch.ones(2, dtype=DTYPE), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["mean_shape", "lower_shape", "dtype", "mean_nan", "lower_inf", "upper_triangle",
    "zero_diagonal", "negative_diagonal", "days_shape", "days_inf", "days_negative", "days_repeat", "origin_last", "origin_negative", "origin_bool"])
def test_mean_rejects_invalid_support_and_shapes(kind):
    _, mean, lower, days, origin, _, _ = inputs()
    if kind == "mean_shape": mean = mean[:, :11]
    elif kind == "lower_shape": lower = lower[:, :, :11]
    elif kind == "dtype": lower = lower.float()
    elif kind == "mean_nan": mean[0, 0] = torch.nan
    elif kind == "lower_inf": lower[0, 2, 1] = torch.inf
    elif kind == "upper_triangle": lower[0, 1, 2] = 1e-30
    elif kind == "zero_diagonal": lower[0, 1, 1] = 0
    elif kind == "negative_diagonal": lower[0, 1, 1] = -.1
    elif kind == "days_shape": days = days[:, None]
    elif kind == "days_inf": days[4] = torch.inf
    elif kind == "days_negative": days[0] = -1
    elif kind == "days_repeat": days[2] = days[1]
    elif kind == "origin_last": origin = 4
    elif kind == "origin_negative": origin = -1
    else: origin = True
    with pytest.raises(ValueError): fast_basis_increment_mean(mean, lower, days, origin)


@pytest.mark.parametrize("kind", ["target_shape", "scale_shape", "weight_shape", "scale_zero", "scale_inf", "negative_weight", "weight_nan", "target_integer"])
def test_mse_rejects_invalid_frozen_scoring_inputs(kind):
    target, mean, lower, days, origin, scale, weight = inputs()
    if kind == "target_shape": target = target[:, :, :2]
    elif kind == "scale_shape": scale = scale[:2]
    elif kind == "weight_shape": weight = weight[:2]
    elif kind == "scale_zero": scale[0, 0] = 0
    elif kind == "scale_inf": scale[0, 0] = torch.inf
    elif kind == "negative_weight": weight[0, 0] = -.1
    elif kind == "weight_nan": weight[0, 0] = torch.nan
    else: target = target.long()
    with pytest.raises(ValueError): fast_basis_increment_mean_mse(target, mean, lower, days, origin, scale, weight)


@pytest.mark.parametrize("kind", ["coefficient_overflow", "coefficient_underflow", "variance_overflow", "tail_mass_underflow", "startup_mass_underflow"])
def test_positive_means_and_interval_masses_fail_closed_outside_float64_support(kind):
    _, mean, lower, days, origin, _, _ = inputs()
    if kind == "coefficient_overflow": mean[0, 0] = 1000
    elif kind == "coefficient_underflow": mean[0, 0] = -1000
    elif kind == "variance_overflow": lower[0, 0, 0] = 1e200
    elif kind == "tail_mass_underflow": days = torch.tensor([60000., 60001.], dtype=DTYPE); origin = 0
    else: days = torch.tensor([0., 1e-200], dtype=DTYPE); origin = 0
    with pytest.raises(FloatingPointError): fast_basis_increment_mean(mean, lower, days, origin)


@pytest.mark.parametrize("kind", ["raw_difference", "normalized_error", "squared_error", "weighted_error"])
def test_active_finite_target_arithmetic_overflow_cannot_become_missing(kind):
    target, mean, lower, days, origin, scale, weight = inputs()
    if kind == "raw_difference": target[:, origin] = -1e308; target[:, origin + 1:] = 1e308; scale[:] = 1e308
    elif kind == "normalized_error": target[:, origin + 1:] = 1e200; scale[:] = 1e-200
    elif kind == "squared_error": target[:, origin + 1:] = 1e200
    else: target[:, origin + 1:] = 1e100; weight[:] = 1e200
    with pytest.raises(FloatingPointError): fast_basis_increment_mean_mse(target, mean, lower, days, origin, scale, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_fixed_input_value_gradient_parity_and_device_validation(dtype):
    cpu = list(inputs(dtype)); cuda = [value.cuda() if isinstance(value, torch.Tensor) else value for value in cpu]
    for values in (cpu, cuda): values[1].requires_grad_(); values[2].requires_grad_()
    a, b = fast_basis_increment_mean_mse(*cpu), fast_basis_increment_mean_mse(*cuda)
    assert b.device.type == "cuda" and b.dtype == DTYPE
    torch.testing.assert_close(b.cpu(), a, rtol=2e-12, atol=2e-13)
    ga, gb = torch.autograd.grad(a.sum(), cpu[1:3]), torch.autograd.grad(b.sum(), cuda[1:3])
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(right.cpu(), left, rtol=3e-6 if dtype == torch.float32 else 3e-12, atol=2e-13)
    cuda[3] = cpu[3]
    with pytest.raises(ValueError): fast_basis_increment_mean_mse(*cuda)
