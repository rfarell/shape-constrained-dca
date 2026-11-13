"""Independent initialization, covariance-support and gradient oracles."""
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import tarfile

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pytest
import torch

from joint_production.covariance_parameterization import (
    initial_diagonal_anchor, normalized_row_cholesky, normalized_row_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
ROWS, COLUMNS = np.tril_indices(12)
DIAGONAL = ROWS == COLUMNS
ANCHOR = struct.unpack("f", struct.pack("f", math.log(.3)))[0]


def initial_raw(shape=(), dtype=torch.float32, device="cpu"):
    raw = torch.zeros((*shape, 78), dtype=dtype, device=device)
    raw[..., DIAGONAL] = ANCHOR
    return raw


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_initial_factor_and_local_jacobian_match_frozen_equation_without_archival_files(device):
    # This independent equation comes from the pinned archived transform
    # checked above. It keeps the CUDA initialization requirement mandatory
    # even when the historical build archive is absent on a remote machine.
    raw = initial_raw(dtype=torch.float32, device=device).requires_grad_()
    mask = torch.tensor(DIAGONAL, device=device)

    def frozen_original_equation(value):
        return torch.where(mask, .025 + .975 * torch.sigmoid(value), .2 * torch.tanh(value))

    def candidate_coordinates(value):
        lower = normalized_row_cholesky(value)
        rows, columns = torch.tril_indices(12, 12, device=device)
        return lower[rows, columns]

    original = frozen_original_equation(raw)
    candidate = candidate_coordinates(raw)
    torch.testing.assert_close(candidate, original, rtol=0, atol=0)
    legacy_sigma = .025 + .975 * torch.sigmoid(torch.tensor(ANCHOR, dtype=torch.float32, device=device))
    torch.testing.assert_close(candidate[mask], legacy_sigma.expand(12), rtol=0, atol=0)
    original_jacobian = torch.autograd.functional.jacobian(frozen_original_equation, raw)
    candidate_jacobian = torch.autograd.functional.jacobian(candidate_coordinates, raw)
    torch.testing.assert_close(candidate_jacobian, original_jacobian, rtol=2e-7, atol=3e-8)
    diagonal = torch.where(mask, torch.tensor(9 / 52, device=device), torch.tensor(.2, device=device))
    torch.testing.assert_close(candidate_jacobian, torch.diag(diagonal), rtol=2e-7, atol=3e-8)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_initial_anchor_records_original_float32_backend_rounding_without_rng(device):
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    raw = torch.tensor(ANCHOR, dtype=torch.float32, device=device)
    old_sigma = .025 + .975 * torch.sigmoid(raw)
    anchor = initial_diagonal_anchor(device)
    assert anchor.dtype == torch.float64 and anchor.ndim == 0 and anchor.device.type == device
    assert not anchor.requires_grad
    expected = math.log((old_sigma.item() - .025) / (1.2 - old_sigma.item()))
    assert anchor.item() == pytest.approx(expected, rel=0, abs=5e-16)
    metadata = normalized_row_metadata(anchor)
    assert metadata["covariance_diagonal_anchor_logit"] == anchor.item()
    assert metadata["covariance_initial_legacy_std_float32"] == old_sigma.item()
    assert metadata["covariance_diagonal_anchor_nominal_logit"] == math.log(9 / 38)
    assert "persisted_for_fit_prediction_and_device_moves" in metadata["covariance_diagonal_anchor_rule"]
    assert torch.equal(torch.random.get_rng_state(), before)
    if device == "cuda": assert torch.equal(torch.cuda.get_rng_state(), cuda_before)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_supplied_cpu_or_cuda_legacy_anchor_is_preserved_on_either_device(device):
    # Replay both values seen in preflight without depending on the current
    # backend to reproduce the historical rounding convention.
    for sigma in (.25, float(np.nextafter(np.float32(.25), np.float32(np.inf)))):
        stored = torch.tensor(math.log((sigma - .025) / (1.2 - sigma)), dtype=torch.float64)
        raw = initial_raw((2,), device=device)
        decoded = normalized_row_cholesky(raw, diagonal_anchor_logit=stored)
        expected = torch.eye(12, dtype=torch.float32, device=device)[None].expand(2, -1, -1) * sigma
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
        scalar = normalized_row_cholesky(raw, diagonal_anchor_logit=stored.item())
        torch.testing.assert_close(scalar, decoded, rtol=0, atol=0)
        metadata = normalized_row_metadata(stored)
        assert metadata["covariance_diagonal_anchor_logit"] == stored.item()
        assert metadata["covariance_initial_legacy_std_float32"] == sigma
        assert stored.device.type == "cpu" and not stored.requires_grad


@pytest.mark.parametrize("bad", [torch.tensor(-1.4, dtype=torch.float32), torch.tensor([-1.4], dtype=torch.float64),
    torch.tensor(-1.4, dtype=torch.float64, requires_grad=True), torch.tensor(float("nan"), dtype=torch.float64),
    float("inf"), -float("inf"), float("nan"), 1, True, [-1.4]])
def test_nonfinite_nonscalar_or_trainable_anchor_is_rejected(bad):
    with pytest.raises(ValueError): normalized_row_cholesky(initial_raw(), diagonal_anchor_logit=bad)
    with pytest.raises(ValueError): normalized_row_metadata(bad)


def independent_forward(raw):
    """Scalar logistic and math.hypot oracle at ordinary finite coordinates."""
    shape = raw.shape[:-1]
    result = np.zeros((*shape, 12, 12), dtype=np.float64)
    for index in np.ndindex(shape):
        for i in range(12):
            entries = raw[index][ROWS == i].astype(np.float64)
            argument = (235 / 247) * (entries[-1] - ANCHOR) + math.log(9 / 38)
            sigma = .025 + 1.175 / (1 + math.exp(-argument))
            vector = [.8 * x for x in entries[:-1]] + [1.]
            norm = math.hypot(*vector)
            result[index][i, :i+1] = [sigma * x / norm for x in vector]
    return result


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_arbitrary_leading_axes_match_independent_row_norm_and_covariance_oracle(shape, dtype):
    count = int(np.prod(shape)) if shape else 1
    raw = torch.sin(torch.arange(count * 78, dtype=dtype) / 9).reshape(*shape, 78) * 4
    untouched = raw.clone()
    lower = normalized_row_cholesky(raw)
    assert lower.shape == (*shape, 12, 12) and lower.dtype == dtype and lower.device == raw.device
    expected = independent_forward(raw.numpy())
    np.testing.assert_allclose(lower.double().numpy(), expected, rtol=8e-8 if dtype == torch.float32 else 3e-15, atol=1e-16)
    assert torch.count_nonzero(torch.triu(lower, diagonal=1)) == 0
    assert torch.all(lower.diagonal(dim1=-2, dim2=-1) > 0)
    sigma = np.linalg.norm(expected, axis=-1)
    direction = lower.double().numpy() / sigma[..., :, None]
    np.testing.assert_allclose(np.sum(direction**2, axis=-1), 1., rtol=2e-7 if dtype == torch.float32 else 4e-15)
    covariance = lower.double() @ lower.double().transpose(-1, -2)
    np.testing.assert_allclose(covariance.diagonal(dim1=-2, dim2=-1).numpy(), sigma**2,
        rtol=2e-7 if dtype == torch.float32 else 4e-15)
    assert np.all((sigma > .025) & (sigma < 1.2))
    torch.testing.assert_close(raw, untouched, rtol=0, atol=0)


def inverse_factor(lower):
    """Invert via observed row norms and diagonal ratios, independently."""
    lower = np.asarray(lower, dtype=np.float64)
    norms = np.sqrt(np.sum(lower**2, axis=-1))
    raw = np.empty(78)
    for k, (i, j) in enumerate(zip(ROWS, COLUMNS, strict=True)):
        if i == j:
            raw[k] = ANCHOR + (247 / 235) * (math.log((norms[i] - .025) / (1.2 - norms[i])) - math.log(9 / 38))
        else: raw[k] = lower[i, j] / (.8 * lower[i, i])
    return raw


def test_inverse_represents_arbitrary_covariance_under_every_phase_relabeling():
    standard_deviations = np.linspace(.03, 1.17, 12)
    signs = np.where(np.arange(12) % 3, 1., -1.)
    correlation = .2 * np.eye(12) + .8 * np.outer(signs, signs)
    covariance = standard_deviations[:, None] * correlation * standard_deviations[None]
    raw_rows, orders = [], []
    for phases in itertools.permutations(range(3)):
        order = np.concatenate([np.arange(4*p, 4*p+4) for p in phases])
        factor = np.linalg.cholesky(covariance[np.ix_(order, order)])
        raw_rows.append(inverse_factor(factor)); orders.append(order)
    decoded = normalized_row_cholesky(torch.tensor(np.stack(raw_rows))).numpy()
    for lower, order in zip(decoded, orders, strict=True):
        rebuilt = lower @ lower.T
        np.testing.assert_allclose(rebuilt, covariance[np.ix_(order, order)], rtol=3e-14, atol=3e-15)
        inverse = np.argsort(order)
        np.testing.assert_allclose(rebuilt[np.ix_(inverse, inverse)], covariance, rtol=3e-14, atol=3e-15)
        np.testing.assert_allclose(np.linalg.norm(lower, axis=1), standard_deviations[order], rtol=4e-15)


def test_original_two_coordinate_ordering_counterexample_is_now_representable():
    original = np.eye(12) * .25
    original[0, 0], original[4, 0], original[4, 4] = .8, .16, .12
    covariance = original @ original.T
    order = np.r_[4:8, 0:4, 8:12]
    permuted = np.linalg.cholesky(covariance[np.ix_(order, order)])
    np.testing.assert_allclose(permuted[np.ix_([0, 4], [0, 4])], [[.2, 0.], [.64, .48]], rtol=2e-15)
    assert permuted[4, 0] > .2
    raw = inverse_factor(permuted)
    assert raw[(ROWS == 4) & (COLUMNS == 0)][0] == pytest.approx(5 / 3, rel=2e-15)
    decoded = normalized_row_cholesky(torch.tensor(raw))
    np.testing.assert_allclose(decoded.numpy(), permuted, rtol=3e-15, atol=2e-16)
    np.testing.assert_allclose((decoded @ decoded.T).numpy(), covariance[np.ix_(order, order)], rtol=3e-15, atol=2e-16)


def test_diagonal_marginal_scale_controls_do_not_change_other_rows_or_correlation_directions():
    raw = torch.sin(torch.arange(78, dtype=torch.float64) / 8) * 3
    original = normalized_row_cholesky(raw)
    changed = raw.clone(); changed[(ROWS == 7) & DIAGONAL] += 1.2
    second = normalized_row_cholesky(changed)
    torch.testing.assert_close(second[torch.arange(12) != 7], original[torch.arange(12) != 7], rtol=0, atol=0)
    assert torch.linalg.vector_norm(second[7]) > torch.linalg.vector_norm(original[7])
    torch.testing.assert_close(second[7] / torch.linalg.vector_norm(second[7]),
        original[7] / torch.linalg.vector_norm(original[7]), rtol=3e-15, atol=3e-16)


def test_near_singularity_is_allowed_without_a_cholesky_diagonal_or_eigenvalue_floor():
    raw = initial_raw(dtype=torch.float64)
    raw[(ROWS == 1) & (COLUMNS == 0)] = 1e12
    lower = normalized_row_cholesky(raw)
    assert 0 < lower[1, 1] < 1e-12
    torch.testing.assert_close(torch.linalg.vector_norm(lower, dim=-1), torch.full((12,), .25, dtype=torch.float64), rtol=3e-15, atol=0)
    assert lower[1, 0] == pytest.approx(.25, rel=2e-15)
    # Positive diagonal is the promised support; an eigenvalue floor would
    # exclude this increasingly singular but mathematically valid factor.
    assert lower[1, 1] < .025 / 1e9


def test_off_initialization_float64_gradcheck_includes_cross_coordinate_normalization():
    raw = (torch.sin(torch.arange(78, dtype=torch.float64) / 11) * 2.7).requires_grad_()
    assert torch.autograd.gradcheck(normalized_row_cholesky, (raw,), eps=2e-6, atol=3e-8, rtol=3e-6)
    jacobian = torch.autograd.functional.jacobian(normalized_row_cholesky, raw)
    off_raw_index = np.flatnonzero((ROWS == 7) & (COLUMNS == 1))[0]
    assert abs(float(jacobian[7, 7, off_raw_index])) > 1e-5
    assert abs(float(jacobian[7, 2, off_raw_index])) > 1e-5
    assert torch.count_nonzero(jacobian[torch.arange(12) != 7, :, off_raw_index]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_maximum_finite_raw_values_have_finite_factors_gradients_and_positive_diagonals(dtype, sign):
    raw = torch.full((2, 78), sign * torch.finfo(dtype).max, dtype=dtype, requires_grad=True)
    lower = normalized_row_cholesky(raw)
    assert torch.isfinite(lower).all() and torch.all(lower.diagonal(dim1=-2, dim2=-1) > 0)
    gradient, = torch.autograd.grad(lower.sum(), raw)
    assert torch.isfinite(gradient).all()
    sigma = torch.linalg.vector_norm(lower.double(), dim=-1)
    expected = .025 if sign < 0 else 1.2
    torch.testing.assert_close(sigma, torch.full_like(sigma, expected), rtol=8e-8 if dtype == torch.float32 else 4e-15, atol=0)


@pytest.mark.parametrize("value", [0., 1e-300, -1e-300, 1e150, -1e150])
def test_zero_tiny_and_extreme_offdiagonal_raw_values_retain_finite_gradients(value):
    raw = initial_raw((2,), dtype=torch.float64)
    raw[:, ~DIAGONAL] = value
    raw.requires_grad_()
    lower = normalized_row_cholesky(raw)
    gradient, = torch.autograd.grad((lower * torch.linspace(-1., 2., 12)).sum(), raw)
    assert torch.isfinite(lower).all() and torch.isfinite(gradient).all()
    assert torch.all(lower.diagonal(dim1=-2, dim2=-1) > 0)


@pytest.mark.parametrize("bad", [torch.zeros(77), torch.zeros(2, 79), torch.zeros(0, 78), torch.tensor(1.),
    torch.zeros(78, dtype=torch.int64), torch.zeros(78, dtype=torch.float16), torch.zeros(78, dtype=torch.complex64),
    np.zeros(78), [0.] * 78])
def test_invalid_shape_type_and_dtype_are_rejected(bad):
    with pytest.raises(ValueError): normalized_row_cholesky(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_raw_values_are_rejected(value):
    raw = initial_raw(); raw[5] = value
    with pytest.raises(ValueError): normalized_row_cholesky(raw)


def test_numerically_nonpositive_returned_diagonal_is_rejected(monkeypatch):
    original = torch.linalg.vector_norm
    def failed_norm(values, *args, **kwargs):
        # Simulate a failed backend norm without introducing a covariance floor.
        return torch.full_like(original(values, *args, **kwargs), torch.inf)
    monkeypatch.setattr(torch.linalg, "vector_norm", failed_norm)
    with pytest.raises(FloatingPointError): normalized_row_cholesky(initial_raw())


def test_metadata_is_json_safe_fresh_and_states_the_pointwise_support_limit():
    metadata = normalized_row_metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata["covariance_marginal_std_bounds"] == [.025, 1.2]
    assert metadata["covariance_diagonal_anchor_raw_float32"] == ANCHOR
    assert metadata["covariance_factor_raw_coordinates"] == 78
    assert "finite_MLP_function_class_nesting_not_claimed" in metadata["covariance_support_limit"]
    assert "near_singularity_allowed" in metadata["covariance_conditioning_limit"]
    metadata["covariance_marginal_std_bounds"][0] = 999.
    assert normalized_row_metadata()["covariance_marginal_std_bounds"] == [.025, 1.2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_off_initialization_fixed_values_and_gradient_parity(dtype):
    cpu = (torch.sin(torch.arange(2 * 78, dtype=dtype).reshape(2, 78)) * 8).requires_grad_()
    cuda = cpu.detach().cuda().requires_grad_()
    # Both backends evaluate the same persisted initialization convention.
    # Their standalone defaults deliberately reproduce their respective old
    # float32 initial factors and therefore need not use identical anchors.
    stored_anchor = initial_diagonal_anchor("cpu")
    a = normalized_row_cholesky(cpu, diagonal_anchor_logit=stored_anchor)
    b = normalized_row_cholesky(cuda, diagonal_anchor_logit=stored_anchor)
    assert b.dtype == dtype and b.device.type == "cuda"
    torch.testing.assert_close(b.cpu(), a, rtol=3e-6 if dtype == torch.float32 else 2e-13, atol=2e-8 if dtype == torch.float32 else 3e-15)
    ga, = torch.autograd.grad(a.square().sum() + a.sum(), cpu)
    gb, = torch.autograd.grad(b.square().sum() + b.sum(), cuda)
    torch.testing.assert_close(gb.cpu(), ga, rtol=5e-6 if dtype == torch.float32 else 5e-13, atol=3e-8 if dtype == torch.float32 else 3e-15)
