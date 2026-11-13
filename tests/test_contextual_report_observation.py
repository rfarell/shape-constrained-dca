"""Per-well fixed noise scales: joint Gaussian, gradient, and legacy invariants."""
import hashlib
import os
from pathlib import Path
import tarfile
import types

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pytest
import torch

from joint_production.torch_observation import sample_accumulated_reports


def inputs(dtype=torch.float64, *, wells=3, samples=3, device="cpu"):
    days = torch.tensor([0., 11., 47., 120., 315., 730.], dtype=dtype, device=device)
    scale = torch.tensor([[.5, 2., 1.2], [1.4, .7, 2.1], [.8, 1.6, .9]], dtype=dtype, device=device)[:wells]
    base = days / 100 + days / (200 + days)
    q = (base[None, None, :, None] * scale[:, None, None]).expand(wells, samples, 6, 3).clone()
    history = torch.full((3, 6, 3), float("nan"), dtype=dtype, device=device)
    history[[0, 2], 0] = 0.
    history[0, 1, 0], history[0, 3, 0], history[0, 2, 1] = -.35, .4, .7
    history[1, 1, 1], history[1, 3, 1], history[1, 2, 2] = .8, .45, -.2
    history[2, 2, 0], history[2, 3, 1] = .55, -.15
    history[2, 1:4, 2] = torch.tensor([.1, .05, 0.], dtype=dtype, device=device)
    a = torch.tensor([[.4, .8, .3], [1.1, .2, .7], [.6, .5, 1.3]], dtype=dtype, device=device)[:wells]
    r = torch.tensor([[.1, .23, .05], [.3, .11, .17], [.09, .31, .2]], dtype=dtype, device=device)[:wells]
    normals = torch.sin(torch.arange(q.numel(), dtype=dtype, device=device) * .31).reshape_as(q)
    return q, history[:wells], days, a, r, normals


def dense_conditioned_gaussian(q, history, days, a, r):
    """Condition one dense independent-well/phase covariance, without bridges."""
    q, history, days, a, r = [value.detach().cpu().double().numpy() for value in (q, history, days, a, r)]
    wells, times = q.shape[:2]
    covariance = np.zeros((wells * times * 3, wells * times * 3))
    for well in range(wells):
        for phase in range(3):
            clock = a[well, phase]**2 * days / 360 + (r[well, phase] * q[well, :, phase])**2
            index = well * times * 3 + np.arange(times) * 3 + phase
            covariance[np.ix_(index, index)] = np.minimum(clock[:, None], clock[None, :])
    mean = q.flatten()
    known = np.flatnonzero((np.isfinite(history) & (days[None, :, None] <= 120)
                           & (days[None, :, None] > 0)).flatten())
    gain = np.linalg.solve(covariance[np.ix_(known, known)], covariance[known]).T
    return mean + gain @ (history.flatten()[known] - mean[known]), covariance - gain @ covariance[known]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wells", [1, 3])
def test_per_well_bridges_match_joint_dense_mean_covariance_with_independent_normal_basis(dtype, wells):
    dimension = wells * 6 * 3
    q, history, days, a, r, _ = inputs(dtype, wells=wells, samples=dimension + 1)
    # Basis coordinates span every well, avoiding artificial cross-well correlation.
    normals = torch.cat((torch.zeros((1, dimension), dtype=dtype), torch.eye(dimension, dtype=dtype)))
    normals = normals.reshape(dimension + 1, wells, 6, 3).permute(1, 0, 2, 3)
    actual = sample_accumulated_reports(q, history, days, a, r, cutoff=120, standard_normals=normals)
    expected_mean, expected_covariance = dense_conditioned_gaussian(q[:, 0], history, days, a, r)
    mean = actual[:, 0].reshape(-1)
    factor = (actual[:, 1:] - actual[:, :1]).permute(1, 0, 2, 3).reshape(dimension, dimension)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    np.testing.assert_allclose(mean.numpy(), expected_mean, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose((factor.T @ factor).numpy(), expected_covariance, rtol=tolerance, atol=tolerance)
    known = torch.isfinite(history) & (days[None, :, None] <= 120)
    torch.testing.assert_close(actual[known[:, None].expand_as(actual)],
                               history[:, None].expand_as(actual)[known[:, None].expand_as(actual)], rtol=0, atol=0)
    assert torch.all(actual[:, :, 0] == 0) and (actual < 0).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wells", [1, 3])
def test_constant_per_well_scales_preserve_global_paths_and_generator_state_exactly(dtype, wells):
    q, history, days, a, r, normals = inputs(dtype, wells=wells)
    generators = [torch.Generator().manual_seed(151) for _ in range(2)]
    before = torch.get_rng_state().clone()
    global_result = sample_accumulated_reports(q, history, days, a[0], r[0], cutoff=120, generator=generators[0])
    per_well = sample_accumulated_reports(q, history, days, a[:1].expand(wells, -1), r[:1].expand(wells, -1),
                                         cutoff=120, generator=generators[1])
    torch.testing.assert_close(per_well, global_result, rtol=0, atol=0)
    torch.testing.assert_close(generators[0].get_state(), generators[1].get_state(), rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    # Supplied innovations bypass generation and retain the same exact relation.
    torch.testing.assert_close(
        sample_accumulated_reports(q, history, days, a[0], r[0], cutoff=120, standard_normals=normals),
        sample_accumulated_reports(q, history, days, a[:1].expand(wells, -1), r[:1].expand(wells, -1),
                                   cutoff=120, standard_normals=normals), rtol=0, atol=0)


def test_equal_well_and_sample_counts_do_not_confuse_noise_broadcast_axes():
    q, history, days, a, r, normals = inputs(wells=3, samples=3)
    together = sample_accumulated_reports(q, history, days, a, r, cutoff=120, standard_normals=normals)
    separate = torch.cat([sample_accumulated_reports(q[i:i + 1], history[i:i + 1], days,
        a[i:i + 1], r[i:i + 1], cutoff=120, standard_normals=normals[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(together, separate, rtol=0, atol=0)


def test_noise_gradients_are_well_local_and_sum_to_global_vector_gradient():
    q, history, days, a, r, normals = inputs()
    global_a, global_r = a[0].clone().requires_grad_(), r[0].clone().requires_grad_()
    local_a = global_a.detach()[None].expand(3, -1).clone().requires_grad_()
    local_r = global_r.detach()[None].expand(3, -1).clone().requires_grad_()
    global_result = sample_accumulated_reports(q, history, days, global_a, global_r, cutoff=120, standard_normals=normals)
    local_result = sample_accumulated_reports(q, history, days, local_a, local_r, cutoff=120, standard_normals=normals)
    global_gradient = torch.autograd.grad(global_result.square().sum(), (global_a, global_r))
    local_gradient = torch.autograd.grad(local_result.square().sum(), (local_a, local_r), retain_graph=True)
    for global_value, local_value in zip(global_gradient, local_gradient, strict=True):
        torch.testing.assert_close(global_value, local_value.sum(0), rtol=3e-14, atol=3e-13)
    first_well = torch.autograd.grad(local_result[0].square().sum(), (local_a, local_r))
    for gradient in first_well:
        assert torch.all(gradient[1:] == 0) and gradient[0].abs().sum() > 0


def test_shared_scale_on_both_noise_terms_changes_deviations_but_not_conditional_mean():
    q, history, days, a, r, normals = inputs()
    g = torch.tensor([[.7, 1.2, 1.4], [1.5, .8, 1.1], [.6, 1.3, .9]], dtype=q.dtype, requires_grad=True)
    mean = sample_accumulated_reports(q, history, days, a, r, cutoff=120, standard_normals=torch.zeros_like(q))
    base = sample_accumulated_reports(q, history, days, a, r, cutoff=120, standard_normals=normals)
    scaled_mean = sample_accumulated_reports(q, history, days, a * g, r * g, cutoff=120, standard_normals=torch.zeros_like(q))
    scaled = sample_accumulated_reports(q, history, days, a * g, r * g, cutoff=120, standard_normals=normals)
    torch.testing.assert_close(scaled_mean, mean, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(scaled, mean + g[:, None, None] * (base - mean), rtol=2e-14, atol=2e-14)
    weights = torch.linspace(.2, 1.4, q.numel(), dtype=q.dtype).reshape_as(q)
    gradient = torch.autograd.grad((scaled * weights).sum(), g, retain_graph=True)[0]
    torch.testing.assert_close(gradient, ((base - mean) * weights).sum(dim=(1, 2)), rtol=3e-13, atol=3e-13)
    known = torch.isfinite(history) & (days[None, :, None] <= 120)
    known_loss = scaled[known[:, None].expand_as(q)].sum() + scaled[:, :, 0].sum()
    pinned_gradient = torch.autograd.grad(known_loss, g)[0]
    assert torch.all(pinned_gradient == 0)


def test_gradcheck_through_positive_curve_increments_per_well_noise_and_shared_multiplier():
    q, history, days, a, r, normals = inputs(samples=2)
    log_increments = q[:, :, 1:].diff(dim=2, prepend=q[:, :, :1]).log().requires_grad_()
    a, r = a.requires_grad_(), r.requires_grad_()
    g = torch.full((3, 3), 1.1, dtype=q.dtype, requires_grad=True)
    def draw(log_q, abs_scale, rel_scale, multiplier):
        cumulative = torch.cat((torch.zeros_like(log_q[:, :, :1]), log_q.exp().cumsum(2)), dim=2)
        return sample_accumulated_reports(cumulative, history, days, abs_scale * multiplier, rel_scale * multiplier,
                                          cutoff=120, standard_normals=normals)
    assert torch.autograd.gradcheck(draw, (log_increments, a, r, g), fast_mode=True, atol=2e-5, rtol=2e-4)
    gradients = torch.autograd.grad(draw(log_increments, a, r, g).square().mean(), (log_increments, a, r, g))
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)


def test_per_well_noise_retains_phase_units_and_ignores_future_history_poison():
    q, history, days, a, r, normals = inputs()
    units = torch.tensor([8., .125, 16.], dtype=q.dtype)
    original = sample_accumulated_reports(q, history, days, a, r, cutoff=120, standard_normals=normals)
    converted = sample_accumulated_reports(q * units, history * units, days, a * units, r,
                                          cutoff=120, standard_normals=normals)
    torch.testing.assert_close(converted, original * units, rtol=2e-14, atol=2e-13)
    poison = history.clone()
    poison[:, days > 120] = torch.tensor([float("inf"), -float("inf"), 1e300], dtype=q.dtype)
    replay = sample_accumulated_reports(q, poison, days, a, r, cutoff=120, standard_normals=normals)
    torch.testing.assert_close(replay, original, rtol=0, atol=0)


@pytest.mark.parametrize("shape_a,shape_r", [((3,), (3, 3)), ((3, 3), (3,)), ((1, 3), (1, 3)),
                                          ((3, 1, 3), (3, 1, 3)), ((3, 2), (3, 2))])
def test_mixed_or_incorrect_contextual_noise_shapes_are_rejected(shape_a, shape_r):
    q, history, days, _, _, normals = inputs()
    with pytest.raises(ValueError, match="both have shape"):
        sample_accumulated_reports(q, history, days, torch.ones(shape_a, dtype=q.dtype), torch.ones(shape_r, dtype=q.dtype),
                                    cutoff=120, standard_normals=normals)
    if shape_a == (3,) and shape_r == (3, 3):
        with pytest.raises(ValueError, match="both have shape"):
            sample_accumulated_reports(q[:1], history[:1], days, torch.ones(3, dtype=q.dtype), torch.ones((1, 3), dtype=q.dtype),
                                        cutoff=120, standard_normals=normals[:1])


@pytest.mark.parametrize("kind", ["a_dtype", "r_dtype", "a_device", "r_device", "zero", "infinite", "nan"])
def test_contextual_noise_dtype_device_and_positive_support_checks(kind):
    q, history, days, a, r, normals = inputs()
    if kind == "a_dtype": a = a.float()
    elif kind == "r_dtype": r = r.long()
    elif kind == "a_device": a = a.to("meta")
    elif kind == "r_device": r = r.to("meta")
    elif kind == "zero": a[1, 2] = 0
    elif kind == "infinite": r[2, 0] = float("inf")
    else: a[0, 1] = float("nan")
    with pytest.raises(ValueError, match="dtype and device|finite and strictly positive"):
        sample_accumulated_reports(q, history, days, a, r, cutoff=120, standard_normals=normals)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only contextual noise value/gradient parity")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_contextual_noise_matches_cpu_values_and_gradients(dtype):
    results = []
    for device in ("cpu", "cuda"):
        q, history, days, a, r, normals = inputs(dtype, device=device)
        a, r = a.requires_grad_(), r.requires_grad_()
        actual = sample_accumulated_reports(q, history, days, a, r, cutoff=120, standard_normals=normals)
        gradients = torch.autograd.grad(actual.square().mean(), (a, r))
        global_result = sample_accumulated_reports(q, history, days, a[0], r[0], cutoff=120, standard_normals=normals)
        constant = sample_accumulated_reports(q, history, days, a[:1].expand(3, -1), r[:1].expand(3, -1),
                                              cutoff=120, standard_normals=normals)
        torch.testing.assert_close(constant, global_result, rtol=0, atol=0)
        results.append((actual.detach().cpu(), *(gradient.cpu() for gradient in gradients)))
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    for cpu, cuda in zip(*results, strict=True):
        torch.testing.assert_close(cpu, cuda, rtol=tolerance, atol=tolerance)
