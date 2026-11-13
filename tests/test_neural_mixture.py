"""Synthetic distribution and compatibility checks for the isolated K2 prototype.

Hash-verified archived source supplies an optional legacy oracle. Tests never
read field reports or modify the main repository.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import sys
import tarfile
from types import ModuleType

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pytest
import torch

from joint_production import neural


FROZEN_NEURAL_SHA256 = "d827cdff7755ae41c5e5fbab0bd3a89c6c3272542df8a89bda5dda1f8c6749ce"
FIT_SETTINGS = dict(epochs=2, hidden_width=8, rank=2, batch_size=7, seed=91,
                    history_rate_features="last_interval", log_every=0)


def dense_mixture_nll(target, location, variance, factor, mask, logits):
    """Independent dense masked Gaussian formula, then mixture logsumexp."""
    outputs = []
    for well in range(len(target)):
        keep = mask[well]
        if not keep.any():
            outputs.append(sum(value[well].sum() * 0 for value in
                               (target, location, variance, factor, logits)))
            continue
        covariance = torch.diag(variance[well, keep]) + factor[well, keep] @ factor[well, keep].T
        residual = target[well, keep] - location[well, :, keep]
        quadratic = (residual * torch.linalg.solve(covariance, residual.T).T).sum(-1)
        component_nll = .5 * (quadratic + torch.linalg.slogdet(covariance)[1]
                              + int(keep.sum()) * math.log(2 * math.pi))
        outputs.append(-torch.logsumexp(torch.log_softmax(logits[well], -1) - component_nll, -1))
    return torch.stack(outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_mixture_likelihood_and_all_parameter_gradients_match_dense_oracle(dtype):
    rng = np.random.default_rng(311)
    target = rng.normal(size=(4, 5))
    target[0, 0] = target[1, 1] = target[2, 2] = 0.0
    arrays = [target, rng.normal(size=(4, 2, 5)), np.exp(rng.normal(0, .3, size=(4, 5))),
              rng.normal(0, .25, size=(4, 5, 2)), rng.normal(size=(4, 2))]
    mask = torch.tensor([[True] * 5, [False, True, False, True, False],
                         [False, False, True, False, False], [False] * 5])
    parameters = [torch.tensor(value, dtype=dtype, requires_grad=True) for value in arrays]
    y, location, variance, factor, logits = parameters
    actual = neural._masked_mixture_gaussian_nll(y, location, variance, factor, mask, logits)
    actual_gradients = torch.autograd.grad(actual.sum(), parameters)
    reference_parameters = [value.detach().double().requires_grad_() for value in parameters]
    expected = dense_mixture_nll(*reference_parameters[:4], mask, reference_parameters[4])
    expected_gradients = torch.autograd.grad(expected.sum(), reference_parameters)
    torch.testing.assert_close(actual.double(), expected, atol=1e-10, rtol=1e-10)
    for gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient.double(), expected_gradient, atol=1e-7, rtol=2e-6)
        torch.testing.assert_close(gradient[-1], torch.zeros_like(gradient[-1]), atol=0, rtol=0)
    assert actual[-1].item() == 0.0
    # Zero is an observed report; every masked-out parameter/target coordinate vanishes.
    assert actual[2].item() != 0
    torch.testing.assert_close(actual_gradients[0][~mask], torch.zeros_like(y[~mask]), atol=0, rtol=0)
    torch.testing.assert_close(actual_gradients[1][~mask[:, None].expand_as(location)],
                               torch.zeros_like(location[~mask[:, None].expand_as(location)]), atol=0, rtol=0)
    torch.testing.assert_close(actual_gradients[2][~mask], torch.zeros_like(variance[~mask]), atol=0, rtol=0)
    torch.testing.assert_close(actual_gradients[3][~mask], torch.zeros_like(factor[~mask]), atol=0, rtol=0)
    changed = y.detach().clone()
    changed[~mask] = 1e4
    other = neural._masked_mixture_gaussian_nll(changed, location, variance, factor, mask, logits)
    torch.testing.assert_close(other, actual, atol=1e-11, rtol=1e-11)


def test_extreme_mixture_logits_are_stable_and_component_labels_are_exchangeable():
    dtype = torch.float64
    target = torch.tensor([[0., 1., -2.]], dtype=dtype)
    location = torch.tensor([[[-1., 1., 2.], [2., -1., 0.]]], dtype=dtype)
    variance = torch.tensor([[.02, .3, .1]], dtype=dtype)
    factor = torch.tensor([[[.2], [-.4], [.1]]], dtype=dtype)
    mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[1000., -1000.]], dtype=dtype, requires_grad=True)
    result = neural._masked_mixture_gaussian_nll(target, location, variance, factor, mask, logits)
    expected = neural._masked_gaussian_nll(target, location[:, 0], variance, factor, mask)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    permuted = neural._masked_mixture_gaussian_nll(target, location.flip(1), variance, factor, mask, logits.flip(1))
    torch.testing.assert_close(permuted, result, atol=0, rtol=0)
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_two_component_initialization_preserves_center_and_exact_fixed_perturbation():
    arguments = dict(input_dim=9, output_dim=15, width=8, rank=2,
                     mean_initial=np.linspace(-.4, .6, 15), min_std=.025, max_std=3., max_factor=1.)
    with torch.random.fork_rng():
        torch.manual_seed(23)
        original = neural._JointReportNetwork(**arguments)
        torch.manual_seed(23)
        mixture = neural._JointMixtureReportNetwork(**arguments)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(mixture.state_dict()[name], value, atol=0, rtol=0)
    context = torch.linspace(-1., 1., 27).reshape(3, 9)
    center, expected_variance, expected_factor = original(context)
    location, variance, factor, logits = mixture(context)
    direction = .05 * (1 - 2 * (torch.arange(15) % 2))
    assert location.shape == (3, 2, 15)
    torch.testing.assert_close(mixture.mean_offset.weight, torch.zeros_like(mixture.mean_offset.weight), atol=0, rtol=0)
    torch.testing.assert_close(mixture.mean_offset.bias, direction, atol=0, rtol=0)
    torch.testing.assert_close(location[:, 0], center - direction, atol=0, rtol=0)
    torch.testing.assert_close(location[:, 1], center + direction, atol=0, rtol=0)
    torch.testing.assert_close(variance, expected_variance, atol=0, rtol=0)
    torch.testing.assert_close(factor, expected_factor, atol=0, rtol=0)
    torch.testing.assert_close(logits, torch.zeros_like(logits), atol=0, rtol=0)


@pytest.fixture(scope="module")
def small_reports():
    rng = np.random.default_rng(17)
    days = np.asarray([0., 30., 60., 90., 180.])
    amplitude = np.exp(rng.normal(1.7, .3, size=(18, 1, 3)))
    reports = amplitude * (1 - np.exp(-days[None, :, None] / 70))
    reports += rng.normal(0, .12, size=reports.shape)
    reports[:, 0] = 0
    reports[0, 1, 2] = -.2
    reports[1, 2, 1] = np.nan
    reports[2, 1, 0] = 0
    return reports, days


@pytest.fixture(scope="module")
def fitted(small_reports):
    reports, days = small_reports
    models = {
        "default": neural.ContextualGaussianNeuralBaseline(**FIT_SETTINGS),
        "one": neural.ContextualGaussianNeuralBaseline(**FIT_SETTINGS, mixture_components=1),
        "two": neural.ContextualGaussianNeuralBaseline(**FIT_SETTINGS, mixture_components=2),
    }
    for model in models.values():
        model.fit(reports, days, origins=[60., 90.])
    return models


def test_k2_tiny_fit_has_finite_scores_gradients_parameters_and_mixture_diagnostics(fitted, small_reports):
    model = fitted["two"]
    diagnostics = model.fit_diagnostics_
    assert diagnostics["mixture_components"] == 2
    assert diagnostics["training_wells"] == 18
    assert diagnostics["optimizer_updates"] == 12
    assert diagnostics["total_training_contexts"] == 72
    assert diagnostics["context_mask_sha256_by_epoch"] == fitted["one"].fit_diagnostics_["context_mask_sha256_by_epoch"]
    for key in ("conditional_report_nll_by_epoch", "max_gradient_norm_before_clipping_by_epoch"):
        assert len(diagnostics[key]) == 2 and np.isfinite(diagnostics[key]).all()
    assert len(diagnostics["mixture_diagnostics_by_epoch"]) == 2
    for entry in diagnostics["mixture_diagnostics_by_epoch"]:
        weights = np.asarray(entry["mean_component_weights"])
        assert weights.shape == (2,) and (weights > 0).all()
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-12)
        assert 0 <= entry["mean_weight_entropy_nats"] <= math.log(2) + 1e-12
        assert 0 <= entry["fraction_contexts_min_weight_below_0_01"] <= 1
        assert entry["mean_component_separation_rms_transformed"] > 0
    for parameter in model.network_.parameters():
        assert torch.isfinite(parameter).all()
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    reports, _ = small_reports
    parameters = model.predict_transformed_parameters(reports[:9], 60)
    assert parameters["location"].shape == (9, 2, 15)
    assert parameters["diagonal_variance"].shape == (9, 15)
    assert parameters["factor"].shape == (9, 15, 2)
    assert parameters["mixing_logits"].shape == parameters["mixture_weights"].shape == (9, 2)
    assert parameters["observed_mask"].shape == (9, 15)
    np.testing.assert_allclose(parameters["mixture_weights"].sum(-1), 1, atol=1e-15)


def test_k2_uses_one_component_for_the_whole_joint_draw_and_hidden_origin(fitted, small_reports):
    model = copy.deepcopy(fitted["two"])
    with torch.no_grad():
        model.network_.mean.weight.zero_()
        model.network_.mean.bias.zero_()
        model.network_.mean_offset.weight.zero_()
        direction = 2. * (1 - 2 * (torch.arange(15) % 2))
        model.network_.mean_offset.bias.copy_(direction)
        model.network_.factor.weight.zero_()
        model.network_.factor.bias.zero_()
        model.network_.std.weight.zero_()
        model.network_.std.bias.fill_(-30)
        model.network_.mixing_logits.weight.zero_()
        model.network_.mixing_logits.bias.copy_(torch.log(torch.tensor([.25, .75])))
    reports, days = small_reports
    history = reports[:1].copy()
    history[:, days > 60] = np.nan
    history[:, :, 1] = np.nan
    parameters = model.predict_transformed_parameters(history, 60)
    assert not parameters["observed_mask"][0, 7]
    draws = model.predict_samples(history, 60, 4096, np.random.default_rng(71))
    assert draws.shape == (1, 4096, 5, 3) and np.isfinite(draws).all()
    flat = np.arcsinh(draws / model.phase_scales_).reshape(4096, 15)
    missing = ~parameters["observed_mask"][0]
    signs = (flat[:, missing] * direction.numpy()[missing]) > 0
    assert np.array_equal(signs, np.broadcast_to(signs[:, :1], signs.shape))
    assert abs(signs[:, 0].mean() - .75) < .025
    assert (draws[:, :, days > 60] < 0).any() and (draws[:, :, days > 60] > 0).any()
    assert draws[0, :, 2, 1].std() > 0
    np.testing.assert_array_equal(draws[:, :, 0], 0)
    observed = parameters["observed_mask"].reshape(1, 5, 3)
    supplied = history.copy()
    supplied[:, 0] = 0
    mask = np.broadcast_to(observed[:, None], draws.shape)
    np.testing.assert_array_equal(draws[mask], np.broadcast_to(supplied[:, None], draws.shape)[mask])
    np.testing.assert_allclose(draws[:, :, 4] - draws[:, :, 2],
                               (draws[:, :, 4] - draws[:, :, 3]) + (draws[:, :, 3] - draws[:, :, 2]),
                               atol=1e-12, rtol=1e-12)


def test_k2_future_infinities_are_ignored_observed_infinities_rejected_and_zero_retained(fitted, small_reports):
    model = fitted["two"]
    reports, days = small_reports
    history = reports[:3].copy()
    history[:, days > 60] = np.nan
    zero = model.predict_transformed_parameters(history, 60)
    assert zero["observed_mask"][2, 3]
    assert history[2, 1, 0] == 0
    changed = history.copy()
    changed[:, days > 60] = np.inf
    np.testing.assert_array_equal(model.predict_samples(history, 60, 13, np.random.default_rng(19)),
                                  model.predict_samples(changed, 60, 13, np.random.default_rng(19)))
    np.testing.assert_array_equal(history, reports[:3].copy() * np.where((days <= 60)[None, :, None], 1., np.nan))
    changed[0, 1, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        model.predict_samples(changed, 60, 13, np.random.default_rng(19))


def test_k1_and_k2_leave_the_same_gaussian_rng_state_after_multiple_chunks(fitted, small_reports):
    history, _ = small_reports
    first_rng, second_rng = np.random.default_rng(841), np.random.default_rng(841)
    first_before, second_before = copy.deepcopy(first_rng.bit_generator.state), copy.deepcopy(second_rng.bit_generator.state)
    first = fitted["one"].predict_samples(history[:9], 60, 17, first_rng)
    second = fitted["two"].predict_samples(history[:9], 60, 17, second_rng)
    assert first_rng.bit_generator.state != first_before
    assert second_rng.bit_generator.state != second_before
    assert first_rng.bit_generator.state == second_rng.bit_generator.state
    assert not np.array_equal(first, second)


def test_k2_checkpoint_roundtrip_preserves_the_entire_conditional_law(fitted, small_reports, tmp_path):
    model = fitted["two"]
    path = tmp_path / "mixture.pt"
    model.save(path)
    stored = torch.load(path, map_location="cpu", weights_only=True)
    assert stored["configuration"]["mixture_components"] == 2
    restored = neural.ContextualGaussianNeuralBaseline.load(path)
    assert restored.mixture_components == 2
    assert restored.fit_diagnostics_ == model.fit_diagnostics_
    reports, _ = small_reports
    before, after = model.predict_transformed_parameters(reports[:9], 60), restored.predict_transformed_parameters(reports[:9], 60)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    np.testing.assert_array_equal(model.predict_samples(reports[:9], 60, 17, np.random.default_rng(3)),
                                  restored.predict_samples(reports[:9], 60, 17, np.random.default_rng(3)))


@pytest.mark.parametrize("logits", [[18., 0.], [0., 18.]])
def test_nearly_saturated_mixture_weights_keep_float64_probability_resolution(fitted, small_reports, logits):
    model = copy.deepcopy(fitted["two"])
    with torch.no_grad():
        model.network_.mixing_logits.weight.zero_()
        model.network_.mixing_logits.bias.copy_(torch.tensor(logits))
    reports, _ = small_reports
    parameters = model.predict_transformed_parameters(reports[:1], 60)
    assert parameters["mixing_logits"].dtype == np.float64
    assert parameters["mixture_weights"].dtype == np.float64
    weights = parameters["mixture_weights"][0]
    assert weights.min() > 0 and weights.max() < 1
    expected = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=-1).numpy()
    np.testing.assert_allclose(weights, expected, atol=1e-16, rtol=1e-15)


@pytest.mark.parametrize("components", [0, 3, 1.0, True, "2"])
def test_invalid_component_counts_fail_explicitly(components):
    with pytest.raises(ValueError, match="mixture_components"):
        neural.ContextualGaussianNeuralBaseline(mixture_components=components)
