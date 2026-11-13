"""Tiny synthetic training checks for integrating only marginal report noise."""
import copy
import hashlib
import os
from pathlib import Path
import tarfile
import types
from unittest.mock import patch

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pytest
import torch

import joint_production.conditional_rates as conditional
from joint_production.gaussian_increment_score import accumulated_increment_u_crps


ROOT = Path(__file__).resolve().parents[1]
MODE = "gaussian_report_integrated"
DAYS = np.array([0., 30., 60., 90., 180.])
ORIGINS = [0., 30., 60.]
SETTINGS = dict(epochs=2, hidden_width=8, batch_size=6, train_samples=4,
                seed=31, log_every=0, history_rate_features="last_interval",
                covariate_preprocessing="external", noise_regularization=.03)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def assert_same_prediction_law(sampled, integrated):
    assert "marginal_score_estimator" not in sampled
    assert integrated["marginal_score_estimator"] == MODE
    numerical_and_law_fields = dict(integrated)
    numerical_and_law_fields.pop("marginal_score_estimator")
    assert_nested_equal(sampled, numerical_and_law_fields)


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(913)
    amplitudes = rng.lognormal(0., .2, (12, 1, 3)) * [140., 510., 110.]
    reports = amplitudes * (1-np.exp(-DAYS[None, :, None]/130.))
    reports[:, 0] = 0.
    reports[0, 1, 0] = -.12500000000000003
    reports[0, 2, 1] = np.nan
    reports[1, 1, 2] = np.nan
    features = np.concatenate((rng.normal(size=(12, 17)), rng.integers(0, 2, (12, 17))), axis=1)
    return reports, features


def fit_with_normal_trace(cls, family, reports, features, extra=None):
    normals = []
    original = torch.randn

    def draw(*args, **kwargs):
        result = original(*args, **kwargs)
        normals.append(result.detach().cpu().clone())
        return result

    before = torch.get_rng_state().clone()
    with patch("torch.randn", side_effect=draw):
        model = cls(family, **SETTINGS, **(extra or {})).fit(reports, DAYS, origins=ORIGINS, covariates=features)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    return model, normals


@pytest.fixture(scope="module", params=["basis", "dca"])
def models(request, inputs):
    reports, features = inputs
    result = {"family": request.param}
    for name, extra in (("omitted", {}), ("sampled", {"marginal_score_estimator": "sampled"}),
                        ("integrated", {"marginal_score_estimator": MODE})):
        result[name] = fit_with_normal_trace(conditional.ConditionalRateForecaster,
                                             request.param, reports, features, extra)
    return result


def test_omitted_and_explicit_default_remain_bitwise_identical(models, inputs, tmp_path):
    omitted, draws = models["omitted"]
    explicit, explicit_draws = models["sampled"]
    assert omitted.marginal_score_estimator == explicit.marginal_score_estimator == "sampled"
    assert "marginal_score_estimator" not in omitted.configuration
    assert "marginal_score_estimator" not in omitted.fit_diagnostics_
    assert_nested_equal(omitted.configuration, explicit.configuration)
    assert_nested_equal(omitted.state_dict(), explicit.state_dict())
    assert_nested_equal(omitted.fit_diagnostics_, explicit.fit_diagnostics_)
    assert_nested_equal(draws, explicit_draws)
    reports, features = inputs
    rng_a, rng_b = np.random.default_rng(57), np.random.default_rng(57)
    assert_nested_equal(omitted.predict_distribution(reports[:2], 60, 12, rng_a, features[:2]),
                        explicit.predict_distribution(reports[:2], 60, 12, rng_b, features[:2]))
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    path = tmp_path / "default.pt"
    explicit.save(path)
    saved = torch.load(path, weights_only=True)
    assert "marginal_score_estimator" not in saved
    assert "marginal_score_estimator_meaning" not in saved


def test_integrated_mode_has_same_initial_parameters_context_masks_and_normal_streams(models):
    sampled, sampled_normals = models["sampled"]
    integrated, integrated_normals = models["integrated"]
    assert integrated.marginal_score_estimator == MODE
    assert integrated.configuration["marginal_score_estimator"] == MODE
    assert integrated.fit_diagnostics_["marginal_score_estimator"] == MODE
    for key in ("initial_parameter_sha256", "context_mask_sha256_by_epoch", "rng_seeds",
                "increment_scales_native", "increment_observation_counts", "phase_scales_native",
                "optimizer_updates", "total_training_contexts", "score_coefficients",
                "uniform_origin_marginal_weights", "missing_target_weighting"):
        assert sampled.fit_diagnostics_[key] == integrated.fit_diagnostics_[key]
    assert [(key, value.shape) for key, value in sampled.named_parameters()] == [
        (key, value.shape) for key, value in integrated.named_parameters()]
    assert sampled.state_dict().keys() == integrated.state_dict().keys()
    assert_nested_equal(sampled_normals, integrated_normals)
    assert sampled.observation_law == integrated.observation_law == "conditional_gaussian_accumulated_bridge"
    assert any(not torch.equal(value, sampled.state_dict()[name]) for name, value in integrated.state_dict().items())
    for row in integrated.fit_diagnostics_["optimization"]:
        assert np.isfinite([row[key] for key in ("loss", "macro_increment_crps", "joint_energy", "max_gradient_norm")]).all()


def controlled_fit(family, estimator, reports, features, device="cpu"):
    """Freeze updates to compare both modes on exactly the same production graph."""
    trace = {"normals": [], "contexts": [], "batches": [], "analytic_calls": 0, "gradient_checks": 0}
    cls = conditional.ConditionalRateForecaster
    model = cls(family, **dict(SETTINGS, epochs=1, batch_size=len(reports), device=device),
                marginal_score_estimator=estimator)
    old_context, old_sample = cls._encoder_context, cls._sample_reports
    old_u, old_analytic = conditional._u_scores, conditional.accumulated_increment_u_crps
    old_randn, old_backward, old_clip = torch.randn, torch.Tensor.backward, torch.nn.utils.clip_grad_norm_

    def encoder(self, *args, **kwargs):
        value = old_context(self, *args, **kwargs)
        trace["contexts"].append(value.detach().cpu().clone())
        return value

    def sample(self, q, history, days, a, r, **kwargs):
        value = old_sample(self, q, history, days, a, r, **kwargs)
        trace["current_q"] = q
        trace["current_history"] = history
        trace["batches"].append({"q": q.detach().cpu().clone(), "reports": value.detach().cpu().clone()})
        return value

    def u_score(target, draws, observed, days, origin_index, scales, weights):
        marginal, energy, dimension = old_u(target, draws, observed, days, origin_index, scales, weights)
        # These distinct offsets expose accidentally retaining sampled C or dropping E.
        marginal, energy = marginal + 17., energy + 5.
        trace["current_score"] = (target, observed, days, origin_index, scales, weights, marginal, energy)
        assert torch.equal(torch.isfinite(trace["current_history"]), observed)
        trace["batches"][-1].update(origin_index=origin_index, observed=observed.detach().cpu().clone(),
            energy=energy.detach().cpu().clone(), sampled_c=marginal.detach().cpu().clone())
        return marginal, energy, dimension

    def analytic(target, q, days, origin_index, a, r, scales, weights):
        actual, _, known_days, known_origin, known_scales, known_weights, _, _ = trace["current_score"]
        assert q is trace["current_q"] and target is actual and days is known_days
        assert origin_index == known_origin
        torch.testing.assert_close(scales, known_scales, rtol=0, atol=0)
        torch.testing.assert_close(weights, known_weights, rtol=0, atol=0)
        trace["analytic_calls"] += 1
        return old_analytic(target, q, days, origin_index, a, r, scales, weights)

    def draw(*args, **kwargs):
        value = old_randn(*args, **kwargs)
        trace["normals"].append(value.detach().cpu().clone())
        return value

    def backward(loss, *args, **kwargs):
        target, _, days, origin, scales, weights, sampled_c, energy = trace["current_score"]
        c = (accumulated_increment_u_crps(target, trace["current_q"], days, origin,
                model.noise_abs, model.noise_rel, scales, weights) if estimator == MODE else sampled_c)
        multiplier = float((DAYS > float(days[origin])).sum()) / np.mean([(DAYS > s).sum() for s in ORIGINS])
        penalty = .5*model.noise_regularization * (
            torch.log(model.noise_abs/model.initial_noise_abs).square().sum()
            + torch.log(model.noise_rel/model.initial_noise_rel).square().sum())
        expected = (c*multiplier+energy).mean()+penalty
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        trace["expected_gradients"] = torch.autograd.grad(expected, tuple(model.parameters()), retain_graph=True)
        trace["batches"][-1].update(loss=float(loss.detach()), weighted_c=float((c*multiplier).mean().detach()),
                                    energy_mean=float(energy.mean().detach()), penalty=float(penalty.detach()))
        return old_backward(loss, *args, **kwargs)

    def clip(parameters, *args, **kwargs):
        parameters = tuple(parameters)
        for parameter, expected in zip(parameters, trace["expected_gradients"], strict=True):
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-6, atol=2e-7)
        trace["gradient_checks"] += 1
        return old_clip(parameters, *args, **kwargs)

    with patch.object(cls, "_encoder_context", encoder), patch.object(cls, "_sample_reports", sample), \
            patch.object(conditional, "_u_scores", u_score), \
            patch.object(conditional, "accumulated_increment_u_crps", analytic), \
            patch("torch.randn", side_effect=draw), patch.object(torch.Tensor, "backward", backward), \
            patch("torch.nn.utils.clip_grad_norm_", side_effect=clip), \
            patch.object(torch.optim.Adam, "step", lambda self, *args, **kwargs: None):
        model.fit(reports, DAYS, origins=ORIGINS, covariates=features)
    for key in ("current_q", "current_history", "current_score", "expected_gradients"):
        trace.pop(key)
    return model, trace


@pytest.mark.parametrize("family", ["basis", "dca"])
def test_actual_training_objective_gradients_and_shared_report_streams(family, inputs):
    reports, features = inputs
    sampled, left = controlled_fit(family, "sampled", reports, features)
    integrated, right = controlled_fit(family, MODE, reports, features)
    assert left["analytic_calls"] == 0 and right["analytic_calls"] == len(ORIGINS)
    assert left["gradient_checks"] == right["gradient_checks"] == len(ORIGINS)
    assert_nested_equal(left["contexts"], right["contexts"])
    assert_nested_equal(left["normals"], right["normals"])
    assert len(left["normals"]) == 2*len(ORIGINS)  # One latent and one report bank per context batch.
    assert_nested_equal(sampled.state_dict(), integrated.state_dict())
    for old, new in zip(left["batches"], right["batches"], strict=True):
        for key in ("q", "reports", "observed", "energy", "sampled_c", "origin_index"):
            assert_nested_equal(old[key], new[key])
        assert old["weighted_c"] != new["weighted_c"]
    for model, trace in ((sampled, left), (integrated, right)):
        entry = model.fit_diagnostics_["optimization"][0]
        for diagnostic, key in (("loss", "loss"), ("macro_increment_crps", "weighted_c"),
                                ("joint_energy", "energy_mean"), ("noise_penalty", "penalty")):
            assert entry[diagnostic] == pytest.approx(np.mean([row[key] for row in trace["batches"]]), abs=1e-12)
    before, after = np.random.default_rng(43), np.random.default_rng(43)
    assert_same_prediction_law(sampled.predict_distribution(reports[:2], 60, 12, before, features[:2]),
                               integrated.predict_distribution(reports[:2], 60, 12, after, features[:2]))
    assert before.bit_generator.state == after.bit_generator.state


@pytest.mark.parametrize("origin", [0., 60.])
def test_integrated_training_preserves_prediction_law_missing_origin_and_cutoff_isolation(models, inputs, origin):
    model = models["integrated"][0]
    history = inputs[0][:2].copy(); history[:, DAYS > origin] = np.nan
    history[:, DAYS > 0, 1] = np.nan
    if origin > 0:
        history[0, DAYS == origin, 0] = np.nan
    poison = history.copy(); poison[:, DAYS > origin] = np.inf
    rng_a, rng_b = np.random.default_rng(71), np.random.default_rng(71)
    first = model.predict_distribution(history, origin, 32, rng_a, inputs[1][:2])
    second = model.predict_distribution(poison, origin, 32, rng_b, inputs[1][:2])
    assert_nested_equal(first, second)
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    known = np.isfinite(history) & (DAYS[None, :, None] <= origin)
    mask = np.broadcast_to(known[:, None], first["report_cumulative"].shape)
    np.testing.assert_array_equal(first["report_cumulative"][mask],
        np.broadcast_to(history[:, None], first["report_cumulative"].shape)[mask])
    assert np.isfinite(first["report_cumulative"]).all()
    if origin > 0:
        oi = int(np.flatnonzero(DAYS == origin)[0])
        assert first["report_cumulative"][0, :, oi, 0].std() > 0
        assert first["report_cumulative"][0, :, oi, 1].std() > 0
    assert first["marginal_score_estimator"] == MODE
    assert "marginal_score_estimator" not in first["conditional_diagnostics"]
    # At the identical trained state, the estimator option has no prediction effect.
    same_state = copy.deepcopy(model); same_state.marginal_score_estimator = "sampled"
    replay_rng = np.random.default_rng(71)
    assert_same_prediction_law(same_state.predict_distribution(history, origin, 32, replay_rng, inputs[1][:2]), first)


def test_integrated_mode_checkpoint_roundtrip_preserves_metadata_state_and_issued_paths(models, inputs, tmp_path):
    model = models["integrated"][0]
    path = tmp_path / "integrated.pt"; model.save(path)
    saved = torch.load(path, weights_only=True)
    for metadata in (saved, saved["configuration"], saved["fit_diagnostics"], saved["fit_diagnostics"]["configuration"]):
        assert metadata["marginal_score_estimator"] == MODE
    assert saved["marginal_score_estimator_meaning"] == saved["fit_diagnostics"]["marginal_score_estimator_meaning"]
    restored = conditional.ConditionalRateForecaster.load(path)
    assert restored.marginal_score_estimator == MODE
    assert_nested_equal(restored.state_dict(), model.state_dict())
    assert_nested_equal(restored.fit_diagnostics_, model.fit_diagnostics_)
    for origin in (0., 60.):
        a, b = np.random.default_rng(97), np.random.default_rng(97)
        assert_nested_equal(model.predict_distribution(inputs[0][:2], origin, 12, a, inputs[1][:2]),
                            restored.predict_distribution(inputs[0][:2], origin, 12, b, inputs[1][:2]))
        assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize("mutation", ["top_mode", "missing_top", "configuration", "fit_mode", "nested_configuration", "missing_nested", "top_meaning", "fit_meaning"])
def test_checkpoint_rejects_inconsistent_estimator_metadata_including_nested_configuration(models, tmp_path, mutation):
    path = tmp_path / "corrupted.pt"; models["integrated"][0].save(path)
    saved = torch.load(path, weights_only=True)
    saved["fit_diagnostics"]["configuration"] = copy.deepcopy(saved["fit_diagnostics"]["configuration"])
    if mutation == "top_mode": saved["marginal_score_estimator"] = "sampled"
    elif mutation == "missing_top": saved.pop("marginal_score_estimator")
    elif mutation == "configuration": saved["configuration"].pop("marginal_score_estimator")
    elif mutation == "fit_mode": saved["fit_diagnostics"]["marginal_score_estimator"] = "sampled"
    elif mutation == "nested_configuration": saved["fit_diagnostics"]["configuration"]["marginal_score_estimator"] = "sampled"
    elif mutation == "missing_nested": saved["fit_diagnostics"]["configuration"].pop("marginal_score_estimator")
    elif mutation == "top_meaning": saved["marginal_score_estimator_meaning"] = "another objective"
    else: saved["fit_diagnostics"]["marginal_score_estimator_meaning"] = "another objective"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="estimator|score"):
        conditional.ConditionalRateForecaster.load(path)


@pytest.mark.parametrize("extra", [
    {"marginal_score_estimator": "unknown"}, {"marginal_score_estimator": None},
    {"marginal_score_estimator": True},
    {"marginal_score_estimator": MODE, "mixture_mode": "tied"},
    {"marginal_score_estimator": MODE, "mixture_mode": "two"},
    {"marginal_score_estimator": MODE, "report_half_life_days": 180.},
])
def test_invalid_estimator_mixture_or_report_kernel_is_rejected_before_fit(extra):
    with pytest.raises(ValueError, match="marginal_score_estimator|accumulated|mixture"):
        conditional.ConditionalRateForecaster(**dict(SETTINGS, **extra))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only training objective and checkpoint replay")
@pytest.mark.parametrize("family", ["basis", "dca"])
def test_cuda_initialization_objective_and_checkpoint_replay(family, inputs, tmp_path):
    reports, features = inputs
    sampled, left = controlled_fit(family, "sampled", reports, features, device="cuda")
    integrated, right = controlled_fit(family, MODE, reports, features, device="cuda")
    assert_nested_equal(sampled.state_dict(), integrated.state_dict())
    assert_nested_equal(left["normals"], right["normals"])
    assert right["gradient_checks"] == len(ORIGINS)
    path = tmp_path / "cuda.pt"; integrated.save(path)
    restored = conditional.ConditionalRateForecaster.load(path, device="cuda")
    assert restored.device.type == "cuda" and restored.marginal_score_estimator == MODE
    a, b = np.random.default_rng(37), np.random.default_rng(37)
    assert_nested_equal(integrated.predict_distribution(reports[:2], 60, 12, a, features[:2]),
                        restored.predict_distribution(reports[:2], 60, 12, b, features[:2]))
    assert a.bit_generator.state == b.bit_generator.state
