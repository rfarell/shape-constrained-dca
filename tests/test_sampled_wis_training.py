"""Synthetic fixed-bank training objectives, stream parity, and checkpoint checks."""
import copy
import hashlib
import os
from pathlib import Path
import tarfile
import types
from unittest.mock import patch

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pytest
import torch

import joint_production.conditional_rates as conditional
from joint_production.metrics import weighted_interval_score


ROOT = Path(__file__).resolve().parents[1]
DAYS = np.array([0., 30., 60., 90., 180.])
ORIGINS = [0., 60.]
SETTINGS = dict(epochs=1, hidden_width=6, batch_size=4, train_samples=4, seed=37,
                log_every=0, history_rate_features="last_interval", covariate_preprocessing="external",
                history_dropout=.25, phase_dropout=.2, noise_regularization=.07)
SCORE_KEYS = ("marginal_objective", "energy_samples", "marginal_objective_meaning", "energy_sample_selection")


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def bank_inputs(dtype=torch.float64, device="cpu"):
    target = torch.tensor([
        [[0., 0., 0.], [.7, -.25, 1.2], [1.1, 1.8, 1.5], [.9, 2.5, 2.7], [3.1, 4., 3.5]],
        [[0., 0., 0.], [.9, 1.3, .7], [1.7, 1.4, float("nan")], [2.1, 2.7, 1.9], [3.7, 3.1, 2.4]],
    ], dtype=dtype, device=device)
    index = torch.arange(2 * 256 * 5 * 3, dtype=dtype, device=device).reshape(2, 256, 5, 3)
    samples = torch.sin(index * .41) * .9 + torch.arange(5, dtype=dtype, device=device)[None, None, :, None] * .6
    times = torch.tensor(DAYS, dtype=dtype, device=device)
    observed = torch.isfinite(target) & (times <= 60)[None, :, None]
    observed[0, 2, 0] = False  # Hidden raw-finite origin remains a marginal target.
    observed[0, 1, 2] = False  # Earlier history hole belongs to energy only.
    observed[1, 1:3, 1] = False
    samples = torch.where(observed[:, None], target[:, None], samples)
    scales = torch.tensor([[1.2, 2.3, .7], [2.1, 3.4, 1.7]], dtype=dtype, device=device)
    weights = torch.tensor([[1.5, 1., 2.], [1., 1.25, 0.]], dtype=dtype, device=device)
    return target, samples, observed, times, 2, scales, weights


def explicit_scores(target, samples, observed, times, origin, scales, weights, objective):
    """Full pairwise marginal and vector-energy oracle, independent of sorting."""
    future = times > times[origin]
    finite = torch.isfinite(target)
    mask = finite[:, future] & finite[:, origin, None]
    clean = torch.where(finite, target, torch.zeros_like(target))
    actual = torch.where(mask, clean[:, future] - clean[:, origin, None], torch.zeros_like(clean[:, future])) / scales
    increments = (samples[:, :, future] - samples[:, :, origin, None]) / scales
    if objective == "crps":
        first, second = torch.triu_indices(samples.shape[1], samples.shape[1], offset=1, device=samples.device)
        cells = (increments - actual[:, None]).abs().mean(1)
        cells = cells - (increments[:, first] - increments[:, second]).abs().sum(1) / (samples.shape[1] * (samples.shape[1] - 1))
    else:
        # Derive the linear order interpolation explicitly, not torch.quantile.
        ordered = increments.sort(dim=1).values
        cells = torch.zeros_like(actual)
        for probability in (.025, .1, .5, .9, .975):
            rank = probability * (samples.shape[1] - 1)
            lower, upper = int(np.floor(rank)), int(np.ceil(rank))
            quantile = ordered[:, lower] * (upper - rank) + ordered[:, upper] * (rank - lower) if lower != upper else ordered[:, lower]
            error = actual - quantile
            cells = cells + probability * error.clamp_min(0) + (1 - probability) * (-error).clamp_min(0)
        cells = cells / 2.5
    marginal = (torch.where(mask, cells, torch.zeros_like(cells)) * weights).mean(dim=(-2, -1))
    scored = finite & ~observed & (times > 0)[None, :, None]
    residual = torch.where(scored[:, None], samples[:, :16] - clean[:, None], torch.zeros_like(samples[:, :16])).flatten(2)
    first, second = torch.triu_indices(16, 16, offset=1, device=samples.device)
    energy = torch.linalg.vector_norm(residual, dim=-1).mean(1)
    energy = energy - torch.linalg.vector_norm(residual[:, first] - residual[:, second], dim=-1).sum(1) / 240
    dimension = scored.sum(dim=(-2, -1))
    energy = energy / dimension.clamp_min(1).to(energy.dtype).sqrt()
    return marginal, energy, dimension


@pytest.mark.parametrize("objective", ["crps", "empirical_wis"])
def test_fixed_bank_values_and_gradients_match_full_pair_and_numpy_oracles(objective):
    arguments = list(bank_inputs())
    arguments[0].requires_grad_()
    arguments[1].requires_grad_()
    actual = conditional._sampled_bank_scores(*arguments, marginal_objective=objective)
    expected = explicit_scores(*arguments, objective)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=5e-14, atol=5e-14)
    target, samples, observed, times, origin, scales, weights = arguments
    for coordinate in (0, 1):
        left = torch.autograd.grad(actual[coordinate].sum(), (target, samples), retain_graph=True)
        right = torch.autograd.grad(expected[coordinate].sum(), (target, samples), retain_graph=True)
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, rtol=5e-13, atol=5e-13)
        assert torch.isfinite(left[0]).all() and torch.isfinite(left[1]).all()
        if coordinate == 1:
            assert torch.all(left[1][:, 16:] == 0)
    if objective == "empirical_wis":
        y, x, c, w = [value.detach().numpy() for value in (target, samples, scales, weights)]
        increments = (y[:, 3:] - y[:, 2, None]) / c
        sample_increments = (x[:, :, 3:] - x[:, :, 2, None]) / c
        score = weighted_interval_score(increments, sample_increments)
        oracle = np.where(np.isfinite(increments), score, 0.) * w
        np.testing.assert_allclose(actual[0].detach().numpy(), oracle.mean(axis=(-2, -1)), rtol=5e-14, atol=5e-14)
    # A missing water origin suppresses marginal water; finite future water still enters energy.
    assert actual[2][1] == 8


@pytest.mark.parametrize("objective", ["crps", "empirical_wis"])
def test_hidden_origin_cancels_from_same_draw_marginal_but_not_report_energy(objective):
    arguments = list(bank_inputs())
    original = conditional._sampled_bank_scores(*arguments, marginal_objective=objective)
    changed = arguments[1].clone()
    shift = torch.linspace(-1., 2., 256, dtype=changed.dtype)
    changed[0, :, 2:, 0] += shift[:, None]
    shifted = conditional._sampled_bank_scores(arguments[0], changed, *arguments[2:], marginal_objective=objective)
    torch.testing.assert_close(shifted[0], original[0], rtol=5e-14, atol=5e-14)
    assert not torch.equal(shifted[1], original[1])
    samples = arguments[1].detach().clone().requires_grad_()
    marginal = conditional._sampled_bank_scores(arguments[0], samples, *arguments[2:], marginal_objective=objective)[0]
    gradient = torch.autograd.grad(marginal.sum(), samples)[0]
    torch.testing.assert_close(gradient[:, :, 2], -gradient[:, :, 3:].sum(2), rtol=0, atol=2e-16)
    assert torch.all(gradient[:, :, :2] == 0)
    assert torch.all(gradient[1, :, :, 2] == 0)


def test_all_256_draws_score_wis_but_only_first_16_score_energy(monkeypatch):
    target = torch.zeros((1, 2, 3), dtype=torch.float64)
    samples = torch.zeros((1, 256, 2, 3), dtype=torch.float64)
    samples[:, 16:, 1] = 10.
    observed = torch.tensor([[[True] * 3, [False] * 3]])
    times = torch.tensor([0., 30.], dtype=torch.float64)
    scales = weights = torch.ones((1, 3), dtype=torch.float64)
    calls = []
    original = conditional._u_scores
    def energy_bank(y, draws, *args):
        calls.append(draws)
        assert draws.shape[1] == 16
        assert draws.untyped_storage().data_ptr() == samples.untyped_storage().data_ptr()
        return original(y, draws, *args)
    monkeypatch.setattr(conditional, "_u_scores", energy_bank)
    marginal, energy, dimension = conditional._sampled_bank_scores(
        target, samples, observed, times, 0, scales, weights, marginal_objective="empirical_wis")
    torch.testing.assert_close(marginal, torch.tensor([6.1], dtype=torch.float64), rtol=0, atol=2e-15)
    torch.testing.assert_close(energy, torch.zeros(1, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(dimension, torch.tensor([3]))
    assert len(calls) == 1


@pytest.mark.parametrize("objective", ["crps", "empirical_wis"])
@pytest.mark.parametrize("failure", ["target_subtraction", "sample_subtraction", "normalization"])
def test_active_overflow_cannot_be_reclassified_as_missing_target(objective, failure):
    target, samples, observed, times, origin, scales, weights = bank_inputs()
    if failure == "target_subtraction":
        target[0, origin, 0] = -.75 * torch.finfo(target.dtype).max
        target[0, 3:, 0] = .75 * torch.finfo(target.dtype).max
    elif failure == "sample_subtraction":
        samples[0, :, origin, 0] = -.75 * torch.finfo(samples.dtype).max
        samples[0, :, 3:, 0] = .75 * torch.finfo(samples.dtype).max
    else:
        scales[0, 0] = 1e-320
    with pytest.raises(FloatingPointError, match="subtraction|normalization"):
        conditional._sampled_bank_scores(target, samples, observed, times, origin, scales, weights,
                                          marginal_objective=objective)


@pytest.fixture(scope="module")
def inputs():
    amplitude = np.array([[140., 500., 110.], [165., 620., 130.], [190., 710., 160.], [120., 440., 95.]])
    reports = amplitude[:, None] * (1 - np.exp(-DAYS[None, :, None] / 130.))
    reports[0, 1, 0] = -.12500000000000003
    reports[0, 2, 1] = np.nan
    reports[1, 1, 2] = np.nan
    features = np.arange(4 * 34, dtype=float).reshape(4, 34) / 100
    return reports, features


def fit_default(cls, family, inputs, explicit=False):
    reports, features = inputs
    normals = []
    original = torch.randn
    def draw(*args, **kwargs):
        result = original(*args, **kwargs)
        normals.append(result.detach().cpu().clone())
        return result
    extra = {"marginal_objective": "crps", "energy_samples": None} if explicit else {}
    before = torch.get_rng_state().clone()
    with patch("torch.randn", side_effect=draw):
        model = cls(family, **SETTINGS, **extra).fit(reports, DAYS, origins=ORIGINS, covariates=features)
    equal(torch.get_rng_state(), before)
    return model, normals


@pytest.fixture(scope="module", params=["basis", "dca"])
def defaults(request, inputs):
    return request.param, fit_default(conditional.ConditionalRateForecaster, request.param, inputs), \
        fit_default(conditional.ConditionalRateForecaster, request.param, inputs, explicit=True)


def test_omitted_and_explicit_defaults_preserve_config_state_streams_and_prediction(defaults, inputs, tmp_path):
    _, (omitted, normals), (explicit, explicit_normals) = defaults
    equal(omitted.configuration, explicit.configuration)
    equal(omitted.state_dict(), explicit.state_dict())
    equal(omitted.fit_diagnostics_, explicit.fit_diagnostics_)
    equal(normals, explicit_normals)
    assert omitted.marginal_objective == "crps" and omitted.energy_samples is None
    for container in (omitted.configuration, omitted.fit_diagnostics_):
        assert not any(key in container for key in SCORE_KEYS)
    first, second = np.random.default_rng(73), np.random.default_rng(73)
    left = omitted.predict_distribution(inputs[0][:2], 60, 9, first, inputs[1][:2])
    right = explicit.predict_distribution(inputs[0][:2], 60, 9, second, inputs[1][:2])
    equal(left, right)
    assert not any(key in left for key in SCORE_KEYS)
    assert first.bit_generator.state == second.bit_generator.state
    path = tmp_path / "default.pt"
    omitted.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert not any(key in checkpoint for key in SCORE_KEYS)


def controlled_fit(family, objective, inputs, device="cpu"):
    """Audit the real fit graph while freezing Adam updates for stream pairing."""
    cls = conditional.ConditionalRateForecaster
    model = cls(family, **dict(SETTINGS, train_samples=256, device=device),
                marginal_objective=objective, energy_samples=16)
    trace = {"contexts": [], "normals": [], "rng_states_after": [], "batches": [], "gradient_checks": 0}
    original_init, original_context, original_sample = cls._initialize, cls._encoder_context, cls._sample_reports
    original_score, original_draw = conditional._sampled_bank_scores, torch.randn
    original_backward, original_clip = torch.Tensor.backward, torch.nn.utils.clip_grad_norm_

    def initialize(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        # Move noise away from its penalty center, so its loss and gradient matter.
        with torch.no_grad():
            self.noise_abs_raw.add_(.3)
            self.noise_rel_raw.sub_(.2)
        return result

    def context(self, *args, **kwargs):
        value = original_context(self, *args, **kwargs)
        trace["contexts"].append(value.detach().cpu().clone())
        return value

    def sample(self, cumulative, history, *args, **kwargs):
        value = original_sample(self, cumulative, history, *args, **kwargs)
        trace["current_history"] = history
        trace["batches"].append({"latent": cumulative.detach().cpu().clone(), "reports": value.detach().cpu().clone()})
        return value

    def score(*args, **kwargs):
        assert kwargs == {"marginal_objective": objective} and args[1].shape[1] == 256
        value = original_score(*args, **kwargs)
        trace["current_arguments"] = args
        equal(torch.isfinite(trace["current_history"]), args[2])
        trace["batches"][-1].update(observed=args[2].detach().cpu().clone(), origin_index=args[4])
        return value

    def draw(*args, **kwargs):
        value = original_draw(*args, **kwargs)
        trace["normals"].append(value.detach().cpu().clone())
        trace["rng_states_after"].append(kwargs["generator"].get_state().cpu().clone())
        return value

    def backward(loss, *args, **kwargs):
        arguments = trace["current_arguments"]
        marginal, energy, _ = explicit_scores(*arguments, objective)
        origin_day = float(arguments[3][arguments[4]])
        origin_weight = float((DAYS > origin_day).sum()) / np.mean([(DAYS > origin).sum() for origin in ORIGINS])
        penalty = .5 * model.noise_regularization * (
            torch.log(model.noise_abs / model.initial_noise_abs).square().sum()
            + torch.log(model.noise_rel / model.initial_noise_rel).square().sum())
        assert penalty.detach().item() > .005
        expected = (marginal * origin_weight + energy).mean() + penalty
        torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-7)
        trace["expected_gradients"] = torch.autograd.grad(expected, tuple(model.parameters()), retain_graph=True)
        trace["batches"][-1].update(loss=float(expected.detach()), marginal=float((marginal * origin_weight).mean().detach()),
                                    energy=float(energy.mean().detach()), penalty=float(penalty.detach()))
        return original_backward(loss, *args, **kwargs)

    def clip(parameters, *args, **kwargs):
        parameters = tuple(parameters)
        for parameter, expected in zip(parameters, trace["expected_gradients"], strict=True):
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-6)
        trace["gradient_checks"] += 1
        return original_clip(parameters, *args, **kwargs)

    before = torch.get_rng_state().clone()
    with patch.object(cls, "_initialize", initialize), patch.object(cls, "_encoder_context", context), \
            patch.object(cls, "_sample_reports", sample), patch.object(conditional, "_sampled_bank_scores", score), \
            patch("torch.randn", side_effect=draw), patch.object(torch.Tensor, "backward", backward), \
            patch("torch.nn.utils.clip_grad_norm_", side_effect=clip), \
            patch.object(torch.optim.Adam, "step", lambda self, *args, **kwargs: None):
        model.fit(inputs[0], DAYS, origins=ORIGINS, covariates=inputs[1])
    equal(torch.get_rng_state(), before)
    for key in ("current_history", "current_arguments", "expected_gradients"):
        trace.pop(key)
    return model, trace


@pytest.fixture(scope="module", params=["basis", "dca"])
def paired(request, inputs):
    return {"family": request.param,
            **{objective: controlled_fit(request.param, objective, inputs) for objective in ("crps", "empirical_wis")}}


def test_controlled_real_fit_gradients_and_identical_context_normal_streams(paired):
    crps, left = paired["crps"]
    wis, right = paired["empirical_wis"]
    equal(crps.state_dict(), wis.state_dict())
    equal(left["contexts"], right["contexts"])
    equal(left["normals"], right["normals"])
    equal(left["rng_states_after"], right["rng_states_after"])
    assert len(left["normals"]) == 2 * len(ORIGINS)
    assert left["gradient_checks"] == right["gradient_checks"] == len(ORIGINS)
    for key in ("initial_parameter_sha256", "context_mask_sha256_by_epoch", "rng_seeds",
                "increment_scales_native", "increment_observation_counts", "phase_scales_native",
                "optimizer_updates", "total_training_contexts", "score_coefficients",
                "uniform_origin_marginal_weights", "missing_target_weighting"):
        equal(crps.fit_diagnostics_[key], wis.fit_diagnostics_[key])
    for old, new in zip(left["batches"], right["batches"], strict=True):
        for key in ("latent", "reports", "observed", "origin_index", "energy", "penalty"):
            equal(old[key], new[key])
        assert old["marginal"] != new["marginal"]
    for objective in ("crps", "empirical_wis"):
        model, trace = paired[objective]
        row = model.fit_diagnostics_["optimization"][0]
        score_key = "macro_increment_crps" if objective == "crps" else "macro_increment_wis"
        other = "macro_increment_wis" if objective == "crps" else "macro_increment_crps"
        assert other not in row
        assert row["marginal_objective"] == objective and row["energy_samples"] == 16
        for recorded, expected in ((score_key, "marginal"), ("joint_energy", "energy"),
                                   ("noise_penalty", "penalty"), ("loss", "loss")):
            assert row[recorded] == pytest.approx(np.mean([batch[expected] for batch in trace["batches"]]), rel=2e-6, abs=2e-7)


def test_same_trained_state_issues_same_law_with_only_training_metadata_changed(paired, inputs):
    crps, wis = paired["crps"][0], paired["empirical_wis"][0]
    history = inputs[0][:2].copy()
    history[:, DAYS > 0, 1] = np.nan
    history[0, DAYS == 60, 0] = np.nan
    poison = history.copy()
    poison[:, DAYS > 60] = np.inf
    rngs = [np.random.default_rng(97) for _ in range(3)]
    first = crps.predict_distribution(history, 60, 23, rngs[0], inputs[1][:2])
    second = wis.predict_distribution(history, 60, 23, rngs[1], inputs[1][:2])
    poisoned = wis.predict_distribution(poison, 60, 23, rngs[2], inputs[1][:2])
    equal(second, poisoned)
    for model, result in ((crps, first), (wis, second)):
        for key, value in model._training_score_metadata().items():
            assert result[key] == value
        assert not any(key in result["conditional_diagnostics"] for key in SCORE_KEYS)
    equal({key: value for key, value in first.items() if key not in SCORE_KEYS},
          {key: value for key, value in second.items() if key not in SCORE_KEYS})
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state == rngs[2].bit_generator.state
    known = np.isfinite(history) & (DAYS[None, :, None] <= 60)
    for row in range(2):
        np.testing.assert_array_equal(second["report_cumulative"][row][:, known[row]],
                                      np.broadcast_to(history[row][known[row]], (23, known[row].sum())))
    assert second["report_cumulative"][0, :, 2, 0].std() > 0


@pytest.mark.parametrize("objective", ["crps", "empirical_wis"])
def test_fixed_bank_checkpoint_metadata_and_prediction_roundtrip(paired, inputs, tmp_path, objective):
    model = paired[objective][0]
    path = tmp_path / "fixed_bank.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    expected = {"marginal_objective": objective, "energy_samples": 16,
                "marginal_objective_meaning": conditional.SAMPLED_OBJECTIVE_MEANINGS[objective],
                "energy_sample_selection": "first_16_of_same_256_iid_joint_report_trajectories"}
    assert model._training_score_metadata() == expected
    for metadata in (checkpoint, checkpoint["fit_diagnostics"]):
        for key, value in expected.items():
            assert metadata[key] == value
    for config in (checkpoint["configuration"], checkpoint["fit_diagnostics"]["configuration"]):
        assert config["marginal_objective"] == objective and config["energy_samples"] == 16
        assert config["train_samples"] == 256
    restored = conditional.ConditionalRateForecaster.load(path)
    equal(model.state_dict(), restored.state_dict())
    equal(model.fit_diagnostics_, restored.fit_diagnostics_)
    a, b = np.random.default_rng(103), np.random.default_rng(103)
    equal(model.predict_distribution(inputs[0][:2], 60, 11, a, inputs[1][:2]),
          restored.predict_distribution(inputs[0][:2], 60, 11, b, inputs[1][:2]))
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize("mutation", ["top_mode", "top_missing", "top_energy", "fit_mode", "nested_mode",
    "nested_energy", "configuration_mode", "top_meaning", "fit_selection", "objective", "draw_count", "log_label"])
def test_checkpoint_rejects_tampered_score_contract_including_nested_config(paired, tmp_path, mutation):
    path = tmp_path / "tampered.pt"
    paired["empirical_wis"][0].save(path)
    saved = torch.load(path, weights_only=True)
    saved["fit_diagnostics"]["configuration"] = copy.deepcopy(saved["fit_diagnostics"]["configuration"])
    if mutation == "top_mode": saved["marginal_objective"] = "crps"
    elif mutation == "top_missing": saved.pop("marginal_objective")
    elif mutation == "top_energy": saved["energy_samples"] = 256
    elif mutation == "fit_mode": saved["fit_diagnostics"]["marginal_objective"] = "crps"
    elif mutation == "nested_mode": saved["fit_diagnostics"]["configuration"]["marginal_objective"] = "crps"
    elif mutation == "nested_energy": saved["fit_diagnostics"]["configuration"].pop("energy_samples")
    elif mutation == "configuration_mode": saved["configuration"]["marginal_objective"] = "crps"
    elif mutation == "top_meaning": saved["marginal_objective_meaning"] = "changed"
    elif mutation == "fit_selection": saved["fit_diagnostics"]["energy_sample_selection"] = "new_16_draws"
    elif mutation == "objective": saved["fit_diagnostics"]["objective"] = "changed"
    elif mutation == "draw_count": saved["fit_diagnostics"]["train_samples_iid"] = 16
    else:
        saved["fit_diagnostics"]["optimization"][0]["macro_increment_crps"] = 0.
    torch.save(saved, path)
    with pytest.raises(ValueError, match="objective|energy|score|sample count"):
        conditional.ConditionalRateForecaster.load(path)


@pytest.mark.parametrize("extra", [
    {"marginal_objective": "unknown"}, {"marginal_objective": None},
    {"marginal_objective": "empirical_wis", "energy_samples": None},
    {"energy_samples": 8}, {"energy_samples": True}, {"energy_samples": 16., "train_samples": 256},
    {"energy_samples": 16, "train_samples": 16}, {"energy_samples": 16, "train_samples": 256.},
    {"mixture_mode": "two"}, {"history_head_mode": "no_history"},
    {"coefficient_parameterization": "log_expectation"}, {"tree_static_mode": "fixed_empty_history"},
    {"earlier_history_features": "zero"}, {"report_half_life_days": 180.},
    {"history_rate_features": "none"}, {"marginal_score_estimator": "gaussian_report_integrated"},
])
def test_invalid_fixed_bank_modes_fail_before_fit(extra):
    configuration = dict(SETTINGS, train_samples=256, marginal_objective="crps", energy_samples=16)
    configuration.update(extra)
    with pytest.raises(ValueError):
        conditional.ConditionalRateForecaster(**configuration)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only controlled objective and checkpoint replay")
@pytest.mark.parametrize("family", ["basis", "dca"])
def test_cuda_controlled_objective_streams_and_checkpoint_replay(family, inputs, tmp_path):
    crps, first = controlled_fit(family, "crps", inputs, device="cuda")
    wis, second = controlled_fit(family, "empirical_wis", inputs, device="cuda")
    equal(crps.state_dict(), wis.state_dict())
    equal(first["normals"], second["normals"])
    equal(first["rng_states_after"], second["rng_states_after"])
    assert first["gradient_checks"] == second["gradient_checks"] == len(ORIGINS)
    path = tmp_path / "cuda_wis.pt"
    wis.save(path)
    restored = conditional.ConditionalRateForecaster.load(path, device="cuda")
    a, b = np.random.default_rng(109), np.random.default_rng(109)
    equal(wis.predict_distribution(inputs[0][:2], 60, 11, a, inputs[1][:2]),
          restored.predict_distribution(inputs[0][:2], 60, 11, b, inputs[1][:2]))
    assert a.bit_generator.state == b.bit_generator.state
