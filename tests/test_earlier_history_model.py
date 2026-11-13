"""Synthetic integration and legacy-law checks for earlier interval history."""
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


ROOT = Path(__file__).resolve().parents[1]
DAYS = np.array([0., 30., 60., 90., 180., 360., 720., 1080.])
ORIGINS = [0., 90., 360.]
WIDTH = 45
END_DAYS = [30, 60, 90, 180, 360]
LAYOUT = [f"{day}:{phase}:{feature}" for day in END_DAYS for phase in ("oil", "gas", "water")
          for feature in ("asinh_signed_rate", "duration_360", "availability")]
SETTINGS = dict(epochs=2, hidden_width=8, batch_size=6, train_samples=4,
                seed=31, log_every=0, history_rate_features="last_interval",
                covariate_preprocessing="external")


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


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(811)
    amplitudes = rng.lognormal(0., .2, (12, 1, 3))*[160., 620., 130.]
    reports = amplitudes*(1-np.exp(-DAYS[None, :, None]/210.))
    reports[:, 0] = 0.
    reports[0, 1, 0] = -.12500000000000003
    reports[0, 2, 1] = np.nan
    reports[1, 3, 2] = np.nan
    features = np.column_stack((rng.normal(size=(12, 17)), rng.integers(0, 2, (12, 17))))
    return reports, features


def initial(family, mode, inputs, device="cpu"):
    model = conditional.ConditionalRateForecaster(family, **SETTINGS,
                earlier_history_features=mode, device=device)
    reports, features = inputs
    model._covariates(features, len(features), fit=True)
    model._initialize(DAYS, [160., 620., 130.], np.ones((sum((DAYS > o).sum() for o in ORIGINS), 3)))
    model.origins_ = np.asarray(ORIGINS)
    model.fit_diagnostics_ = {"synthetic_initial_law": True}
    return model


def fit_with_trace(cls, family, mode, inputs):
    reports, features = inputs
    normals, extra_gradients = [], []
    original_draw, original_clip = torch.randn, torch.nn.utils.clip_grad_norm_
    extra = {} if mode is None else {"earlier_history_features": mode}
    model = cls(family, **SETTINGS, **extra)

    def draw(*args, **kwargs):
        result = original_draw(*args, **kwargs)
        normals.append(result.detach().cpu().clone())
        return result

    def clip(parameters, *args, **kwargs):
        if mode in {"actual", "zero"}:
            extra_gradients.append(model.network.hidden[0].weight.grad[:, -WIDTH:].detach().cpu().clone())
        return original_clip(parameters, *args, **kwargs)

    before = torch.get_rng_state().clone()
    with patch("torch.randn", side_effect=draw), patch("torch.nn.utils.clip_grad_norm_", side_effect=clip):
        model.fit(reports, DAYS, origins=ORIGINS, covariates=features)
    equal(torch.get_rng_state(), before)
    return model, normals, extra_gradients


@pytest.fixture(scope="module", params=["basis", "dca"])
def models(request, inputs):
    result = {"family": request.param}
    for mode in (None, "none", "zero", "actual"):
        result[mode] = fit_with_trace(conditional.ConditionalRateForecaster, request.param, mode, inputs)
    return result


def test_omitted_default_and_explicit_none_preserve_model_state_and_random_streams(models, inputs):
    omitted, normals, _ = models[None]
    explicit, explicit_normals, _ = models["none"]
    assert omitted.earlier_history_features == explicit.earlier_history_features == "none"
    assert "earlier_history_features" not in omitted.configuration
    assert "earlier_history_features" not in omitted.fit_diagnostics_
    equal(omitted.configuration, explicit.configuration)
    equal(omitted.state_dict(), explicit.state_dict())
    equal(omitted.fit_diagnostics_, explicit.fit_diagnostics_)
    equal(normals, explicit_normals)
    a, b = np.random.default_rng(37), np.random.default_rng(37)
    equal(omitted.predict_distribution(inputs[0][:2], 90, 12, a, inputs[1][:2]),
          explicit.predict_distribution(inputs[0][:2], 90, 12, b, inputs[1][:2]))
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize("family", ["basis", "dca"])
def test_zero_padding_preserves_common_initialization_and_actual_zero_full_state(family, inputs):
    before = torch.get_rng_state().clone()
    old, zero, actual = [initial(family, mode, inputs) for mode in ("none", "zero", "actual")]
    equal(torch.get_rng_state(), before)
    assert old.network.hidden[0].in_features == 105
    assert zero.network.hidden[0].in_features == actual.network.hidden[0].in_features == 150
    assert actual.initial_parameter_sha256_ == zero.initial_parameter_sha256_
    assert actual.common_initial_parameter_sha256_ == zero.common_initial_parameter_sha256_ == old.initial_parameter_sha256_
    equal(actual.state_dict(), zero.state_dict())
    common = dict(old.named_parameters())
    for name, parameter in actual.named_parameters():
        if name == "network.hidden.0.weight":
            equal(parameter[:, :-WIDTH], common[name])
            assert torch.count_nonzero(parameter[:, -WIDTH:]) == 0
        else:
            equal(parameter, common[name])
    history, features = inputs
    for origin in ORIGINS:
        parameters = [model.conditional_parameters(history[:2], origin, features[:2]) for model in (old, zero, actual)]
        equal(parameters[0], parameters[1]); equal(parameters[0], parameters[2])
        predictions = []
        rng_states = []
        for model in (old, zero, actual):
            rng = np.random.default_rng(71)
            predictions.append(model.predict_distribution(history[:2], origin, 12, rng, features[:2]))
            rng_states.append(rng.bit_generator.state)
        assert rng_states[0] == rng_states[1] == rng_states[2]
        for key in ("latent_parameters", "latent_rate", "latent_cumulative", "latent_total_volume", "report_cumulative"):
            equal(predictions[0][key], predictions[1][key]); equal(predictions[0][key], predictions[2][key])
        equal(predictions[0]["conditional_diagnostics"], predictions[1]["conditional_diagnostics"])
        equal(predictions[0]["conditional_diagnostics"], predictions[2]["conditional_diagnostics"])


def test_paired_feature_training_keeps_masks_scales_and_normals_but_only_actual_columns_learn(models):
    zero, zero_normals, zero_gradients = models["zero"]
    actual, actual_normals, actual_gradients = models["actual"]
    for key in ("initial_parameter_sha256", "common_initial_parameter_sha256", "context_mask_sha256_by_epoch",
                "rng_seeds", "increment_scales_native", "increment_observation_counts", "phase_scales_native",
                "optimizer_updates", "total_training_contexts"):
        assert zero.fit_diagnostics_[key] == actual.fit_diagnostics_[key]
    equal(zero_normals, actual_normals)
    # Zero output heads initially block all trunk gradients, even with real features.
    assert torch.count_nonzero(actual_gradients[0]) == torch.count_nonzero(zero_gradients[0]) == 0
    assert all(torch.count_nonzero(gradient) == 0 for gradient in zero_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in actual_gradients[1:])
    assert torch.count_nonzero(zero.network.hidden[0].weight[:, -WIDTH:]) == 0
    assert torch.count_nonzero(actual.network.hidden[0].weight[:, -WIDTH:]) > 0
    assert len(actual_gradients) == len(zero_gradients) == actual.fit_diagnostics_["optimizer_updates"]


def context(model, native, observed, cutoff, features):
    tensor = torch.as_tensor(native, dtype=torch.float64, device=model.device)
    mask = torch.as_tensor(observed, dtype=torch.bool, device=model.device)
    normalized = (tensor/model.volume_scale).float()
    encoded = model._encoder_context(normalized, mask, model.report_days.float(), cutoff,
        native_reports=tensor, covariates=torch.as_tensor(features, dtype=torch.float32, device=model.device))
    return encoded


def test_context_appends_postmask_native_earlier_intervals_without_changing_existing_slots(inputs):
    old, zero, actual = [initial("basis", mode, inputs) for mode in ("none", "zero", "actual")]
    native, features = inputs
    native = native[:2].copy(); features = features[:2]
    observed = np.isfinite(native) & (DAYS[None, :, None] <= 360.)
    observed[0, 2, 0] = False  # Oil earlier interval must span the dropped day 60 endpoint.
    observed[0, 1:, 1] = False  # Entire positive-time gas history is hidden.
    original = context(old, native, observed, 360., features)
    empty = context(zero, native, observed, 360., features)
    full = context(actual, native, observed, 360., features)
    assert original.shape == (2, 105) and empty.shape == full.shape == (2, 150)
    equal(full[:, :105], original); equal(empty[:, :105], original)
    assert torch.count_nonzero(empty[:, 105:]) == 0
    tail = full[:, 105:].reshape(2, 5, 3, 3)
    assert torch.count_nonzero(tail) > 0
    assert torch.count_nonzero(tail[0, :, 1]) == 0
    # The latest surviving pair is already covered by the existing last-interval
    # feature. It must not also appear in the earlier-history slots.
    assert torch.count_nonzero(tail[:, -1]) == 0
    assert torch.count_nonzero(tail[0, 1, 0]) == 0
    assert float(tail[0, 2, 0, 1]) == pytest.approx(60/360)
    assert float(tail[0, 2, 0, 2]) == 1.
    expected_rate = np.arcsinh(360*(native[0, 3, 0]-native[0, 1, 0])/(60*160.))
    assert float(tail[0, 2, 0, 0]) == pytest.approx(expected_rate, rel=2e-7)
    # Dropping the latest endpoint changes which pair is excluded, independently
    # for each phase. The newly latest 90->180 oil pair is removed.
    dropped = observed.copy(); dropped[0, 5, 0] = False
    changed = context(actual, native, dropped, 360., features)[:, 105:].reshape(2, 5, 3, 3)
    assert torch.count_nonzero(tail[0, 3, 0]) > 0
    assert torch.count_nonzero(changed[0, 3, 0]) == 0
    equal(changed[0, :3, 0], tail[0, :3, 0])
    # All unavailable raw endpoints can be poisoned without entering either block.
    poison = native.copy(); poison[~observed] = np.inf
    equal(context(actual, poison, observed, 360., features), full)
    equal(context(zero, poison, observed, 360., features), empty)


@pytest.mark.parametrize("mode", ["zero", "actual"])
def test_feature_context_requires_original_float64_history_even_for_zero_control(inputs, mode):
    model = initial("basis", mode, inputs)
    native = torch.as_tensor(inputs[0][:2], dtype=torch.float64)
    observed = torch.isfinite(native) & (model.report_days <= 360)[None, :, None]
    normalized = (native/model.volume_scale).float()
    with pytest.raises(ValueError, match="float64|native|Native"):
        model._encoder_context(normalized, observed, model.report_days.float(), 360.,
            native_reports=native.float(), covariates=torch.as_tensor(inputs[1][:2], dtype=torch.float32))


def assert_feature_metadata(metadata, mode):
    assert metadata["earlier_history_features"] == mode
    assert metadata["earlier_history_feature_count"] == WIDTH
    assert metadata["earlier_history_feature_end_days"] == END_DAYS
    assert metadata["earlier_history_feature_order"] == "end_day_then_oil_gas_water_then_rate_duration_availability"
    assert metadata["earlier_history_feature_layout"] == LAYOUT
    assert isinstance(metadata["earlier_history_feature_meaning"], str) and metadata["earlier_history_feature_meaning"]


@pytest.mark.parametrize("mode", ["zero", "actual"])
@pytest.mark.parametrize("origin", [0., 90., 360.])
def test_prediction_masks_cutoffs_and_preserves_known_signed_reports_and_feature_metadata(models, inputs, mode, origin):
    model = models[mode][0]
    assert_feature_metadata(model.fit_diagnostics_, mode)
    history = inputs[0][:2].copy()
    history[:, DAYS > origin] = np.nan
    history[:, DAYS > 0, 1] = np.nan
    if origin > 0:
        history[0, DAYS == origin, 2] = np.nan
    poisoned = history.copy(); poisoned[:, DAYS > origin] = np.inf
    a, b = np.random.default_rng(83), np.random.default_rng(83)
    before = model.predict_distribution(history, origin, 24, a, inputs[1][:2])
    after = model.predict_distribution(poisoned, origin, 24, b, inputs[1][:2])
    equal(before, after)
    assert a.bit_generator.state == b.bit_generator.state
    assert_feature_metadata(before, mode)
    assert before["earlier_history_feature_meaning"] == model.fit_diagnostics_["earlier_history_feature_meaning"]
    known = np.isfinite(history) & (DAYS[None, :, None] <= origin)
    mask = np.broadcast_to(known[:, None], before["report_cumulative"].shape)
    np.testing.assert_array_equal(before["report_cumulative"][mask],
        np.broadcast_to(history[:, None], before["report_cumulative"].shape)[mask])
    if origin > 0:
        oi = int(np.flatnonzero(DAYS == origin)[0])
        assert before["report_cumulative"][0, :, oi, 1].std() > 0
        assert before["report_cumulative"][0, :, oi, 2].std() > 0
        assert before["report_cumulative"][0, 0, 1, 0] == -.12500000000000003
    assert np.isfinite(before["report_cumulative"]).all()
    assert before["observation_law"] == "conditional_gaussian_accumulated_bridge"


@pytest.mark.parametrize("mode", ["zero", "actual"])
def test_checkpoint_restores_feature_contract_initialization_and_exact_prediction(models, inputs, tmp_path, mode):
    model = models[mode][0]
    path = tmp_path / "earlier.pt"; model.save(path)
    saved = torch.load(path, weights_only=True)
    assert saved["configuration"]["earlier_history_features"] == mode
    assert_feature_metadata(saved, mode)
    assert_feature_metadata(saved["fit_diagnostics"], mode)
    restored = conditional.ConditionalRateForecaster.load(path)
    equal(restored.state_dict(), model.state_dict())
    equal(restored.configuration, model.configuration)
    equal(restored.fit_diagnostics_, model.fit_diagnostics_)
    assert restored.common_initial_parameter_sha256_ == model.common_initial_parameter_sha256_
    assert restored.network.hidden[0].in_features == 150
    a, b = np.random.default_rng(89), np.random.default_rng(89)
    equal(model.predict_distribution(inputs[0][:2], 360, 12, a, inputs[1][:2]),
          restored.predict_distribution(inputs[0][:2], 360, 12, b, inputs[1][:2]))
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize("mutation", ["top_mode", "configuration", "nested_configuration", "fit_mode", "layout", "end_days", "count", "meaning", "common_hash", "both_common_hashes", "network_width"])
def test_checkpoint_rejects_feature_metadata_and_network_substitution(models, tmp_path, mutation):
    path = tmp_path / "bad.pt"; models["actual"][0].save(path)
    saved = torch.load(path, weights_only=True)
    saved["fit_diagnostics"]["configuration"] = copy.deepcopy(saved["fit_diagnostics"]["configuration"])
    if mutation == "top_mode": saved["earlier_history_features"] = "zero"
    elif mutation == "configuration": saved["configuration"].pop("earlier_history_features")
    elif mutation == "nested_configuration": saved["fit_diagnostics"]["configuration"]["earlier_history_features"] = "zero"
    elif mutation == "fit_mode": saved["fit_diagnostics"]["earlier_history_features"] = "zero"
    elif mutation == "layout": saved["earlier_history_feature_layout"] = LAYOUT[::-1]
    elif mutation == "end_days": saved["earlier_history_feature_end_days"] = [30, 60, 90, 180, 720]
    elif mutation == "count": saved["earlier_history_feature_count"] = 44
    elif mutation == "meaning": saved["earlier_history_feature_meaning"] = "latest intervals included"
    elif mutation == "common_hash": saved["common_initial_parameter_sha256"] = "a"*64
    elif mutation == "both_common_hashes":
        saved["common_initial_parameter_sha256"] = "a"*64
        saved["fit_diagnostics"]["common_initial_parameter_sha256"] = "a"*64
    else: saved["state_dict"]["network.hidden.0.weight"] = saved["state_dict"]["network.hidden.0.weight"][:, :-WIDTH]
    torch.save(saved, path)
    with pytest.raises(ValueError):
        conditional.ConditionalRateForecaster.load(path)


@pytest.mark.parametrize("options", [
    {"earlier_history_features": "unknown"}, {"earlier_history_features": None},
    {"earlier_history_features": True}, {"mixture_mode": "tied"}, {"mixture_mode": "two"},
    {"history_head_mode": "no_history"}, {"coefficient_parameterization": "log_expectation"},
    {"tree_static_mode": "fixed_empty_history"}, {"history_rate_features": "none"},
    {"history_rate_features": "zero"}, {"report_half_life_days": 180.},
    {"marginal_score_estimator": "gaussian_report_integrated"},
])
def test_invalid_mode_or_unapproved_combination_is_rejected_before_fit(options):
    with pytest.raises(ValueError):
        conditional.ConditionalRateForecaster(**{**SETTINGS, "earlier_history_features": "actual", **options})


@pytest.mark.parametrize("mode", ["zero", "actual"])
def test_nondefault_features_require_the_canonical_eight_day_grid(inputs, mode):
    days = DAYS.copy(); days[3] = 91.
    model = conditional.ConditionalRateForecaster(**SETTINGS, earlier_history_features=mode)
    with pytest.raises(ValueError, match="canonical|Canonical|eight|report.*days"):
        model.fit(inputs[0], days, origins=[0., 360.], covariates=inputs[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only padding and checkpoint replay")
def test_cuda_initialization_zero_padding_and_checkpoint_replay(inputs, tmp_path):
    before = torch.cuda.get_rng_state().clone()
    old, zero, actual = [initial("basis", mode, inputs, "cuda") for mode in ("none", "zero", "actual")]
    equal(torch.cuda.get_rng_state(), before)
    equal(zero.state_dict(), actual.state_dict())
    assert actual.common_initial_parameter_sha256_ == old.initial_parameter_sha256_
    assert torch.count_nonzero(actual.network.hidden[0].weight[:, -WIDTH:]) == 0
    trained = conditional.ConditionalRateForecaster(**dict(SETTINGS, epochs=1), earlier_history_features="actual", device="cuda")
    trained.fit(inputs[0], DAYS, origins=ORIGINS, covariates=inputs[1])
    path = tmp_path / "cuda.pt"; trained.save(path)
    restored = conditional.ConditionalRateForecaster.load(path, device="cuda")
    a, b = np.random.default_rng(31), np.random.default_rng(31)
    equal(trained.predict_distribution(inputs[0][:2], 360, 12, a, inputs[1][:2]),
          restored.predict_distribution(inputs[0][:2], 360, 12, b, inputs[1][:2]))
    assert a.bit_generator.state == b.bit_generator.state
