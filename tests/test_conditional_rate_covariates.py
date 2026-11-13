"""Static context, attribute-only forecasts, and portable rate checkpoints."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import hashlib
from pathlib import Path
import tarfile
import types

import numpy as np
import pytest
import torch

from joint_production.conditional_rates import ConditionalRateForecaster


DAYS = np.array([0., 30., 60., 90., 180.])
SETTINGS = dict(epochs=2, hidden_width=12, batch_size=6, train_samples=4,
                seed=31, log_every=0, history_rate_features="last_interval")


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(319)
    amplitude = rng.lognormal(0, .25, (12, 1, 3)) * [150., 600., 120.]
    reports = amplitude * (1-np.exp(-DAYS[None, :, None]/120))
    reports[:, 0] = 0
    reports[0, 1, 0] = -.12500000000000003
    features = np.concatenate((rng.normal(size=(12, 17)), rng.integers(0, 2, (12, 17))), axis=1)
    return reports, features


@pytest.fixture(scope="module", params=["basis", "dca"])
def model(request, inputs):
    reports, features = inputs
    return ConditionalRateForecaster(request.param, covariate_preprocessing="external", **SETTINGS).fit(
        reports, DAYS, origins=[0, 30, 60], covariates=features)


def test_external_features_enter_context_unchanged_and_affect_full_curves(model, inputs):
    reports, features = inputs
    features_before = features.copy()
    encoded = model._covariates(features, len(features), fit=False)
    np.testing.assert_array_equal(encoded, features.astype(np.float32))
    assert model.n_covariates_ == model.covariate_feature_count_ == 34
    assert model.covariate_center_.size == model.covariate_scale_.size == 0
    assert model.network.hidden[0].in_features == 8*len(DAYS)+1+6+34
    history = np.full_like(reports[:2], np.nan)
    original = model.conditional_parameters(history, 0, features[:2])[0]
    changed_features = features[:2].copy()
    changed_features[:, 0] += 3
    changed = model.conditional_parameters(history, 0, changed_features)[0]
    assert not torch.equal(original, changed)
    np.testing.assert_array_equal(features, features_before)


def test_attribute_only_draws_ignore_all_positive_day_reports(model, inputs):
    reports, features = inputs
    empty = np.full_like(reports[:2], np.nan)
    unavailable = empty.copy()
    unavailable[:, 1:] = np.inf
    a = model.predict_distribution(empty, 0, 12, np.random.default_rng(71), features[:2])
    b = model.predict_distribution(unavailable, 0, 12, np.random.default_rng(71), features[:2])
    for key in ("latent_parameters", "latent_rate", "latent_cumulative", "latent_total_volume", "report_cumulative"):
        np.testing.assert_array_equal(a[key], b[key])
        assert np.isfinite(a[key]).all()
    np.testing.assert_array_equal(a["report_cumulative"][:, :, 0], 0)
    assert (a["latent_rate"] >= 0).all()
    assert (np.diff(a["latent_cumulative"], axis=2) >= 0).all()
    assert (a["latent_total_volume"] >= a["latent_cumulative"][:, :, -1]).all()
    assert a["conditional_diagnostics"]["history_observed_counts_by_phase"] == [[0, 0, 0], [0, 0, 0]]


def test_prefix_masking_keeps_native_history_and_same_sample_hidden_origin(model, inputs):
    reports, features = inputs
    history = reports[:2].copy()
    history[:, DAYS > 30] = np.nan
    history[:, :, 1] = np.nan
    poison = history.copy()
    poison[:, DAYS > 30] = -np.inf
    a = model.predict_distribution(history, 30, 64, np.random.default_rng(23), features[:2])
    b = model.predict_distribution(poison, 30, 64, np.random.default_rng(23), features[:2])
    np.testing.assert_array_equal(a["report_cumulative"], b["report_cumulative"])
    drawn = a["report_cumulative"]
    known = np.isfinite(history)
    np.testing.assert_array_equal(drawn.transpose(0, 2, 3, 1)[known],
                                  np.repeat(history[known, None], 64, axis=1))
    assert drawn[0, :, 1, 1].std() > 0
    np.testing.assert_allclose(drawn[:, :, 4]-drawn[:, :, 1],
        drawn[:, :, 4]-drawn[:, :, 3]+drawn[:, :, 3]-drawn[:, :, 1], atol=1e-10)
    query = model.evaluate_trajectories(a["latent_parameters"], [30., 120., 180.])
    np.testing.assert_allclose(query["latent_cumulative"][:, :, [0, 2]],
                               a["latent_cumulative"][:, :, [1, 4]], rtol=1e-12)


def test_feature_and_zero_controls_share_scales_initialization_and_masks(model, inputs):
    reports, features = inputs
    zero = ConditionalRateForecaster(model.rate_family, covariate_preprocessing="external", **SETTINGS).fit(
        reports, DAYS, origins=[0, 30, 60], covariates=np.zeros_like(features))
    for name in ("initial_parameter_sha256", "context_mask_sha256_by_epoch", "rng_seeds",
                 "increment_scales_native", "increment_observation_counts", "phase_scales_native"):
        assert zero.fit_diagnostics_[name] == model.fit_diagnostics_[name]
    for item in model.fit_diagnostics_["optimization"]:
        assert np.isfinite([item[k] for k in ("loss", "macro_increment_crps", "joint_energy", "max_gradient_norm")]).all()


def test_external_checkpoint_restores_feature_context_and_joint_law(model, inputs, tmp_path):
    reports, features = inputs
    file = tmp_path / "features.pt"
    model.save(file)
    stored = torch.load(file, weights_only=True)
    assert stored["configuration"]["covariate_preprocessing"] == "external"
    assert stored["covariate_metadata"] == dict(n_covariates=34, feature_count=34, center=[], scale=[])
    restored = ConditionalRateForecaster.load(file)
    assert restored.fit_diagnostics_ == model.fit_diagnostics_
    for origin in (0, 30):
        a = model.predict_distribution(reports[:2], origin, 8, np.random.default_rng(39), features[:2])
        b = restored.predict_distribution(reports[:2], origin, 8, np.random.default_rng(39), features[:2])
        for key in ("latent_parameters", "latent_cumulative", "report_cumulative"):
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("problem", ["missing", "width", "nan", "inf", "float32_overflow"])
def test_external_query_rejects_missing_or_invalid_features(model, inputs, problem):
    reports, features = inputs
    bad = features[:2].copy()
    if problem == "missing":
        bad = None
    elif problem == "width":
        bad = bad[:, :-1]
    else:
        bad[0, 0] = {"nan": np.nan, "inf": np.inf, "float32_overflow": 1e100}[problem]
    with pytest.raises(ValueError, match="[Cc]ovariate"):
        model.predict_samples(reports[:2], 0, 3, np.random.default_rng(2), bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e100])
def test_external_training_features_fail_before_network_initialization(inputs, value):
    reports, features = inputs
    invalid = features.copy()
    invalid[0, 0] = value
    model = ConditionalRateForecaster(covariate_preprocessing="external", **SETTINGS)
    with pytest.raises(ValueError, match="[Cc]ovariate"):
        model.fit(reports, DAYS, origins=[0, 30], covariates=invalid)
    assert model.network is None


def test_internal_preprocessing_is_train_only_and_portable(inputs, tmp_path):
    reports, _ = inputs
    values = np.column_stack((np.arange(len(reports)), np.full(len(reports), np.nan)))
    model = ConditionalRateForecaster(**SETTINGS).fit(reports, DAYS, origins=[0, 30], covariates=values)
    np.testing.assert_array_equal(model.covariate_center_, [5.5, 0.])
    np.testing.assert_array_equal(model.covariate_scale_, [np.std(np.arange(12)), 1.])
    assert model.covariate_feature_count_ == 4
    transformed = model._covariates(np.array([[100., np.nan]]), 1, fit=False)
    np.testing.assert_allclose(transformed, [[(100-5.5)/np.std(np.arange(12)), 0, 1, 0]], rtol=1e-6)
    center, scale = model.covariate_center_.copy(), model.covariate_scale_.copy()
    path = tmp_path / "internal.pt"
    model.save(path)
    loaded = ConditionalRateForecaster.load(path)
    a = model.predict_samples(reports[:2], 0, 5, np.random.default_rng(9), values[:2])
    b = loaded.predict_samples(reports[:2], 0, 5, np.random.default_rng(9), values[:2])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(model.covariate_center_, center)
    np.testing.assert_array_equal(model.covariate_scale_, scale)


@pytest.mark.parametrize("mutation", ["missing_metadata", "wrong_width", "wrong_mode", "bad_scale"])
def test_corrupt_checkpoint_cannot_silently_change_feature_encoding(model, tmp_path, mutation):
    path = tmp_path / "bad.pt"
    model.save(path)
    state = torch.load(path, weights_only=True)
    if mutation == "missing_metadata":
        state.pop("covariate_metadata")
    elif mutation == "wrong_width":
        state["covariate_metadata"]["feature_count"] += 1
    elif mutation == "wrong_mode":
        state["configuration"]["covariate_preprocessing"] = "internal"
    else:
        state["covariate_metadata"]["scale"] = [1.]
    torch.save(state, path)
    with pytest.raises(ValueError, match="covariate"):
        ConditionalRateForecaster.load(path)


def test_two_component_model_uses_available_covariates_without_changing_path_scope(inputs, tmp_path):
    reports, features = inputs
    model = ConditionalRateForecaster(mixture_mode="two", covariate_preprocessing="external", **SETTINGS).fit(
        reports, DAYS, origins=[0, 30], covariates=features)
    means, lower, weights = model.conditional_mixture_parameters(reports[:2], 0, features[:2])
    assert means.shape == (2, 2, 12) and lower.shape == (2, 12, 12) and weights.shape == (2, 2)
    result = model.predict_distribution(reports[:2], 0, 12, np.random.default_rng(12), features[:2])
    assert result["mixture_component"].shape == (2, 12)
    assert result["component_scope"] == "one_component_for_entire_joint_latent_trajectory"
    path = tmp_path / "mixture.pt"
    model.save(path)
    loaded = ConditionalRateForecaster.load(path)
    repeated = loaded.predict_distribution(reports[:2], 0, 12, np.random.default_rng(12), features[:2])
    for key in ("latent_parameters", "report_cumulative", "mixture_component"):
        np.testing.assert_array_equal(result[key], repeated[key])


def test_invalid_preprocessing_mode_fails():
    with pytest.raises(ValueError, match="covariate_preprocessing"):
        ConditionalRateForecaster(covariate_preprocessing="unrecognized")
