"""Static-feature contracts for the conditional signed-report comparator."""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pytest
import torch

from joint_production.neural import ContextualGaussianNeuralBaseline


SETTINGS = dict(epochs=2, hidden_width=10, rank=2, batch_size=24,
                seed=31, history_rate_features="last_interval", log_every=0)


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(481)
    days = np.array([0., 30., 60., 90., 180.])
    raw = rng.normal(size=(12, 17))
    raw[::3, 2] = np.nan
    masks = np.isfinite(raw)
    # These stand in for preprocessing already fitted on the caller's train
    # partition. Their 17 masks are part of the 34-column external contract.
    centers = np.nanmedian(raw, axis=0)
    scales = np.nanstd(raw, axis=0)
    features = np.concatenate((np.where(masks, (raw - centers) / scales, 0), masks), axis=1)
    amplitude = np.exp(1 + .3 * raw[:, :1, None]) * np.array([1., 2., .6])
    reports = amplitude * (1 - np.exp(-days[None, :, None] / 80))
    reports += rng.normal(0, .1, size=reports.shape)
    reports[:, 0] = 0
    reports[0, 1, 0] = -.12345678901234567
    reports[1, 1, 1] = np.nan
    return reports, days, features


@pytest.fixture(scope="module")
def external_model(synthetic_data):
    reports, days, features = synthetic_data
    return ContextualGaussianNeuralBaseline(**SETTINGS).fit(reports, days, origins=[0, 30, 60], covariates=features)


def test_external_features_reach_network_unchanged_without_added_masks(external_model, synthetic_data):
    reports, days, features = synthetic_data
    model = external_model
    seen = []
    hook = model.network_.register_forward_pre_hook(lambda _, arguments: seen.append(arguments[0].clone()))
    try:
        model.predict_transformed_parameters(reports[:2], 0, features[:2])
    finally:
        hook.remove()
    assert model.n_covariates_ == 34
    assert model.network_.hidden[0].in_features == 8 * len(days) + 1 + 6 + 34
    np.testing.assert_array_equal(seen[0][:, -40:-6].numpy(), features[:2].astype(np.float32))
    assert model.fit_diagnostics_["covariate_preprocessing"] == "external"
    assert model.fit_diagnostics_["covariate_feature_count"] == 34


def test_secondary_only_origin_ignores_future_and_conditions_on_features(external_model, synthetic_data):
    reports, days, features = synthetic_data
    model = external_model
    no_history = np.full_like(reports[:2], np.nan)
    future_inf = no_history.copy()
    future_inf[:, days > 0] = np.inf
    scales_before = model.phase_scales_.copy()
    features_before = features.copy()
    first = model.predict_samples(no_history, 0, 128, np.random.default_rng(2), features[:2])
    second = model.predict_samples(future_inf, 0, 128, np.random.default_rng(2), features[:2])
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 128, len(days), 3)
    assert first.dtype == np.float64 and np.isfinite(first).all()
    np.testing.assert_array_equal(first[:, :, 0], 0)
    assert np.all(first[:, :, 1:].std(axis=1) > 0)
    before = model.predict_transformed_parameters(no_history, 0, features[:2])
    changed = model.predict_transformed_parameters(no_history, 0, np.zeros_like(features[:2]))
    assert not np.array_equal(before["location"], changed["location"])
    np.testing.assert_array_equal(model.phase_scales_, scales_before)
    np.testing.assert_array_equal(features, features_before)
    assert np.isnan(no_history).all()


def test_origin_30_restores_signed_float64_prefix_and_joint_missing_phase(external_model, synthetic_data):
    reports, days, features = synthetic_data
    history = reports[:2].copy()
    history[:, days > 30] = np.inf
    snapshot = history.copy()
    draws = external_model.predict_samples(history, 30, 128, np.random.default_rng(5), features[:2])
    observed = np.isfinite(history) & (days[None, :, None] <= 30)
    expanded = np.broadcast_to(observed[:, None], draws.shape)
    expected = np.broadcast_to(history[:, None], draws.shape)
    np.testing.assert_array_equal(draws[expanded], expected[expanded])
    assert draws[1, :, 1, 1].std() > 0
    np.testing.assert_allclose(draws[:, :, 4] - draws[:, :, 1],
                               (draws[:, :, 4] - draws[:, :, 2])
                               + (draws[:, :, 2] - draws[:, :, 1]), atol=1e-12)
    np.testing.assert_array_equal(history, snapshot)
    history[0, 1, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        external_model.predict_transformed_parameters(history, 30, features[:2])


def test_external_feature_sampler_matches_joint_transformed_law(external_model, synthetic_data):
    _, days, features = synthetic_data
    model = external_model
    history = np.full((1, len(days), 3), np.nan)
    parameters = model.predict_transformed_parameters(history, 0, features[:1])
    samples = model.predict_samples(history, 0, 40000, np.random.default_rng(181), features[:1])
    missing = ~parameters["observed_mask"][0]
    transformed = np.arcsinh(samples[0] / model.phase_scales_).reshape(40000, -1)[:, missing]
    factor = parameters["factor"][0, missing]
    covariance = np.diag(parameters["diagonal_variance"][0, missing]) + factor @ factor.T
    mean = parameters["location"][0, missing]
    np.testing.assert_allclose(transformed.mean(0), mean, atol=.012, rtol=0)
    np.testing.assert_allclose(np.cov(transformed.T), covariance, atol=.01, rtol=.04)
    assert (samples[:, :, 1:] < 0).any(), "The report law retains signed support"


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1e40, -1e40])
def test_external_nonfinite_and_float32_overflow_rejected_during_fit_and_prediction(synthetic_data, invalid):
    reports, days, features = synthetic_data
    model = ContextualGaussianNeuralBaseline(**SETTINGS)
    altered = features.copy()
    altered[0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        model.fit(reports, days, origins=[0], covariates=altered)
    model.n_covariates_ = features.shape[1]
    with pytest.raises(ValueError, match="finite"):
        model._covariates(altered, len(altered), fit=False)


def test_external_prediction_requires_training_feature_width(external_model, synthetic_data):
    reports, _, features = synthetic_data
    with pytest.raises(ValueError, match="requires cutoff"):
        external_model.predict_transformed_parameters(reports[:2], 0)
    for invalid in (features[:2, :-1], np.empty((2, 0))):
        with pytest.raises(ValueError, match="dimension differs"):
            external_model.predict_transformed_parameters(reports[:2], 0, invalid)
    for invalid in (features[:1], features[:2, :, None]):
        with pytest.raises(ValueError, match="shape"):
            external_model.predict_transformed_parameters(reports[:2], 0, invalid)


def test_external_checkpoint_preserves_mode_width_and_full_law(external_model, synthetic_data, tmp_path):
    reports, _, features = synthetic_data
    path = tmp_path / "external.pt"
    external_model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["configuration"]["covariate_preprocessing"] == "external"
    assert checkpoint["metadata"]["n_covariates"] == 34
    restored = ContextualGaussianNeuralBaseline.load(path)
    assert restored.covariate_preprocessing == "external"
    assert restored.fit_diagnostics_ == external_model.fit_diagnostics_
    for origin in (0, 30):
        before = external_model.predict_transformed_parameters(reports[:2], origin, features[:2])
        after = restored.predict_transformed_parameters(reports[:2], origin, features[:2])
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
        np.testing.assert_array_equal(
            external_model.predict_samples(reports[:2], origin, 17, np.random.default_rng(14), features[:2]),
            restored.predict_samples(reports[:2], origin, 17, np.random.default_rng(14), features[:2]))


@pytest.mark.parametrize("corruption", ["mode", "count", "network"])
def test_checkpoint_rejects_inconsistent_covariate_metadata(external_model, tmp_path, corruption):
    path = tmp_path / "corrupt.pt"
    external_model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    if corruption == "mode":
        checkpoint["configuration"]["covariate_preprocessing"] = "internal"
    elif corruption == "count":
        checkpoint["metadata"]["n_covariates"] = -1
    else:
        checkpoint["metadata"]["n_covariates"] = 33
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="covariate"):
        ContextualGaussianNeuralBaseline.load(path)


def test_invalid_covariate_preprocessing_mode_is_explicit():
    for options in (dict(covariate_preprocessing="internal"), dict(history_rate_features="none")):
        with pytest.raises(ValueError, match="last_interval history features and external covariates"):
            ContextualGaussianNeuralBaseline(**options)
