"""CUDA startup checks for shared external static conditioning; synthetic only."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pytest
import torch

from joint_production.conditional_rates import ConditionalRateForecaster
from joint_production.neural import ContextualGaussianNeuralBaseline


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
DAYS = np.array([0., 30., 60., 90., 180.])


@pytest.fixture(scope="module", params=["basis", "dca", "neural"])
def fitted(request):
    rng = np.random.default_rng(262)
    features = np.concatenate((rng.normal(size=(8, 17)), rng.integers(0, 2, (8, 17))), axis=1)
    amplitude = rng.lognormal(0, .2, (8, 1, 3)) * [150., 600., 120.]
    reports = amplitude * (1 - np.exp(-DAYS[None, :, None]/120))
    reports[:, 0] = 0
    reports[0, 1, 0] = -.12500000000000003
    settings = dict(epochs=2, hidden_width=12, batch_size=8, seed=52,
                    log_every=0, device="cuda", covariate_preprocessing="external",
                    history_rate_features="last_interval")
    if request.param == "neural":
        model = ContextualGaussianNeuralBaseline(rank=2, **settings)
    else:
        model = ConditionalRateForecaster(request.param, train_samples=4, **settings)
    model.fit(reports, DAYS, origins=[0, 30, 60], covariates=features)
    return request.param, model, reports, features


def distribution(kind, model, history, origin, features, seed=77):
    rng = np.random.default_rng(seed)
    if kind == "neural":
        return {"report_cumulative": model.predict_samples(history, origin, 16, rng, features)}
    result = model.predict_distribution(history, origin, 16, rng, features)
    return {name: result[name] for name in ("latent_parameters", "latent_rate", "latent_cumulative",
                                           "latent_total_volume", "report_cumulative")}


def test_cuda_feature_fit_uses_device_and_has_finite_optimization(fitted):
    kind, model, _, _ = fitted
    parameters = list(model.network_.parameters() if kind == "neural" else model.parameters())
    assert parameters and all(p.device.type == "cuda" for p in parameters)
    assert all(bool(torch.isfinite(p).all()) for p in parameters)
    assert model.n_covariates_ == model.covariate_feature_count_ == 34
    assert model.fit_diagnostics_["covariate_preprocessing"] == "external"
    assert model.fit_diagnostics_["origins_days"] == [0., 30., 60.]
    if kind == "neural":
        losses = model.fit_diagnostics_["conditional_report_nll_by_epoch"]
    else:
        losses = [row["loss"] for row in model.fit_diagnostics_["optimization"]]
    assert len(losses) == 2 and np.isfinite(losses).all()


def test_cuda_secondary_only_ignores_future_reports_and_uses_features(fitted):
    kind, model, reports, features = fitted
    empty = np.full_like(reports[:2], np.nan)
    poison = empty.copy()
    poison[:, DAYS > 0] = np.inf
    a = distribution(kind, model, empty, 0, features[:2])
    b = distribution(kind, model, poison, 0, features[:2])
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
        assert np.isfinite(a[name]).all()
    np.testing.assert_array_equal(a["report_cumulative"][:, :, 0], 0)
    assert a["report_cumulative"].shape == (2, 16, len(DAYS), 3)
    changed = features[:2].copy()
    changed[:, :17] += 2
    c = distribution(kind, model, empty, 0, changed)
    assert not np.array_equal(a["report_cumulative"], c["report_cumulative"])
    if kind != "neural":
        assert (a["latent_rate"] >= 0).all()
        assert (np.diff(a["latent_cumulative"], axis=2) >= 0).all()


def test_cuda_prefix_pins_native_reports_and_draws_hidden_origin_jointly(fitted):
    kind, model, reports, features = fitted
    history = reports[:2].copy()
    history[0, 1, 0] = 200123.49
    history[:, :, 1] = np.nan
    history[:, DAYS > 30] = np.inf
    result = distribution(kind, model, history, 30, features[:2])
    samples = result["report_cumulative"]
    known = np.isfinite(history) & (DAYS <= 30)[None, :, None]
    np.testing.assert_array_equal(samples.transpose(0, 2, 3, 1)[known],
                                  np.repeat(history[known, None], 16, axis=1))
    assert samples[0, 0, 1, 0] != float(np.float32(history[0, 1, 0]))
    assert samples[0, :, 1, 1].std() > 0
    np.testing.assert_allclose(samples[:, :, 4]-samples[:, :, 1],
        (samples[:, :, 4]-samples[:, :, 3])+(samples[:, :, 3]-samples[:, :, 1]), atol=1e-8)


def test_cuda_checkpoint_restores_complete_draws_and_feature_contract(fitted, tmp_path):
    kind, model, reports, features = fitted
    path = tmp_path / "static_cuda.pt"
    model.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert all(t.device.type == "cpu" for t in checkpoint["state_dict"].values())
    cls = ContextualGaussianNeuralBaseline if kind == "neural" else ConditionalRateForecaster
    restored = cls.load(path, device="cuda")
    assert restored.covariate_preprocessing == "external"
    assert restored.n_covariates_ == restored.covariate_feature_count_ == 34
    for origin in (0, 30, 60):
        original = distribution(kind, model, reports[:2], origin, features[:2])
        replay = distribution(kind, restored, reports[:2], origin, features[:2])
        for name in original:
            np.testing.assert_array_equal(original[name], replay[name])
