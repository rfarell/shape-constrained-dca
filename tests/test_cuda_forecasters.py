"""Bounded synthetic CUDA checks for the GPU forecast paths.

Six two-epoch fits are shared across the module. These checks establish
device execution and accounting/conditioning invariants, not field accuracy
or numerical identity between CPU and CUDA training or predictions.
"""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pytest
import torch

from joint_production.conditional_rates import ConditionalRateForecaster
from joint_production.neural import ContextualGaussianNeuralBaseline


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
DAYS = np.array([0., 30., 60., 90., 180., 360., 720., 1080.])
KINDS = ("basis", "dca", "neural", "basis_tied", "basis_mixture", "neural_mixture")


def _synthetic_reports():
    rng = np.random.default_rng(191)
    amplitudes = rng.lognormal(0, .3, (12, 1, 3)) * [150., 700., 120.]
    reports = amplitudes * (1 - np.exp(-DAYS[None, :, None] / 300))
    reports += rng.normal(0, .1, reports.shape)
    reports[:, 0] = 0
    reports[0, 1, 0] = -.12500000000000003  # Native float64 signed report.
    reports[1, 2, 2] = np.nan
    return reports


def _fit_model(kind, device):
    common = dict(epochs=2, hidden_width=16, batch_size=6, seed=17,
                  log_every=0, device=device, history_rate_features="last_interval")
    if kind.startswith("neural"):
        model = ContextualGaussianNeuralBaseline(rank=4,
            mixture_components=2 if kind == "neural_mixture" else 1, **common)
    elif kind in {"basis_tied", "basis_mixture"}:
        model = ConditionalRateForecaster("basis", train_samples=16,
            mixture_mode="tied" if kind == "basis_tied" else "two", **common)
    else:
        model = ConditionalRateForecaster(kind, train_samples=4, **common)
    return model.fit(_synthetic_reports(), DAYS, origins=[60, 90])


@pytest.fixture(scope="module")
def cuda_models():
    return {kind: _fit_model(kind, "cuda") for kind in KINDS}


@pytest.fixture(params=KINDS)
def forecaster(request, cuda_models):
    return cuda_models[request.param]


@pytest.fixture(params=("basis", "dca", "basis_tied", "basis_mixture"))
def rate_forecaster(request, cuda_models):
    return cuda_models[request.param]


def _masked_history():
    history = _synthetic_reports()[:2].copy()
    history[:, DAYS > 90] = np.nan
    history[:, 1:, 1] = np.nan  # Entire gas phase missing except accounting zero.
    history[1, DAYS == 90, 2] = np.nan  # Additional missing water origin.
    return history


def _network(model):
    return model.network if isinstance(model, ConditionalRateForecaster) else model.network_


def _draws(model, history, count=32):
    rng = np.random.default_rng(2026)
    if isinstance(model, ConditionalRateForecaster):
        result = model.predict_distribution(history, 90, count, rng)
        return {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    return {"report_cumulative": model.predict_samples(history, 90, count, rng)}


def _check_report_invariants(draws, history):
    assert draws.shape == (2, 32, len(DAYS), 3)
    assert np.isfinite(draws).all()
    observed = np.isfinite(history) & (DAYS <= 90)[None, :, None]
    np.testing.assert_array_equal(draws.transpose(0, 2, 3, 1)[observed],
                                  np.repeat(history[observed, None], 32, axis=1))
    np.testing.assert_array_equal(draws[:, :, 0], 0)
    # Missing origins must remain sampled coordinates of the same trajectory.
    assert (draws[:, :, 3, 1].std(axis=1) > 1e-10).all()
    assert draws[1, :, 3, 2].std() > 1e-10
    assert (draws[:, :, -1].std(axis=1) > 1e-10).all()
    np.testing.assert_allclose(draws[:, :, -1] - draws[:, :, 3],
                               (draws[:, :, -1] - draws[:, :, 5])
                               + (draws[:, :, 5] - draws[:, :, 3]),
                               rtol=1e-12, atol=1e-9)


def test_cuda_fit_and_conditional_report_invariants(forecaster):
    assert all(parameter.device.type == "cuda" for parameter in _network(forecaster).parameters())
    assert all(torch.isfinite(parameter).all() for parameter in _network(forecaster).parameters())
    diagnostic = forecaster.fit_diagnostics_
    assert diagnostic["optimizer_updates"] > 0
    if isinstance(forecaster, ConditionalRateForecaster):
        losses = [entry["loss"] for entry in diagnostic["optimization"]]
        gradients = [entry["max_gradient_norm"] for entry in diagnostic["optimization"]]
    else:
        losses = diagnostic["conditional_report_nll_by_epoch"]
        gradients = diagnostic["max_gradient_norm_before_clipping_by_epoch"]
    assert len(losses) == len(gradients) == 2
    assert np.isfinite(losses).all() and np.isfinite(gradients).all()
    assert (np.asarray(gradients) > 0).all()
    history = _masked_history()
    unchanged = history.copy()
    result = _draws(forecaster, history)
    contaminated = history.copy()
    contaminated[:, DAYS > 90] = np.inf
    second = _draws(forecaster, contaminated)
    for key in result:
        np.testing.assert_array_equal(result[key], second[key])
    np.testing.assert_array_equal(history, unchanged)
    _check_report_invariants(result["report_cumulative"], history)


def test_cuda_checkpoint_replay_and_explicit_cpu_load(forecaster, tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    forecaster.save(checkpoint)
    restored = type(forecaster).load(checkpoint, device="cuda")
    assert all(parameter.device.type == "cuda" for parameter in _network(restored).parameters())
    assert restored.fit_diagnostics_ == forecaster.fit_diagnostics_
    history = _masked_history()
    original, copied = _draws(forecaster, history), _draws(restored, history)
    for key in original:
        np.testing.assert_array_equal(original[key], copied[key])

    # CPU loading must preserve learned tensors and preprocessing. CPU/GPU
    # floating-point inference is allowed to differ; no cross-device draw
    # equality or field-performance claim follows from this portability check.
    portable = type(forecaster).load(checkpoint, device="cpu")
    assert all(parameter.device.type == "cpu" for parameter in _network(portable).parameters())
    assert portable.fit_diagnostics_ == forecaster.fit_diagnostics_
    assert portable.history_rate_features == "last_interval"
    for name, tensor in _network(forecaster).state_dict().items():
        torch.testing.assert_close(tensor.cpu(), _network(portable).state_dict()[name], rtol=0, atol=0)
    if isinstance(forecaster, ConditionalRateForecaster):
        for name in ("volume_scale", "increment_scales", "report_days", "noise_abs_raw", "noise_rel_raw"):
            torch.testing.assert_close(getattr(forecaster, name).cpu(), getattr(portable, name), rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(forecaster.phase_scales_, portable.phase_scales_)
        np.testing.assert_array_equal(forecaster.report_days_, portable.report_days_)
    _check_report_invariants(_draws(portable, history)["report_cumulative"], history)


def test_cuda_rate_integrals_match_returned_cumulative_paths(rate_forecaster):
    result = _draws(rate_forecaster, _masked_history(), count=4)
    assert all(np.isfinite(value).all() for value in result.values())
    assert (result["latent_rate"] >= 0).all()
    assert (np.diff(result["latent_cumulative"], axis=2) >= 0).all()
    np.testing.assert_array_equal(result["latent_cumulative"][:, :, 0], 0)
    assert (result["latent_total_volume"] >= result["latent_cumulative"][:, :, -1]).all()

    # Independent numerical integration includes an off-grid interval and
    # the DCA terminal regime, split explicitly at its 1080-day boundary.
    endpoints = np.array([0., 30., 135., 180., 1080., 1440.])
    nodes, weights = np.polynomial.legendre.leggauss(64)
    half_width = np.diff(endpoints) / 2
    midpoint = (endpoints[1:] + endpoints[:-1]) / 2
    query = midpoint[:, None] + half_width[:, None] * nodes
    latent = result["latent_parameters"]
    quadrature = rate_forecaster.evaluate_trajectories(latent, query.ravel())["latent_rate"]
    quadrature = quadrature.reshape(2, 4, len(half_width), len(nodes), 3)
    integrated = (quadrature * weights[None, None, None, :, None]).sum(axis=3)
    integrated *= half_width[None, None, :, None]
    exact = rate_forecaster.evaluate_trajectories(latent, endpoints)["latent_cumulative"]
    np.testing.assert_allclose(integrated, np.diff(exact, axis=2), rtol=1e-8, atol=1e-8)
    replay = rate_forecaster.evaluate_trajectories(latent, DAYS)
    for key in replay:
        np.testing.assert_array_equal(replay[key], result[key])
