"""Synthetic invariants for the shared-covariance joint latent mixture."""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

from unittest.mock import patch

import numpy as np
import pytest
import torch

from joint_production.conditional_rates import (ConditionalRateForecaster, MIXTURE_INITIAL_DELTA,
                                               _ContextNetwork, _MixtureContextNetwork, _u_scores)
from joint_production.torch_observation import sample_accumulated_reports


DAYS = np.array([0., 30., 60., 90., 180., 360., 720., 1080.])


def _reports():
    rng = np.random.default_rng(371)
    amplitudes = rng.lognormal(0, .3, (12, 1, 3)) * [150., 700., 120.]
    reports = amplitudes * (1 - np.exp(-DAYS[None, :, None] / 300))
    reports += rng.normal(0, .1, reports.shape)
    reports[:, 0] = 0
    reports[0, 1, 0] = -.12500000000000003
    reports[1, 2, 2] = np.nan
    return reports


def _history():
    history = _reports()[:2].copy()
    history[:, DAYS > 90] = np.nan
    history[:, 1:, 1] = np.nan
    history[1, DAYS == 90, 2] = np.nan
    return history


@pytest.fixture(scope="module")
def models():
    return {mode: ConditionalRateForecaster(epochs=2, hidden_width=16, batch_size=6,
        train_samples=16, seed=19, log_every=0, history_rate_features="last_interval",
        mixture_mode=mode).fit(_reports(), DAYS, origins=[60, 90])
        for mode in ("single", "tied", "two")}


@pytest.mark.parametrize("mode", ["tied", "two"])
def test_synthetic_fits_are_finite_and_context_masks_match_single(models, mode):
    model = models[mode]
    assert model.fit_diagnostics_["context_mask_sha256_by_epoch"] == models["single"].fit_diagnostics_["context_mask_sha256_by_epoch"]
    np.testing.assert_array_equal(model.increment_scales, models["single"].increment_scales)
    assert model.fit_diagnostics_["train_samples_independent_per_component"] == 8
    assert "train_samples_iid" not in model.fit_diagnostics_
    assert model.fit_diagnostics_["component_scope"] == "one_component_for_entire_joint_latent_trajectory"
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
    for entry in model.fit_diagnostics_["optimization"]:
        assert all(np.isfinite(entry[key]) for key in ("loss", "macro_increment_crps", "joint_energy", "max_gradient_norm"))
        assert entry["max_gradient_norm"] > 0
    if mode == "two":
        for head in (model.network.delta, model.network.logits):
            assert head.weight.grad is not None
            assert torch.isfinite(head.weight.grad).all()
            assert float(head.weight.grad.abs().sum()) > 0


def test_initialization_shares_original_parameters_and_fixed_symmetric_delta():
    initialized = {}
    for mode in ("single", "tied", "two"):
        model = ConditionalRateForecaster(hidden_width=8, seed=99, mixture_mode=mode)
        model._initialize(DAYS, [1., 2., 3.], np.ones((11, 3)))
        initialized[mode] = model
    original = dict(initialized["single"].network.named_parameters())
    for mode in ("tied", "two"):
        changed = dict(initialized[mode].network.named_parameters())
        for name in original:
            torch.testing.assert_close(original[name], changed[name], rtol=0, atol=0)
    mixture = initialized["two"].network
    context = torch.zeros((3, mixture.hidden[0].in_features))
    means, lower, weights = mixture(context)
    signs = torch.where(torch.arange(12) % 2 == 0, 1., -1.)
    torch.testing.assert_close(means[:, 1] - means[:, 0], (2 * MIXTURE_INITIAL_DELTA * signs)[None].expand(3, -1))
    torch.testing.assert_close(weights, torch.full((3, 2), .5, dtype=torch.float64), rtol=0, atol=0)
    assert lower.shape == (3, 12, 12)
    tied_means, _, tied_weights = initialized["tied"]._mixture_network_parameters(context)
    torch.testing.assert_close(tied_means[:, 0], tied_means[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(tied_weights, weights, rtol=0, atol=0)


def test_extra_head_initialization_preserves_rng_state_and_does_not_seed_cuda():
    arguments = (65, 8, torch.zeros(12), .025, 1., .25, .2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(381)
        original = _ContextNetwork(*arguments)
        expected_state = torch.random.get_rng_state()
        torch.random.default_generator.manual_seed(381)
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CPU heads must not seed CUDA")):
            mixture = _MixtureContextNetwork(*arguments, head_seed=4382)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_state, rtol=0, atol=0)
        parameters = dict(mixture.named_parameters())
        for name, value in original.named_parameters():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cpu_extra_head_initialization_preserves_all_cuda_rng_states():
    before = torch.cuda.get_rng_state_all()
    _MixtureContextNetwork(65, 8, torch.zeros(12), .025, 1., .25, .2, head_seed=4382)
    after = torch.cuda.get_rng_state_all()
    for original, updated in zip(before, after, strict=True):
        torch.testing.assert_close(original, updated, rtol=0, atol=0)


@pytest.mark.parametrize("logits", [(8., -8.), (-8., 8.), (500., -500.)])
def test_near_saturated_probabilities_share_float64_scoring_and_sampling_law(logits):
    model = ConditionalRateForecaster(hidden_width=8, mixture_mode="two")
    model._initialize(DAYS, [1., 2., 3.], np.ones((11, 3)))
    with torch.no_grad():
        model.network.logits.bias.copy_(torch.tensor(logits))
    context = torch.zeros((2, model.network.hidden[0].in_features))
    _, _, weights = model._mixture_network_parameters(context)
    expected = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0).expand(2, -1)
    assert weights.dtype == torch.float64
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, dtype=torch.float64), rtol=0, atol=2e-16)
    torch.testing.assert_close(1 - weights[:, 0], weights[:, 1], rtol=0, atol=2e-16)
    loss = weights[:, 0].sum()
    loss.backward()
    assert torch.isfinite(model.network.logits.bias.grad).all()


def test_training_latent_and_bridge_innovations_share_exact_schedules():
    streams = {}
    original_randn = torch.randn
    for mode in ("single", "tied", "two"):
        draws = []
        def record(*args, **kwargs):
            result = original_randn(*args, **kwargs)
            draws.append(result.clone())
            return result
        with patch("joint_production.conditional_rates.torch.randn", side_effect=record):
            ConditionalRateForecaster(epochs=1, hidden_width=8, batch_size=6, train_samples=16,
                seed=43, log_every=0, mixture_mode=mode).fit(_reports(), DAYS, origins=[60])
        streams[mode] = draws
    assert len(streams["single"]) == 4  # Two batches, latent and report draw each.
    for mode in ("tied", "two"):
        for reference, candidate in zip(streams["single"], streams[mode], strict=True):
            torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
    # Independent stratum draws, never copied or antithetically reused.
    for draw in streams["two"]:
        assert not torch.equal(draw[:, :8], draw[:, 8:])


@pytest.mark.parametrize("mode", ["tied", "two"])
def test_cutoff_missing_origins_and_float64_known_reports(models, mode):
    model = models[mode]
    history = _history()
    contaminated = history.copy()
    contaminated[:, DAYS > 90] = np.inf
    first = model.predict_distribution(history, 90, 64, np.random.default_rng(29))
    second = model.predict_distribution(contaminated, 90, 64, np.random.default_rng(29))
    for key in first:
        if isinstance(first[key], np.ndarray):
            np.testing.assert_array_equal(first[key], second[key])
    observed = np.isfinite(history)
    np.testing.assert_array_equal(first["report_cumulative"].transpose(0, 2, 3, 1)[observed],
                                  np.repeat(history[observed, None], 64, axis=1))
    assert (first["report_cumulative"][:, :, 3, 1].std(axis=1) > 0).all()
    assert first["report_cumulative"][1, :, 3, 2].std() > 0
    assert first["mixture_component"].shape == (2, 64)
    assert first["mixture_component_probabilities"].shape == (2, 2)
    assert len(set(first["prediction_rng_seeds"].values())) == 3
    mixture_diagnostic = first["conditional_diagnostics"]["mixture"]
    assert np.isfinite(mixture_diagnostic["component_entropy_nats"]).all()
    assert np.isfinite(mixture_diagnostic["component_mean_separation_l2"]).all()
    if mode == "tied":
        np.testing.assert_array_equal(mixture_diagnostic["component_mean_separation_l2"], 0)
    with pytest.raises(ValueError, match="not one Gaussian"):
        model.conditional_parameters(history, 90)


@pytest.mark.parametrize("mode", ["tied", "two"])
def test_sampler_uses_one_component_and_replays_independent_bridge(models, mode):
    model = models[mode]
    history = _history()
    result = model.predict_distribution(history, 90, 128, np.random.default_rng(183))
    means, lower, weights = model.conditional_mixture_parameters(history, 90)
    component_rng = np.random.default_rng(result["prediction_rng_seeds"]["component"])
    components = component_rng.random((2, 128)) >= weights[:, 0, None].numpy()
    np.testing.assert_array_equal(result["mixture_component"], components)
    latent_rng = np.random.default_rng(result["prediction_rng_seeds"]["latent"])
    epsilon = torch.tensor(latent_rng.standard_normal((2, 128, 12)))
    selected = means[torch.arange(2)[:, None], torch.tensor(components, dtype=torch.long)]
    expected_z = selected + torch.einsum("nij,nsj->nsi", lower, epsilon)
    np.testing.assert_array_equal(result["latent_parameters"], expected_z.numpy())
    report_rng = np.random.default_rng(result["prediction_rng_seeds"]["report"])
    q = torch.tensor(result["latent_cumulative"]) / model.volume_scale
    normalized = torch.tensor(history) / model.volume_scale
    with torch.no_grad():
        expected = sample_accumulated_reports(q, normalized, model.report_days,
            model.noise_abs.double(), model.noise_rel.double(), cutoff=90,
            standard_normals=torch.tensor(report_rng.standard_normal(q.shape))) * model.volume_scale
    expected = np.where(np.isfinite(history)[:, None], history[:, None], expected.numpy())
    np.testing.assert_array_equal(result["report_cumulative"], expected)


def test_mixture_sampler_matches_joint_mean_covariance_and_component_frequency(models):
    model = models["two"]
    result = model.predict_distribution(_history()[:1], 90, 24000, np.random.default_rng(208))
    means, lower, weights = (x.numpy()[0] for x in model.conditional_mixture_parameters(_history()[:1], 90))
    expected_mean = weights @ means
    centered = means - expected_mean
    expected_covariance = lower @ lower.T + np.einsum("k,ki,kj->ij", weights, centered, centered)
    draws = result["latent_parameters"][0]
    np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=.007, rtol=0)
    np.testing.assert_allclose(np.cov(draws.T), expected_covariance, atol=.005, rtol=0)
    assert abs(result["mixture_component"].mean() - weights[1]) < .012


@pytest.mark.parametrize("mode", ["tied", "two"])
def test_positive_integrals_offgrid_tail_and_same_sample_ratios(models, mode):
    model = models[mode]
    result = model.predict_distribution(_history(), 90, 4, np.random.default_rng(23))
    assert (result["latent_rate"] >= 0).all()
    assert (np.diff(result["latent_cumulative"], axis=2) >= 0).all()
    assert (result["latent_total_volume"] >= result["latent_cumulative"][:, :, -1]).all()
    boundaries = np.array([0., 30., 135., 180., 1080., 1440.])
    nodes, weights = np.polynomial.legendre.leggauss(64)
    half_width = np.diff(boundaries) / 2
    query = (boundaries[1:] + boundaries[:-1])[:, None] / 2 + half_width[:, None] * nodes
    latent = result["latent_parameters"]
    rates = model.evaluate_trajectories(latent, query.ravel())["latent_rate"].reshape(2, 4, 5, 64, 3)
    integrated = (rates * weights[None, None, None, :, None]).sum(axis=3) * half_width[None, None, :, None]
    cumulative = model.evaluate_trajectories(latent, boundaries)["latent_cumulative"]
    np.testing.assert_allclose(integrated, np.diff(cumulative, axis=2), rtol=1e-9, atol=1e-9)
    report = result["report_cumulative"]
    np.testing.assert_allclose(report[:, :, -1] - report[:, :, 3],
        report[:, :, -1] - report[:, :, 5] + report[:, :, 5] - report[:, :, 3], rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("mode", ["single", "tied", "two"])
def test_checkpoint_metadata_and_same_rng_replay(models, mode, tmp_path):
    model = models[mode]
    path = tmp_path / "model.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["schema_version"] == (1 if mode == "single" else 2)
    if mode == "single":
        assert "mixture_mode" not in checkpoint["configuration"]
    else:
        assert checkpoint["configuration"]["mixture_mode"] == mode
    restored = ConditionalRateForecaster.load(path)
    assert restored.mixture_mode == mode
    assert restored.fit_diagnostics_ == model.fit_diagnostics_
    first = model.predict_distribution(_history(), 90, 16, np.random.default_rng(78))
    second = restored.predict_distribution(_history(), 90, 16, np.random.default_rng(78))
    for key in first:
        if isinstance(first[key], np.ndarray):
            np.testing.assert_array_equal(first[key], second[key])


def test_default_uses_original_iid_score_and_prediction_stream():
    with patch("joint_production.conditional_rates._stratified_scores", side_effect=AssertionError("default must use iid score")), \
            patch("joint_production.conditional_rates._u_scores", wraps=_u_scores) as score:
        model = ConditionalRateForecaster(epochs=1, hidden_width=8, batch_size=12, log_every=0).fit(_reports(), DAYS, origins=[90])
    assert score.call_count == 1
    assert score.call_args.args[1].shape[1] == 16
    history = _history()
    result = model.predict_distribution(history, 90, 5, np.random.default_rng(87))
    mean, lower = model.conditional_parameters(history, 90)
    epsilon = torch.tensor(np.random.default_rng(87).standard_normal((2, 5, 12)))
    expected = mean[:, None] + torch.einsum("nij,nsj->nsi", lower, epsilon)
    np.testing.assert_array_equal(result["latent_parameters"], expected.numpy())
    assert "mixture_component" not in result


@pytest.mark.parametrize("kwargs", [{"mixture_mode": "unknown"}, {"mixture_mode": "two", "train_samples": 3},
                                   {"mixture_mode": "tied", "train_samples": 2}])
def test_invalid_mixture_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ConditionalRateForecaster(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("mode", ["tied", "two"])
def test_cuda_mixture_fit_bridge_and_checkpoint_portability(mode, tmp_path):
    model = ConditionalRateForecaster(epochs=2, hidden_width=16, batch_size=6,
        train_samples=16, seed=19, log_every=0, history_rate_features="last_interval",
        mixture_mode=mode, device="cuda").fit(_reports(), DAYS, origins=[60, 90])
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    history = _history()
    result = model.predict_distribution(history, 90, 32, np.random.default_rng(841))
    contaminated = history.copy()
    contaminated[:, DAYS > 90] = np.inf
    second = model.predict_distribution(contaminated, 90, 32, np.random.default_rng(841))
    for key in result:
        if isinstance(result[key], np.ndarray):
            np.testing.assert_array_equal(result[key], second[key])
    observed = np.isfinite(history)
    np.testing.assert_array_equal(result["report_cumulative"].transpose(0, 2, 3, 1)[observed],
                                  np.repeat(history[observed, None], 32, axis=1))
    assert (result["latent_rate"] >= 0).all()
    assert (np.diff(result["latent_cumulative"], axis=2) >= 0).all()
    assert (result["report_cumulative"][:, :, 3, 1].std(axis=1) > 0).all()
    path = tmp_path / "cuda_mixture.pt"
    model.save(path)
    restored = ConditionalRateForecaster.load(path, device="cuda")
    replay = restored.predict_distribution(history, 90, 32, np.random.default_rng(841))
    for key in result:
        if isinstance(result[key], np.ndarray):
            np.testing.assert_array_equal(result[key], replay[key])
    portable = ConditionalRateForecaster.load(path, device="cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), portable.state_dict()[key], rtol=0, atol=0)
    assert portable.fit_diagnostics_ == model.fit_diagnostics_
    assert np.isfinite(portable.predict_samples(history, 90, 32, np.random.default_rng(841))).all()
