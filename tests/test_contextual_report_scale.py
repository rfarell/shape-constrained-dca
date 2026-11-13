"""Synthetic context-fixed report clocks, initialization and saved-law contracts."""
import copy
import os
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pytest
import torch

from joint_production.conditional_rates import ConditionalRateForecaster, _u_scores
from joint_production.torch_observation import sample_accumulated_reports


DAYS = np.array([0., 30., 60., 90., 180.])
SETTINGS = dict(seed=37, epochs=1, hidden_width=8, batch_size=4, train_samples=16,
                log_every=0, history_rate_features="last_interval", covariate_preprocessing="external",
                history_dropout=.3, phase_dropout=.2)
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def inputs():
    rng = np.random.default_rng(991)
    y = rng.uniform(30, 130, (8, 1, 3)) * (1-np.exp(-DAYS[None, :, None]/80))
    y[0, 1, 0] = 0.
    y[1, 2, 1] = y[1, 1, 1] - 2.  # Signed reporting correction.
    x = np.column_stack((rng.normal(size=(8, 17)), np.ones((8, 17))))
    return y, x


def initialized(mode="contextual_multiplier", device="cpu"):
    model = ConditionalRateForecaster(report_scale_mode=mode, device=device, **SETTINGS)
    y, x = inputs()
    model._covariates(x, len(y), fit=True)
    model._initialize(DAYS, [100., 300., 80.], np.ones((6, 3)))
    model.origins_ = np.array([0., 60.])
    model.fit_diagnostics_ = {"synthetic_initialization": True}
    return model


def context(model, history, origin, x):
    reports = model._history(history, origin)
    normalized = reports / model.volume_scale.cpu().numpy()
    values = torch.as_tensor(normalized, dtype=torch.float32, device=model.device)
    observed = torch.isfinite(values)
    times = model.report_days.float()
    covariates = torch.as_tensor(model._covariates(x, len(reports), fit=False), device=model.device)
    native = torch.as_tensor(reports, dtype=torch.float64, device=model.device)
    return model._encoder_context(values, observed, times, origin, native_reports=native,
                                  covariates=covariates), values, observed, times


@pytest.mark.parametrize("device", DEVICES)
def test_zero_head_matches_complete_global_initial_law_and_rng(device):
    torch.manual_seed(827)
    cpu_state = torch.random.get_rng_state().clone()
    gpu_state = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    old, new = initialized("global", device), initialized(device=device)
    assert torch.equal(torch.random.get_rng_state(), cpu_state)
    if gpu_state is not None:
        assert torch.equal(torch.cuda.get_rng_state(), gpu_state)
    assert old.initial_parameter_sha256_ == new.common_initial_parameter_sha256_
    assert old.initial_parameter_sha256_ != new.initial_parameter_sha256_
    assert new.report_scale_head.bias is None
    assert new.report_scale_head.weight.device.type == device
    assert torch.count_nonzero(new.report_scale_head.weight) == 0
    shared = dict(old.named_parameters())
    assert set(dict(new.named_parameters())) == set(shared) | {"report_scale_head.weight"}
    for name, value in shared.items():
        assert torch.equal(value, dict(new.named_parameters())[name])
    y, x = inputs()
    y[1, :, 2] = np.nan
    for origin in (0, 60):
        left_rng, right_rng = np.random.default_rng(14), np.random.default_rng(14)
        left = old.predict_distribution(y, origin, 16, left_rng, x)
        right = new.predict_distribution(y, origin, 16, right_rng, x)
        assert left_rng.bit_generator.state == right_rng.bit_generator.state
        for key in ("latent_parameters", "latent_cumulative", "latent_rate", "latent_total_volume", "report_cumulative"):
            np.testing.assert_array_equal(left[key], right[key])
        np.testing.assert_array_equal(right["report_scale_multiplier"], np.ones((8, 3)))
        for key in ("historical_latent_rmse_scaled_by_phase", "historical_abs_standardized_innovation_median_by_phase",
                    "historical_abs_standardized_innovation_gt3_fraction_by_phase"):
            assert left["conditional_diagnostics"][key] == right["conditional_diagnostics"][key]
    assert "report_scale_mode" not in old.configuration
    assert "report_scale_multiplier" not in left
    assert not any(key.startswith("report_scale") for key in old.state_dict())


def test_explicit_global_omission_and_original_effective_tensors():
    y, x = inputs()
    default = ConditionalRateForecaster(**SETTINGS).fit(y, DAYS, origins=[0, 60], covariates=x)
    explicit = ConditionalRateForecaster(report_scale_mode="global", **SETTINGS).fit(y, DAYS, origins=[0, 60], covariates=x)
    assert default.configuration == explicit.configuration
    assert default.fit_diagnostics_ == explicit.fit_diagnostics_
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    a, r, g = default.effective_noise(None)
    assert torch.equal(a, default.noise_abs) and torch.equal(r, default.noise_rel) and g is None


@pytest.mark.parametrize("option", [dict(rate_family="dca"), dict(mixture_mode="tied"),
    dict(history_head_mode="no_history"), dict(coefficient_parameterization="log_expectation"),
    dict(tree_static_mode="fixed_empty_history"), dict(earlier_history_features="actual"),
    dict(report_half_life_days=180.), dict(history_rate_features="none"),
    dict(covariate_preprocessing="internal"), dict(marginal_score_estimator="gaussian_report_integrated"),
    dict(train_samples=4), dict(train_samples=256, energy_samples=16),
    dict(train_samples=256, energy_samples=16, marginal_objective="empirical_wis")])
def test_closed_mode_combinations_are_rejected(option):
    with pytest.raises(ValueError):
        ConditionalRateForecaster(**(SETTINGS | option), report_scale_mode="contextual_multiplier")


def test_nontrivial_scale_formula_bounds_and_masked_future_poisoning():
    model = initialized()
    y, x = inputs()
    with torch.no_grad():
        model.report_scale_head.weight.copy_(torch.linspace(-.07, .09, model.report_scale_head.weight.numel()).reshape_as(model.report_scale_head.weight))
    history = y.copy()
    history[0, :, 1] = np.nan
    history[1, 2, 2] = np.nan  # Hidden cutoff endpoint.
    clean = history.copy()
    clean[:, DAYS > 60] = np.nan
    poison = clean.astype(object)
    poison[:, DAYS > 60] = "future must never enter encoder"
    left = model.predict_distribution(clean, 60, 16, np.random.default_rng(3), x)
    right = model.predict_distribution(poison, 60, 16, np.random.default_rng(3), x)
    for key in ("report_scale_multiplier", "noise_abs_scaled", "noise_rel", "latent_parameters", "report_cumulative"):
        np.testing.assert_array_equal(left[key], right[key])
    c, _, _, _ = context(model, clean, 60, x)
    expected = torch.exp(np.log(4.)*torch.tanh(c @ model.report_scale_head.weight.T)).detach().numpy()
    np.testing.assert_array_equal(left["report_scale_multiplier"], expected.astype(float))
    assert np.all((expected >= .25) & (expected <= 4)) and not np.all(expected == 1)
    for key, base in (("noise_abs_scaled", model.noise_abs), ("noise_rel", model.noise_rel)):
        exact = (expected*base.detach().numpy()).astype(float)
        np.testing.assert_array_equal(left[key], exact)
    known = np.isfinite(clean)
    np.testing.assert_array_equal(left["report_cumulative"].transpose(0, 2, 3, 1)[known], np.repeat(clean[known, None], 16, axis=1))
    invalid = clean.copy()
    invalid[0, 1, 0] = np.inf
    with pytest.raises(ValueError):
        model.predict_distribution(invalid, 60, 16, np.random.default_rng(3), x)


def test_same_context_noise_drives_sampled_loss_gradients_and_history_standardization():
    model = initialized()
    y, x = inputs()
    c, values, observed, times = context(model, y, 60, x)
    mean, lower = model._network_parameters(c, observed, times)
    generator = torch.Generator().manual_seed(88)
    z = mean[:, None]+torch.einsum("nij,nsj->nsi", lower, torch.randn((8, 16, 12), generator=generator))
    q = model.rate_model.cumulative(z, times)
    a, r, g = model.effective_noise(c)
    draws = model._sample_reports(q, values, times, a, r, cutoff=60., generator=generator)
    actual = torch.as_tensor(y/model.volume_scale.cpu().numpy(), dtype=torch.float32)
    marginal, energy, _ = _u_scores(actual, draws, observed, times, 2, torch.ones((2, 3)), torch.ones((2, 3)))
    gradients = torch.autograd.grad((marginal+energy).sum(), tuple(model.parameters()))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    named = dict(zip(dict(model.named_parameters()), gradients, strict=True))
    for key in ("report_scale_head.weight", "noise_abs_raw", "noise_rel_raw", "network.mean.bias", "network.cholesky.bias"):
        assert named[key].abs().sum() > 0
    # Scaling both innovation SDs leaves conditional bridge fractions unchanged;
    # the historical standardized discrepancy changes by exactly inverse g.
    reports = model._history(y, 60)
    native_q = q.detach().numpy()*model.volume_scale.numpy()
    base = model._history_discrepancies(reports, native_q, noise_abs=model.noise_abs, noise_rel=model.noise_rel)
    scaled = model._history_discrepancies(reports, native_q, noise_abs=2*model.noise_abs.expand(8, 3), noise_rel=2*model.noise_rel.expand(8, 3))
    np.testing.assert_allclose(scaled["historical_abs_standardized_innovation_median_by_phase"],
        .5*np.asarray(base["historical_abs_standardized_innovation_median_by_phase"]), rtol=0, atol=0)
    with pytest.raises(ValueError):
        model._history_discrepancies(reports, native_q)


@pytest.fixture(scope="module")
def fitted():
    y, x = inputs()
    models, streams = {}, {}
    original = torch.randn
    for mode in ("global", "contextual_multiplier"):
        draws = []
        def record(*args, **kwargs):
            value = original(*args, **kwargs)
            draws.append(value.clone())
            return value
        with patch("torch.randn", side_effect=record):
            models[mode] = ConditionalRateForecaster(report_scale_mode=mode, **SETTINGS).fit(y, DAYS, origins=[0, 60], covariates=x)
        streams[mode] = draws
    return models, streams


def test_fit_pairing_and_actual_nonzero_head_training(fitted):
    models, streams = fitted
    old, new = models.values()
    assert torch.count_nonzero(new.report_scale_head.weight) > 0
    assert old.initial_parameter_sha256_ == new.common_initial_parameter_sha256_
    for key in ("context_mask_sha256_by_epoch", "rng_seeds", "phase_scales_native", "increment_scales_native", "optimizer_updates"):
        assert old.fit_diagnostics_[key] == new.fit_diagnostics_[key]
    for a, b in zip(*streams.values(), strict=True):
        assert torch.equal(a, b)
    assert "report_scale_mode" not in old.fit_diagnostics_
    assert new.fit_diagnostics_["report_scale_mode"] == "contextual_multiplier"


def test_checkpoint_replay_and_exact_nontrivial_effective_clock(fitted, tmp_path):
    model = fitted[0]["contextual_multiplier"]
    path = tmp_path/"model.pt"
    model.save(path)
    restored = ConditionalRateForecaster.load(path)
    y, x = inputs()
    for origin in (0, 60):
        a = model.predict_distribution(y, origin, 16, np.random.default_rng(73), x)
        b = restored.predict_distribution(y, origin, 16, np.random.default_rng(73), x)
        for key in ("latent_parameters", "report_cumulative", "report_scale_multiplier", "noise_abs_scaled", "noise_rel"):
            np.testing.assert_array_equal(a[key], b[key])
        assert not np.all(a["report_scale_multiplier"] == 1)
        # Independently replay the issued report bank using saved effective SDs.
        rng = np.random.default_rng(73)
        rng.standard_normal((8, 16, 12))
        report_normals = torch.tensor(rng.standard_normal((8, 16, 5, 3)))
        history = model._history(y, origin)
        phase = model.volume_scale.numpy()
        replay = sample_accumulated_reports(torch.tensor(a["latent_cumulative"]/phase), torch.tensor(history/phase),
            model.report_days, torch.tensor(a["noise_abs_scaled"]), torch.tensor(a["noise_rel"]),
            cutoff=origin, standard_normals=report_normals).numpy()*phase
        replay = np.where(np.isfinite(history)[:, None], history[:, None], replay)
        np.testing.assert_array_equal(replay, a["report_cumulative"])


def test_checkpoint_preserves_recorded_base_rounding_for_device_portability(fitted, tmp_path):
    """Simulate a one-ulp training-device transform without requiring CUDA."""
    path = tmp_path/"model.pt"
    fitted[0]["contextual_multiplier"].save(path)
    checkpoint = torch.load(path, weights_only=True)
    for key in ("noise_abs_scaled", "noise_rel"):
        saved = np.asarray(checkpoint["fit_diagnostics"][key], dtype=np.float32)
        checkpoint["fit_diagnostics"][key] = np.nextafter(saved, np.float32(np.inf)).astype(float).tolist()
    torch.save(checkpoint, path)
    restored = ConditionalRateForecaster.load(path)
    y, x = inputs()
    output = restored.predict_distribution(y, 60, 16, np.random.default_rng(3), x)
    for key in ("noise_abs_scaled", "noise_rel"):
        base = np.asarray(checkpoint["fit_diagnostics"][key], dtype=np.float32)
        expected = (output["report_scale_multiplier"].astype(np.float32)*base).astype(float)
        np.testing.assert_array_equal(output[key], expected)
    with torch.no_grad():
        restored.noise_abs_raw.add_(.01)
    with pytest.raises(ValueError, match="stale"):
        restored.predict_distribution(y, 60, 16, np.random.default_rng(3), x)


def test_refit_refreshes_recorded_bases(fitted):
    model = copy.deepcopy(fitted[0]["contextual_multiplier"])
    y, x = inputs()
    model.fit(y*1.1, DAYS, origins=[0, 60], covariates=x)
    assert model.fit_diagnostics_["report_scale_base_raw_sha256"] == model._report_scale_base_raw_sha256()
    result = model.predict_distribution(y, 60, 16, np.random.default_rng(3), x)
    assert np.isfinite(result["report_cumulative"]).all()


def test_post_mask_encoder_and_scale_ignore_excluded_native_values():
    model = initialized()
    y, x = inputs()
    native = torch.tensor(y, dtype=torch.float64)
    values = (native/model.volume_scale).float()
    times = model.report_days.float()
    observed = torch.isfinite(values) & (times <= 60)[None, :, None]
    observed[0, :, 1] = False
    observed[1, 2, :] = False
    observed[:, 0] = True
    kwargs = dict(covariates=torch.tensor(x, dtype=torch.float32))
    first = model._encoder_context(values, observed, times, 60., native_reports=native, **kwargs)
    native[~observed] = float("inf")
    values[~observed] = float("inf")
    second = model._encoder_context(values, observed, times, 60., native_reports=native, **kwargs)
    assert torch.equal(first, second)
    with torch.no_grad():
        model.report_scale_head.weight.fill_(.03)
    for a, b in zip(model.effective_noise(first), model.effective_noise(second), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("corruption", ["mode", "fit_mode", "fit_config", "bounds", "meaning", "arithmetic", "head_missing",
    "head_shape", "head_nan", "head_dtype", "head_value", "head_hash", "common_hash", "base_noise", "sample_count"])
def test_checkpoint_mutations_fail_closed(fitted, tmp_path, corruption):
    path = tmp_path/"model.pt"
    fitted[0]["contextual_multiplier"].save(path)
    checkpoint = torch.load(path, weights_only=True)
    if corruption == "mode": checkpoint["configuration"].pop("report_scale_mode")
    elif corruption == "fit_mode": checkpoint["fit_diagnostics"]["report_scale_mode"] = "global"
    elif corruption == "fit_config": checkpoint["fit_diagnostics"]["configuration"]["report_scale_mode"] = "global"
    elif corruption == "bounds": checkpoint["report_scale_multiplier_bounds"] = [.1, 10.]
    elif corruption == "meaning": checkpoint["report_scale_interpretation"] = "identified meter noise"
    elif corruption == "arithmetic": checkpoint["report_scale_arithmetic"] = "float64 products"
    elif corruption == "head_missing": checkpoint["state_dict"].pop("report_scale_head.weight")
    elif corruption == "head_shape": checkpoint["state_dict"]["report_scale_head.weight"] = torch.zeros((3, 1))
    elif corruption == "head_nan": checkpoint["state_dict"]["report_scale_head.weight"][0, 0] = float("nan")
    elif corruption == "head_dtype": checkpoint["state_dict"]["report_scale_head.weight"] = checkpoint["state_dict"]["report_scale_head.weight"].double()
    elif corruption == "head_value": checkpoint["state_dict"]["report_scale_head.weight"][0, 0] += 1
    elif corruption == "head_hash": checkpoint["report_scale_head_sha256"] = "0"*64
    elif corruption == "common_hash":
        checkpoint["common_initial_parameter_sha256"] = "0"*64
        checkpoint["fit_diagnostics"]["common_initial_parameter_sha256"] = "0"*64
    elif corruption == "base_noise": checkpoint["fit_diagnostics"]["noise_abs_scaled"][0] *= 2
    elif corruption == "sample_count": checkpoint["fit_diagnostics"]["train_samples_iid"] = 32
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        ConditionalRateForecaster.load(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_checkpoint_cpu_portability(tmp_path):
    y, x = inputs()
    model = ConditionalRateForecaster(report_scale_mode="contextual_multiplier", device="cuda", **SETTINGS).fit(y, DAYS, origins=[0, 60], covariates=x)
    path = tmp_path/"gpu.pt"
    model.save(path)
    restored = ConditionalRateForecaster.load(path, device="cpu")
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), restored.state_dict()[key])
    assert np.isfinite(restored.predict_distribution(y, 60, 16, np.random.default_rng(1), x)["report_cumulative"]).all()
