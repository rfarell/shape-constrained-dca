"""Exact first-moment loss integration, unchanged draw schedules and checkpoints."""
import copy
import os
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pytest
import torch

import joint_production.conditional_rates as core


DAYS = np.array([0., 30., 60., 90., 180.])
ORIGINS = [0., 60., 90.]
SETTINGS = dict(seed=57, epochs=1, hidden_width=8, batch_size=4, train_samples=16, log_every=0,
    history_rate_features="last_interval", covariate_preprocessing="external", history_dropout=.3, phase_dropout=.2)
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def inputs():
    rng = np.random.default_rng(281)
    y = rng.uniform(30, 150, (8, 1, 3))*(1-np.exp(-DAYS[None, :, None]/90))
    y[0, 1, 0] = 0.
    y[1, 2, 1] = y[1, 1, 1]-3.
    y[2, 2, 2] = np.nan
    x = np.column_stack((rng.normal(size=(8, 17)), np.ones((8, 17))))
    return y, x


def initialized(mode="analytic_mse", device="cpu"):
    model = core.ConditionalRateForecaster(mean_score_mode=mode, device=device, **SETTINGS)
    y, x = inputs()
    model._covariates(x, len(y), fit=True)
    model._initialize(DAYS, [100., 300., 80.], np.ones((7, 3)))
    model.origins_ = np.asarray(ORIGINS)
    model.fit_diagnostics_ = {"synthetic_initial_law": True}
    return model


def independent_mse(targets, mean, lower, days, origin_index, scales, weights):
    """Independent dense covariance and Erlang survival-polynomial oracle."""
    y, mu, chol, t, scale, weight = [value.double() for value in (targets, mean, lower, days, scales, weights)]
    covariance = chol @ chol.transpose(-1, -2)
    coefficients = (mu+.5*torch.diagonal(covariance, dim1=-2, dim2=-1)).exp().reshape(-1, 3, 4)
    u = t[:, None]/t.new_tensor([30., 180., 360., 720.])
    survival = torch.exp(-u)*torch.stack((torch.ones_like(u[:, 0]), torch.ones_like(u[:, 1]), 1+u[:, 2], 1+u[:, 3]), dim=-1)
    masses = survival[origin_index]-survival[origin_index+1:]
    predicted = torch.einsum("npk,tk->ntp", coefficients, masses)
    valid = torch.isfinite(y[:, origin_index, None]) & torch.isfinite(y[:, origin_index+1:])
    safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
    actual = safe[:, origin_index+1:]-safe[:, origin_index, None]
    error = torch.where(valid, predicted-actual, torch.zeros_like(predicted))/scale
    return (error.square()*weight).mean(dim=(-2, -1))


@pytest.mark.parametrize("device", DEVICES)
def test_no_new_parameters_rng_or_initial_issued_law(device):
    torch.manual_seed(718)
    cpu_before = torch.random.get_rng_state().clone()
    gpu_before = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    old, new = initialized("none", device), initialized(device=device)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    if gpu_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), gpu_before)
    assert old.initial_parameter_sha256_ == new.initial_parameter_sha256_
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    y, x = inputs()
    for origin in ORIGINS:
        a_rng, b_rng = np.random.default_rng(83), np.random.default_rng(83)
        a = old.predict_distribution(y, origin, 16, a_rng, x)
        b = new.predict_distribution(y, origin, 16, b_rng, x)
        assert a_rng.bit_generator.state == b_rng.bit_generator.state
        for key in ("latent_parameters", "latent_cumulative", "latent_rate", "latent_total_volume", "report_cumulative"):
            np.testing.assert_array_equal(a[key], b[key])
    assert not any(key.startswith("mean_score_") for key in a)
    assert b["mean_score_coefficient"] == 1.


@pytest.mark.parametrize("changes", [dict(rate_family="dca"), dict(mixture_mode="tied"),
    dict(history_head_mode="no_history"), dict(coefficient_parameterization="log_expectation"),
    dict(tree_static_mode="fixed_empty_history"), dict(earlier_history_features="actual"),
    dict(report_half_life_days=180.), dict(history_rate_features="zero"),
    dict(covariate_preprocessing="internal"), dict(marginal_score_estimator="gaussian_report_integrated"),
    dict(train_samples=4), dict(train_samples=256, energy_samples=16),
    dict(train_samples=256, energy_samples=16, marginal_objective="empirical_wis"),
    dict(report_scale_mode="contextual_multiplier")])
def test_only_fixed_retained_configuration_supports_mean_objective(changes):
    with pytest.raises(ValueError):
        core.ConditionalRateForecaster(**(SETTINGS | changes), mean_score_mode="analytic_mse")


@pytest.mark.parametrize("device", DEVICES)
def test_full_covariance_mean_mse_value_and_gradients_match_independent_oracle(device):
    model = initialized(device=device)
    y, _ = inputs()
    target = torch.tensor(y/model.volume_scale.cpu().numpy(), dtype=torch.float32, device=device)
    times = model.report_days.float()
    c = torch.linspace(-.8, 1., 8*model.network.hidden[0].in_features, device=device).reshape(8, -1)
    with torch.no_grad():
        model.network.cholesky.bias.add_(torch.linspace(-.5, .8, 78, device=device))
    mean, lower = model.network(c)
    assert torch.tril(lower, -1).abs().sum() > 0
    scales = torch.tensor([[.7, 1.3, .2], [.9, 1.2, .4]], device=device)
    weights = torch.tensor([[1., 1., 8/7], [1., 1., 8/7]], device=device)
    actual = core.fast_basis_increment_mean_mse(target, mean, lower, times, 2, scales, weights)
    expected = independent_mse(target, mean, lower, times, 2, scales, weights)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    gradient_inputs = (mean, lower, model.network.mean.bias, model.network.cholesky.bias)
    a = torch.autograd.grad(actual.sum(), gradient_inputs, retain_graph=True)
    b = torch.autograd.grad(expected.sum(), gradient_inputs, retain_graph=True)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-6)
        assert torch.isfinite(left).all() and left.abs().sum() > 0
    assert torch.tril(a[1], -1).abs().sum() > 0
    assert torch.autograd.grad(actual.sum(), (model.noise_abs_raw, model.noise_rel_raw), allow_unused=True) == (None, None)


@pytest.fixture(scope="module")
def fitted():
    y, x = inputs()
    models, random_banks, terms = {}, {}, []
    original_randn = torch.randn
    original_mse = core.fast_basis_increment_mean_mse
    original_scores = core._u_scores
    for mode in ("none", "analytic_mse"):
        banks, scores = [], []
        def randn(*args, **kwargs):
            output = original_randn(*args, **kwargs)
            banks.append(output.clone())
            return output
        def score(*args):
            output = original_scores(*args)
            scores.append((args, tuple(value.detach().clone() for value in output)))
            return output
        def mse(*args):
            output = original_mse(*args)
            oracle = independent_mse(*args)
            torch.testing.assert_close(output, oracle, rtol=2e-12, atol=2e-12)
            score_args, score_values = scores[-1]
            assert torch.equal(args[0], score_args[0]) or torch.allclose(args[0], score_args[0], equal_nan=True)
            assert args[4] == score_args[4]
            assert torch.equal(args[5], score_args[5]) and torch.equal(args[6], score_args[6])
            terms.append(dict(n=len(args[0]), origin=int(args[4]),
                mse=output.detach().numpy(), marginal=score_values[0].numpy(), energy=score_values[1].numpy()))
            return output
        with patch("torch.randn", side_effect=randn), patch.object(core, "_u_scores", side_effect=score), patch.object(core, "fast_basis_increment_mean_mse", side_effect=mse):
            models[mode] = core.ConditionalRateForecaster(mean_score_mode=mode, **SETTINGS).fit(y, DAYS, origins=ORIGINS, covariates=x)
        random_banks[mode] = banks
    return models, random_banks, terms


def test_exact_mask_count_origin_weighting_and_logged_objective(fitted):
    models, random_banks, terms = fitted
    old, new = models.values()
    for key in ("initial_parameter_sha256", "context_mask_sha256_by_epoch", "rng_seeds", "phase_scales_native",
                "increment_scales_native", "increment_observation_counts", "optimizer_updates"):
        assert old.fit_diagnostics_[key] == new.fit_diagnostics_[key]
    for a, b in zip(*random_banks.values(), strict=True):
        assert torch.equal(a, b)
    weights = {0: 4/(7/3), 2: 2/(7/3), 3: 1/(7/3)}
    n = sum(term["n"] for term in terms)
    mse = sum(np.sum(term["mse"]*weights[term["origin"]]) for term in terms)/n
    marginal = sum(float(torch.tensor(term["marginal"]).__mul__(weights[term["origin"]]).mean())*term["n"] for term in terms)/n
    energy = sum(float(torch.tensor(term["energy"]).mean())*term["n"] for term in terms)/n
    entry = new.fit_diagnostics_["optimization"][0]
    assert entry["macro_increment_mean_mse"] == pytest.approx(mse, rel=1e-14)
    assert entry["macro_increment_crps"] == marginal
    assert entry["joint_energy"] == energy
    assert entry["loss"] == pytest.approx(marginal+energy+mse, rel=2e-7)
    assert new.fit_diagnostics_["score_coefficients"] == [1., 1., 1.]
    assert old.fit_diagnostics_["score_coefficients"] == [1., 1.]
    assert "mean_score_mode" not in old.fit_diagnostics_


def test_zero_added_term_leaves_entire_old_training_path_exact():
    y, x = inputs()
    old = core.ConditionalRateForecaster(**SETTINGS).fit(y, DAYS, origins=ORIGINS, covariates=x)
    with patch.object(core, "fast_basis_increment_mean_mse", side_effect=lambda target, mean, *args: mean.double().sum(-1)*0):
        candidate = core.ConditionalRateForecaster(mean_score_mode="analytic_mse", **SETTINGS).fit(y, DAYS, origins=ORIGINS, covariates=x)
    for key, value in old.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    assert old.fit_diagnostics_["optimization"][0]["loss"] == candidate.fit_diagnostics_["optimization"][0]["loss"]


def test_checkpoint_and_future_mask_replay(fitted, tmp_path):
    model = fitted[0]["analytic_mse"]
    path = tmp_path/"candidate.pt"
    model.save(path)
    restored = core.ConditionalRateForecaster.load(path)
    assert restored.fit_diagnostics_ == model.fit_diagnostics_
    y, x = inputs()
    y[0, :, 1] = np.nan
    for origin in ORIGINS:
        clean = y.copy()
        clean[:, DAYS > origin] = np.nan
        poisoned = clean.copy()
        poisoned[:, DAYS > origin] = np.inf
        a = model.predict_distribution(clean, origin, 16, np.random.default_rng(25), x)
        b = restored.predict_distribution(poisoned, origin, 16, np.random.default_rng(25), x)
        for key in ("latent_parameters", "report_cumulative", "latent_cumulative"):
            np.testing.assert_array_equal(a[key], b[key])
        assert a["mean_score_mode"] == b["mean_score_mode"] == "analytic_mse"


@pytest.mark.parametrize("corruption", ["config_mode", "fit_mode", "fit_configuration", "coefficient", "meaning", "arithmetic",
    "objective", "coefficients", "count", "interpretation", "log_missing", "log_nan", "log_negative"])
def test_checkpoint_mean_score_metadata_mutations_rejected(fitted, tmp_path, corruption):
    path = tmp_path/"candidate.pt"
    fitted[0]["analytic_mse"].save(path)
    checkpoint = torch.load(path, weights_only=True)
    fit = checkpoint["fit_diagnostics"]
    if corruption == "config_mode": checkpoint["configuration"].pop("mean_score_mode")
    elif corruption == "fit_mode": fit["mean_score_mode"] = "none"
    elif corruption == "fit_configuration": fit["configuration"]["mean_score_mode"] = "none"
    elif corruption == "coefficient": checkpoint["mean_score_coefficient"] = .5
    elif corruption == "meaning": fit["mean_score_meaning"] = "squared Monte Carlo mean error"
    elif corruption == "arithmetic": checkpoint["mean_score_arithmetic"] = "all float32"
    elif corruption == "objective": fit["objective"] = "mse only"
    elif corruption == "coefficients": fit["score_coefficients"] = [1., 1.]
    elif corruption == "count": fit["train_samples_iid"] = 256
    elif corruption == "interpretation": fit["interpretation"] = "identified latent truth"
    elif corruption == "log_missing": fit["optimization"][0].pop("macro_increment_mean_mse")
    elif corruption == "log_nan": fit["optimization"][0]["macro_increment_mean_mse"] = float("nan")
    elif corruption == "log_negative": fit["optimization"][0]["macro_increment_mean_mse"] = -1.
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        core.ConditionalRateForecaster.load(path)


def test_default_omits_and_rejects_stray_mean_metadata(fitted, tmp_path):
    model = fitted[0]["none"]
    path = tmp_path/"control.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    for metadata in (checkpoint, checkpoint["configuration"], checkpoint["fit_diagnostics"]):
        assert not any(key.startswith("mean_score_") for key in metadata)
    restored = core.ConditionalRateForecaster.load(path)
    assert restored.fit_diagnostics_ == model.fit_diagnostics_
    checkpoint["mean_score_coefficient"] = 1.
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        core.ConditionalRateForecaster.load(path)


@pytest.mark.parametrize("remaining", ["objective", "coefficients", "optimization"])
def test_stripped_analytic_mode_cannot_mislabel_a_checkpoint_as_default(fitted, tmp_path, remaining):
    path = tmp_path/"mislabeled.pt"
    fitted[0]["analytic_mse"].save(path)
    checkpoint = torch.load(path, weights_only=True)
    fit = checkpoint["fit_diagnostics"]
    for metadata in (checkpoint, checkpoint["configuration"], fit, fit["configuration"]):
        for key in list(metadata):
            if key.startswith("mean_score_"):
                metadata.pop(key)
    if remaining != "objective": fit["objective"] = "macro_scaled_increment_crps_u + mean_origin_report_energy_u/sqrt(d)"
    if remaining != "coefficients": fit["score_coefficients"] = [1., 1.]
    if remaining != "optimization":
        for row in fit["optimization"]:
            row.pop("macro_increment_mean_mse")
            row.pop("mean_score_mode")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="contradicts analytic"):
        core.ConditionalRateForecaster.load(path)
