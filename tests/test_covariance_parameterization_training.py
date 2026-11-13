"""Covariance-map integration, same-law initialization, and checkpoint provenance."""
import hashlib
import math
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

import joint_production.conditional_rates as core


ROOT = Path(__file__).resolve().parents[1]
DAYS = np.array([0., 30., 60., 90., 180.])
ORIGINS = [0., 60., 90.]
SETTINGS = dict(seed=57, epochs=2, hidden_width=8, batch_size=4, train_samples=16,
    log_every=0, history_rate_features="last_interval", covariate_preprocessing="external",
    history_dropout=.3, phase_dropout=.2)
MODE = "marginal_correlation"
ANCHOR_BUFFER = "network.covariance_diagonal_anchor_logit"
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def inputs():
    rng = np.random.default_rng(281)
    y = rng.uniform(30, 150, (8, 1, 3)) * (1 - np.exp(-DAYS[None, :, None] / 90))
    y[0, 1, 0] = 0.
    y[1, 2, 1] = y[1, 1, 1] - 3.
    y[2, 2, 2] = np.nan
    x = np.column_stack((rng.normal(size=(8, 17)), np.ones((8, 17))))
    return y, x


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


def initialized(mode="bounded_cholesky", device="cpu"):
    model = core.ConditionalRateForecaster(covariance_mode=mode, device=device, **SETTINGS)
    y, x = inputs()
    model._covariates(x, len(y), fit=True)
    model._initialize(DAYS, [100., 300., 80.], np.ones((7, 3)))
    model.origins_ = np.asarray(ORIGINS)
    model.fit_diagnostics_ = {"synthetic_initial_law": True}
    model.eval()
    return model


def equal_common_state(old, new):
    candidate = new.state_dict()
    anchor = candidate.pop(ANCHOR_BUFFER)
    assert anchor.dtype == torch.float64 and anchor.shape == () and not anchor.requires_grad
    assert torch.isfinite(anchor)
    equal(old.state_dict(), candidate)


@pytest.mark.parametrize("device", DEVICES)
def test_no_new_parameters_rng_and_exact_complete_initial_law(device):
    torch.manual_seed(718)
    cpu_before = torch.random.get_rng_state().clone()
    gpu_before = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    old, new = initialized(device=device), initialized(MODE, device)
    equal(torch.random.get_rng_state(), cpu_before)
    if gpu_before is not None:
        equal(torch.cuda.get_rng_state(), gpu_before)
    assert old.initial_parameter_sha256_ == new.initial_parameter_sha256_
    equal_common_state(old, new)
    equal(dict(old.named_parameters()), dict(new.named_parameters()))
    y, x = inputs()
    for origin in ORIGINS:
        old_mean, old_lower = old.conditional_parameters(y, origin, x)
        new_mean, new_lower = new.conditional_parameters(y, origin, x)
        equal(old_mean, new_mean)
        equal(old_lower, new_lower)
        if device == "cpu":
            equal(new_lower, .25 * torch.eye(12, device=device, dtype=new_lower.dtype).expand(8, -1, -1))
        equal(new_lower.diagonal(dim1=-2, dim2=-1), torch.full((8, 12),
            new._covariance_metadata()["covariance_initial_legacy_std_float32"], device=device, dtype=new_lower.dtype))
        a_rng, b_rng = np.random.default_rng(83), np.random.default_rng(83)
        a = old.predict_distribution(y, origin, 16, a_rng, x)
        b = new.predict_distribution(y, origin, 16, b_rng, x)
        assert a_rng.bit_generator.state == b_rng.bit_generator.state
        for key in ("latent_parameters", "latent_cumulative", "latent_rate", "latent_total_volume", "report_cumulative"):
            equal(a[key], b[key])
        assert not any(key.startswith("covariance_") for key in a)
        for metadata in (b, b["conditional_diagnostics"]):
            for key, value in new._covariance_metadata().items():
                equal(metadata[key], value)


@pytest.mark.parametrize("changes", [dict(rate_family="dca"), dict(mixture_mode="tied"),
    dict(mixture_mode="two"), dict(history_head_mode="no_history"),
    dict(coefficient_parameterization="log_expectation"), dict(tree_static_mode="fixed_empty_history"),
    dict(earlier_history_features="actual"), dict(report_half_life_days=180.),
    dict(history_rate_features="zero"), dict(history_rate_features="none"),
    dict(covariate_preprocessing="internal"), dict(marginal_score_estimator="gaussian_report_integrated"),
    dict(train_samples=4), dict(train_samples=256, energy_samples=16),
    dict(train_samples=256, energy_samples=16, marginal_objective="empirical_wis"),
    dict(report_scale_mode="contextual_multiplier"), dict(mean_score_mode="analytic_mse"),
    dict(initial_latent_std=.3), dict(min_latent_std=.02), dict(max_latent_std=1.2), dict(max_offdiag=.3)])
def test_candidate_rejects_other_experimental_modes_and_changed_scales(changes):
    with pytest.raises(ValueError):
        core.ConditionalRateForecaster(**(SETTINGS | changes), covariance_mode=MODE)


def test_invalid_covariance_mode_rejected():
    with pytest.raises(ValueError, match="covariance_mode"):
        core.ConditionalRateForecaster(**SETTINGS, covariance_mode="unbounded")


def fit_trace(cls=core.ConditionalRateForecaster, mode=None, device="cpu", freeze=False):
    y, x = inputs()
    kwargs = {} if mode is None else {"covariance_mode": mode}
    model = cls(**SETTINGS, device=device, **kwargs)
    trace = {"normals": [], "contexts": [], "gradients": []}
    original_randn, original_context = torch.randn, cls._encoder_context
    original_clip = torch.nn.utils.clip_grad_norm_
    def draw(*args, **kwargs):
        result = original_randn(*args, **kwargs)
        trace["normals"].append(result.detach().cpu().clone())
        return result
    def context(self, *args, **kwargs):
        result = original_context(self, *args, **kwargs)
        trace["contexts"].append(result.detach().cpu().clone())
        return result
    def clip(parameters, *args, **kwargs):
        values = list(parameters)
        trace["gradients"].append([None if p.grad is None else p.grad.detach().cpu().clone() for p in values])
        return original_clip(values, *args, **kwargs)
    with patch("torch.randn", side_effect=draw), patch.object(cls, "_encoder_context", context), patch("torch.nn.utils.clip_grad_norm_", side_effect=clip):
        if freeze:
            with patch("torch.optim.Adam.step", return_value=None):
                model.fit(y, DAYS, origins=ORIGINS, covariates=x)
        else:
            model.fit(y, DAYS, origins=ORIGINS, covariates=x)
    return model, trace


@pytest.fixture(scope="module")
def fitted():
    return {mode: fit_trace(mode=mode) for mode in (None, "bounded_cholesky", MODE)}


def test_matched_draws_information_and_fixed_objective_with_real_updates(fitted):
    old, a = fitted[None]
    new, b = fitted[MODE]
    for key in ("initial_parameter_sha256", "context_mask_sha256_by_epoch", "rng_seeds", "phase_scales_native",
        "increment_scales_native", "increment_observation_counts", "optimizer_updates", "total_training_contexts",
        "uniform_origin_marginal_weights", "score_coefficients", "objective", "train_samples_iid"):
        equal(old.fit_diagnostics_[key], new.fit_diagnostics_[key])
    equal(a["normals"], b["normals"])
    equal(a["contexts"], b["contexts"])
    assert old.fit_diagnostics_["score_coefficients"] == [1., 1.]
    assert old.observation_law == new.observation_law == "conditional_gaussian_accumulated_bridge"
    assert any(not torch.equal(value, old.state_dict()[key]) for key, value in new.state_dict().items() if key != ANCHOR_BUFFER)
    for row in new.fit_diagnostics_["optimization"]:
        assert row["covariance_mode"] == MODE
        assert np.isfinite([row[key] for key in ("loss", "macro_increment_crps", "joint_energy", "max_gradient_norm")]).all()
    for step in b["gradients"]:
        assert all(value is None or torch.isfinite(value).all() for value in step)


@pytest.mark.parametrize("device", DEVICES)
def test_initial_training_graph_losses_and_gradients_match_with_updates_frozen(device):
    old, a = fit_trace(device=device, freeze=True)
    new, b = fit_trace(mode=MODE, device=device, freeze=True)
    equal_common_state(old, new)
    equal(a["contexts"], b["contexts"])
    equal(a["normals"], b["normals"])
    for left, right in zip(old.fit_diagnostics_["optimization"], new.fit_diagnostics_["optimization"], strict=True):
        for key in ("loss", "macro_increment_crps", "joint_energy", "noise_penalty"):
            assert left[key] == right[key]
    for left, right in zip(a["gradients"], b["gradients"], strict=True):
        for old_grad, new_grad in zip(left, right, strict=True):
            if old_grad is None:
                assert new_grad is None
            else:
                # Decoder Jacobians agree to float32 precision; accumulation
                # in the unchanged score graph may add float32 rounding.
                torch.testing.assert_close(old_grad, new_grad, rtol=8 * torch.finfo(torch.float32).eps,
                                           atol=8 * torch.finfo(torch.float32).eps)


def test_checkpoint_future_masking_and_same_atom_integrable_paths(fitted, tmp_path):
    model = fitted[MODE][0]
    path = tmp_path / "candidate.pt"
    model.save(path)
    saved = torch.load(path, weights_only=True)
    assert saved["schema_version"] == 3
    restored = core.ConditionalRateForecaster.load(path)
    equal(model.fit_diagnostics_, restored.fit_diagnostics_)
    equal(model.state_dict(), restored.state_dict())
    y, x = inputs()
    y[0, :, 1] = np.nan
    for origin in ORIGINS:
        clean = y.copy()
        clean[:, DAYS > origin] = np.nan
        poisoned = clean.copy()
        poisoned[:, DAYS > origin] = np.inf
        a = model.predict_distribution(clean, origin, 16, np.random.default_rng(25), x)
        b = restored.predict_distribution(poisoned, origin, 16, np.random.default_rng(25), x)
        equal(a, b)
        observed = np.isfinite(clean)
        np.testing.assert_array_equal(a["report_cumulative"].transpose(1, 0, 2, 3)[:, observed],
                                      np.broadcast_to(clean[observed], (16, observed.sum())))
        assert np.isfinite(a["latent_total_volume"]).all() and (a["latent_total_volume"] > 0).all()
        assert (a["latent_rate"] >= 0).all()
        assert (np.diff(a["latent_cumulative"], axis=2) >= 0).all()
        assert (a["latent_cumulative"] <= a["latent_total_volume"][:, :, None] * (1 + 1e-14)).all()
        query = np.array([0., 5., 35., 720., 900., 1080.])
        same_atoms = restored.evaluate_trajectories(torch.tensor(a["latent_parameters"], dtype=torch.float64), query)
        equal(same_atoms["latent_total_volume"], a["latent_total_volume"])
        assert (np.diff(same_atoms["latent_rate"][:, :, 3:], axis=2) < 0).all()


@pytest.mark.parametrize("corruption", ["config_mode", "fit_mode", "fit_configuration", "checkpoint_mode",
    "bounds", "formula", "arithmetic", "initial_law", "moment_scope", "objective", "coefficients", "samples",
    "interpretation", "optimization_mode", "optimization_missing", "initial_parameter_hash", "schema_downgrade",
    "anchor_missing", "anchor_value", "anchor_sigma", "anchor_rule", "buffer_missing", "buffer_dtype", "buffer_shape",
    "buffer_value", "buffer_nan", "buffer_grad", "consistent_large_anchor"])
def test_checkpoint_covariance_metadata_mutations_rejected(fitted, tmp_path, corruption):
    path = tmp_path / "corrupted.pt"
    fitted[MODE][0].save(path)
    checkpoint = torch.load(path, weights_only=True)
    fit = checkpoint["fit_diagnostics"]
    if corruption == "config_mode": checkpoint["configuration"].pop("covariance_mode")
    elif corruption == "fit_mode": fit["covariance_mode"] = "bounded_cholesky"
    elif corruption == "fit_configuration": fit["configuration"]["covariance_mode"] = "bounded_cholesky"
    elif corruption == "checkpoint_mode": checkpoint.pop("covariance_mode")
    elif corruption == "bounds": checkpoint["covariance_marginal_std_bounds"] = [.025, 2.]
    elif corruption == "formula": fit["covariance_factor_formula"] = "unbounded lower entries"
    elif corruption == "arithmetic": checkpoint["covariance_factor_arithmetic"] = "float32"
    elif corruption == "initial_law": checkpoint["covariance_initial_law"] = "Sigma=.25I"
    elif corruption == "moment_scope": fit["covariance_moment_scope"] = "population moment guarantee"
    elif corruption == "objective": fit["objective"] = "mse only"
    elif corruption == "coefficients": fit["score_coefficients"] = [1., 1., 1.]
    elif corruption == "samples": fit["train_samples_iid"] = 256
    elif corruption == "interpretation": fit["interpretation"] = "identified latent truth"
    elif corruption == "optimization_mode": fit["optimization"][0].pop("covariance_mode")
    elif corruption == "optimization_missing": fit["optimization"] = []
    elif corruption == "initial_parameter_hash": fit["initial_parameter_sha256"] = "0" * 64
    elif corruption == "schema_downgrade": checkpoint["schema_version"] = 1
    elif corruption == "anchor_missing": checkpoint.pop("covariance_diagonal_anchor_logit")
    elif corruption == "anchor_value": checkpoint["covariance_diagonal_anchor_logit"] += .1
    elif corruption == "anchor_sigma": fit["covariance_initial_legacy_std_float32"] = .3
    elif corruption == "anchor_rule": fit["covariance_diagonal_anchor_rule"] = "recompute on current device"
    elif corruption == "buffer_missing": checkpoint["state_dict"].pop(ANCHOR_BUFFER)
    elif corruption == "buffer_dtype": checkpoint["state_dict"][ANCHOR_BUFFER] = checkpoint["state_dict"][ANCHOR_BUFFER].float()
    elif corruption == "buffer_shape": checkpoint["state_dict"][ANCHOR_BUFFER] = checkpoint["state_dict"][ANCHOR_BUFFER].reshape(1)
    elif corruption == "buffer_value": checkpoint["state_dict"][ANCHOR_BUFFER] += 1e-10
    elif corruption == "buffer_nan": checkpoint["state_dict"][ANCHOR_BUFFER].fill_(float("nan"))
    elif corruption == "buffer_grad": checkpoint["state_dict"][ANCHOR_BUFFER].requires_grad_()
    elif corruption == "consistent_large_anchor":
        value = checkpoint["covariance_diagonal_anchor_logit"] + .1
        checkpoint["state_dict"][ANCHOR_BUFFER].fill_(value)
        metadata = fitted[MODE][0]._covariance_metadata(diagonal_anchor_logit=value)
        checkpoint.update(metadata); fit.update(metadata)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        core.ConditionalRateForecaster.load(path)


@pytest.mark.parametrize("remaining", ["schema", "optimization", "buffer"])
def test_stripped_covariance_metadata_cannot_accidentally_downgrade_checkpoint(fitted, tmp_path, remaining):
    path = tmp_path / "stripped.pt"
    fitted[MODE][0].save(path)
    checkpoint = torch.load(path, weights_only=True)
    fit = checkpoint["fit_diagnostics"]
    containers = (checkpoint, checkpoint["configuration"], fit, fit["configuration"])
    if remaining != "optimization":
        containers = (*containers, *fit["optimization"])
    for metadata in containers:
        for key in list(metadata):
            if key.startswith("covariance_"):
                metadata.pop(key)
    if remaining != "schema":
        checkpoint["schema_version"] = 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="schema|covariance"):
        core.ConditionalRateForecaster.load(path)


def test_recorded_nonlocal_anchor_is_preserved_without_a_cuda_runtime(fitted, tmp_path):
    """Use the observed CUDA initial scalar as a synthetic portability fixture."""
    path = tmp_path / "nonlocal.pt"
    fitted[MODE][0].save(path)
    model = core.ConditionalRateForecaster.load(path)
    sigma = float(np.nextafter(np.float32(.25), np.float32(np.inf)))
    anchor = math.log((sigma - .025) / (1.2 - sigma))
    model.network.covariance_diagonal_anchor_logit.fill_(anchor)
    model.fit_diagnostics_.update(model._covariance_metadata())
    model.save(path)
    restored = core.ConditionalRateForecaster.load(path)
    assert restored.network.covariance_diagonal_anchor_logit.item() == anchor
    assert restored.fit_diagnostics_["covariance_initial_legacy_std_float32"] == sigma
    y, x = inputs()
    for origin in ORIGINS:
        equal(model.predict_distribution(y, origin, 16, np.random.default_rng(87), x),
              restored.predict_distribution(y, origin, 16, np.random.default_rng(87), x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cross-device checkpoint test requires CUDA")
@pytest.mark.parametrize("training_device", ["cpu", "cuda"])
def test_actual_cross_device_checkpoint_and_move_preserve_frozen_anchor_and_round_trip_law(tmp_path, training_device):
    destination = "cuda" if training_device == "cpu" else "cpu"
    model, _ = fit_trace(mode=MODE, device=training_device)
    original_anchor = model.network.covariance_diagonal_anchor_logit.detach().cpu().clone()
    path = tmp_path / "original.pt"
    model.save(path)
    restored = core.ConditionalRateForecaster.load(path, device=destination)
    equal(restored.network.covariance_diagonal_anchor_logit.cpu(), original_anchor)
    equal(restored._covariance_metadata(), model._covariance_metadata())
    restored.to(training_device)
    equal(restored.network.covariance_diagonal_anchor_logit.cpu(), original_anchor)
    restored.to(destination)
    moved_path = tmp_path / "moved.pt"
    restored.save(moved_path)
    round_trip = core.ConditionalRateForecaster.load(moved_path, device=training_device)
    equal(round_trip.state_dict(), model.state_dict())
    equal(round_trip._covariance_metadata(), model._covariance_metadata())
    y, x = inputs()
    for origin in ORIGINS:
        equal(model.predict_distribution(y, origin, 16, np.random.default_rng(89), x),
              round_trip.predict_distribution(y, origin, 16, np.random.default_rng(89), x))


def test_explicit_default_is_omitted_and_stray_covariance_metadata_rejected(fitted, tmp_path):
    omitted, a = fitted[None]
    explicit, b = fitted["bounded_cholesky"]
    equal(omitted.configuration, explicit.configuration)
    equal(omitted.state_dict(), explicit.state_dict())
    equal(omitted.fit_diagnostics_, explicit.fit_diagnostics_)
    equal(a, b)
    y, x = inputs()
    equal(omitted.predict_distribution(y, 60, 16, np.random.default_rng(12), x),
          explicit.predict_distribution(y, 60, 16, np.random.default_rng(12), x))
    path = tmp_path / "default.pt"
    omitted.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["schema_version"] == 1
    for metadata in (checkpoint, checkpoint["configuration"], checkpoint["fit_diagnostics"], *checkpoint["fit_diagnostics"]["optimization"]):
        assert not any(key.startswith("covariance_") for key in metadata)
    core.ConditionalRateForecaster.load(path)
    checkpoint["covariance_marginal_std_bounds"] = [.025, 1.2]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="covariance"):
        core.ConditionalRateForecaster.load(path)


