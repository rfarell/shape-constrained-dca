"""Small synthetic checks of one finite joint prior, posterior and report law."""
from copy import deepcopy
import json
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import joblib
import numpy as np
import pytest
import torch
from scipy.special import logsumexp

import joint_production.finite_forest_rates as finite
from joint_production.baselines import ProbabilisticTrajectoryForest
from joint_production.static_features import STATIC_FEATURE_NAMES, StaticFeaturePreprocessor
from joint_production.tree_static_prior import fast_basis_cumulative_matrix, fit_projection_scales

DAYS = np.array([0., 30., 60., 90., 180., 360., 720., 1080.])


@pytest.fixture
def population():
    records = [{"row_id": f"r{i}", "group_id": f"g{i//2}", "split": "train",
        "covariates_raw": {name: str(1+.2*i+.04*j) for j, name in enumerate(STATIC_FEATURE_NAMES)},
        "reports": "must never read targets from records"} for i in range(40)]
    rng = np.random.default_rng(34)
    coefficients = rng.uniform(1, 30, (40, 3, 4))*(1+np.arange(40)[:, None, None]/5)
    y = np.einsum("npk,tk->ntp", coefficients, fast_basis_cumulative_matrix(DAYS))
    y[:, 2, 1] -= 2
    return records, y


@pytest.fixture
def small_forests(monkeypatch):
    calls = []
    def factory(**settings):
        calls.append(dict(settings))
        return ProbabilisticTrajectoryForest(**dict(settings, n_estimators=5, n_jobs=1))
    monkeypatch.setattr(finite, "ProbabilisticTrajectoryForest", factory)
    return calls


@pytest.fixture
def fitted(population, small_forests):
    records, reports = population
    return finite.FiniteForestRateForecaster().fit(reports, DAYS, records=records)


def test_bank_factory_only_one_teacher_all_donors_and_complement_scaling(population, small_forests):
    records, y = population
    records, y = records[6:], y[6:]
    bank = finite.fit_forest_rate_prior(records, y, DAYS, seed=42, n_jobs=2)
    assert len(small_forests) == 1
    assert small_forests[0] == dict(finite.FOREST_SETTINGS, random_state=42, n_jobs=2)
    assert bank.metadata()["training_row_ids"] == [record["row_id"] for record in records]
    assert bank.processor.to_dict() == StaticFeaturePreprocessor.fit(records).to_dict()
    endpoint, phase, _ = fit_projection_scales(y)
    np.testing.assert_array_equal(bank.endpoint_scales_native, endpoint)
    np.testing.assert_array_equal(bank.phase_scales_native, phase)
    np.testing.assert_array_equal(bank.forest.train_reports_, y)
    np.testing.assert_array_equal(bank.forest.donor_row_indices_, np.arange(len(y)))
    assert np.all(bank.coefficients_native >= np.repeat(phase*1e-6, 4))
    bank.validate()
    changed = bank.metadata(); changed["training_rows"] = -1
    assert bank.metadata()["training_rows"] == len(y)


def test_exact_static_weights_receipt_and_query_reports_ids_ignored(population, fitted, monkeypatch):
    records, _ = population
    queries = [records[0], records[-1]]
    forest = fitted.bank_.forest
    monkeypatch.setattr(forest, "predict_samples", lambda *a, **kw: pytest.fail("No prior Monte Carlo"))
    monkeypatch.setattr(forest.origin_models_[0.0].forest, "predict", lambda *a, **kw: pytest.fail("No inverse transformed regressor"))
    weights, receipt = fitted.bank_.exact_weights(queries, return_receipt=True)
    rebuilt = np.zeros_like(weights)
    labels = receipt["query_tree_leaf_ids"]
    for i, (tree, leaf) in enumerate(zip(receipt["receipt_tree_indices"], receipt["receipt_leaf_ids"], strict=True)):
        donors = receipt["receipt_donor_indices"][receipt["receipt_member_offsets"][i]:receipt["receipt_member_offsets"][i+1]]
        assert len(donors) == len(np.unique(donors))
        rebuilt[np.ix_(np.flatnonzero(labels[:, tree] == leaf), donors)] += 1/(labels.shape[1]*len(donors))
    np.testing.assert_array_equal(receipt["forest_weights"], rebuilt)
    np.testing.assert_array_equal(weights, .95*rebuilt+.05/len(forest.train_reports_))
    changed = [{"covariates_raw": record["covariates_raw"], "row_id": "altered", "split": "test", "reports": object()}
               for record in queries]
    np.testing.assert_array_equal(fitted.bank_.exact_weights(changed), weights)
    assert not np.array_equal(weights[0], weights[1])


def dense_native_likelihood(y, q, days, a, r):
    result = np.zeros(len(q))
    for atom in range(len(q)):
        for phase in range(3):
            selected = np.flatnonzero(np.isfinite(y[:, phase]) & (days > 0))
            if not len(selected): continue
            clock = a[phase]**2*days[selected]/360 + (r[phase]*q[atom, selected, phase])**2
            covariance = np.minimum(clock[:, None], clock[None])
            residual = y[selected, phase]-q[atom, selected, phase]
            sign, logdet = np.linalg.slogdet(covariance)
            assert sign == 1
            result[atom] -= .5*(len(selected)*np.log(2*np.pi)+logdet+residual @ np.linalg.solve(covariance, residual))
    return result


def test_exact_joint_posterior_matches_independent_dense_missing_phase_oracle(population, fitted):
    records, reports = population
    y = reports[[5]].copy()
    y[:, 4:] = np.inf  # future values excluded before validation
    y[0, 1, 0] = -2
    y[0, 2, 1] = np.nan
    y[0, 3, 1] = np.nan  # unknown gas cutoff
    y[0, 1:4, 2] = np.nan  # entirely missing water history
    result = fitted.conditional_probabilities(y, 90, records=[records[5]])
    native_a = fitted.noise_abs.numpy()*fitted.bank_.phase_scales_native
    expected_ll = dense_native_likelihood(y[0, :4], fitted.bank_.cumulative_native[:, :4], DAYS[:4], native_a, fitted.noise_rel.numpy())
    np.testing.assert_allclose(result["atom_log_likelihood_native"][0], expected_ll, atol=1e-10, rtol=1e-12)
    prior = result["prior_atom_probabilities"][0]
    expected_logjoint = np.log(prior)+expected_ll
    np.testing.assert_allclose(result["posterior_atom_probabilities"][0], np.exp(expected_logjoint-logsumexp(expected_logjoint)), atol=1e-13)
    assert result["history_log_evidence_native"][0] == pytest.approx(logsumexp(expected_logjoint), abs=1e-11)


def test_empty_history_has_same_static_law_at_zero_and_positive_cutoffs(population, fitted):
    records, _ = population
    y = np.full((2, 8, 3), np.nan)
    expected = fitted.bank_.exact_weights(records[:2])
    for origin in (0, 30, 90, 360):
        result = fitted.conditional_probabilities(y, origin, records=records[:2])
        np.testing.assert_allclose(result["posterior_atom_probabilities"], expected, atol=2e-16, rtol=2e-15)
        np.testing.assert_array_equal(result["atom_log_likelihood_native"], 0)
        np.testing.assert_allclose(result["history_log_evidence_native"], 0, atol=2e-15)
    y[:, 1, 0] = 0  # observed zero contributes an actual likelihood
    changed = fitted.conditional_probabilities(y, 30, records=records[:2])
    assert not np.allclose(changed["posterior_atom_probabilities"], expected)


def test_future_poison_invariance_exact_known_reports_and_one_joint_atom(population, fitted):
    records, y = population
    history = y[:2].copy(); history[0, 3, 1] = np.nan; history[1, 1:4, 2] = np.nan
    poisoned = history.copy(); poisoned[:, 4:] = np.inf
    result = fitted.predict_distribution(history, 90, 256, np.random.default_rng(13), records=records[:2])
    again = fitted.predict_distribution(poisoned, 90, 256, np.random.default_rng(13), records=records[:2])
    for key in ("posterior_atom_probabilities", "latent_parameters", "latent_coefficients_native", "latent_cumulative", "report_cumulative"):
        np.testing.assert_array_equal(result[key], again[key])
    indices = result["posterior_atom_indices"]
    np.testing.assert_array_equal(result["latent_coefficients_native"], fitted.bank_.coefficients_native[indices])
    np.testing.assert_allclose(result["latent_cumulative"], fitted.bank_.cumulative_native[indices], rtol=1e-15, atol=1e-12)
    permitted = np.isfinite(history) & (DAYS[None, :, None] <= 90)
    for row in range(2):
        np.testing.assert_array_equal(result["report_cumulative"][row][:, permitted[row]], np.broadcast_to(history[row][permitted[row]], (256, permitted[row].sum())))
    assert np.std(result["report_cumulative"][0, :, 3, 1]) > 0  # unknown cutoff is sampled
    assert np.isfinite(result["report_cumulative"]).all()
    np.testing.assert_allclose(result["latent_total_volume"], result["latent_coefficients_native"].reshape(2, 256, 3, 4).sum(-1))
    query = fitted.evaluate_trajectories(result["latent_parameters"], [0, 7, 90, 360, np.inf])
    assert query["latent_cumulative"].shape == (2, 256, 5, 3)
    assert np.all(query["latent_rate"] >= 0) and np.all(np.diff(query["latent_cumulative"], axis=2) >= 0)
    np.testing.assert_allclose(query["latent_cumulative"][:, :, -1], result["latent_total_volume"], rtol=1e-15, atol=1e-11)
    np.testing.assert_array_equal(query["latent_rate"][:, :, -1], 0)
    # Exact increments retain the same hidden origin in each issued report draw.
    increment = result["report_cumulative"][:, :, 5]-result["report_cumulative"][:, :, 3]
    assert np.isfinite(increment).all()


def test_native_units_change_only_likelihood_jacobian(population, fitted):
    records, reports = population
    y = reports[[3]].copy(); y[0, 2, 1] = np.nan; y[:, 4:] = np.nan
    original = fitted.conditional_probabilities(y, 90, records=[records[3]])
    converted = deepcopy(fitted)
    units = np.array([10., 1000., .2])
    converted.bank_.cumulative_native *= units
    converted.bank_.coefficients_native *= np.repeat(units, 4)
    converted.bank_.phase_scales_native *= units
    converted.bank_.forest.train_reports_ *= units
    converted.volume_scale = torch.tensor(converted.bank_.phase_scales_native, dtype=torch.float64)
    actual = converted.conditional_probabilities(y*units, 90, records=[records[3]])
    mask = np.isfinite(y[0]) & (DAYS[:, None] > 0)
    jacobian = mask.sum(0) @ np.log(units)
    np.testing.assert_allclose(actual["atom_log_likelihood_native"], original["atom_log_likelihood_native"]-jacobian, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(actual["posterior_atom_probabilities"], original["posterior_atom_probabilities"], atol=1e-13)


def test_large_common_likelihood_offset_preserves_posterior():
    prior = np.array([[.1, .3, .6], [.3, .2, .5]])
    ll = np.array([[0., -1., -100.], [-3., -7., -2.]])
    p, logp, logz = finite.normalized_finite_posterior(prior, ll)
    for offset in (-1e12, 1e12):
        actual = finite.normalized_finite_posterior(prior, ll+offset)
        np.testing.assert_array_equal(actual[0], p)
        np.testing.assert_array_equal(actual[1], logp)
        np.testing.assert_array_equal(actual[2], logz+offset)


@pytest.mark.parametrize("mutation", ["infinite", "zero_day", "records", "features"])
def test_available_input_or_feature_contract_errors(population, fitted, mutation):
    records, reports = population
    y = reports[:2].copy(); query = records[:2]
    features = fitted.bank_.processor.transform(query)
    if mutation == "infinite": y[0, 1, 0] = np.inf
    elif mutation == "zero_day": y[0, 0, 0] = 1
    elif mutation == "records": query = query[:1]
    else: features[0, 0] += .01
    with pytest.raises(ValueError):
        fitted.predict_distribution(y, 90, 2, np.random.default_rng(1), records=query, covariates=features)


def test_checkpoint_replay_complete_law_and_metadata(population, fitted, tmp_path):
    records, reports = population
    path = tmp_path / "model.pt"
    fitted.save(path)
    restored = finite.FiniteForestRateForecaster.load(path)
    for model in (fitted, restored):
        json.dumps(model.fit_diagnostics_, allow_nan=False)
    assert restored.bank_.metadata() == fitted.bank_.metadata()
    original = fitted.predict_distribution(reports[:2], 90, 20, np.random.default_rng(91), records=records[:2])
    again = restored.predict_distribution(reports[:2], 90, 20, np.random.default_rng(91), records=records[:2])
    for key in original:
        if isinstance(original[key], np.ndarray): np.testing.assert_array_equal(original[key], again[key])
    with pytest.raises(FileExistsError): fitted.save(path)
    checkpoint = joblib.load(path)
    checkpoint["bank"].provenance["uniform_background_probability"] = .2
    joblib.dump(checkpoint, tmp_path / "corrupt.pt")
    with pytest.raises(ValueError, match="metadata"):
        finite.FiniteForestRateForecaster.load(tmp_path / "corrupt.pt")


@pytest.mark.parametrize("field", ["fixed_noise", "interpretation", "observation_law", "seed", "origins"])
def test_checkpoint_cannot_relabel_fitted_law_or_change_fixed_noise(fitted, tmp_path, field):
    path = tmp_path / "model.pt"
    fitted.save(path)
    checkpoint = joblib.load(path)
    if field == "fixed_noise":
        checkpoint["noise_abs"] = np.full(3, .2)
        checkpoint["fit_diagnostics"]["noise_abs_normalized"] = [.2]*3
    elif field == "origins":
        checkpoint["origins"] = [17.]
        checkpoint["fit_diagnostics"]["origins_days"] = [17.]
    else:
        checkpoint["fit_diagnostics"][field] = -1 if field == "seed" else "changed"
    joblib.dump(checkpoint, path)
    with pytest.raises(ValueError, match="Checkpoint|checkpoint"):
        finite.FiniteForestRateForecaster.load(path)


def test_unresolved_oof_fit_saved_without_full_bank_or_validation(population, monkeypatch, tmp_path):
    from joint_production.finite_forest_noise import NoiseFitResult
    import joint_production.finite_forest_noise as noise
    records, y = population
    result = NoiseFitResult(np.full(3, .05), np.full(3, .1), "unresolved_test", False,
        {"status": "unresolved_test"}, np.full(len(y), np.nan))
    seen = []
    def fit(rows, reports, days, **kwargs):
        seen.extend(rows)
        assert all("reports" not in row for row in rows)
        np.testing.assert_array_equal(reports, y)
        return result
    monkeypatch.setattr(noise, "fit_oof_forest_noise", fit)
    monkeypatch.setattr(finite, "fit_forest_rate_prior", lambda *a, **kw: pytest.fail("No full bank after unresolved OOF"))
    model = finite.FiniteForestRateForecaster(noise_mode="oof_fitted").fit(y, DAYS, records=records)
    assert len(seen) == len(records) and not model.fit_diagnostics_["converged"] and model.bank_ is None
    model.save(tmp_path / "unresolved.pt")
    restored = finite.FiniteForestRateForecaster.load(tmp_path / "unresolved.pt")
    with pytest.raises(RuntimeError, match="not ready"):
        restored.predict_distribution(y[:1], 0, 4, np.random.default_rng(1), records=records[:1])


def test_converged_oof_noise_is_served_and_portable_without_fold_payload(population, small_forests, monkeypatch, tmp_path):
    from joint_production.finite_forest_noise import NoiseFitResult
    import joint_production.finite_forest_noise as noise
    records, y = population
    result = NoiseFitResult(np.array([.12, .2, .3]), np.array([.15, .2, .25]), "converged_test", True,
        {"status": "converged_test"}, np.zeros(len(y)), fold_data=("synthetic_large_payload",))
    monkeypatch.setattr(noise, "fit_oof_forest_noise", lambda *a, **kw: result)
    model = finite.FiniteForestRateForecaster(noise_mode="oof_fitted").fit(y, DAYS, records=records)
    assert len(small_forests) == 1 and model.fit_diagnostics_["converged"]
    np.testing.assert_array_equal(model.noise_abs.numpy(), result.noise_abs)
    model.save(tmp_path / "ready.pt")
    restored = finite.FiniteForestRateForecaster.load(tmp_path / "ready.pt")
    assert result.fold_data == ("synthetic_large_payload",) and restored.noise_fit_.fold_data == ()
    np.testing.assert_array_equal(restored.noise_rel.numpy(), result.noise_rel)
    checkpoint = joblib.load(tmp_path / "ready.pt")
    checkpoint["noise_fit"].noise_abs = np.full(3, .8)
    joblib.dump(checkpoint, tmp_path / "mismatched_result.pt")
    with pytest.raises(ValueError, match="retained OOF"):
        finite.FiniteForestRateForecaster.load(tmp_path / "mismatched_result.pt")


def test_real_five_fold_bank_noise_integration_retains_exclusions_and_receipts(population, small_forests, tmp_path):
    from joint_production.finite_forest_noise import fit_oof_forest_noise
    records, y = population
    records = [dict(record, group_id=f"group-{i}") for i, record in enumerate(records)]
    result = fit_oof_forest_noise(records, y, DAYS, seed=42, n_jobs=2,
        max_evaluations=1, max_elapsed_seconds=60)
    # A one-evaluation synthetic budget verifies transport, not convergence.
    assert len(small_forests) == 5 and len(result.fold_data) == 5
    assert not result.converged
    assert sorted(np.concatenate([fold.query_indices for fold in result.fold_data]).tolist()) == list(range(len(y)))
    for fold in result.fold_data:
        assert set(fold.metadata["fit_group_ids"]).isdisjoint(fold.metadata["query_group_ids"])
        assert fold.metadata["bank"]["training_row_ids"] == fold.metadata["fit_row_ids"]
        np.testing.assert_array_equal(fold.prior_probabilities,
            .95*fold.forest_probabilities+.05/len(fold.cumulative_native))
        assert "query_tree_leaf_ids" in fold.weight_receipt
        assert np.isfinite(fold.cumulative_native).all()
    files = result.save_artifacts(tmp_path, "synthetic")
    assert len(files) == 6
    for fold in result.fold_data:
        with np.load(tmp_path/f"synthetic_noise_fold{fold.fold}.npz", allow_pickle=False) as bank:
            np.testing.assert_array_equal(bank["prior_probabilities"], fold.prior_probabilities)
            np.testing.assert_array_equal(bank["query_indices"], fold.query_indices)
            assert bank["coefficients_native"].shape[-1] == 12
            assert set(bank["donor_group_ids"]).isdisjoint(bank["query_group_ids"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only finite law integration")
def test_cuda_density_bridge_and_cpu_checkpoint_portability(population, small_forests, tmp_path):
    records, reports = population
    model = finite.FiniteForestRateForecaster(device="cuda").fit(reports, DAYS, records=records)
    assert model.noise_abs.device.type == "cuda" and model.report_days.device.type == "cuda"
    y = reports[:2].copy(); y[0, 3, 1] = np.nan; y[:, 4:] = np.inf
    result = model.predict_distribution(y, 90, 8, np.random.default_rng(2), records=records[:2])
    assert np.isfinite(result["report_cumulative"]).all()
    model.save(tmp_path / "gpu.pt")
    cpu = finite.FiniteForestRateForecaster.load(tmp_path / "gpu.pt", device="cpu")
    actual = cpu.predict_distribution(y, 90, 8, np.random.default_rng(2), records=records[:2])
    np.testing.assert_allclose(actual["posterior_atom_probabilities"], result["posterior_atom_probabilities"], atol=1e-12)
    np.testing.assert_array_equal(actual["posterior_atom_indices"], result["posterior_atom_indices"])
    np.testing.assert_allclose(actual["report_cumulative"], result["report_cumulative"], rtol=1e-12, atol=1e-10)
