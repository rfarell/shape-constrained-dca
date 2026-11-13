"""Synthetic exact-law, group exclusion, units and provenance checks; no field fits."""
from copy import deepcopy
import hashlib
import itertools

import numpy as np
import pytest
import torch

from joint_production.baselines import ProbabilisticTrajectoryForest
from joint_production.rates import PositiveBasisRate
from joint_production.static_features import STATIC_FEATURE_NAMES, StaticFeaturePreprocessor
import joint_production.tree_static_prior as prior

DAYS = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=float)


@pytest.fixture
def population():
    n = 90
    records = [{"row_id": f"row-{i}", "group_id": f"group-{i // 2}", "split": "train",
                "covariates_raw": {name: str(1 + .1 * i + .03 * j) for j, name in enumerate(STATIC_FEATURE_NAMES)}}
               for i in range(n)]
    weights = (1 + np.arange(n)[:, None, None] / 60) * np.array(
        [[30, 100, 70, 10], [60, 200, 80, 90], [10, 70, 20, 30]])[None]
    reports = np.einsum("npk,tk->ntp", weights, prior.fast_basis_cumulative_matrix(DAYS))
    assert set(prior.assign_group_folds(records).tolist()) == set(range(5))
    return records, reports


@pytest.fixture
def small_forests(monkeypatch):
    calls = []
    def factory(**settings):
        calls.append(dict(settings))
        return ProbabilisticTrajectoryForest(**dict(settings, n_estimators=5))
    monkeypatch.setattr(prior, "ProbabilisticTrajectoryForest", factory)
    return calls


@pytest.fixture
def fitted(population, small_forests):
    records, reports = population
    return prior.TreeStaticPrior(seed=42, n_jobs=1).fit(records, reports, DAYS)


def test_exact_forest_mean_equals_explicit_tree_and_unique_donor_enumeration(population, monkeypatch):
    records, reports = population
    cov = StaticFeaturePreprocessor.fit(records).transform(records)
    forest = ProbabilisticTrajectoryForest(n_estimators=5, min_samples_leaf=3, mask_augmentation=2,
        random_state=21, covariate_preprocessing="external").fit(reports, DAYS, cov, origins=[0])
    query = cov[[0, 17, 89]]
    history = np.full((3, 8, 3), np.nan)
    history[:, 0] = 0
    origin = forest.origin_models_[0.0]
    features = np.column_stack((forest._history_features(history, 0, origin.feature_scale), query))
    labels = origin.forest.apply(features)
    oracle = np.zeros((3, 8, 3))
    for row in range(3):
        for tree in range(5):
            members = origin.leaves[tree][int(labels[row, tree])]
            assert len(members) == len(np.unique(members))
            for member in members:
                oracle[row] += forest.train_reports_[member] / (5 * len(members))
    def forbidden(*args, **kwargs):
        raise AssertionError("Exact law mean must not call regressor prediction or Monte Carlo draws")
    monkeypatch.setattr(origin.forest, "predict", forbidden)
    monkeypatch.setattr(forest, "predict_samples", forbidden)
    actual = prior.exact_static_forest_mean(forest, query)
    np.testing.assert_allclose(actual, oracle, rtol=2e-15, atol=1e-12)
    np.testing.assert_array_equal(actual[:, 0], 0)
    forest.scale_bounds = (2., 4.)
    with pytest.raises(ValueError, match="bracketing one"):
        prior.exact_static_forest_mean(forest, query)
    forest.scale_bounds = (.25, 4.)
    forest.train_reports_[0, 0, 0] = 1
    with pytest.raises(ValueError, match="accounting zero"):
        prior.exact_static_forest_mean(forest, query)


@pytest.fixture
def fingerprint_forest(population):
    records, reports = population
    covariates = StaticFeaturePreprocessor.fit(records).transform(records)
    return ProbabilisticTrajectoryForest(n_estimators=5, min_samples_leaf=3,
        random_state=21, covariate_preprocessing="external").fit(reports, DAYS, covariates, origins=[0])


def test_origin0_law_fingerprint_matches_reference_with_other_origins(fingerprint_forest, population):
    records, reports = population
    covariates = StaticFeaturePreprocessor.fit(records).transform(records)
    reference = ProbabilisticTrajectoryForest(n_estimators=5, min_samples_leaf=3,
        random_state=21, n_jobs=2, covariate_preprocessing="external").fit(reports, DAYS, covariates)
    actual = prior.origin0_forest_law_fingerprint(fingerprint_forest)
    assert len(reference.origin_models_) == 7
    assert actual == prior.origin0_forest_law_fingerprint(reference)
    assert actual["fields"]["train_reports"]["dtype"] == "<f8"
    assert actual["fields"]["train_reports"]["shape"] == list(reports.shape)
    assert actual["fields"]["tree.0.feature"]["dtype"] == "<i8"
    reference.origin_models_[90.0].forest.estimators_[0].tree_.threshold[:] += 10
    assert actual == prior.origin0_forest_law_fingerprint(reference)


@pytest.mark.parametrize("field", ["donor_report", "donor_row", "threshold", "leaf_donor", "missing_route"])
def test_origin0_law_fingerprint_changes_with_issued_law_fields(fingerprint_forest, field):
    original = prior.origin0_forest_law_fingerprint(fingerprint_forest)
    origin = fingerprint_forest.origin_models_[0.0]
    tree = origin.forest.estimators_[0].tree_
    if field == "donor_report":
        fingerprint_forest.train_reports_[0, 1, 0] += 1
    elif field == "donor_row":
        fingerprint_forest.donor_row_indices_ += 1000
    elif field == "threshold":
        tree.threshold[np.flatnonzero(tree.children_left >= 0)[0]] += .1
    elif field == "leaf_donor":
        table = origin.leaves[0]
        label = sorted(table)[0]
        outside = next(i for i in range(len(fingerprint_forest.train_reports_)) if i not in table[label])
        table[label] = np.append(table[label], outside)
    else:
        if not hasattr(tree, "missing_go_to_left"):
            pytest.skip("Installed sklearn tree does not expose missing-value routing")
        split = np.flatnonzero(tree.children_left >= 0)[0]
        tree.missing_go_to_left[split] = 1 - tree.missing_go_to_left[split]
    changed = prior.origin0_forest_law_fingerprint(fingerprint_forest)
    assert changed["sha256"] != original["sha256"]


def test_origin0_law_fingerprint_ignores_node_padding_unused_values_and_leaf_order(fingerprint_forest):
    original = prior.origin0_forest_law_fingerprint(fingerprint_forest)
    origin = fingerprint_forest.origin_models_[0.0]
    touched_padding = False
    for estimator in origin.forest.estimators_:
        tree = estimator.tree_
        state = tree.__getstate__()
        nodes = state["nodes"].copy()
        meaningful = set()
        for dtype, offset in nodes.dtype.fields.values():
            meaningful.update(range(offset, offset + dtype.itemsize))
        padding = sorted(set(range(nodes.dtype.itemsize)) - meaningful)
        if padding:
            touched_padding = True
            raw = nodes.view(np.uint8).reshape(len(nodes), nodes.dtype.itemsize)
            raw[:, padding] ^= 0xff
            state["nodes"] = nodes
            tree.__setstate__(state)
        tree.value[:] += 100
    assert touched_padding, "This oracle requires the installed sklearn structured node padding"
    for table in origin.leaves:
        for label in table:
            table[label] = table[label][::-1]
    assert original == prior.origin0_forest_law_fingerprint(fingerprint_forest)


def test_fold_assignment_is_stable_under_order_and_keeps_groups_together(population):
    records, _ = population
    folds = prior.assign_group_folds(records)
    np.testing.assert_array_equal(prior.assign_group_folds(records[::-1])[::-1], folds)
    for group in {r["group_id"] for r in records}:
        assert len({int(f) for r, f in zip(records, folds) if r["group_id"] == group}) == 1
    expected = [int(hashlib.sha256(
        f"bakken-tree-static-prior-group-fold-v1:271828:{row['group_id']}".encode()).hexdigest(), 16) % 5
        for row in records]
    np.testing.assert_array_equal(folds, expected)


def test_all_six_forests_have_retained_settings_and_each_fold_excludes_held_groups(fitted, population, small_forests):
    records, reports = population
    assert len(small_forests) == 6
    for settings in small_forests:
        assert settings["n_estimators"] == 200
        assert settings["min_samples_leaf"] == 4
        assert settings["covariate_preprocessing"] == "external"
    for fold, metadata in enumerate(fitted.metadata_["folds"]):
        take = fitted.folds_ != fold
        allowed = [r for r, keep in zip(records, take) if keep]
        assert set(metadata["training_group_ids"]).isdisjoint(metadata["held_out_group_ids"])
        assert metadata["preprocessing"] == StaticFeaturePreprocessor.fit(allowed).to_dict()
        endpoint, phase, details = prior.fit_projection_scales(reports[take])
        np.testing.assert_array_equal(metadata["endpoint_scales_native"], endpoint)
        assert metadata["projection_scaling"] == details
        np.testing.assert_array_equal(metadata["coefficient_floor_native"], prior.COEFFICIENT_FLOOR * phase)
    assert fitted.full_teacher_.forest.origin_models_.keys() == {0.0}
    assert fitted.oof_coefficients_native_.shape == (len(records), 12)
    assert fitted.oof_mean_cumulative_native_.shape == (len(records), 8, 3)


def test_mutating_a_held_groups_future_targets_cannot_change_its_oof_priors(fitted, population, small_forests):
    records, reports = population
    held = fitted.folds_ == 0
    changed = reports.copy()
    changed[held, 1:] *= 100
    second = prior.TreeStaticPrior(seed=42, n_jobs=1).fit(records, changed, DAYS)
    np.testing.assert_array_equal(second.oof_coefficients_native_[held], fitted.oof_coefficients_native_[held])
    np.testing.assert_array_equal(second.oof_tree_mean_cumulative_native_[held], fitted.oof_tree_mean_cumulative_native_[held])
    assert second.metadata_["folds"][0] == fitted.metadata_["folds"][0]
    assert second.metadata_["full_train"]["training_reports_sha256"] != fitted.metadata_["full_train"]["training_reports_sha256"]


def test_prediction_reads_only_static_fields_and_allows_duplicate_queries(fitted, population):
    records, _ = population
    query = [deepcopy(records[1]), deepcopy(records[1]), deepcopy(records[60])]
    original = fitted.predict_coefficients(query)
    for i, row in enumerate(query):
        row.update(reports="unreadable future", report_mask=object(), row_id=None, group_id=None,
                   split="test", first_prod_date_raw=object(), api_text=object())
        row["covariates_raw"].update(Latitude="Infinity", Longitude="not an allowed feature")
    np.testing.assert_array_equal(fitted.predict_coefficients(query), original)
    np.testing.assert_array_equal(original[0], original[1])
    assert fitted.predict_coefficients([]).shape == (0, 12)
    assert fitted.predict_mean_cumulative([]).shape == (0, 8, 3)


def test_prediction_always_serves_full_teacher_even_with_train_identity(fitted, population):
    records, _ = population
    query = records[:6]
    actual = fitted.predict_components(query)
    expected = fitted.full_teacher_.predict(query)
    for result, oracle in zip(actual, expected):
        np.testing.assert_array_equal(result, oracle)
    assert np.max(np.abs(actual[0] - fitted.oof_coefficients_native_[:6])) > 1e-3


def test_teacher_projects_exact_mean_before_nnls_not_each_donor(population):
    records, _ = population
    records = records[:2]
    preprocessing = StaticFeaturePreprocessor.fit(records)
    basis = prior.fast_basis_cumulative_matrix(DAYS)
    curve = np.einsum("pk,tk->tp", np.full((3, 4), 10.), basis)
    reports = np.stack((-curve, 3 * curve))
    endpoint, phase, _ = prior.fit_projection_scales(reports)
    forest = ProbabilisticTrajectoryForest(n_estimators=3, min_samples_leaf=2,
        random_state=21, covariate_preprocessing="external").fit(
            reports, DAYS, preprocessing.transform(records), origins=[0])
    teacher = prior._Teacher(preprocessing, forest, endpoint, phase, {})
    coefficients, cumulative, mean = teacher.predict(records[:1])
    np.testing.assert_allclose(mean, curve[None], rtol=2e-15, atol=1e-12)
    projected_mean, projected_cumulative = prior.project_positive_coefficients(
        curve[None], DAYS, endpoint, phase)
    np.testing.assert_allclose(coefficients, projected_mean, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(cumulative, projected_cumulative, rtol=1e-12, atol=1e-10)
    donor_coefficients, _ = prior.project_positive_coefficients(reports, DAYS, endpoint, phase)
    assert np.max(np.abs(coefficients[0] - donor_coefficients.mean(axis=0))) > 1


def test_positive_floor_projection_reconstructs_basis_and_native_integration():
    scale = np.array([100., 500., 20.])
    normalized = np.array([[[.2, .4, .1, .3], [.5, .1, .8, .4], [.1, .3, .2, .7]]])
    coefficient = normalized * scale[None, :, None]
    mean = np.einsum("npk,tk->ntp", coefficient, prior.fast_basis_cumulative_matrix(DAYS))
    endpoint = np.maximum(mean[0, 1:], 1)
    fitted, cumulative = prior.project_positive_coefficients(mean, DAYS, endpoint, scale)
    np.testing.assert_allclose(fitted.reshape(1, 3, 4), coefficient, rtol=3e-11, atol=1e-10)
    np.testing.assert_allclose(cumulative, mean, rtol=2e-13, atol=1e-11)
    rate = PositiveBasisRate("fast")
    z = torch.tensor(np.log(fitted.reshape(1, 3, 4) / scale[None, :, None]).reshape(1, 12), dtype=torch.float64)
    with torch.no_grad():
        replay = rate.cumulative(z, DAYS).numpy() * scale
        total = rate.total_volume(z).numpy() * scale
    np.testing.assert_allclose(replay, cumulative, rtol=3e-14, atol=1e-11)
    np.testing.assert_allclose(total, fitted.reshape(1, 3, 4).sum(axis=-1))
    assert np.all(total >= cumulative[:, -1])
    np.testing.assert_array_equal(prior.fast_basis_cumulative_matrix([np.inf]), np.ones((1, 4)))


def test_shifted_nnls_matches_independent_active_subset_oracle_and_keeps_zero_targets_positive():
    phase = np.array([100., 1000., 50.])
    scales = np.arange(1., 8.)[:, None] * phase[None]
    basis = prior.fast_basis_cumulative_matrix(DAYS)[1:]
    mean = np.zeros((2, 8, 3))
    mean[1, 1:] = np.array([.7, .5, 2, .9, 5, 2, 7])[:, None] * phase
    coefficients, reconstruction = prior.project_positive_coefficients(mean, DAYS, scales, phase)
    normalized = coefficients.reshape(2, 3, 4) / phase[None, :, None]
    np.testing.assert_array_equal(normalized[0], np.full((3, 4), prior.COEFFICIENT_FLOOR))
    assert np.all(normalized >= prior.COEFFICIENT_FLOOR)
    for p in range(3):
        A = basis * (phase[p] / scales[:, p, None])
        y = mean[1, 1:, p] / scales[:, p] - A @ np.full(4, prior.COEFFICIENT_FLOOR)
        objectives = [float(y @ y)]
        for size in range(1, 5):
            for active in itertools.combinations(range(4), size):
                solution = np.linalg.lstsq(A[:, active], y, rcond=None)[0]
                if np.all(solution >= 0):
                    objectives.append(float(np.sum((A[:, active] @ solution - y) ** 2)))
        actual = float(np.sum(((reconstruction[1, 1:, p] - mean[1, 1:, p]) / scales[:, p]) ** 2))
        assert actual == pytest.approx(min(objectives), rel=1e-12, abs=1e-12)


def test_phase_unit_conversion_commutes_with_fitted_priors(fitted, population, small_forests):
    records, reports = population
    factors = np.array([8., 1024., .125])
    converted = prior.TreeStaticPrior(seed=42, n_jobs=1).fit(records, reports * factors, DAYS)
    coefficient_factors = np.repeat(factors, 4)
    np.testing.assert_allclose(converted.oof_coefficients_native_ / coefficient_factors,
                               fitted.oof_coefficients_native_, rtol=3e-11, atol=1e-10)
    query = [records[6], records[66]]
    np.testing.assert_allclose(converted.predict_coefficients(query) / coefficient_factors,
                               fitted.predict_coefficients(query), rtol=3e-11, atol=1e-10)


@pytest.mark.parametrize("fault", ["validation", "calibration", "test", "duplicate", "no_group", "no_id", "zero_origin", "missing_report", "infinite_terminal", "days"])
def test_invalid_fitting_inputs_fail_before_any_forest(fault, population, small_forests):
    records, reports = population
    records, reports, days = deepcopy(records), reports.copy(), DAYS.copy()
    if fault in {"validation", "calibration", "test"}:
        records[0]["split"] = fault
    elif fault == "duplicate":
        records[1]["row_id"] = records[0]["row_id"]
    elif fault == "no_group":
        records[0].pop("group_id")
    elif fault == "no_id":
        records[0].pop("row_id")
    elif fault == "zero_origin":
        reports[0, 0, 0] = 1
    elif fault == "missing_report":
        reports[0, 2, 1] = np.nan
    elif fault == "infinite_terminal":
        reports[0, -1, 2] = np.inf
    else:
        days[1] = 31
    with pytest.raises(ValueError):
        prior.TreeStaticPrior().fit(records, reports, days)
    assert not small_forests


def test_insufficient_populated_folds_and_corrupted_group_membership_are_rejected(population, small_forests, fitted):
    records, reports = population
    with pytest.raises(ValueError, match="five populated"):
        prior.TreeStaticPrior().fit(records[:4], reports[:4], DAYS)
    fitted.metadata_["folds"][0]["held_out_group_ids"].append(fitted.metadata_["folds"][0]["training_group_ids"][0])
    with pytest.raises(ValueError, match="group exclusion"):
        fitted.predict_coefficients(records[:1])


def test_native_serialization_replays_and_protects_existing_checkpoints(fitted, population, tmp_path):
    records, _ = population
    path = tmp_path / "prior.joblib"
    expected = fitted.predict_coefficients(records[:5])
    fitted.save(path)
    restored = prior.TreeStaticPrior.load(path)
    np.testing.assert_array_equal(restored.oof_coefficients_native_, fitted.oof_coefficients_native_)
    np.testing.assert_array_equal(restored.predict_coefficients(records[:5]), expected)
    assert restored.metadata() == fitted.metadata()
    separate = restored.metadata()
    separate["fold_assignment"][0]["fold"] = 999
    assert restored.metadata() == fitted.metadata()
    with pytest.raises(FileExistsError):
        restored.save(path)
    restored.oof_coefficients_native_[0, 0] *= 2
    with pytest.raises(ValueError, match="bank"):
        restored.save(tmp_path / "corrupted.joblib")


def test_full_teacher_metadata_contains_and_enforces_law_fingerprint(fitted, population):
    assert fitted.metadata_["full_train"]["origin0_forest_law_fingerprint"] == prior.origin0_forest_law_fingerprint(
        fitted.full_teacher_.forest)
    tree = fitted.full_teacher_.forest.origin_models_[0.0].forest.estimators_[0].tree_
    tree.threshold[np.flatnonzero(tree.children_left >= 0)[0]] += .1
    with pytest.raises(ValueError, match="fingerprint"):
        fitted.predict_components(population[0][:1])


@pytest.mark.parametrize("setting,value", [("SHAPES", (1, 1, 2, 3)),
    ("SCALES_DAYS", (30., 180., 360., 721.)), ("COEFFICIENT_FLOOR", 2e-6),
    ("FOLD_NAMESPACE", "future-group-fold-rule"), ("FOLD_COUNT", 6)])
def test_changed_serving_defaults_reject_saved_priors(fitted, tmp_path, monkeypatch, setting, value):
    path = tmp_path / "prior.joblib"
    fitted.save(path)
    monkeypatch.setattr(prior, setting, value)
    with pytest.raises(ValueError, match="serving contract"):
        prior.TreeStaticPrior.load(path)
