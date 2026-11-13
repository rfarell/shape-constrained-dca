"""Cross-fitted tree means inside the fixed positive-integrable rate basis.

Only TRAIN groups fit teachers or preprocessing. Prediction reads the selected
17 raw static fields, never the query's reports, masks, identifiers or dates.
Coefficients are returned in native phase-volume units, flattened phase-major
(oil four, gas four, water four). These are conditional mean coefficients, not
identified latent truth and not a predictive uncertainty distribution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import nnls
from scipy.special import gammainc

from .baselines import ProbabilisticTrajectoryForest
from .data_audit import REPORT_DAYS
from .metrics import fit_increment_scales
from .static_features import StaticFeaturePreprocessor

FOLD_COUNT = 5
FOLD_NAMESPACE = "bakken-tree-static-prior-group-fold-v1"
COEFFICIENT_FLOOR = 1e-6
PHASE_ORDER = ("oil", "gas", "water")
SHAPES = (1, 1, 2, 2)
SCALES_DAYS = (30.0, 180.0, 360.0, 720.0)
FOREST_SETTINGS = {
    "n_estimators": 200, "min_samples_leaf": 4, "max_features": 1.0,
    "max_depth": None, "scale_adjustment": 0.5, "scale_bounds": (0.25, 4.0),
    "mask_augmentation": 1, "report_mask_probability": 0.15,
    "phase_mask_probability": 0.15, "covariate_preprocessing": "external",
}


def _digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False) + "\n").encode()).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(json.dumps(list(array.shape)).encode() + b"\n" + array.tobytes()).hexdigest()


def _integer_seed(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _identities(records):
    if not isinstance(records, Sequence) or not records:
        raise ValueError("Require a nonempty sequence of TRAIN records")
    rows, groups = [], []
    for record in records:
        if not isinstance(record, Mapping) or record.get("split") != "train":
            raise ValueError("Every fitting record must belong to TRAIN")
        row, group = record.get("row_id"), record.get("group_id")
        if not isinstance(row, str) or not row or not isinstance(group, str) or not group:
            raise ValueError("TRAIN records require nonempty text row_id and group_id")
        rows.append(row)
        groups.append(group)
    if len(set(rows)) != len(rows):
        raise ValueError("TRAIN row identities must be unique, including within groups")
    return rows, groups


def assign_group_folds(records, *, fold_seed=271828):
    """SHA256(namespace:fold_seed:group_id) modulo five, independent of row order."""
    fold_seed = _integer_seed(fold_seed, "fold_seed")
    _, groups = _identities(records)
    return np.asarray([int(hashlib.sha256(f"{FOLD_NAMESPACE}:{fold_seed}:{group}".encode()).hexdigest(), 16)
                       % FOLD_COUNT for group in groups], dtype=np.int8)


def _canonical_days(days):
    days = np.asarray(days, dtype=np.float64)
    if days.shape != (8,) or not np.array_equal(days, np.asarray(REPORT_DAYS, dtype=np.float64)):
        raise ValueError("Require canonical report days [0,30,60,90,180,360,720,1080]")
    return days


def fast_basis_cumulative_matrix(days):
    """Analytic nonnegative unit-mass Erlang integrals, including day zero/infinity."""
    days = np.asarray(days, dtype=np.float64)
    if days.ndim != 1 or np.isnan(days).any() or np.any(days < 0):
        raise ValueError("Query days must be a nonnegative vector, with positive infinity allowed")
    return gammainc(np.asarray(SHAPES), days[:, None] / np.asarray(SCALES_DAYS))


def origin0_forest_law_fingerprint(model):
    """Canonical origin-zero issued-law fingerprint, independent of other origins.

    Hash each apply-relevant sklearn node field separately. Structured node
    padding, unused regressor values, execution parallelism and later-origin
    estimators do not define this law and are excluded. Donors within a leaf
    are canonicalized as a sorted set because sampling is uniform over unique
    donors. The upstream raw-static preprocessor is audited separately.
    Equality establishes the same stored law under the same baselines source;
    inequality need not imply different distributions (e.g. permuted trees).
    """
    if (not hasattr(model, "origin_models_") or 0.0 not in model.origin_models_
            or model.origin_models_[0.0].index != 0):
        raise ValueError("Fingerprint requires a fitted origin-zero forest")
    days = _canonical_days(model.report_days_)
    reports = np.asarray(model.train_reports_)
    donor_rows = np.asarray(model.donor_row_indices_)
    if (reports.ndim != 3 or reports.shape[1:] != (8, 3) or not len(reports)
            or not np.isfinite(reports).all() or np.any(reports[:, 0] != 0)
            or donor_rows.shape != (len(reports),) or not np.issubdtype(donor_rows.dtype, np.integer)
            or np.any(donor_rows < 0) or len(np.unique(donor_rows)) != len(donor_rows)):
        raise ValueError("Fingerprint requires finite accounting-zero donors and unique donor row indices")
    origin = model.origin_models_[0.0]
    estimators = origin.forest.estimators_
    if not len(estimators) or len(estimators) != len(origin.leaves):
        raise ValueError("Fingerprint tree/leaf sampling metadata differ")
    fields = {}

    def add(name, value, dtype):
        array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite fingerprint field {name}")
        descriptor = {"dtype": array.dtype.str, "shape": list(array.shape)}
        header = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        fields[name] = {**descriptor, "sha256": hashlib.sha256(header + array.tobytes()).hexdigest()}

    add("report_days", days, "<f8")
    add("train_reports", reports, "<f8")
    add("donor_row_indices", donor_rows, "<i8")
    add("origin0.feature_scale", origin.feature_scale, "<f8")
    add("covariate_center", model.covariate_center_, "<f8")
    add("covariate_scale", model.covariate_scale_, "<f8")
    trees = []
    for index, (estimator, leaf_table) in enumerate(zip(estimators, origin.leaves)):
        tree = estimator.tree_
        prefix = f"tree.{index}"
        for field in ("children_left", "children_right", "feature"):
            add(f"{prefix}.{field}", getattr(tree, field), "<i8")
        add(f"{prefix}.threshold", tree.threshold, "<f8")
        missing = getattr(tree, "missing_go_to_left", None)
        if missing is not None:
            add(f"{prefix}.missing_go_to_left", missing, "|u1")
        leaf_ids, offsets, all_members = [], [0], []
        actual_leaves = set(np.flatnonzero(tree.children_left == -1).tolist())
        if set(leaf_table) != actual_leaves:
            raise ValueError("Fingerprint donor table must cover exactly the tree leaves")
        for leaf in sorted(leaf_table):
            members = np.asarray(leaf_table[leaf])
            if (members.ndim != 1 or not len(members) or not np.issubdtype(members.dtype, np.integer)
                    or np.any(members < 0) or np.any(members >= len(reports))
                    or len(np.unique(members)) != len(members)):
                raise ValueError("Fingerprint leaf donors must be distinct valid indices")
            leaf_ids.append(int(leaf))
            all_members.extend(np.sort(members).tolist())
            offsets.append(len(all_members))
        add(f"{prefix}.leaf_ids", leaf_ids, "<i8")
        add(f"{prefix}.leaf_donor_offsets", offsets, "<i8")
        add(f"{prefix}.leaf_donors", all_members, "<i8")
        trees.append({"index": index, "node_count": int(tree.node_count),
                      "n_features": int(tree.n_features), "has_missing_route": missing is not None})
    settings = {name: getattr(model, name) for name in (
        "n_estimators", "min_samples_leaf", "max_features", "max_depth", "random_state",
        "scale_adjustment", "scale_bounds", "mask_augmentation", "report_mask_probability", "phase_mask_probability")}
    settings = {key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in settings.items()}
    settings["scale_bounds"] = [float(value) for value in settings["scale_bounds"]]
    contract = {
        "phase_order": list(PHASE_ORDER), "report_units": ["BBL", "MCF", "BBL"], "origin": 0,
        "accounting_zero": True, "tree_selection": "uniform", "leaf_donor_selection": "uniform_unique",
        "history_features": "baselines.ProbabilisticTrajectoryForest._history_features",
        "covariate_preprocessing": getattr(model, "covariate_preprocessing", "internal"),
        "n_covariates": int(model.n_covariates_), "n_features_in": int(origin.forest.n_features_in_),
        "complete_donor_count": int(model.complete_donor_count_),
        "dropped_incomplete_count": int(model.dropped_incomplete_count_),
        "settings": settings, "tree_count": len(estimators), "trees": trees,
    }
    content = {"schema_version": 1, "kind": "origin0_forest_issued_law", "contract": contract, "fields": fields}
    return {**content, "sha256": _digest(content)}


def exact_static_forest_mean(forest, covariates):
    """Exact native mean under uniform-tree / unique-leaf-donor sampling at zero.

    The accounting-zero query makes every donor history multiplier exactly one.
    Averaging the forest regressor's transformed targets would define a different
    mean; this averages the actual native donor paths in its sampling law.
    """
    if (not hasattr(forest, "origin_models_") or 0.0 not in forest.origin_models_
            or forest.origin_models_[0.0].index != 0
            or getattr(forest, "covariate_preprocessing", "internal") != "external"):
        raise ValueError("Require a fitted external-covariate forest at origin zero")
    bounds = np.asarray(forest.scale_bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] <= 1 <= bounds[1]:
        raise ValueError("Exact accounting-zero donor multiplier requires scale bounds bracketing one")
    days = _canonical_days(forest.report_days_)
    donors = forest.train_reports_
    if (donors.ndim != 3 or donors.shape[1:] != (8, 3) or not len(donors)
            or not np.isfinite(donors).all() or np.any(donors[:, 0] != 0)):
        raise ValueError("Origin-zero donor trajectories must be complete with exact accounting zero")
    covariates = np.asarray(covariates, dtype=np.float64)
    if (covariates.ndim != 2 or covariates.shape[1] != forest.n_covariates_
            or not np.isfinite(covariates).all()):
        raise ValueError("Static forest covariates must be finite and match fitted width")
    count = len(covariates)
    if not count:
        return np.empty((0, len(days), 3), dtype=np.float64)
    history = np.full((count, len(days), 3), np.nan)
    history[:, 0] = 0.0
    model = forest.origin_models_[0.0]
    encoded = forest._history_features(history, 0, model.feature_scale)
    encoded = np.column_stack((encoded, forest._covariate_features(covariates, count)))
    leaves = model.forest.apply(encoded)
    if leaves.shape[1] != len(model.leaves) or not model.leaves:
        raise ValueError("Forest tree/leaf sampling metadata differ")
    result = np.zeros((count, len(days), 3), dtype=np.float64)
    for tree, table in enumerate(model.leaves):
        for leaf in np.unique(leaves[:, tree]):
            members = np.asarray(table[int(leaf)])
            if (members.ndim != 1 or not len(members) or not np.issubdtype(members.dtype, np.integer)
                    or np.any(members < 0) or np.any(members >= len(donors))
                    or len(np.unique(members)) != len(members)):
                raise ValueError("Each sampled leaf must contain distinct valid donor indices")
            # Divide before summing to avoid overflow from a finite native mean.
            mean = np.sum(donors[members] / len(members), axis=0)
            result[leaves[:, tree] == leaf] += mean / len(model.leaves)
    if not np.isfinite(result).all() or np.any(result[:, 0] != 0):
        raise FloatingPointError("Exact static forest mean exceeded finite support or accounting zero")
    return result


def fit_projection_scales(train_reports):
    """Endpoint weights and numerical floor units fitted only on a teacher's rows."""
    reports = np.asarray(train_reports, dtype=np.float64)
    if reports.ndim != 3 or reports.shape[1:] != (8, 3) or not len(reports) or not np.isfinite(reports).all():
        raise ValueError("Projection scaling needs complete teacher-TRAIN reports")
    endpoint, details = fit_increment_scales(reports[:, 1:], return_details=True)
    phase, sources = [], []
    for column in range(3):
        positive = reports[:, -1, column]
        positive = positive[positive > 0]
        source = "positive_terminal_train_report_median"
        if not len(positive):
            positive = reports[:, :, column]
            positive = positive[positive > 0]
            source = "pooled_positive_train_report_median"
        phase.append(float(np.median(positive)) if len(positive) else 1.0)
        sources.append(source if len(positive) else "one_native_unit_fallback")
    phase = np.asarray(phase)
    if not np.isfinite(phase).all() or np.any(phase <= 0):
        raise ValueError("Projection phase scales must be finite and positive")
    return endpoint, phase, {"endpoint_scale_details": details, "phase_scales_native": phase.tolist(),
                            "phase_scale_sources": sources, "fit_split": "teacher_training_rows_only"}


def project_positive_coefficients(native_means, report_days, endpoint_scales, phase_scales):
    """Shifted NNLS with coefficient floor 1e-6 in teacher-normalized phase units.

    The objective is the sum of seven squared cumulative residuals divided by
    the frozen teacher-TRAIN endpoint scales squared. Output [N,12] contains native-volume
    coefficients; [N,8,3] is their integrated mean curve. The floor is part of
    this mean construction, not clipping of later stochastic coefficient draws.
    """
    days = _canonical_days(report_days)
    means, scales, phase = [np.asarray(value, dtype=np.float64)
                            for value in (native_means, endpoint_scales, phase_scales)]
    if (means.ndim != 3 or means.shape[1:] != (8, 3) or not np.isfinite(means).all()
            or np.any(means[:, 0] != 0) or scales.shape != (7, 3) or phase.shape != (3,)
            or not np.isfinite(scales).all() or not np.isfinite(phase).all()
            or np.any(scales <= 0) or np.any(phase <= 0)):
        raise ValueError("Require finite accounting-zero mean paths and positive endpoint/phase scales")
    basis = fast_basis_cumulative_matrix(days)
    coefficients = np.empty((len(means), 3, 4), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for p in range(3):
            matrix = basis[1:] * (phase[p] / scales[:, p, None])
            column_scale = np.linalg.norm(matrix, axis=0)
            if not np.isfinite(column_scale).all() or np.any(column_scale <= 0):
                raise FloatingPointError("Projection design exceeded numerical support")
            normalized = matrix / column_scale
            response = means[:, 1:, p] / scales[:, p]
            shifted = response - matrix @ np.full(4, COEFFICIENT_FLOOR)
            for row in range(len(means)):
                amplitude = float(np.max(np.abs(shifted[row]))) or 1.0
                solution, _ = nnls(normalized, shifted[row] / amplitude, maxiter=1000)
                coefficients[row, p] = (COEFFICIENT_FLOOR + solution * amplitude / column_scale) * phase[p]
        reconstructed = np.einsum("npk,tk->ntp", coefficients, basis)
    if (not np.isfinite(coefficients).all() or np.any(coefficients <= 0)
            or not np.isfinite(reconstructed).all()):
        raise FloatingPointError("Projected coefficients or mean paths exceeded finite positive support")
    return coefficients.reshape(len(means), 12), reconstructed


@dataclass
class _Teacher:
    preprocessor: StaticFeaturePreprocessor
    forest: ProbabilisticTrajectoryForest
    endpoint_scales: np.ndarray
    phase_scales: np.ndarray
    metadata: dict

    def predict(self, records):
        features = self.preprocessor.transform(records)
        mean = exact_static_forest_mean(self.forest, features)
        coefficients, cumulative = project_positive_coefficients(
            mean, self.forest.report_days_, self.endpoint_scales, self.phase_scales)
        return coefficients, cumulative, mean


class TreeStaticPrior:
    """Five group-excluded TRAIN teachers plus one full-TRAIN inference teacher.

    Fit requires complete canonical TRAIN reports and five populated group
    folds. Unknown/missing query histories are immaterial: only raw static
    attributes are read. No uncertainty sampler or well-history update is fitted
    by this component. Joblib reload is for trusted local checkpoints only.
    """

    def __init__(self, *, seed=42, fold_seed=271828, n_jobs=2):
        self.seed = _integer_seed(seed, "seed")
        self.fold_seed = _integer_seed(fold_seed, "fold_seed")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        self.n_jobs = n_jobs

    def _fit_teacher(self, records, reports, days, *, label, held_records):
        ids, groups = _identities(records)
        held_ids = [row["row_id"] for row in held_records]
        held_groups = sorted({row["group_id"] for row in held_records})
        if set(ids) & set(held_ids) or set(groups) & set(held_groups):
            raise ValueError("Teacher fitting and held-out rows/groups overlap")
        preprocessing = StaticFeaturePreprocessor.fit(records)
        features = preprocessing.transform(records)
        endpoint, phase, scales = fit_projection_scales(reports)
        forest = ProbabilisticTrajectoryForest(random_state=self.seed, n_jobs=self.n_jobs,
                                               **FOREST_SETTINGS).fit(reports, days, features, origins=[0])
        if forest.dropped_incomplete_count_ != 0 or forest.complete_donor_count_ != len(records):
            raise ValueError("A static teacher unexpectedly changed its complete TRAIN donor cohort")
        metadata = {
            "label": label, "training_rows": len(ids), "training_row_ids": ids,
            "training_group_ids": sorted(set(groups)), "training_row_sha256": _digest(ids),
            "training_group_sha256": _digest(sorted(set(groups))),
            "held_out_row_ids": held_ids, "held_out_group_ids": held_groups,
            "held_out_row_sha256": _digest(held_ids), "held_out_group_sha256": _digest(held_groups),
            "training_reports_sha256": _array_hash(reports), "training_encoded_static_sha256": _array_hash(features),
            "preprocessing": preprocessing.to_dict(), "projection_scaling": scales,
            "endpoint_scales_native": endpoint.tolist(), "coefficient_floor_native": (COEFFICIENT_FLOOR * phase).tolist(),
            "forest_settings": {**FOREST_SETTINGS, "scale_bounds": list(FOREST_SETTINGS["scale_bounds"]),
                                "random_state": self.seed, "n_jobs": self.n_jobs, "origins": [0]},
            "actual_tree_count": len(forest.origin_models_[0.0].leaves),
        }
        return _Teacher(preprocessing, forest, endpoint, phase, metadata)

    def fit(self, train_records, train_reports, report_days):
        ids, groups = _identities(train_records)
        days = _canonical_days(report_days)
        reports = np.asarray(train_reports, dtype=np.float64)
        if (reports.shape != (len(ids), 8, 3) or not np.isfinite(reports).all()
                or np.any(reports[:, 0] != 0)):
            raise ValueError("Require complete finite TRAIN reports with exact zero-origin accounting")
        folds = assign_group_folds(train_records, fold_seed=self.fold_seed)
        if set(folds.tolist()) != set(range(FOLD_COUNT)):
            raise ValueError("Require five populated spatial-group folds; never redraw folds on outcomes")
        oof_coefficients = np.empty((len(ids), 12), dtype=np.float64)
        oof_cumulative = np.empty((len(ids), 8, 3), dtype=np.float64)
        oof_tree_means = np.empty_like(oof_cumulative)
        fold_metadata = []
        for fold in range(FOLD_COUNT):
            fit_indices, held_indices = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            fitting = [train_records[i] for i in fit_indices]
            held = [train_records[i] for i in held_indices]
            teacher = self._fit_teacher(fitting, reports[fit_indices], days, label=f"held_fold_{fold}", held_records=held)
            coefficients, cumulative, mean = teacher.predict(held)
            oof_coefficients[held_indices], oof_cumulative[held_indices], oof_tree_means[held_indices] = coefficients, cumulative, mean
            fold_metadata.append(teacher.metadata)
            del teacher  # Inference needs the full-TRAIN teacher, not five large fold forests.
        full = self._fit_teacher(train_records, reports, days, label="full_train", held_records=[])
        full.metadata["origin0_forest_law_fingerprint"] = origin0_forest_law_fingerprint(full.forest)
        self.full_teacher_ = full
        self.report_days_ = days.copy()
        self.folds_ = folds
        self.train_row_ids_ = list(ids)
        self.train_group_ids_ = list(groups)
        self.oof_coefficients_native_ = oof_coefficients
        self.oof_mean_cumulative_native_ = oof_cumulative
        self.oof_tree_mean_cumulative_native_ = oof_tree_means
        self.metadata_ = {
            "schema_version": 1, "kind": "crossfit_tree_static_positive_basis_prior", "fit_split": "train",
            "seed": self.seed, "fold_seed": self.fold_seed, "fold_count": FOLD_COUNT,
            "fold_namespace": FOLD_NAMESPACE, "fold_rule": "integer SHA256(namespace:fold_seed:group_id) modulo five",
            "fold_assignment": [{"row_id": row, "group_id": group, "fold": int(fold)}
                                for row, group, fold in zip(ids, groups, folds)],
            "training_row_sha256": _digest(ids), "report_days": days.tolist(), "phase_order": list(PHASE_ORDER),
            "coefficient_order": "phase-major oil[4],gas[4],water[4]", "coefficient_units": ["BBL", "MCF", "BBL"],
            "basis_shapes": list(SHAPES), "basis_scales_days": list(SCALES_DAYS),
            "normalized_coefficient_floor": COEFFICIENT_FLOOR,
            "mean_definition": "uniform-tree, unique-leaf-donor arithmetic native report mean at accounting origin zero",
            "projection_objective": "sum of seven squared endpoint residuals divided by teacher-TRAIN endpoint scales squared",
            "projection_solver": "scipy.optimize.nnls after fixed floor shift and column/response normalization",
            "oof_coefficients_sha256": _array_hash(oof_coefficients),
            "oof_mean_cumulative_sha256": _array_hash(oof_cumulative),
            "oof_tree_mean_cumulative_sha256": _array_hash(oof_tree_means),
            "folds": fold_metadata, "full_train": full.metadata,
            "inference_training_size_shift": "OOF teachers fit four folds; deployment teacher fits all TRAIN groups",
            "limitations": ["Coefficient means are model constructions, not identified latent production.",
                            "The numerical positive floor can make multiplicative residual learning difficult near inactive atoms.",
                            "One-native-unit fallback scales in entirely degenerate phases are not unit equivariant unless explicitly converted.",
                            "No conditional uncertainty distribution, history update, calibration or field accuracy is established by preparation."],
        }
        self._validate_fitted()
        return self

    def _validate_fitted(self):
        if not hasattr(self, "full_teacher_"):
            raise ValueError("TreeStaticPrior is not fitted")
        serving_contract = {
            "schema_version": 1, "kind": "crossfit_tree_static_positive_basis_prior", "fit_split": "train",
            "seed": self.seed, "fold_seed": self.fold_seed, "fold_count": FOLD_COUNT,
            "fold_namespace": FOLD_NAMESPACE, "report_days": _canonical_days(self.report_days_).tolist(),
            "phase_order": list(PHASE_ORDER), "coefficient_order": "phase-major oil[4],gas[4],water[4]",
            "coefficient_units": ["BBL", "MCF", "BBL"], "basis_shapes": list(SHAPES),
            "basis_scales_days": list(SCALES_DAYS), "normalized_coefficient_floor": COEFFICIENT_FLOOR,
        }
        if any(self.metadata_.get(key) != value for key, value in serving_contract.items()):
            raise ValueError("Stored tree-prior basis, floor or fold serving contract differs")
        n = len(self.train_row_ids_)
        if (len(set(self.train_row_ids_)) != n or self.folds_.shape != (n,)
                or set(self.folds_.tolist()) != set(range(FOLD_COUNT))
                or self.oof_coefficients_native_.shape != (n, 12)
                or not np.isfinite(self.oof_coefficients_native_).all() or np.any(self.oof_coefficients_native_ <= 0)):
            raise ValueError("Stored TRAIN prior identities or coefficients are invalid")
        for key, value in (("oof_coefficients_sha256", self.oof_coefficients_native_),
                           ("oof_mean_cumulative_sha256", self.oof_mean_cumulative_native_),
                           ("oof_tree_mean_cumulative_sha256", self.oof_tree_mean_cumulative_native_)):
            if self.metadata_.get(key) != _array_hash(value):
                raise ValueError("Stored TRAIN prior bank no longer matches metadata")
        if self.metadata_.get("training_row_sha256") != _digest(self.train_row_ids_):
            raise ValueError("Stored TRAIN row order no longer matches metadata")
        if len(self.train_group_ids_) != n:
            raise ValueError("Stored TRAIN group axis differs")
        identity_records = [{"split": "train", "row_id": row, "group_id": group}
                            for row, group in zip(self.train_row_ids_, self.train_group_ids_)]
        expected_folds = assign_group_folds(identity_records, fold_seed=self.fold_seed)
        assignment = [{"row_id": row["row_id"], "group_id": row["group_id"], "fold": int(fold)}
                      for row, fold in zip(identity_records, expected_folds)]
        if (not np.array_equal(self.folds_, expected_folds) or self.metadata_.get("fold_assignment") != assignment
                or len(self.metadata_.get("folds", [])) != FOLD_COUNT):
            raise ValueError("Stored fold assignments no longer preserve declared group exclusion")
        for fold, metadata in enumerate(self.metadata_["folds"]):
            fitting = [row for row, part in zip(identity_records, expected_folds) if part != fold]
            held = [row for row, part in zip(identity_records, expected_folds) if part == fold]
            if (metadata["training_row_ids"] != [row["row_id"] for row in fitting]
                    or metadata["held_out_row_ids"] != [row["row_id"] for row in held]
                    or metadata["training_group_ids"] != sorted({row["group_id"] for row in fitting})
                    or metadata["held_out_group_ids"] != sorted({row["group_id"] for row in held})
                    or metadata["preprocessing"]["training_row_sha256"] != _digest(metadata["training_row_ids"])):
                raise ValueError("Stored fold preprocessing or memberships violate group exclusion")
        if (self.full_teacher_.metadata != self.metadata_.get("full_train")
                or self.full_teacher_.metadata["training_row_ids"] != self.train_row_ids_
                or self.full_teacher_.preprocessor.training_row_sha256 != _digest(self.train_row_ids_)
                or self.full_teacher_.preprocessor.to_dict() != self.full_teacher_.metadata["preprocessing"]
                or self.full_teacher_.endpoint_scales.tolist() != self.full_teacher_.metadata["endpoint_scales_native"]
                or self.full_teacher_.phase_scales.tolist() != self.full_teacher_.metadata["projection_scaling"]["phase_scales_native"]
                or self.full_teacher_.metadata["training_reports_sha256"] != _array_hash(self.full_teacher_.forest.train_reports_)):
            raise ValueError("Stored full-TRAIN teacher no longer matches its provenance")
        if self.full_teacher_.metadata.get("origin0_forest_law_fingerprint") != origin0_forest_law_fingerprint(self.full_teacher_.forest):
            raise ValueError("Stored full-TRAIN forest law no longer matches its fingerprint")

    def predict_components(self, records):
        """Return full-TRAIN native coefficients, integrated means and raw tree means.

        Shapes are [N,12], [N,8,3] and [N,8,3]. Even queries carrying TRAIN
        identities use the full teacher; OOF training banks are separate fields.
        """
        self._validate_fitted()
        return self.full_teacher_.predict(records)

    def predict_coefficients(self, records):
        """Return full-TRAIN teacher coefficient means [N,12] in native units."""
        return self.predict_components(records)[0]

    def predict_mean_cumulative(self, records):
        """Return integrated coefficient means [N,8,3], including exact day zero."""
        return self.predict_components(records)[1]

    def metadata(self):
        self._validate_fitted()
        return deepcopy(self.metadata_)

    def save(self, path):
        """Write a fresh trusted-local native checkpoint; existing paths are never replaced."""
        self._validate_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            joblib.dump({"schema_version": 1, "model": self, "metadata_sha256": _digest(self.metadata_)}, handle, compress=3)

    @classmethod
    def load(cls, path):
        """Load only a trusted local joblib file; verify stored metadata and OOF banks."""
        payload = joblib.load(path)
        if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                or not isinstance(payload.get("model"), cls)):
            raise ValueError("Unsupported tree static-prior checkpoint")
        model = payload["model"]
        if payload.get("metadata_sha256") != _digest(model.metadata_):
            raise ValueError("Tree static-prior metadata digest differs")
        model._validate_fitted()
        return model
