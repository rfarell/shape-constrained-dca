"""Joint empirical predictive baselines for cumulative production reports.

Phase order is oil, gas, water; reports retain their native BBL/MCF/BBL units.
These are distributions of reported endpoints, including signed corrections,
not distributions of nonnegative latent rates.  A forest is fitted separately
at each forecast origin.  One donor supplies all times and phases of a draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _reports_array(value: np.ndarray, *, times: int | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("reports must have shape (wells, report_times, 3): oil, gas, water")
    if times is not None and value.shape[1] != times:
        raise ValueError("reports do not match fitted report_days")
    if np.isinf(value).any():
        raise ValueError("reports may be finite or NaN, never infinite")
    return value


def _robust_scale(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """A positive training-only scale, including for entirely zero columns."""
    scale = np.nanmedian(np.abs(values), axis=axis)
    fallback = np.nanmean(np.abs(values), axis=axis)
    return np.where(np.isfinite(scale) & (scale > 0), scale,
                    np.where(np.isfinite(fallback) & (fallback > 0), fallback, 1.0))


@dataclass
class _OriginForest:
    index: int
    feature_scale: np.ndarray
    forest: RandomForestRegressor
    leaves: list[dict[int, np.ndarray]]


class ProbabilisticTrajectoryForest:
    """History-conditioned forest-neighborhood distribution of report paths.

    Tree split targets combine future endpoint increments and endpoint levels
    at/after the cutoff, giving the two target families equal total weight.
    Levels are needed to condition an unobserved phase even when its future
    increments are zero.  Each coordinate uses a training-only scale before
    signed ``asinh`` so gas units or late horizons do not dominate. Features include
    every available historical endpoint, phase masks, masked timestamps,
    latest levels, latest observed slopes, and optional numeric covariates.

    A draw chooses a tree uniformly, then a unique training donor uniformly
    from its matching leaf.  Thus the law is a mixture of complete empirical
    paths.  With ``scale_adjustment > 0``, each donor phase receives one positive
    history-fitted amplitude multiplier, shrunk toward one, and a translation
    anchoring its latest available report to the query report.  All observed
    query reports are then clamped to their supplied values.  This is an
    explicit conditional predictive law, not a latent-rate posterior.  Joint
    donor selection preserves empirical dependence; the deterministic history
    adjustment can change its magnitude and must be assessed empirically.

    Only complete training report paths are eligible donors in this first
    implementation; their counts are exposed after fitting.  Arbitrarily
    masked query histories are supported.  Training includes constructed
    historical masks to prepare the forest for those queries.  No report at
    a time later than the prediction origin is used as an input feature or
    in the donor amplitude adjustment.

    Parameters are sklearn-compatible where applicable. ``origins`` in fit
    and ``origin`` in predict_samples are elapsed days, not array indices.
    ``covariate_preprocessing='external'`` consumes finite covariate columns
    unchanged, allowing a shared training-only transform and matched zero
    controls. The default internal transform retains its own scaling/masks.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 12,
        max_features: float | str | int | None = 1.0,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int | None = 1,
        scale_adjustment: float = 0.5,
        scale_bounds: tuple[float, float] = (0.25, 4.0),
        mask_augmentation: int = 1,
        report_mask_probability: float = 0.15,
        phase_mask_probability: float = 0.15,
        covariate_preprocessing: str = "internal",
    ) -> None:
        if not 0 <= scale_adjustment <= 1:
            raise ValueError("scale_adjustment must be between zero and one")
        if not 0 < scale_bounds[0] <= 1 <= scale_bounds[1]:
            raise ValueError("scale_bounds must be positive and bracket one")
        if not isinstance(mask_augmentation, int) or mask_augmentation < 0:
            raise ValueError("mask_augmentation must be a nonnegative integer")
        if not 0 <= report_mask_probability <= 1 or not 0 <= phase_mask_probability <= 1:
            raise ValueError("mask probabilities must be between zero and one")
        if covariate_preprocessing not in {"internal", "external"}:
            raise ValueError("covariate_preprocessing must be internal or external")
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scale_adjustment = scale_adjustment
        self.scale_bounds = scale_bounds
        self.mask_augmentation = mask_augmentation
        self.report_mask_probability = report_mask_probability
        self.phase_mask_probability = phase_mask_probability
        self.covariate_preprocessing = covariate_preprocessing

    def fit(
        self,
        train_reports: np.ndarray,
        report_days: np.ndarray,
        covariates: np.ndarray | None = None,
        *,
        origins: Iterable[float] | None = None,
    ) -> "ProbabilisticTrajectoryForest":
        """Fit only on the supplied training population; return self."""
        reports = _reports_array(train_reports)
        days = np.asarray(report_days, dtype=np.float64)
        if (days.ndim != 1 or len(days) != reports.shape[1] or len(days) < 2
                or not np.isfinite(days).all() or days[0] < 0 or np.any(np.diff(days) <= 0)):
            raise ValueError("report_days must be finite, nonnegative and strictly increasing")
        complete = np.isfinite(reports).all(axis=(1, 2))
        self.complete_donor_count_ = int(complete.sum())
        self.dropped_incomplete_count_ = int((~complete).sum())
        if self.complete_donor_count_ == 0:
            raise ValueError("at least one complete training report trajectory is required")
        self.report_days_ = days.copy()
        self.train_reports_ = reports[complete].copy()
        self.donor_row_indices_ = np.flatnonzero(complete)
        cov = self._fit_covariates(covariates, len(reports), complete)
        requested = list(days[:-1] if origins is None else origins)
        if not requested:
            raise ValueError("at least one origin is required")
        self.origin_models_: dict[float, _OriginForest] = {}
        for origin in requested:
            index = self._origin_index(origin)
            if index == len(days) - 1:
                raise ValueError("an origin must have at least one future report time")
            if float(origin) in self.origin_models_:
                raise ValueError("origins must be distinct")
            rng = np.random.default_rng(np.random.SeedSequence([self.random_state, index]))
            training = self.train_reports_
            # The feature scale uses only training history at this origin.
            scale = _robust_scale(training[:, :index + 1], axis=(0, 1))
            histories = [training]
            for _ in range(self.mask_augmentation):
                masked = training.copy()
                hide = rng.random((len(training), index + 1, 3)) < self.report_mask_probability
                hide |= (rng.random((len(training), 1, 3)) < self.phase_mask_probability)
                masked[:, :index + 1] = np.where(hide, np.nan, masked[:, :index + 1])
                histories.append(masked)
            feature_rows = [self._history_features(h, index, scale) for h in histories]
            features = np.concatenate(feature_rows, axis=0)
            if cov.shape[1]:
                features = np.column_stack([features, np.tile(cov, (len(histories), 1))])
            increments = training[:, index + 1:] - training[:, index:index + 1]
            target_scale = _robust_scale(increments, axis=0)
            increment_targets = np.arcsinh(increments / target_scale).reshape(len(training), -1)
            endpoints = training[:, index:]
            endpoint_scale = _robust_scale(endpoints, axis=0)
            endpoint_targets = np.arcsinh(endpoints / endpoint_scale).reshape(len(training), -1)
            targets = np.column_stack([
                increment_targets / np.sqrt(increment_targets.shape[1]),
                endpoint_targets / np.sqrt(endpoint_targets.shape[1]),
            ])
            targets = np.tile(targets, (len(histories), 1))
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features, max_depth=self.max_depth,
                random_state=int(rng.integers(0, 2**31 - 1)), n_jobs=self.n_jobs,
            ).fit(features, targets)
            assignments = forest.apply(features)
            donor_ids = np.tile(np.arange(len(training)), len(histories))
            leaves: list[dict[int, np.ndarray]] = []
            for tree in range(assignments.shape[1]):
                order = np.argsort(assignments[:, tree], kind="stable")
                labels, starts = np.unique(assignments[order, tree], return_index=True)
                chunks = np.split(order, starts[1:])
                # Mask replicas never multiply a donor's weight within a leaf.
                leaves.append({int(label): np.unique(donor_ids[chunk])
                               for label, chunk in zip(labels, chunks)})
            self.origin_models_[float(origin)] = _OriginForest(index, scale, forest, leaves)
        return self

    def _fit_covariates(self, covariates, rows, complete) -> np.ndarray:
        if covariates is None:
            self.n_covariates_ = 0
            self.covariate_center_ = np.empty(0)
            self.covariate_scale_ = np.empty(0)
            return np.empty((int(complete.sum()), 0))
        cov = np.asarray(covariates, dtype=np.float64)
        if cov.ndim != 2 or len(cov) != rows or np.isinf(cov).any():
            raise ValueError("covariates must be a finite-or-NaN matrix with one row per well")
        if getattr(self, "covariate_preprocessing", "internal") == "external" and not np.isfinite(cov).all():
            raise ValueError("Externally preprocessed covariates must be finite")
        cov = cov[complete]
        self.n_covariates_ = cov.shape[1]
        if getattr(self, "covariate_preprocessing", "internal") == "external":
            self.covariate_center_ = np.zeros(self.n_covariates_)
            self.covariate_scale_ = np.ones(self.n_covariates_)
            return cov.copy()
        self.covariate_center_ = np.array([
            np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
            for column in cov.T
        ])
        self.covariate_scale_ = np.array([
            max(float(np.median(np.abs(column[np.isfinite(column)] - center))), 1e-12)
            if np.isfinite(column).any() else 1.0
            for column, center in zip(cov.T, self.covariate_center_)
        ])
        return self._covariate_features(cov, len(cov))

    def _covariate_features(self, covariates, rows) -> np.ndarray:
        if self.n_covariates_ == 0:
            if covariates is not None and np.asarray(covariates).shape != (rows, 0):
                raise ValueError("model was fitted without covariates")
            return np.empty((rows, 0))
        cov = np.asarray(covariates, dtype=np.float64)
        if cov.shape != (rows, self.n_covariates_) or np.isinf(cov).any():
            raise ValueError("prediction covariates must match the fitted covariate matrix")
        if getattr(self, "covariate_preprocessing", "internal") == "external":
            if not np.isfinite(cov).all():
                raise ValueError("Externally preprocessed covariates must be finite")
            return cov.copy()
        mask = np.isfinite(cov)
        scaled = np.where(mask, (cov - self.covariate_center_) / self.covariate_scale_, 0)
        return np.column_stack([np.arcsinh(scaled), mask])

    def _origin_index(self, origin: float) -> int:
        matches = np.flatnonzero(self.report_days_ == float(origin))
        if len(matches) != 1:
            raise ValueError("origin must equal a canonical report day")
        return int(matches[0])

    def _history_features(self, reports, index, scale) -> np.ndarray:
        history = reports[:, :index + 1]
        mask = np.isfinite(history)
        levels = np.arcsinh(np.where(mask, history, 0.0) / scale)
        days = self.report_days_[:index + 1]
        time_scale = max(float(days[-1]), 1.0)
        timestamps = mask * (days / time_scale)[None, :, None]
        latest = np.zeros((len(reports), 3))
        age = np.ones_like(latest)
        slope = np.zeros_like(latest)
        counts = mask.sum(axis=1)
        for phase in range(3):
            last_index = np.where(mask[:, :, phase], np.arange(index + 1), -1).max(axis=1)
            present = last_index >= 0
            row = np.flatnonzero(present)
            at = last_index[present]
            latest[row, phase] = levels[row, at, phase]
            age[row, phase] = (days[-1] - days[at]) / time_scale
            previous = mask[:, :, phase] & (np.arange(index + 1)[None, :] < last_index[:, None])
            prev_index = np.where(previous, np.arange(index + 1), -1).max(axis=1)
            row = np.flatnonzero(prev_index >= 0)
            at, prev = last_index[row], prev_index[row]
            rate = (history[row, at, phase] - history[row, prev, phase]) / (days[at] - days[prev])
            slope[row, phase] = np.arcsinh(rate * time_scale / scale[phase])
        return np.column_stack([
            levels.reshape(len(reports), -1), mask.reshape(len(reports), -1),
            timestamps.reshape(len(reports), -1), latest, age, slope,
            counts / (index + 1),
        ])

    def predict_samples(
        self,
        history_reports: np.ndarray,
        origin: float,
        n_samples: int,
        rng: np.random.Generator | int | None = None,
        covariates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return joint report samples with shape ``(N, S, T, 3)``.

        Values supplied after origin are ignored, even if they are finite.
        Supplied historical observations are exactly preserved in every draw.
        NaNs denote absent reports, never zero production or shut-in.
        """
        if not hasattr(self, "origin_models_"):
            raise ValueError("fit must be called before predict_samples")
        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        history = _reports_array(history_reports, times=len(self.report_days_))
        if float(origin) not in self.origin_models_:
            raise ValueError("origin was not included in fit")
        model = self.origin_models_[float(origin)]
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        if len(history) == 0:
            return np.empty((0, n_samples, len(self.report_days_), 3))
        features = self._history_features(history, model.index, model.feature_scale)
        cov = self._covariate_features(covariates, len(history))
        if cov.shape[1]:
            features = np.column_stack([features, cov])
        assignments = model.forest.apply(features)
        samples = np.empty((len(history), n_samples, len(self.report_days_), 3))
        for row in range(len(history)):
            tree_ids = generator.integers(0, len(model.leaves), size=n_samples)
            donor_ids = np.empty(n_samples, dtype=int)
            for tree in np.unique(tree_ids):
                chosen = np.flatnonzero(tree_ids == tree)
                members = model.leaves[tree][int(assignments[row, tree])]
                donor_ids[chosen] = generator.choice(members, size=len(chosen))
            draws = self.train_reports_[donor_ids].copy()
            for phase in range(3):
                observed = np.flatnonzero(np.isfinite(history[row, :model.index + 1, phase]))
                if len(observed) == 0:
                    continue
                anchor = observed[-1]
                actual = history[row, observed, phase]
                donor_history = draws[:, observed, phase]
                # A scale-relative ridge shrinks a noisy short-history estimate
                # to one and remains equivariant to phase unit conversion.
                ridge = (0.1 * model.feature_scale[phase]) ** 2 * len(observed)
                multiplier = ((donor_history @ actual + ridge)
                              / ((donor_history ** 2).sum(axis=1) + ridge))
                multiplier = np.clip(multiplier, *self.scale_bounds) ** self.scale_adjustment
                draws[:, :, phase] = (history[row, anchor, phase]
                                      + multiplier[:, None]
                                      * (draws[:, :, phase] - draws[:, anchor:anchor + 1, phase]))
                draws[:, observed, phase] = actual
            samples[row] = draws
        return samples
