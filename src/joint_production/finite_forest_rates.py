"""Finite static forest prior and exact joint history-conditioned rate posterior.

Each atom is one complete positive, integrable oil/gas/water basis trajectory,
obtained by projecting one TRAIN donor. Exact origin-zero forest probabilities
receive a fixed 5% uniform background over every donor. Conditional on an atom,
reports follow the accumulated Gaussian endpoint law; all observed endpoints
enter its joint likelihood once. Enumerating this finite support gives an exact
categorical posterior under the declared model, without an importance proposal.

Donor-estimation and fitted population uncertainty are not integrated out.
Finite tails assume continued operation; historical agreement and coherent
sampling do not establish identified latent production or calibration.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
import torch

from .baselines import ProbabilisticTrajectoryForest
from .observation import gaussian_endpoint_log_likelihood
from .static_features import StaticFeaturePreprocessor
from .torch_observation import sample_accumulated_reports
from .tree_static_prior import (
    COEFFICIENT_FLOOR, FOREST_SETTINGS, PHASE_ORDER, SCALES_DAYS, SHAPES,
    exact_static_forest_mean, fast_basis_cumulative_matrix, fit_projection_scales,
    origin0_forest_law_fingerprint, project_positive_coefficients,
)

REPORT_DAYS = np.array([0., 30., 60., 90., 180., 360., 720., 1080.])
BACKGROUND = .05
OBSERVATION_LAW = "conditional_gaussian_accumulated_bridge"
INTERPRETATION = "exact_categorical_posterior_under_static_conditioned_finite_joint_rate_prior"


def _hash_array(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(json.dumps(list(array.shape)).encode()+b"\n"+array.tobytes()).hexdigest()


def _hash_ids(value):
    return hashlib.sha256(json.dumps(list(value), separators=(",", ":")).encode()).hexdigest()


def _static_records(records, *, training=False):
    if not isinstance(records, Sequence) or not len(records):
        raise ValueError("Require a nonempty sequence of raw static records")
    result = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("covariates_raw"), Mapping):
            raise ValueError("Raw static query records require covariates_raw")
        static = {"covariates_raw": record["covariates_raw"]}
        if training:
            if (record.get("split") != "train" or not isinstance(record.get("row_id"), str)
                    or not record["row_id"] or not isinstance(record.get("group_id"), str) or not record["group_id"]):
                raise ValueError("Fitting requires explicit TRAIN row and group identities")
            static.update({key: record[key] for key in ("row_id", "group_id", "split")})
        result.append(static)
    if training and len({record["row_id"] for record in result}) != len(result):
        raise ValueError("Fitting TRAIN row identities must be unique")
    return result


def _days(days):
    value = np.asarray(days, dtype=np.float64)
    if not np.array_equal(value, REPORT_DAYS):
        raise ValueError("Finite forest prior requires the canonical eight report days")
    return value.copy()


@dataclass
class ForestRatePriorBank:
    processor: StaticFeaturePreprocessor
    forest: ProbabilisticTrajectoryForest
    coefficients_native: np.ndarray
    cumulative_native: np.ndarray
    endpoint_scales_native: np.ndarray
    phase_scales_native: np.ndarray
    provenance: dict

    def exact_weights(self, query_records, *, return_receipt=False):
        """Return [.95 forest + .05/M] without reading query reports or IDs."""
        features = self.processor.transform(_static_records(query_records))
        # This validates unique donor tables, accounting zero, and unit history
        # multipliers. It is the mean of the issued law, not sklearn.predict.
        reference = exact_static_forest_mean(self.forest, features)
        model = self.forest.origin_models_[0.0]
        history = np.full((len(features), 8, 3), np.nan)
        history[:, 0] = 0.
        encoded = np.column_stack((self.forest._history_features(history, 0, model.feature_scale),
                                   self.forest._covariate_features(features, len(features))))
        labels = model.forest.apply(encoded)
        weights = np.zeros((len(features), len(self.coefficients_native)), dtype=np.float64)
        trees, leaves, offsets, members = [], [], [0], []
        for tree, table in enumerate(model.leaves):
            for leaf in np.unique(labels[:, tree]):
                donors = np.asarray(table[int(leaf)])
                weights[np.ix_(np.flatnonzero(labels[:, tree] == leaf), donors)] += 1./(len(model.leaves)*len(donors))
                if return_receipt:
                    trees.append(tree); leaves.append(int(leaf))
                    members.extend(donors.tolist()); offsets.append(len(members))
        mean = np.einsum("nm,mtp->ntp", weights, self.forest.train_reports_)
        if (not np.allclose(weights.sum(1), 1., atol=5e-13, rtol=0)
                or not np.allclose(mean, reference, atol=1e-10, rtol=5e-14)):
            raise FloatingPointError("Exact forest donor probabilities failed issued-law verification")
        prior = (1-BACKGROUND)*weights + BACKGROUND/weights.shape[1]
        if not np.isfinite(prior).all() or np.any(prior <= 0):
            raise FloatingPointError("Finite prior must retain every donor with positive probability")
        if return_receipt:
            return prior, {"forest_weights": weights, "query_tree_leaf_ids": labels.astype(np.int64),
                "receipt_tree_indices": np.asarray(trees, dtype=np.int64), "receipt_leaf_ids": np.asarray(leaves, dtype=np.int64),
                "receipt_member_offsets": np.asarray(offsets, dtype=np.int64), "receipt_donor_indices": np.asarray(members, dtype=np.int64)}
        return prior

    def metadata(self):
        return deepcopy(self.provenance)

    def validate(self):
        contract = {"schema_version": 1, "kind": "finite_forest_positive_basis_prior", "fit_split": "train",
            "phase_order": list(PHASE_ORDER), "units": ["BBL", "MCF", "BBL"], "report_days": REPORT_DAYS.tolist(),
            "basis_shapes": list(SHAPES), "basis_scales_days": list(SCALES_DAYS),
            "coefficient_floor_normalized": COEFFICIENT_FLOOR, "uniform_background_probability": BACKGROUND,
            "amplitude_augmentation": False}
        if any(self.provenance.get(key) != value for key, value in contract.items()):
            raise ValueError("Finite prior metadata differs from the declared serving law")
        n = len(self.forest.train_reports_)
        arrays = {"coefficients_native": self.coefficients_native, "cumulative_native": self.cumulative_native,
                  "endpoint_scales_native": self.endpoint_scales_native, "phase_scales_native": self.phase_scales_native}
        shapes = {"coefficients_native": (n, 12), "cumulative_native": (n, 8, 3),
                  "endpoint_scales_native": (7, 3), "phase_scales_native": (3,)}
        for key, value in arrays.items():
            if (value.shape != shapes[key] or not np.isfinite(value).all()
                    or _hash_array(value) != self.provenance["array_sha256"][key]):
                raise ValueError("Finite prior bank array/provenance mismatch: "+key)
        if (np.any(self.coefficients_native <= 0) or np.any(self.endpoint_scales_native <= 0)
                or np.any(self.phase_scales_native <= 0)
                or not np.array_equal(self.forest.donor_row_indices_, np.arange(n))
                or _hash_array(self.forest.train_reports_) != self.provenance["training_reports_sha256"]
                or self.processor.to_dict() != self.provenance["preprocessing"]
                or origin0_forest_law_fingerprint(self.forest) != self.provenance["origin0_forest_law_fingerprint"]):
            raise ValueError("Finite bank donor/scale/forest/preprocessing state differs from its provenance")
        reconstruction = np.einsum("mpk,tk->mtp", self.coefficients_native.reshape(n, 3, 4), fast_basis_cumulative_matrix(REPORT_DAYS))
        if not np.array_equal(reconstruction, self.cumulative_native):
            raise ValueError("Finite donor trajectories differ from saved coefficients")


def fit_forest_rate_prior(records, reports, days, seed, n_jobs=2):
    """Fit one teacher on exactly these TRAIN rows, including every donor floor.

    The caller is responsible for whole-fold exclusion when using this factory
    for cross-fitting. Raw static preprocessing and every scale fit these rows
    only; the explicit reports argument is the sole production target input.
    """
    rows = _static_records(records, training=True)
    days = _days(days)
    y = np.asarray(reports, dtype=np.float64)
    if (y.shape != (len(rows), 8, 3) or not np.isfinite(y).all() or np.any(y[:, 0] != 0)
            or isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("Require complete finite TRAIN reports, accounting zero, and a nonnegative seed")
    processor = StaticFeaturePreprocessor.fit(rows)
    features = processor.transform(rows)
    endpoint, phase, scale_metadata = fit_projection_scales(y)
    forest = ProbabilisticTrajectoryForest(random_state=int(seed), n_jobs=n_jobs, **FOREST_SETTINGS).fit(
        y, days, features, origins=[0])
    if (forest.dropped_incomplete_count_ or forest.complete_donor_count_ != len(rows)
            or not np.array_equal(forest.donor_row_indices_, np.arange(len(rows)))
            or not np.array_equal(forest.train_reports_, y)):
        raise ValueError("Finite support must retain every fitting donor in order")
    coefficients, cumulative = project_positive_coefficients(y, days, endpoint, phase)
    row_ids, group_ids = [row["row_id"] for row in rows], [row["group_id"] for row in rows]
    arrays = dict(coefficients_native=coefficients, cumulative_native=cumulative,
                  endpoint_scales_native=endpoint, phase_scales_native=phase)
    provenance = {"schema_version": 1, "kind": "finite_forest_positive_basis_prior", "fit_split": "train",
        "seed": int(seed), "training_rows": len(rows), "training_row_ids": row_ids,
        "training_group_ids": group_ids, "training_row_sha256": _hash_ids(row_ids),
        "training_group_sha256": _hash_ids(sorted(set(group_ids))), "training_reports_sha256": _hash_array(y),
        "preprocessing": processor.to_dict(), "projection_scaling": scale_metadata,
        "array_sha256": {name: _hash_array(value) for name, value in arrays.items()},
        "origin0_forest_law_fingerprint": origin0_forest_law_fingerprint(forest),
        "phase_order": list(PHASE_ORDER), "units": ["BBL", "MCF", "BBL"], "report_days": days.tolist(),
        "coefficient_order": "phase-major oil[4],gas[4],water[4]", "basis_shapes": list(SHAPES),
        "basis_scales_days": list(SCALES_DAYS), "coefficient_floor_normalized": COEFFICIENT_FLOOR,
        "uniform_background_probability": BACKGROUND, "amplitude_augmentation": False,
        "prior_law": "0.95*uniform_tree_uniform_unique_leaf_donor + 0.05/uniform_all_fitting_donors",
        "mean_preservation": False, "empirical_donor_estimation_uncertainty_integrated": False}
    bank = ForestRatePriorBank(processor, forest, coefficients, cumulative, endpoint, phase, provenance)
    bank.validate()
    return bank


def normalized_finite_posterior(prior, log_likelihood):
    """Stable exact categorical normalization; retain every support slot."""
    prior, ll = np.asarray(prior, dtype=np.float64), np.asarray(log_likelihood, dtype=np.float64)
    if (prior.ndim != 2 or not all(prior.shape) or ll.shape != prior.shape
            or not np.isfinite(prior).all() or np.any(prior <= 0) or not np.isfinite(ll).all()
            or not np.allclose(prior.sum(1), 1., atol=5e-13, rtol=0)):
        raise ValueError("Finite posterior requires normalized positive prior and every finite likelihood")
    offset = ll.max(1, keepdims=True)
    centered = np.log(prior)+(ll-offset)
    normalizer = logsumexp(centered, axis=1, keepdims=True)
    logp = centered-normalizer
    p = np.exp(logp)
    logz = (normalizer+offset)[:, 0]
    if (not np.isfinite(logp).all() or not np.isfinite(logz).all()
            or not np.allclose(p.sum(1), 1., atol=5e-13, rtol=0)):
        raise FloatingPointError("Finite posterior normalization exceeded numerical support")
    return p, logp, logz


class FiniteForestRateForecaster:
    rate_family = "basis"
    basis_dictionary = "fast"
    latent_size = 12
    observation_law = OBSERVATION_LAW
    report_half_life_days = None

    def __init__(self, *, seed=42, noise_mode="fixed", device="cpu", n_jobs=2,
                 uniform_background_probability=.05, coefficient_floor_normalized=1e-6,
                 absolute_standard_deviation_normalized=.05, relative_standard_deviation=.1):
        if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0
                or noise_mode not in {"fixed", "oof_fitted"} or str(device) not in {"cpu", "cuda"}
                or isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1):
            raise ValueError("Require a nonnegative seed, fixed/oof_fitted noise, CPU/CUDA, and positive n_jobs")
        if (uniform_background_probability != BACKGROUND or coefficient_floor_normalized != COEFFICIENT_FLOOR
                or absolute_standard_deviation_normalized != .05 or relative_standard_deviation != .1):
            raise ValueError("Finite prior and fixed noise constants differ from the frozen protocol")
        self.seed, self.noise_mode, self.device, self.n_jobs = int(seed), noise_mode, torch.device(device), int(n_jobs)
        self.configuration = {"seed": self.seed, "noise_mode": noise_mode, "device": str(self.device), "n_jobs": self.n_jobs,
            "uniform_background_probability": BACKGROUND, "coefficient_floor_normalized": COEFFICIENT_FLOOR,
            "absolute_standard_deviation_normalized": .05, "relative_standard_deviation": .1}

    def _covariates(self, records, covariates, bank):
        if covariates is not None:
            values = np.asarray(covariates, dtype=np.float64)
            expected = bank.processor.transform(_static_records(records))
            if values.shape != expected.shape or not np.isfinite(values).all() or not np.array_equal(values, expected):
                raise ValueError("External features differ from this full-TRAIN static preprocessor")

    def fit(self, reports, report_days, *, records, covariates=None, origins=None):
        rows = _static_records(records, training=True)
        days = _days(report_days)
        y = np.asarray(reports, dtype=np.float64)
        if y.shape != (len(rows), 8, 3) or not np.isfinite(y).all() or np.any(y[:, 0] != 0):
            raise ValueError("Require complete finite explicit TRAIN reports with deterministic day zero")
        self.report_days_ = days
        self.report_days = torch.as_tensor(days, dtype=torch.float64, device=self.device)
        self.origins_ = list(days[:-1] if origins is None else origins)
        if not self.origins_ or len(set(self.origins_)) != len(self.origins_) or any(origin not in days for origin in self.origins_):
            raise ValueError("Requested origins must be unique canonical report days")
        self.bank_, self.noise_fit_ = None, None
        self.fit_diagnostics_ = {"interpretation": INTERPRETATION, "noise_mode": self.noise_mode,
            "observation_law": OBSERVATION_LAW, "phase_order": list(PHASE_ORDER), "training_rows": len(rows),
            "training_row_sha256": _hash_ids([row["row_id"] for row in rows]),
            "origins_days": list(map(float, self.origins_)), "converged": False,
            "latent_parameterization": "log_phase_normalized_positive_basis_coefficient", "seed": self.seed}
        if self.noise_mode == "oof_fitted":
            from .finite_forest_noise import fit_oof_forest_noise
            self.noise_fit_ = fit_oof_forest_noise(rows, y, days, seed=self.seed,
                                                 device=str(self.device), n_jobs=self.n_jobs)
            self.fit_diagnostics_["noise_fit"] = deepcopy(self.noise_fit_.diagnostics)
            self.fit_diagnostics_["status"] = self.noise_fit_.status
            if not self.noise_fit_.converged:
                self.training_diagnostics_ = self.fit_diagnostics_
                return self
            absolute, relative = self.noise_fit_.noise_abs, self.noise_fit_.noise_rel
        else:
            absolute, relative = np.full(3, .05), np.full(3, .1)
        for value in (absolute, relative):
            if np.shape(value) != (3,) or not np.isfinite(value).all() or np.any(np.asarray(value) <= 0):
                raise ValueError("Converged finite noise must have finite positive phase vectors")
        self.bank_ = fit_forest_rate_prior(rows, y, days, self.seed, self.n_jobs)
        self._covariates(rows, covariates, self.bank_)
        self.volume_scale = torch.as_tensor(self.bank_.phase_scales_native, dtype=torch.float64, device=self.device)
        self.noise_abs = torch.as_tensor(absolute, dtype=torch.float64, device=self.device)
        self.noise_rel = torch.as_tensor(relative, dtype=torch.float64, device=self.device)
        self.fit_diagnostics_.update({"converged": True, "status": "ready", "finite_support_size": len(y),
            "noise_abs_normalized": self.noise_abs.cpu().tolist(), "noise_rel": self.noise_rel.cpu().tolist(),
            "phase_scales_native": self.bank_.phase_scales_native.tolist(),
            "endpoint_scales_native": self.bank_.endpoint_scales_native.tolist(),
            "finite_prior": self.bank_.metadata(),
            "origin0_forest_law_fingerprint": self.bank_.provenance["origin0_forest_law_fingerprint"],
            "sampling_law": "One exact posterior atom for all phases and times, followed by the same accumulated Gaussian conditional bridge",
            "posterior_evaluation": "All support atoms; native float64 likelihood; query chunks of at most16",
            "rng_law": "One NumPy stream: per-query categorical atoms, then one full joint report-normal array"})
        self.training_diagnostics_ = self.fit_diagnostics_
        return self

    def _require_ready(self):
        if not getattr(self, "fit_diagnostics_", {}).get("converged") or self.bank_ is None:
            raise RuntimeError("Finite model is not ready: a successful fixed or converged OOF noise fit is required before prediction")

    def _history(self, history, origin):
        self._require_ready()
        if not np.isfinite(origin) or origin < 0:
            raise ValueError("Cutoff must be finite and nonnegative")
        y = np.asarray(history, dtype=np.float64)
        if y.ndim != 3 or not len(y) or y.shape[1:] != (8, 3):
            raise ValueError("History must be [nonempty queries,8,3]")
        y = y.copy()
        y[:, self.report_days_ > origin] = np.nan  # Before any value validation.
        if np.isinf(y).any() or np.any(np.isfinite(y[:, 0]) & (y[:, 0] != 0)):
            raise ValueError("Permitted reports must be finite or missing, with deterministic day zero")
        y[:, 0] = 0.
        return y

    def conditional_probabilities(self, history, origin, *, records, covariates=None):
        y = self._history(history, origin)
        static = _static_records(records)
        if len(static) != len(y):
            raise ValueError("One static record is required for each query row")
        self._covariates(static, covariates, self.bank_)
        prior = self.bank_.exact_weights(static)
        length = np.count_nonzero(self.report_days_ <= origin)
        q = torch.as_tensor(self.bank_.cumulative_native[:, None, :length], dtype=torch.float64, device=self.device)
        days = self.report_days[:length]
        ll = np.empty_like(prior)
        with torch.no_grad():
            for start in range(0, len(y), 16):
                prefix = torch.as_tensor(y[start:start+16, :length], dtype=torch.float64, device=self.device)
                mask = torch.isfinite(prefix)
                probability = gaussian_endpoint_log_likelihood(prefix, q.expand(-1, len(prefix), -1, -1),
                    self.noise_abs*self.volume_scale, self.noise_rel, mask, days, law="accumulated")
                ll[start:start+len(prefix)] = probability.T.cpu().numpy()
        posterior, log_posterior, logz = normalized_finite_posterior(prior, ll)
        return {"prior_atom_probabilities": prior, "posterior_atom_probabilities": posterior,
                "posterior_atom_log_probabilities": log_posterior, "atom_log_likelihood_native": ll,
                "history_log_evidence_native": logz}

    def evaluate_trajectories(self, latent_parameters, query_days):
        self._require_ready()
        if isinstance(latent_parameters, torch.Tensor):
            latent_parameters = latent_parameters.detach().cpu().numpy()
        if isinstance(query_days, torch.Tensor):
            query_days = query_days.detach().cpu().numpy()
        z, days = np.asarray(latent_parameters, dtype=np.float64), np.asarray(query_days, dtype=np.float64)
        if z.ndim < 1 or z.shape[-1] != 12 or not np.isfinite(z).all() or days.ndim != 1 or np.isnan(days).any() or np.any(days < 0):
            raise ValueError("Require finite joint log coefficients and nonnegative query times, with positive infinity allowed")
        with np.errstate(over="raise", invalid="raise"):
            coefficients = np.exp(z.reshape(*z.shape[:-1], 3, 4))*self.bank_.phase_scales_native[:, None]
        return self._coefficient_trajectories(coefficients, days)

    @staticmethod
    def _coefficient_trajectories(coefficients, days):
        finite_days = np.where(np.isposinf(days), 0., days)
        x = finite_days[:, None]/np.asarray(SCALES_DAYS)
        basis_rate = np.where(np.asarray(SHAPES) == 1, 1., x)*np.exp(-x)/np.asarray(SCALES_DAYS)
        basis_rate[np.isposinf(days)] = 0.
        with np.errstate(over="raise", invalid="raise"):
            result = {"latent_rate": np.einsum("...pk,tk->...tp", coefficients, basis_rate),
                "latent_cumulative": np.einsum("...pk,tk->...tp", coefficients, fast_basis_cumulative_matrix(days)),
                "latent_total_volume": coefficients.sum(-1)}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Finite atom query exceeded numeric support; no sample clipping")
        return result

    def predict_distribution(self, history, origin, n_samples, rng, *, records, covariates=None):
        if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("Prediction requires an explicit NumPy Generator")
        y = self._history(history, origin)
        probabilities = self.conditional_probabilities(y, origin, records=records, covariates=covariates)
        p = probabilities["posterior_atom_probabilities"]
        indices = np.stack([rng.choice(p.shape[1], size=n_samples, p=row) for row in p])
        coefficients = self.bank_.coefficients_native[indices]
        paths = self._coefficient_trajectories(coefficients.reshape(len(y), n_samples, 3, 4), self.report_days_)
        z = np.log(coefficients/np.repeat(self.bank_.phase_scales_native, 4))
        normals = torch.as_tensor(rng.standard_normal(paths["latent_cumulative"].shape), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            reports = sample_accumulated_reports(
                torch.as_tensor(paths["latent_cumulative"], dtype=torch.float64, device=self.device),
                torch.as_tensor(y, dtype=torch.float64, device=self.device), self.report_days,
                self.noise_abs*self.volume_scale, self.noise_rel, cutoff=float(origin), standard_normals=normals).cpu().numpy()
        reports = np.where(np.isfinite(y)[:, None], y[:, None], reports)
        if not np.isfinite(reports).all():
            raise FloatingPointError("Finite posterior report sampler exceeded numerical support")
        observed = np.isfinite(y) & (self.report_days_[None, :, None] > 0)
        diagnostics = {"interpretation": INTERPRETATION, "noise_mode": self.noise_mode,
            "categorical_effective_count": (1/np.sum(p*p, axis=1)).tolist(),
            "maximum_posterior_atom_probability": p.max(1).tolist(),
            "history_log_evidence_native": probabilities["history_log_evidence_native"].tolist(),
            "history_observed_counts_by_phase": observed.sum(1).tolist(),
            "diagnostic_scope": "Exact finite posterior concentration, not importance adequacy or latent calibration",
            "cross_cutoff_bayesian_consistency": "Under this fixed finite prior and fixed fitted report law only"}
        self.last_conditional_diagnostics_ = diagnostics
        return {**paths, **probabilities, "latent_parameters": z, "latent_coefficients_native": coefficients,
            "posterior_atom_indices": indices, "report_cumulative": reports, "observed_history_mask": observed,
            "conditional_diagnostics": diagnostics, "interpretation": INTERPRETATION,
            "phase_order": list(PHASE_ORDER), "observation_law": OBSERVATION_LAW, "noise_mode": self.noise_mode,
            "report_half_life_days": None, "finite_support_size": len(self.bank_.coefficients_native)}

    def predict_samples(self, history, origin, n_samples, rng, *, records, covariates=None):
        return self.predict_distribution(history, origin, n_samples, rng, records=records, covariates=covariates)["report_cumulative"]

    def save(self, path):
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the finite model before saving")
        checkpoint = {"schema_version": 1, "kind": "finite_forest_rate", "configuration": self.configuration,
            "fit_diagnostics": self.fit_diagnostics_, "bank": self.bank_,
            "noise_fit": replace(self.noise_fit_, fold_data=()) if self.noise_fit_ is not None else None,
            "report_days": self.report_days_, "origins": self.origins_,
            "noise_abs": self.noise_abs.cpu().numpy() if self.bank_ is not None else None,
            "noise_rel": self.noise_rel.cpu().numpy() if self.bank_ is not None else None,
            "interpretation": INTERPRETATION, "observation_law": OBSERVATION_LAW}
        with Path(path).open("xb") as handle:
            joblib.dump(checkpoint, handle, compress=3)

    def save_evidence(self, output, prefix):
        """Retain separate OOF banks before sealing; fixed noise has none."""
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError("Fit the finite model before saving evidence")
        return {} if self.noise_fit_ is None else self.noise_fit_.save_artifacts(output, prefix)

    @classmethod
    def load(cls, path, *, device="cpu"):
        """Load a trusted local joblib checkpoint; all tensor state is portable."""
        checkpoint = joblib.load(path)
        if (checkpoint.get("schema_version") != 1 or checkpoint.get("kind") != "finite_forest_rate"
                or checkpoint.get("interpretation") != INTERPRETATION or checkpoint.get("observation_law") != OBSERVATION_LAW):
            raise ValueError("Invalid finite forest checkpoint schema or law")
        model = cls(**dict(checkpoint["configuration"], device=device))
        model.fit_diagnostics_ = deepcopy(checkpoint["fit_diagnostics"])
        model.training_diagnostics_ = model.fit_diagnostics_
        model.bank_, model.noise_fit_ = checkpoint["bank"], checkpoint["noise_fit"]
        model.report_days_ = _days(checkpoint["report_days"])
        model.report_days = torch.as_tensor(model.report_days_, dtype=torch.float64, device=model.device)
        model.origins_ = list(checkpoint["origins"])
        expected = {"noise_mode": model.noise_mode, "observation_law": OBSERVATION_LAW,
                    "interpretation": INTERPRETATION, "seed": model.seed, "phase_order": list(PHASE_ORDER),
                    "origins_days": list(map(float, model.origins_))}
        if (any(model.fit_diagnostics_.get(key) != value for key, value in expected.items())
                or not model.origins_ or len(set(model.origins_)) != len(model.origins_)
                or any(origin not in model.report_days_ for origin in model.origins_)):
            raise ValueError("Checkpoint fit metadata differs from its declared configuration and law")
        if model.fit_diagnostics_.get("converged"):
            if model.bank_ is None:
                raise ValueError("Ready checkpoint is missing its finite prior")
            model.bank_.validate()
            if model.bank_.metadata() != model.fit_diagnostics_["finite_prior"]:
                raise ValueError("Checkpoint bank metadata differs from fit diagnostics")
            if (model.bank_.provenance["seed"] != model.seed
                    or model.bank_.provenance["training_rows"] != model.fit_diagnostics_["training_rows"]
                    or model.bank_.provenance["training_row_sha256"] != model.fit_diagnostics_["training_row_sha256"]
                    or not np.array_equal(model.bank_.phase_scales_native, model.fit_diagnostics_["phase_scales_native"])
                    or not np.array_equal(model.bank_.endpoint_scales_native, model.fit_diagnostics_["endpoint_scales_native"])):
                raise ValueError("Checkpoint serving cohort or scales differ from fit metadata")
            model.volume_scale = torch.as_tensor(model.bank_.phase_scales_native, dtype=torch.float64, device=model.device)
            for key, diagnostic_key in (("noise_abs", "noise_abs_normalized"), ("noise_rel", "noise_rel")):
                value = np.asarray(checkpoint[key], dtype=np.float64)
                if (value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0)
                        or not np.array_equal(value, model.fit_diagnostics_[diagnostic_key])):
                    raise ValueError("Checkpoint noise vector differs from fitted law")
                setattr(model, key, torch.as_tensor(value, dtype=torch.float64, device=model.device))
            if model.noise_mode == "oof_fitted" and (model.noise_fit_ is None or not model.noise_fit_.converged):
                raise ValueError("Fitted checkpoint has no converged OOF noise result")
            if model.noise_mode == "fixed":
                if (not np.array_equal(checkpoint["noise_abs"], np.full(3, .05))
                        or not np.array_equal(checkpoint["noise_rel"], np.full(3, .1)) or model.noise_fit_ is not None):
                    raise ValueError("Fixed checkpoint noise differs from the frozen constants")
            elif (not np.array_equal(checkpoint["noise_abs"], model.noise_fit_.noise_abs)
                    or not np.array_equal(checkpoint["noise_rel"], model.noise_fit_.noise_rel)):
                raise ValueError("Serving noise differs from the retained OOF fitting result")
        elif model.bank_ is not None:
            raise ValueError("Unresolved checkpoint unexpectedly contains a serving bank")
        return model
