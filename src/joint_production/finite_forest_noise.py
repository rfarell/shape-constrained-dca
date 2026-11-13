"""Group-excluded empirical noise fitting for an exact finite trajectory prior.

One joint donor supplies all phases and report times. Each fold's forest,
preprocessor, positive-basis projection and scales use its TRAIN complement.
The six shared parameters are log standard deviations in fold-normalized
phase units. The objective is the mean native-unit marginal trajectory NLL,
with one log-sum-exp across joint donors after summing all 21 innovations.

The analytic gradient avoids differentiating forest weights or storing an
autograd graph. The elapsed-time limit is cooperative, checked around banks,
objectives and streamed batches; an external supervisor must enforce a hard
whole-process deadline. Failed convergence never authorizes prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize
import torch

from .artifacts import canonical_hash, sha256_file, write_json
from .tree_static_prior import assign_group_folds, FOLD_COUNT, FOLD_NAMESPACE


REPORT_DAYS = np.array([0., 30., 60., 90., 180., 360., 720., 1080.])
FOLD_SEED = 271828
INITIAL_SDS = np.array([.05, .05, .05, .10, .10, .10])
LOWER_SD, UPPER_SD = 1e-4, 2.
MAX_EVALUATIONS, MAX_ELAPSED_SECONDS = 60, 1800.
GTOL, FTOL, REQUIRED_PROJECTED_GRADIENT = 1e-5, 1e-12, 1e-4
PARAMETER_ORDER = ("log_a_oil", "log_a_gas", "log_a_water",
                   "log_r_oil", "log_r_gas", "log_r_water")


@dataclass
class NoiseFoldData:
    fold: int
    query_indices: np.ndarray
    reports_native: np.ndarray
    cumulative_native: np.ndarray
    prior_probabilities: np.ndarray
    phase_scales_native: np.ndarray
    metadata: dict = field(default_factory=dict)
    coefficients_native: np.ndarray | None = None
    forest_probabilities: np.ndarray | None = None
    weight_receipt: dict = field(default_factory=dict)


@dataclass
class NoiseObjective:
    native_nll: float
    normalized_nll: float
    gradient: np.ndarray
    native_log_evidence: np.ndarray
    normalized_log_evidence: np.ndarray


@dataclass
class NoiseFitResult:
    noise_abs: np.ndarray
    noise_rel: np.ndarray
    status: str
    converged: bool
    diagnostics: dict
    final_native_log_evidence: np.ndarray
    fold_data: tuple[NoiseFoldData, ...] = ()

    def save_artifacts(self, output, prefix):
        """Write replayable arrays into an already-owned run, without overwrite.

        The enclosing runner owns the source/config/data manifest and completion
        seal. These files suffice to recompute the fitted likelihood and its
        gradient without loading a forest or fitting another teacher.
        """
        output = Path(output)
        if not output.is_dir() or not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
            raise ValueError("Require an existing run directory and a plain artifact prefix")
        files = {}
        for fold in self.fold_data:
            path = output/f"{prefix}_noise_fold{fold.fold}.npz"
            metadata = fold.metadata
            arrays = dict(query_indices=fold.query_indices, reports_native=fold.reports_native,
                cumulative_native=fold.cumulative_native, prior_probabilities=fold.prior_probabilities,
                phase_scales_native=fold.phase_scales_native, report_days=REPORT_DAYS,
                query_row_ids=np.asarray(metadata["query_row_ids"]), query_group_ids=np.asarray(metadata["query_group_ids"]),
                donor_row_ids=np.asarray(metadata["fit_row_ids"]), donor_group_ids=np.asarray(metadata["fit_group_ids"]),
                final_native_log_evidence=self.final_native_log_evidence[fold.query_indices],
                noise_abs_normalized=self.noise_abs, noise_rel=self.noise_rel,
                phase_order=np.asarray(["oil", "gas", "water"]), units=np.asarray(["BBL", "MCF", "BBL"]))
            if fold.coefficients_native is not None:
                arrays["coefficients_native"] = fold.coefficients_native
            if fold.forest_probabilities is not None:
                arrays["forest_probabilities"] = fold.forest_probabilities
            for key, value in fold.weight_receipt.items():
                if key in arrays:
                    raise ValueError("Weight receipt collides with an OOF evidence array")
                arrays[key] = np.asarray(value)
            if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
                raise ValueError("OOF evidence must not require object-array unpickling")
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            files[path.name] = sha256_file(path)
        diagnostics_path = output/f"{prefix}_noise_fit.json"
        write_json(diagnostics_path, self.diagnostics)
        files[diagnostics_path.name] = sha256_file(diagnostics_path)
        return files


class _FitLimit(RuntimeError):
    def __init__(self, reason, stage):
        super().__init__(f"{reason} at {stage}; no automatic expansion")
        self.reason, self.stage = reason, stage


def _check_deadline(deadline, stage):
    if deadline is not None and time.monotonic() >= deadline:
        raise _FitLimit("deadline", stage)


def _days(value):
    days = np.asarray(value, dtype=np.float64)
    if not np.array_equal(days, REPORT_DAYS):
        raise ValueError("Noise fitting requires the canonical eight report days")
    return days


def projected_gradient(log_sds, gradient, lower=math.log(LOWER_SD), upper=math.log(UPPER_SD)):
    """Box-constrained minimization KKT residual, without changing the gradient.

    At the lower bound a nonnegative derivative satisfies KKT; at the upper
    bound a nonpositive derivative does. Interior coordinates retain their
    signed derivatives. Bound comparisons use the actual optimizer values.
    """
    x, g = np.asarray(log_sds, dtype=float), np.asarray(gradient, dtype=float)
    if x.shape != (6,) or g.shape != (6,) or not np.isfinite(x).all() or not np.isfinite(g).all():
        raise ValueError("Require six finite log SDs and derivatives")
    return np.where(((x <= lower) & (g >= 0)) | ((x >= upper) & (g <= 0)), 0., g)


def _validate_folds(folds, days):
    if not folds:
        raise ValueError("Require nonempty OOF fold data")
    indices = []
    for fold in folds:
        y, q, p, scales = (np.asarray(value) for value in
            (fold.reports_native, fold.cumulative_native, fold.prior_probabilities, fold.phase_scales_native))
        ids = np.asarray(fold.query_indices)
        if (y.ndim != 3 or y.shape[1:] != (len(days), 3) or not len(y)
                or q.ndim != 3 or q.shape[1:] != y.shape[1:] or not len(q)
                or p.shape != (len(y), len(q)) or scales.shape != (3,)
                or ids.shape != (len(y),) or ids.dtype.kind not in "iu"
                or not all(np.isfinite(a).all() for a in (y, q, p, scales))
                or np.any(scales <= 0) or np.any(p <= 0)
                or not np.allclose(p.sum(1), 1., atol=5e-13, rtol=0)
                or np.any(q < 0) or np.any(np.diff(q, axis=1) < 0)
                or np.any(y[:, 0] != 0) or np.any(q[:, 0] != 0)):
            raise ValueError("Invalid complete fold reports, donor support, probabilities or scales")
        indices.extend(ids.tolist())
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("OOF query indices must partition the supplied TRAIN rows once")
    return len(indices)


@torch.no_grad()
def marginal_nll_and_gradient(log_sds, folds, report_days, *, device="cpu", batch_size=128, deadline=None):
    """Exact finite-mixture mean NLL and its analytic six-log-SD gradient.

    ``prior_probabilities`` already contains the 0.05 uniform component. All
    positive endpoints are observed for this training objective. Differences
    are the exact factorization of the accumulated endpoint Gaussian, not
    independent overlapping triangular increments. Returned evidences follow
    ``query_indices`` in the original TRAIN order.
    """
    _check_deadline(deadline, "objective entry")
    days = _days(report_days)
    x = np.asarray(log_sds, dtype=np.float64)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError("Require six finite log standard deviations")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
        raise ValueError("Streamed batch size must be an integer from one to128")
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("Noise objective supports CPU or CUDA float64 only")
    n = _validate_folds(folds, days)
    dtype = torch.float64
    theta = torch.as_tensor(x, dtype=dtype, device=target_device)
    sds2 = torch.exp(2*theta)
    if not bool(torch.isfinite(sds2).all()) or bool((sds2 <= 0).any()):
        raise FloatingPointError("Noise standard deviations exceeded finite positive support")
    dt = torch.as_tensor(np.diff(days)/360., dtype=dtype, device=target_device)
    evidence = np.empty(n, dtype=np.float64)
    normalized_evidence = np.empty(n, dtype=np.float64)
    gradient = np.zeros(6, dtype=np.float64)
    for fold in folds:
        _check_deadline(deadline, f"fold{fold.fold} objective setup")
        scales = np.asarray(fold.phase_scales_native, dtype=np.float64)
        normalized_q = np.asarray(fold.cumulative_native, dtype=np.float64)/scales
        q = torch.as_tensor(normalized_q, dtype=dtype, device=target_device)
        dq, qsum = q[:, 1:]-q[:, :-1], q[:, 1:]+q[:, :-1]
        absolute = sds2[:3][None, None, :]*dt[None, :, None]
        relative = sds2[3:][None, None, :]*dq*qsum
        variance = absolute+relative
        if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
            raise FloatingPointError("Accumulated innovation variance is not finite positive")
        jacobian = (len(days)-1)*float(np.log(scales).sum())
        for start in range(0, len(fold.reports_native), batch_size):
            _check_deadline(deadline, f"fold{fold.fold} batch{start} entry")
            stop = min(start+batch_size, len(fold.reports_native))
            normalized_y = np.asarray(fold.reports_native[start:stop], dtype=np.float64)/scales
            dy = torch.as_tensor(np.diff(normalized_y, axis=1), dtype=dtype, device=target_device)
            log_likelihood = torch.zeros((stop-start, len(q)), dtype=dtype, device=target_device)
            absolute_gradients, relative_gradients = [], []
            for phase in range(3):
                squared_standardized = (dy[:, None, :, phase]-dq[None, :, :, phase]).square()/variance[None, :, :, phase]
                log_likelihood -= .5*(squared_standardized+torch.log(variance[None, :, :, phase])+math.log(2*math.pi)).sum(-1)
                score = squared_standardized-1.
                absolute_gradients.append((score*(absolute[:, :, phase]/variance[:, :, phase])[None]).sum(-1))
                relative_gradients.append((score*(relative[:, :, phase]/variance[:, :, phase])[None]).sum(-1))
            atom_gradients = torch.stack(absolute_gradients+relative_gradients, dim=-1)
            log_prior = torch.log(torch.as_tensor(fold.prior_probabilities[start:stop], dtype=dtype, device=target_device))
            # Center likelihood before adding log prior to retain normalized
            # probabilities even when every atom has a very negative density.
            offset = log_likelihood.max(1).values
            centered = log_prior+(log_likelihood-offset[:, None])
            log_normalizer = torch.logsumexp(centered, dim=1)
            posterior = torch.exp(centered-log_normalizer[:, None])
            batch_gradient = -(posterior[:, :, None]*atom_gradients).sum((0, 1))/n
            normalized_logz = offset+log_normalizer
            if (not bool(torch.isfinite(normalized_logz).all()) or not bool(torch.isfinite(batch_gradient).all())
                    or not bool(torch.all(torch.abs(posterior.sum(1)-1.) <= 5e-13))):
                raise FloatingPointError("Finite-mixture objective or gradient exceeded numerical support")
            ids = np.asarray(fold.query_indices)[start:stop]
            normalized_evidence[ids] = normalized_logz.cpu().numpy()
            evidence[ids] = normalized_evidence[ids]-jacobian
            gradient += batch_gradient.cpu().numpy()
            _check_deadline(deadline, f"fold{fold.fold} batch{start} completion")
    result = NoiseObjective(float(-np.mean(evidence)), float(-np.mean(normalized_evidence)),
                            gradient, evidence, normalized_evidence)
    if not np.isfinite(result.native_nll) or not np.isfinite(result.normalized_nll):
        raise FloatingPointError("Mean likelihood exceeded finite support")
    _check_deadline(deadline, "objective completion")
    return result


def _point(log_sds, objective=None):
    x = np.asarray(log_sds, dtype=np.float64)
    result = {"log_standard_deviations": x.tolist(), "standard_deviations": np.exp(x).tolist(),
              "lower_bound_indices": np.flatnonzero(x <= math.log(LOWER_SD)).tolist(),
              "upper_bound_indices": np.flatnonzero(x >= math.log(UPPER_SD)).tolist()}
    if objective is not None:
        pg = projected_gradient(x, objective.gradient)
        result.update(native_nll=objective.native_nll, normalized_nll=objective.normalized_nll,
                      gradient=objective.gradient.tolist(), projected_gradient=pg.tolist(),
                      projected_gradient_infinity=float(np.max(np.abs(pg))))
    return result


def fit_oof_forest_noise(records, reports, report_days, *, seed, device="cpu", n_jobs=2,
                         bank_factory=None, max_evaluations=MAX_EVALUATIONS,
                         max_elapsed_seconds=MAX_ELAPSED_SECONDS, batch_size=128):
    """Fit six shared SDs using five fixed whole-group TRAIN holdout folds.

    A returned unresolved result retains completed fold/call diagnostics. The
    caller must refuse validation prediction unless ``converged`` is true.
    Smaller budgets are accepted for tests; larger budgets, restarts and noise
    initializations are not part of this frozen procedure.
    """
    started = time.monotonic()
    days = _days(report_days)
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int)
            or not 1 <= max_evaluations <= MAX_EVALUATIONS
            or isinstance(max_elapsed_seconds, bool) or not np.isfinite(max_elapsed_seconds)
            or not 0 < max_elapsed_seconds <= MAX_ELAPSED_SECONDS
            or n_jobs != 2 or isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**32):
        raise ValueError("Invalid frozen noise-fitting seed, worker count or bounded budget")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
        raise ValueError("Streamed batch size must be an integer from one to128")
    requested_device = torch.device(device)
    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError("Noise fitting supports CPU or CUDA float64 only")
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA noise fitting requested without an available CUDA device")
    assignments = assign_group_folds(records, fold_seed=FOLD_SEED)
    if set(assignments.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("All five fixed whole-group folds must be populated; no outcome-dependent redraw")
    y = np.asarray(reports, dtype=np.float64)
    if y.shape != (len(records), len(days), 3) or not np.isfinite(y).all() or np.any(y[:, 0] != 0):
        raise ValueError("Require complete finite TRAIN reports with deterministic day zero")
    rows = [record["row_id"] for record in records]
    groups = [record["group_id"] for record in records]
    deadline = started+float(max_elapsed_seconds)
    initial = np.log(INITIAL_SDS)
    final_x = initial.copy()
    folds, trace, cached = [], [], {}
    final_objective = None
    status, optimizer, failure = "unresolved_optimizer", {"success": False, "message": "not started"}, None
    diagnostics: dict[str, Any] = {
        "method": "five whole-group TRAIN folds; exact joint finite-mixture native marginal NLL",
        "fit_split": "train", "training_seed": int(seed), "fold_seed": FOLD_SEED,
        "fold_namespace": FOLD_NAMESPACE, "fold_count": FOLD_COUNT,
        "training_row_ids": rows, "training_group_ids": groups,
        "training_row_sha256": canonical_hash(rows), "fold_assignments": assignments.tolist(),
        "parameter_order": list(PARAMETER_ORDER), "bounds_standard_deviation": [LOWER_SD, UPPER_SD],
        "initial": _point(initial), "objective_calls": trace, "folds": [],
        "optimizer_options": {"method": "L-BFGS-B", "gtol": GTOL, "ftol": FTOL,
                              "maxiter": 60, "maxfun": max_evaluations},
        "maximum_objective_evaluations": max_evaluations,
        "maximum_elapsed_seconds": float(max_elapsed_seconds),
        "deadline_kind": "cooperative; external whole-process supervisor required",
        "device": str(device), "dtype": "float64", "batch_size": batch_size,
        "required_projected_gradient_infinity": REQUIRED_PROJECTED_GRADIENT,
        "native_density_jacobian": "subtract seven times sum(log fold TRAIN phase scales) from each normalized log evidence",
        "phase_order": ["oil", "gas", "water"], "units": ["BBL", "MCF", "BBL"],
        "noise_identification": "predictive empirical dispersion; donor estimation, basis and report discrepancies remain confounded",
    }
    try:
        if bank_factory is None:
            from .finite_forest_rates import fit_forest_rate_prior
            bank_factory = fit_forest_rate_prior
        for fold in range(FOLD_COUNT):
            diagnostics["current_stage"] = f"fold{fold} bank"
            _check_deadline(deadline, f"fold{fold} bank entry")
            tick = time.monotonic()
            fit_indices, query_indices = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
            print(f"finite noise: building fold {fold+1}/{FOLD_COUNT}, {len(fit_indices)} donors, {len(query_indices)} queries", flush=True)
            fitting = [records[i] for i in fit_indices]
            querying = [{key: records[i][key] for key in ("row_id", "group_id", "split", "covariates_raw")} for i in query_indices]
            if set(groups[i] for i in fit_indices) & set(groups[i] for i in query_indices):
                raise ValueError("Fold training and query leakage groups overlap")
            bank = bank_factory(fitting, y[fit_indices], days, seed=int(seed), n_jobs=n_jobs)
            _check_deadline(deadline, f"fold{fold} bank fit completion")
            receipt = {}
            if "return_receipt" in inspect.signature(bank.exact_weights).parameters:
                prior, receipt = bank.exact_weights(querying, return_receipt=True)
                receipt = dict(receipt)
            else:
                prior = bank.exact_weights(querying)
            prior = np.asarray(prior, dtype=np.float64)
            _check_deadline(deadline, f"fold{fold} bank query completion")
            q = np.asarray(bank.cumulative_native, dtype=np.float64)
            scale = np.asarray(bank.phase_scales_native, dtype=np.float64)
            if len(q) != len(fit_indices):
                raise ValueError("Each complement TRAIN row must remain one declared joint donor")
            if prior.shape != (len(query_indices), len(fit_indices)) or np.any(prior < .05/len(fit_indices)):
                raise ValueError("Final donor probabilities must include the fixed0.05 uniform background")
            forest_probabilities = receipt.pop("forest_weights", None)
            if forest_probabilities is not None:
                forest_probabilities = np.asarray(forest_probabilities, dtype=np.float64)
                if (forest_probabilities.shape != prior.shape or not np.isfinite(forest_probabilities).all()
                        or np.any(forest_probabilities < 0)
                        or not np.allclose(forest_probabilities.sum(1), 1., atol=5e-13, rtol=0)
                        or not np.array_equal(.95*forest_probabilities+.05/len(q), prior)):
                    raise ValueError("OOF forest evidence and final prior probabilities disagree")
            metadata = {"fold": fold, "fit_row_ids": [rows[i] for i in fit_indices],
                "fit_group_ids": [groups[i] for i in fit_indices],
                "query_row_ids": [rows[i] for i in query_indices], "query_group_ids": [groups[i] for i in query_indices],
                "fit_row_sha256": canonical_hash([rows[i] for i in fit_indices]),
                "query_row_sha256": canonical_hash([rows[i] for i in query_indices]),
                "fit_rows": len(fit_indices), "query_rows": len(query_indices),
                "phase_scales_native": scale.tolist(), "bank": bank.metadata(),
                "whole_group_exclusion_verified": True, "elapsed_seconds": time.monotonic()-tick}
            coefficients = np.asarray(bank.coefficients_native, dtype=np.float64) if hasattr(bank, "coefficients_native") else None
            data = NoiseFoldData(fold, query_indices, y[query_indices].copy(), q.copy(), prior.copy(), scale.copy(), metadata,
                None if coefficients is None else coefficients.copy(),
                None if forest_probabilities is None else forest_probabilities.copy(),
                {key: np.asarray(value).copy() for key, value in receipt.items()})
            folds.append(data)
            diagnostics["folds"].append(metadata)
            del bank
            _check_deadline(deadline, f"fold{fold} bank retention completion")
            print(f"finite noise: fold {fold+1}/{FOLD_COUNT} ready in {metadata['elapsed_seconds']:.2f}s", flush=True)
        _validate_folds(folds, days)

        def evaluate(x):
            nonlocal final_x, final_objective
            diagnostics["current_stage"] = "optimizer objective"
            _check_deadline(deadline, "optimizer objective request")
            if len(trace) >= max_evaluations:
                raise _FitLimit("evaluation_limit", "optimizer objective request")
            entry = {"evaluation": len(trace)+1, **_point(x), "status": "started",
                     "elapsed_seconds": time.monotonic()-started}
            trace.append(entry)
            try:
                value = marginal_nll_and_gradient(x, folds, days, device=device, batch_size=batch_size, deadline=deadline)
                _check_deadline(deadline, "optimizer objective completion")
            except Exception as error:
                entry.update(status="failed", error_type=type(error).__name__, error=str(error),
                             elapsed_seconds=time.monotonic()-started)
                raise
            entry.update(_point(x, value), status="completed", elapsed_seconds=time.monotonic()-started)
            if np.array_equal(np.asarray(x), initial):
                diagnostics["initial"] = _point(initial, value)
            final_x, final_objective = np.array(x, copy=True), value
            cached[np.asarray(x, dtype=np.float64).tobytes()] = value
            print(f"finite noise: evaluation {len(trace)}/{max_evaluations}, native NLL {value.native_nll:.8f}, projected gradient {entry['projected_gradient_infinity']:.6g}", flush=True)
            return value.native_nll, value.gradient

        optimized = minimize(evaluate, initial, jac=True, method="L-BFGS-B",
            bounds=[(math.log(LOWER_SD), math.log(UPPER_SD))]*6,
            options={"gtol": GTOL, "ftol": FTOL, "maxiter": 60, "maxfun": max_evaluations})
        _check_deadline(deadline, "optimizer completion")
        optimizer = {"success": bool(optimized.success), "status": int(optimized.status),
            "message": str(optimized.message), "nit": int(optimized.nit), "nfev": int(optimized.nfev)}
        candidate = np.asarray(optimized.x, dtype=np.float64)
        value = cached.get(candidate.tobytes())
        if (candidate.shape != (6,) or not np.isfinite(candidate).all()
                or np.any(candidate < math.log(LOWER_SD)) or np.any(candidate > math.log(UPPER_SD))
                or value is None):
            raise ValueError("Optimizer final point is invalid or lacks its own retained objective evaluation")
        final_x, final_objective = candidate, value
        kkt = float(np.max(np.abs(projected_gradient(candidate, value.gradient))))
        status = "converged" if optimized.success and kkt <= REQUIRED_PROJECTED_GRADIENT else (
            "unresolved_optimizer" if not optimized.success else "unresolved_kkt")
    except _FitLimit as error:
        status = f"unresolved_{error.reason}"
        failure = {"type": type(error).__name__, "message": str(error), "stage": error.stage}
    except Exception as error:
        status = "unresolved_numerical_or_bank"
        failure = {"type": type(error).__name__, "message": str(error), "stage": diagnostics.get("current_stage")}
    diagnostics.update(status=status, converged=status == "converged", optimizer=optimizer,
        failure=failure, final=_point(final_x, final_objective),
        completed_objective_evaluations=sum(entry["status"] == "completed" for entry in trace),
        attempted_objective_evaluations=len(trace), elapsed_seconds=time.monotonic()-started,
        validation_authorized=status == "converged")
    evidence = (final_objective.native_log_evidence.copy() if final_objective is not None
                else np.full(len(records), np.nan))
    sds = np.exp(final_x)
    return NoiseFitResult(sds[:3], sds[3:], status, status == "converged", diagnostics, evidence, tuple(folds))
