"""Frozen scoring conventions for observed conditional production increments.

Array boundaries use targets ``[well, cell, phase]`` and equally weighted joint
samples ``[well, sample, cell, phase]``. Phase order is oil, gas, water; native
units are BBL, MCF, BBL. A cell identifies a fixed origin/endpoint combination.
Missing targets are excluded, never replaced by zero. Predictions must be finite
at every observed target. Quantiles use NumPy's ``method='linear'`` throughout.

WIS follows Bracher et al. (2021), https://doi.org/10.1371/journal.pcbi.1008618.
The energy score scores the empirical ensemble (the V-statistic convention),
so comparisons must match ensemble size. The variogram score follows Scheuerer
and Hamill (2015), https://doi.org/10.1175/MWR-D-14-00269.1, averaged over unordered
observed coordinate pairs with equal weights and power 0.5. Neither joint score
alone establishes simultaneous or derived-quantity calibration.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

PHASE_NAMES = ("oil", "gas", "water")
DEFAULT_ALPHAS = (0.2, 0.05)


def _reject_numerical_failure(function):
    """A failed forecast computation must fail scoring, never improve its mean."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except FloatingPointError as error:
            raise ValueError(f"Numerical failure while computing {function.__name__}: {error}") from error
    return checked


def _arrays(targets, samples):
    targets = np.asarray(targets, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if targets.ndim != 3 or samples.ndim != 4:
        raise ValueError("Expected targets [well,cell,phase], samples [well,sample,cell,phase]")
    if (samples.shape[0], *samples.shape[2:]) != targets.shape:
        raise ValueError("Target and sample well/cell/phase axes must agree")
    if not samples.shape[1] or not all(targets.shape):
        raise ValueError("Well, sample, cell, and phase axes must be nonempty")
    observed = np.isfinite(targets)
    if np.any(~np.isfinite(samples) & observed[:, None]):
        raise ValueError("Nonfinite forecast sample at an observed target")
    # Sanitize only coordinates already excluded by the target mask. Invalid
    # finite-target predictions were rejected above, not silently discarded.
    return np.where(observed, targets, np.nan), np.where(observed[:, None], samples, np.nan), observed


def _scales(scales, shape):
    scales = np.asarray(scales, dtype=float)
    if scales.shape != shape or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("Scales must be finite, strictly positive [cell,phase]")
    return scales


def _mean(values, axis=None):
    """Finite-only mean with explicit NaN for empty sets, without warnings."""
    values = np.asarray(values, dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite derived scores cannot be excluded from aggregation")
    valid = np.isfinite(values)
    total = np.where(valid, values, 0).sum(axis=axis)
    count = valid.sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _macro(cell_phase_values):
    # Each phase has equal weight even if its number of available cells differs.
    return float(_mean(_mean(cell_phase_values, axis=0)))


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@_reject_numerical_failure
def fit_increment_scales(training_targets, phase_fallback=None, return_details=False):
    """Fit a scale for each cell/phase from TRAINING increments only.

    The primary scale is median absolute observed training increment, including
    reported zero. If it is missing or zero, use the median strictly positive
    absolute training increment pooled over cells of the SAME phase. If the
    entire phase lacks a positive training increment, use the explicit positive
    ``phase_fallback`` (default one native unit per phase). Never pool BBL and
    MCF. Freeze these scales before validation; this function cannot identify
    split membership and must only receive training targets.

    With ``return_details=True``, return ``(scales, JSON-ready provenance)`` with
    training counts and per-cell fallback reasons for the retained manifest.
    Under a unit conversion, convert explicit fallback values as well. Default
    one-unit fallbacks cannot be unit invariant for wholly degenerate phases.
    """
    targets = np.asarray(training_targets, dtype=float)
    if targets.ndim != 3 or not all(targets.shape):
        raise ValueError("Training targets must be nonempty [well,cell,phase]")
    phases = targets.shape[2]
    fallback = np.ones(phases) if phase_fallback is None else np.asarray(phase_fallback, dtype=float)
    if fallback.shape != (phases,) or not np.all(np.isfinite(fallback) & (fallback > 0)):
        raise ValueError("phase_fallback must be finite, positive, and have one value per phase")
    scales = np.empty(targets.shape[1:])
    reasons = np.full(targets.shape[1:], "cell_median_absolute", dtype=object)
    positive_counts = np.zeros(phases, dtype=int)
    for phase in range(phases):
        absolute = np.abs(targets[:, :, phase])
        positive = absolute[np.isfinite(absolute) & (absolute > 0)]
        positive_counts[phase] = positive.size
        phase_scale = float(np.median(positive)) if positive.size else fallback[phase]
        for cell in range(targets.shape[1]):
            values = absolute[:, cell]
            values = values[np.isfinite(values)]
            median = float(np.median(values)) if values.size else np.nan
            scales[cell, phase] = median if np.isfinite(median) and median > 0 else phase_scale
            if not np.isfinite(median) or median <= 0:
                reasons[cell, phase] = "phase_pooled_positive_median" if positive.size else "explicit_native_unit_fallback"
    if return_details:
        return scales, _json_ready({"scales": scales, "training_observation_count": np.isfinite(targets).sum(axis=0),
                                    "phase_positive_training_count": positive_counts,
                                    "phase_fallback": fallback, "scale_source": reasons})
    return scales


@_reject_numerical_failure
def interval_score(targets, lower, upper, alpha):
    """Central interval score: width plus miscoverage penalties, in native units."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    target, lower, upper = np.broadcast_arrays(
        np.asarray(targets, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    )
    observed = np.isfinite(target)
    if np.any(observed & (~np.isfinite(lower) | ~np.isfinite(upper) | (lower > upper))):
        raise ValueError("Observed targets require finite ordered interval bounds")
    target, lower, upper = (np.where(observed, value, 0) for value in (target, lower, upper))
    score = upper - lower + (2 / alpha) * (np.maximum(lower - target, 0) + np.maximum(target - upper, 0))
    return np.where(observed, score, np.nan)


@_reject_numerical_failure
def weighted_interval_score(targets, samples, alphas=DEFAULT_ALPHAS):
    """Return native WIS [well,cell,phase], with central 80%/95% intervals.

    For K levels the normalized formula is
    ``(0.5*abs(y-median) + sum(alpha/2 * IS_alpha)) / (K+0.5)``.
    """
    targets, samples, observed = _arrays(targets, samples)
    alphas = tuple(alphas)
    if not alphas or len(set(alphas)) != len(alphas) or any(not 0 < a < 1 for a in alphas):
        raise ValueError("alphas must be distinct and strictly between zero and one")
    # Replace invalid-target samples only for calculation; they remain unscored.
    clean = np.where(observed[:, None], samples, 0)
    result = 0.5 * np.abs(targets - np.quantile(clean, 0.5, axis=1, method="linear"))
    for alpha in alphas:
        lower, upper = np.quantile(clean, [alpha / 2, 1 - alpha / 2], axis=1, method="linear")
        result += alpha / 2 * interval_score(targets, lower, upper, alpha)
    return np.where(observed, result / (len(alphas) + 0.5), np.nan)


@_reject_numerical_failure
def energy_score(targets, samples, scales):
    """Empirical-ensemble energy score for each scaled joint increment vector.

    All finite target cell/phase coordinates of each well are flattened jointly.
    Compute ``mean ||X-y|| - 0.5*mean_{s,t} ||X_s-X_t||`` INCLUDING diagonal
    pairs (V-statistic). Empty observed vectors return NaN. No division by the
    number of dimensions is made; compare like target masks and ensemble sizes.
    """
    targets, samples, observed = _arrays(targets, samples)
    scales = _scales(scales, targets.shape[1:])
    result = np.full(targets.shape[0], np.nan)
    for well in range(targets.shape[0]):
        valid = observed[well].ravel()
        if not valid.any():
            continue
        y = (targets[well] / scales).ravel()[valid]
        x = (samples[well] / scales).reshape(samples.shape[1], -1)[:, valid]
        # Center first to avoid cancellation for forecasts with a large common level.
        x = x - y
        first = np.linalg.norm(x, axis=1).mean()
        pair_sum = 0.0
        for start in range(0, len(x), 32):
            pair_sum += np.linalg.norm(x[start:start + 32, None, :] - x[None, :, :], axis=2).sum()
        result[well] = first - pair_sum / (2 * len(x) ** 2)
    return result


@_reject_numerical_failure
def variogram_score(targets, samples, scales, power=0.5):
    """Mean unordered-pair variogram score per well, on scaled coordinates.

    ``mean_{i<j} (|y_i-y_j|**p - mean_s |X_si-X_sj|**p)**2``.
    There are no diagonal pairs; fewer than two observed coordinates yield NaN.
    Averaging (rather than summing) fixes the weight convention. This is proper
    but not strictly proper; common shifts can escape variogram detection.
    """
    targets, samples, observed = _arrays(targets, samples)
    scales = _scales(scales, targets.shape[1:])
    if not np.isfinite(power) or not 0 < power < 2:
        raise ValueError("power must lie strictly between zero and two")
    result = np.full(targets.shape[0], np.nan)
    for well in range(targets.shape[0]):
        valid = observed[well].ravel()
        if valid.sum() < 2:
            continue
        y = (targets[well] / scales).ravel()[valid]
        x = (samples[well] / scales).reshape(samples.shape[1], -1)[:, valid]
        left, right = np.triu_indices(len(y), 1)
        observed_difference = np.abs(y[left] - y[right]) ** power
        expected_difference = (np.abs(x[:, left] - x[:, right]) ** power).mean(axis=0)
        result[well] = np.square(observed_difference - expected_difference).mean()
    return result


def safe_ratio(numerator, denominator, minimum_denominator=0.0):
    """Return samplewise ratio, NaN unless finite with denominator > threshold.

    A threshold is an exclusion rule, never an additive epsilon. Ratio units are
    numerator units divided by denominator units. No marginal means or quantiles
    are divided, and invalid ratios must be counted rather than silently filled.
    """
    if not np.isfinite(minimum_denominator) or minimum_denominator < 0:
        raise ValueError("minimum_denominator must be finite and nonnegative")
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > minimum_denominator)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=valid)
    return np.where(np.isfinite(result), result, np.nan)


def sample_derived_quantities(phase_volumes):
    """GOR (MCF/BBL) and water cut (fraction) from the SAME latent volume draws.

    The final input axis is oil/gas/water, in BBL/MCF/BBL. Negative or missing
    involved phase volumes invalidate the ratio. Zero oil invalidates GOR;
    zero combined oil/water invalidates water cut. Reported noisy increments
    must not be presented as latent physical ratios without that distinction.
    """
    volumes = np.asarray(phase_volumes, dtype=float)
    if volumes.ndim < 1 or volumes.shape[-1] != 3:
        raise ValueError("Last volume axis must be oil, gas, water")
    oil, gas, water = np.moveaxis(volumes, -1, 0)
    gor = safe_ratio(np.where(gas >= 0, gas, np.nan), np.where(oil >= 0, oil, np.nan))
    water_cut = safe_ratio(
        np.where(water >= 0, water, np.nan), np.where((oil >= 0) & (water >= 0), oil + water, np.nan)
    )
    return {"gor_mcf_per_bbl": gor, "water_cut": water_cut,
            "gor_valid": np.isfinite(gor), "water_cut_valid": np.isfinite(water_cut)}


@_reject_numerical_failure
def evaluate_forecasts(targets, samples, scales, phase_names=PHASE_NAMES, include_joint_scores=True,
                       joint_cell_groups=None):
    """JSON-ready primary, point, marginal, and joint forecast metrics.

    Compute per-cell phase means over observed wells, then average cells within
    phase and phases equally. R2 uses the SAMPLE MEAN point forecast, separately
    per cell/phase, and is undefined for fewer than two targets or zero target
    variance. Native MAE uses the SAMPLE MEDIAN; native RMSE uses the SAMPLE
    MEAN. Both give cells equal weight (RMSE is the square root of mean cell
    MSE). Undefined cells are explicit JSON nulls;
    counts and numbers of active cells make omissions visible. No values from
    this function may be used to refit training scales. ``joint_cell_groups``
    partitions cells by conditioning origin. Sample rows must be coherent
    WITHIN each group. Joint scores are computed separately for each group and
    equally averaged; independently conditioned origins never form one joint
    vector. The default puts all cells in one group, appropriate for one origin.
    """
    targets, samples, observed = _arrays(targets, samples)
    scales = _scales(scales, targets.shape[1:])
    if len(phase_names) != targets.shape[2] or len(set(phase_names)) != len(phase_names):
        raise ValueError("phase_names must uniquely label the phase axis")
    wis = weighted_interval_score(targets, samples)
    scaled_wis = wis / scales
    cell_wis = _mean(scaled_wis, axis=0)
    point = np.where(observed, np.where(observed[:, None], samples, 0).mean(axis=1), np.nan)
    error = point - targets
    median = np.quantile(np.where(observed[:, None], samples, 0), 0.5, axis=1, method="linear")
    cell_mae = _mean(np.where(observed, np.abs(median - targets), np.nan), axis=0)
    cell_mse = _mean(np.square(error), axis=0)
    count = observed.sum(axis=0)
    centered = targets - _mean(targets, axis=0)
    total_ss = np.where(observed, centered ** 2, 0).sum(axis=0)
    residual_ss = np.where(observed, error ** 2, 0).sum(axis=0)
    r2 = np.full(targets.shape[1:], np.nan)
    valid_r2 = (count >= 2) & (total_ss > 0)
    r2[valid_r2] = 1 - residual_ss[valid_r2] / total_ss[valid_r2]
    result = {"macro_scaled_wis": _macro(cell_wis), "macro_increment_r2": _macro(r2),
              "wells": targets.shape[0], "samples": samples.shape[1],
              "observed_target_count": int(observed.sum()), "phase_order": list(phase_names),
              "point_functionals": {"r2": "sample_mean", "rmse": "sample_mean", "mae": "sample_median"},
              "scales": scales, "per_phase": {},
              "per_cell_phase": {"count": count, "scaled_wis": cell_wis,
                                 "r2": r2, "mae": cell_mae, "rmse": np.sqrt(cell_mse)}}
    for phase, name in enumerate(phase_names):
        result["per_phase"][name] = {"scaled_wis": _mean(cell_wis[:, phase]), "r2": _mean(r2[:, phase]),
                                    "mae": _mean(cell_mae[:, phase]), "rmse": np.sqrt(_mean(cell_mse[:, phase])),
                                    "observed_target_count": int(count[:, phase].sum()),
                                    "scored_cells": int((count[:, phase] > 0).sum()),
                                    "r2_cells": int(valid_r2[:, phase].sum()),
                                    "r2_undefined_fewer_than_two_targets": int((count[:, phase] < 2).sum()),
                                    "r2_undefined_zero_target_variance": int(((count[:, phase] >= 2) & (total_ss[:, phase] == 0)).sum())}
    clean = np.where(observed[:, None], samples, 0)
    for alpha in DEFAULT_ALPHAS:
        label = str(round(100 * (1 - alpha)))
        lower, upper = np.quantile(clean, [alpha / 2, 1 - alpha / 2], axis=1, method="linear")
        coverage = _mean(np.where(observed, (targets >= lower) & (targets <= upper), np.nan), axis=0)
        width = _mean(np.where(observed, upper - lower, np.nan), axis=0)
        result[f"macro_coverage_{label}"] = _macro(coverage)
        result[f"macro_scaled_width_{label}"] = _macro(width / scales)
        result["per_cell_phase"][f"coverage_{label}"] = coverage
        result["per_cell_phase"][f"width_{label}"] = width
        for phase, name in enumerate(phase_names):
            result["per_phase"][name][f"coverage_{label}"] = _mean(coverage[:, phase])
            result["per_phase"][name][f"width_{label}"] = _mean(width[:, phase])
    if include_joint_scores:
        cell_groups = [list(range(targets.shape[1]))] if joint_cell_groups is None else joint_cell_groups
        flat = [cell for group in cell_groups for cell in group]
        if (any(not isinstance(cell, (int, np.integer)) for cell in flat)
                or sorted(flat) != list(range(targets.shape[1])) or any(not len(group) for group in cell_groups)):
            raise ValueError("joint_cell_groups must partition the cell axis exactly once")
        group_scores = []
        for cells in cell_groups:
            cells = list(cells)
            energy = energy_score(targets[:, cells], samples[:, :, cells], scales[cells])
            variogram = variogram_score(targets[:, cells], samples[:, :, cells], scales[cells])
            group_scores.append({"cells": cells, "mean_energy_score": _mean(energy),
                                 "mean_variogram_score_p05": _mean(variogram),
                                 "energy_wells": int(np.isfinite(energy).sum()),
                                 "variogram_wells": int(np.isfinite(variogram).sum()),
                                 "observed_coordinates_per_well": observed[:, cells].reshape(len(targets), -1).sum(axis=1)})
        result["joint"] = {"mean_energy_score": _mean([row["mean_energy_score"] for row in group_scores]),
                           "mean_variogram_score_p05": _mean([row["mean_variogram_score_p05"] for row in group_scores]),
                           "per_cell_group": group_scores, "aggregation": "equal_conditioning_group_means",
                           "energy_estimator": "empirical_ensemble_v_statistic",
                           "variogram_pair_weight": "equal_unordered_pair_mean"}
    return _json_ready(result)
