"""Paired development comparisons; whole groups are the resampling units.

This module never selects training scales or treats ensemble draws as training
seeds. Bootstrap uncertainty is conditional on the two supplied fitted models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .metrics import PHASE_NAMES, _arrays, _scales, _mean, _macro, _json_ready, _reject_numerical_failure, weighted_interval_score


HISTORY_MARKERS = frozenset({
    'scenario', 'mask_seed', 'history_mask', 'history_mask_by_origin', 'origins_days', 'origin_observed',
})


def _history_metadata(result, wells, cells, *, require_days=True):
    """Validate explicit conditioning metadata, without inferring missing fields."""
    if not HISTORY_MARKERS.intersection(result):
        return False
    required = {'scenario', 'mask_seed', 'origins_days', 'cells_origin_endpoint_days'}
    if require_days:
        required.add('report_days')
    missing = required - result.keys()
    if missing or not {'history_mask', 'history_mask_by_origin'}.intersection(result):
        raise ValueError(f'Incomplete history-study metadata: missing {sorted(missing)} or actual history mask')
    scenario, seed, origins = result['scenario'], result['mask_seed'], result['origins_days']
    if scenario.ndim != 0 or scenario.dtype.kind not in 'US' or not scenario.item():
        raise ValueError('scenario must be a nonempty scalar text value')
    if seed.ndim != 0 or seed.dtype.kind not in 'iu':
        raise ValueError('mask_seed must be a scalar integer, not a boolean or float')
    if (origins.ndim != 1 or not len(origins) or origins.dtype.kind not in 'iuf'
            or not np.isfinite(origins).all() or np.any(origins < 0) or len(np.unique(origins)) != len(origins)):
        raise ValueError('origins_days must be distinct finite nonnegative days')
    if 'history_mask' in result:
        mask = result['history_mask']
        if len(origins) != 1 or mask.dtype != bool or mask.ndim != 3 or mask.shape[0] != wells or mask.shape[-1] != 3:
            raise ValueError('history_mask alias requires [well,time,3] and a single origin')
        if 'history_mask_by_origin' in result and not np.array_equal(result['history_mask_by_origin'], mask[None]):
            raise ValueError('Conflicting history_mask aliases')
        result['history_mask_by_origin'] = mask[None]
    mask = result['history_mask_by_origin']
    if (mask.dtype != bool or mask.ndim != 4 or mask.shape[:2] != (len(origins), wells)
            or not mask.shape[2] or mask.shape[-1] != 3):
        raise ValueError('history_mask_by_origin must be boolean [origin,well,time,3]')
    cell_days = result['cells_origin_endpoint_days']
    if (cell_days.shape != (cells, 2) or cell_days.dtype.kind not in 'iuf'
            or not np.isfinite(cell_days).all() or np.any(cell_days[:, 1] <= cell_days[:, 0])
            or len(np.unique(cell_days, axis=0)) != cells or set(cell_days[:, 0]) != set(origins)):
        raise ValueError('cells_origin_endpoint_days must match target cells and explicit conditioning origins')
    if 'origin_observed' in result:
        visible = result['origin_observed']
        if visible.dtype != bool or visible.shape != (wells, len(origins), 3):
            raise ValueError('origin_observed must be boolean [well,origin,3]')
    for key in ('prediction_anchors', 'sampled_origin_reports'):
        if key in result and result[key].shape != (wells, result['samples'].shape[1], len(origins), 3):
            raise ValueError(f'{key} must have [well,sample,origin,3] axes')
    if 'report_days' in result:
        days = result['report_days']
        if (days.shape != (mask.shape[2],) or days.dtype.kind not in 'iuf' or not np.isfinite(days).all()
                or days[0] != 0 or np.any(np.diff(days) <= 0) or not np.isin(cell_days, days).all()):
            raise ValueError('report_days must identify every history-mask timestamp and forecast cell')
        for index, origin in enumerate(origins):
            if mask[index, :, days > origin].any():
                raise ValueError('History mask exposes future reports after its conditioning origin')
            if 'origin_observed' in result:
                at_origin = int(np.flatnonzero(days == origin)[0])
                if not np.array_equal(result['origin_observed'][:, index], mask[index, :, at_origin]):
                    raise ValueError('origin_observed disagrees with actual history mask at the origin')
    return True


def load_predictions(path):
    """Preserve conditioning arrays; verify older studies' shared time grids.

    Original main-pilot archives without history metadata remain unchanged.
    The first mask-study writer put report_days only in history_masks.npz;
    recover it only after its completion-manifest hash and actual masks match.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if HISTORY_MARKERS.intersection(result) and 'report_days' not in result:
        _history_metadata(result, len(result['targets']), result['targets'].shape[1], require_days=False)
        shared_path = path.parent / 'history_masks.npz'
        completion_path = path.parent / 'completion.json'
        if not shared_path.is_file() or not completion_path.is_file():
            raise ValueError('Missing report_days requires a verified parent history_masks.npz archive')
        completion = json.loads(completion_path.read_text())
        digest = hashlib.sha256(shared_path.read_bytes()).hexdigest()
        if completion.get('status') != 'completed' or completion.get('artifacts', {}).get(shared_path.name) != digest:
            raise ValueError('Parent history_masks.npz completion hash mismatch')
        with np.load(shared_path, allow_pickle=False) as shared:
            required = {'report_days', 'origins_days', 'row_ids', 'group_ids', 'scenario_names', 'history_mask_by_scenario_origin'}
            if not required.issubset(shared.files):
                raise ValueError('Parent history mask archive is incomplete')
            identifiers = shared['row_ids'].tolist()
            lookup = {row_id: index for index, row_id in enumerate(identifiers)}
            if len(lookup) != len(identifiers) or not set(result['row_ids']).issubset(lookup):
                raise ValueError('Prediction IDs do not match the parent history mask archive')
            order = [lookup[row_id] for row_id in result['row_ids']]
            at_scenario = np.flatnonzero(shared['scenario_names'] == result['scenario'])
            if (len(at_scenario) != 1 or not np.array_equal(shared['origins_days'], result['origins_days'])
                    or not np.array_equal(shared['group_ids'][order], result['group_ids'])
                    or not np.array_equal(shared['history_mask_by_scenario_origin'][at_scenario[0]][:, order], result['history_mask_by_origin'])):
                raise ValueError('Prediction conditioning differs from the parent history mask archive')
            result['report_days'] = shared['report_days']
        result['_report_days_source_path'] = np.asarray(str(shared_path.resolve()))
        result['_report_days_source_sha256'] = np.asarray(digest)
    return result


def align_predictions(reference, candidate, *, reorder_by_ids=False):
    """Require identical conditioning; deliberate cross-scenario gates are unsupported.

    Reordering is opt-in and uses unique row IDs. Legacy pairs without explicit
    history metadata remain compatible, but cannot be mixed with history studies.
    """
    checked = []
    for prediction in (reference, candidate):
        required = {"targets", "samples", "scales", "row_ids", "group_ids"}
        if not required.issubset(prediction):
            raise ValueError(f"Missing prediction fields: {sorted(required - prediction.keys())}")
        result = {key: np.asarray(value) for key, value in prediction.items()}
        target, sample, observed = _arrays(result['targets'], result['samples'])
        if target.shape[-1] != 3 or np.isinf(result['targets']).any():
            raise ValueError("Targets require three phases and finite values or NaN")
        if not observed.any(axis=(0, 1)).all():
            raise ValueError("Each phase requires at least one observed target")
        _scales(result['scales'], target.shape[1:])
        for key in ('row_ids', 'group_ids'):
            ids = result[key]
            if ids.shape != (len(target),) or ids.dtype.kind not in 'US' or np.any(ids == ''):
                raise ValueError(f"{key} must be nonempty text IDs, one per well")
        if len(np.unique(result['row_ids'])) != len(target):
            raise ValueError("row_ids must be unique")
        if 'mask' in result and (result['mask'].dtype != bool or not np.array_equal(result['mask'], observed)):
            raise ValueError("Explicit mask must equal the finite-target mask")
        _history_metadata(result, len(target), target.shape[1])
        checked.append(result)
    reference, candidate = checked
    if reorder_by_ids:
        lookup = {key: i for i, key in enumerate(candidate['row_ids'])}
        if set(lookup) != set(reference['row_ids']):
            raise ValueError("Different row ID sets")
        order = [lookup[key] for key in reference['row_ids']]
        candidate = dict(candidate)
        for key in ('targets', 'samples', 'row_ids', 'group_ids', 'mask', 'history_mask',
                    'origin_observed', 'prediction_anchors', 'sampled_origin_reports'):
            if key in candidate:
                candidate[key] = candidate[key][order]
        if 'history_mask_by_origin' in candidate:
            candidate['history_mask_by_origin'] = candidate['history_mask_by_origin'][:, order]
    for key in ('row_ids', 'group_ids', 'scales'):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Mismatched {key}; row reordering must be explicit")
    if not np.array_equal(reference['targets'], candidate['targets'], equal_nan=True):
        raise ValueError("Mismatched targets or target mask")
    if reference['samples'].shape != candidate['samples'].shape:
        raise ValueError("Sample axes and ensemble count must match")
    for key in ('scenario', 'mask_seed', 'origins_days', 'cells_origin_endpoint_days',
                'history_mask_by_origin', 'origin_observed', 'report_days'):
        if (key in reference) != (key in candidate):
            raise ValueError(f'Asymmetric conditioning metadata: {key}')
        if key in reference and not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f'Mismatched conditioning information: {key}; cross-scenario success gates are not supported')
    return reference, candidate


def whole_group_weights(groups, replicates, seed):
    """Return well multiplicities: every well in a drawn group has equal weight."""
    groups = np.asarray(groups)
    labels, inverse = np.unique(groups, return_inverse=True)
    if groups.ndim != 1 or len(labels) < 2:
        raise ValueError("At least two distinct spatial groups are required")
    if not isinstance(replicates, (int, np.integer)) or replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    group_weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=replicates)
    return group_weights[:, inverse]


def _relative(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 and denominator == 0 else np.nan


def _summarize(point, draws, confidence):
    draws = np.asarray(draws)
    valid = draws[np.isfinite(draws)]
    if len(valid):
        quantiles = np.quantile(valid, [(1-confidence)/2, (1+confidence)/2, 1-confidence, confidence], method='linear')
    else:
        quantiles = [np.nan] * 4
    return dict(estimate=point, lower=quantiles[0], upper=quantiles[1],
                one_sided_lower=quantiles[2], one_sided_upper=quantiles[3],
                confidence=confidence, valid_replicates=len(valid),
                invalid_replicates=len(draws)-len(valid),
                inference_available=bool(np.isfinite(point) and len(valid) == len(draws)))


@_reject_numerical_failure
def compare_predictions(reference, candidate, criteria, *, replicates=1000,
                        seed=42, reorder_by_ids=False, return_draws=False):
    """Compare one fitted reference/candidate pair, using identical group draws.

    Cell means retain well weights, then cells and phases have equal weight.
    All observations of a well remain together, including overlapping cells.
    Required cells disappearing in a replicate invalidate it, and any invalid
    replicate disables that metric's uncertainty gate (never silently alters
    the estimand). R2 cell membership is frozen to the full comparison sample.
    """
    reference, candidate = align_predictions(reference, candidate, reorder_by_ids=reorder_by_ids)
    if criteria['scope']['interval_levels'] != [0.80, 0.95] or criteria['scope']['phase_order'] != list(PHASE_NAMES):
        raise ValueError('This implementation requires 80/95 intervals and oil/gas/water phase order')
    gates = criteria['practical_gates']
    confidence = gates['paired_group_bootstrap_confidence_level']
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    targets = reference['targets']
    observed = np.isfinite(targets)
    scales = reference['scales']
    required = observed.any(axis=0)
    # Center moments before weighting to avoid large-level cancellation.
    center = _mean(targets, axis=0)
    centered = np.where(observed, targets - center, 0)
    fixed_r2_cells = (observed.sum(axis=0) >= 2) & (np.square(centered).sum(axis=0) > 0)
    model_values = []
    for prediction in (reference, candidate):
        clean = np.where(observed[:, None], prediction['samples'], 0)
        values = {'wis': weighted_interval_score(targets, prediction['samples']) / scales,
                  'sse': np.where(observed, (clean.mean(axis=1) - targets)**2, 0)}
        for level in (80, 95):
            alpha = 1-level/100
            lower, upper = np.quantile(clean, [alpha/2, 1-alpha/2], axis=1, method='linear')
            values[f'coverage_{level}'] = np.where(observed, (targets >= lower) & (targets <= upper), np.nan)
            values[f'width_{level}'] = np.where(observed, (upper-lower)/scales, np.nan)
        model_values.append(values)

    def statistics(weights):
        count = np.tensordot(weights, observed, axes=1)
        valid_cells = not np.any(required & (count == 0))
        def average(values):
            total = np.tensordot(weights, np.where(observed, values, 0), axes=1)
            return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        centered_sum = np.tensordot(weights, centered, axes=1)
        ss = np.tensordot(weights, centered**2, axes=1)
        ss -= np.divide(centered_sum**2, count, out=np.zeros_like(ss), where=count > 0)
        r2_valid = bool(np.all((ss > 0)[fixed_r2_cells]) and np.all((count >= 2)[fixed_r2_cells]))
        summaries = []
        for values in model_values:
            summary = {key: average(value) for key, value in values.items() if key != 'sse'}
            sse = np.tensordot(weights, values['sse'], axes=1)
            summary['r2'] = np.full_like(ss, np.nan)
            if r2_valid:
                summary['r2'][fixed_r2_cells] = 1-sse[fixed_r2_cells]/ss[fixed_r2_cells]
            summaries.append(summary)
        a, b = summaries
        result = {'macro_wis_relative_improvement': _relative(_macro(a['wis'])-_macro(b['wis']), _macro(a['wis'])),
                  'macro_r2_delta': _macro(b['r2'])-_macro(a['r2']),
                  'reference_macro_scaled_wis': _macro(a['wis']), 'candidate_macro_scaled_wis': _macro(b['wis'])}
        for phase, name in enumerate(PHASE_NAMES):
            base, trial = _mean(a['wis'][:, phase]), _mean(b['wis'][:, phase])
            result[f'{name}_wis_relative_degradation'] = _relative(trial-base, base)
        for level in (80, 95):
            result[f'macro_scaled_width_delta_{level}'] = _macro(b[f'width_{level}'])-_macro(a[f'width_{level}'])
            for label, summary in (('reference', a), ('candidate', b)):
                coverage = summary[f'coverage_{level}']
                result[f'{label}_macro_coverage_deficit_{level}'] = max(0.0, level/100-_macro(coverage))
                result[f'{label}_macro_coverage_{level}'] = _macro(coverage)
                for phase, name in enumerate(PHASE_NAMES):
                    result[f'{label}_{name}_coverage_deficit_{level}'] = max(0.0, level/100-_mean(coverage[:, phase]))
        if not valid_cells:
            result = {key: np.nan for key in result}
        return result

    estimate = statistics(np.ones(len(targets)))
    draws = {key: np.empty(replicates) for key in estimate}
    for index, weights in enumerate(whole_group_weights(reference['group_ids'], replicates, seed)):
        for key, value in statistics(weights).items():
            draws[key][index] = value
    summaries = {key: _summarize(value, draws[key], confidence) for key, value in estimate.items()}
    def below(key, threshold, field):
        row = summaries[key]
        return bool(np.isfinite(row[field]) and row[field] <= threshold and
                    (field == 'estimate' or row['inference_available']))
    wis = summaries['macro_wis_relative_improvement']
    decision = {'macro_wis_practical_gain': bool(np.isfinite(wis['estimate']) and wis['estimate'] >= gates['macro_wis_relative_improvement_minimum']),
                'wis_improvement_interval_excludes_zero': bool(wis['inference_available'] and wis['lower'] > 0)}
    for name in PHASE_NAMES:
        key = f'{name}_wis_relative_degradation'
        decision[f'{name}_wis_point_noninferiority'] = below(key, gates['phase_wis_relative_degradation_maximum'], 'estimate')
        decision[f'{name}_wis_uncertainty_noninferiority'] = below(key, gates['phase_wis_relative_degradation_maximum'], 'one_sided_upper')
    r2 = summaries['macro_r2_delta']
    decision['r2_point_noninferiority'] = bool(np.isfinite(r2['estimate']) and r2['estimate'] >= -gates['macro_r2_absolute_degradation_maximum'])
    decision['r2_uncertainty_noninferiority'] = bool(r2['inference_available'] and r2['one_sided_lower'] >= -gates['macro_r2_absolute_degradation_maximum'])
    for key in summaries:
        if key.startswith('candidate_') and 'coverage_deficit' in key:
            decision[key+'_point_gate'] = below(key, gates['macro_and_phase_coverage_deficit_maximum'], 'estimate')
            decision[key+'_uncertainty_gate'] = below(key, gates['macro_and_phase_coverage_deficit_maximum'], 'one_sided_upper')
    required_decisions = {key: value for key, value in decision.items()
                          if (gates['noninferiority_requires_one_sided_uncertainty_bounds'] or 'uncertainty' not in key)
                          and (gates['paired_wis_improvement_interval_excludes_zero'] or key != 'wis_improvement_interval_excludes_zero')}
    result = _json_ready({'status': 'development_only', 'confirmatory_conclusion': None,
        'metrics': summaries, 'criteria': criteria, 'gates': decision,
        'all_development_gates_pass': all(required_decisions.values()),
        'wells': len(targets), 'groups': len(np.unique(reference['group_ids'])),
        'ensemble_count': reference['samples'].shape[1], 'replicates': replicates, 'bootstrap_seed': seed,
        'observed_target_counts': observed.sum(axis=0), 'r2_cell_mask': fixed_r2_cells,
        'phase_order': list(PHASE_NAMES), 'reordered_by_row_ids': reorder_by_ids,
        'method': 'paired_percentile_whole_spatial_group_bootstrap',
        'aggregation': 'well_mean_within_cell_then_equal_cells_within_phase_then_equal_phases',
        'seed_uncertainty': 'not_estimated_single_fitted_pair_conditional_group_uncertainty',
        'limitations': ['Development predictions do not establish confirmation.',
                        'No correction for model selection or repeated development comparisons.',
                        'Intervals are pointwise; the all-gates claim is a prespecified conjunction.',
                        'Prior Bakken exposure remains; no independent confirmation is implied.']})
    if 'scenario' in reference:
        result['conditioning'] = {
            'status': 'matched_explicit_history_study', 'scenario': str(reference['scenario'].item()),
            'mask_seed': int(reference['mask_seed'].item()), 'origins_days': reference['origins_days'].tolist(),
            'report_days': reference['report_days'].tolist(),
            'history_mask_shape_origin_well_time_phase': list(reference['history_mask_by_origin'].shape),
            'history_mask_sha256': hashlib.sha256(np.ascontiguousarray(reference['history_mask_by_origin'], dtype=np.uint8).tobytes()).hexdigest(),
            'cross_scenario_success_gates': 'rejected',
        }
        for label, prediction in (('reference', reference), ('candidate', candidate)):
            if '_report_days_source_path' in prediction:
                result['conditioning'][label + '_report_days_source'] = {
                    'path': str(prediction['_report_days_source_path'].item()),
                    'sha256': str(prediction['_report_days_source_sha256'].item()),
                }
    else:
        result['conditioning'] = {'status': 'legacy_explicit_history_metadata_unavailable',
                                  'scope': 'paired IDs, groups, targets, scales, and sample axes; source protocol establishes history parity'}
    return (result, draws) if return_draws else result
