from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np
import pytest
from joint_production.comparison import align_predictions, compare_predictions, load_predictions, whole_group_weights
from joint_production.metrics import evaluate_forecasts


CRITERIA = tomllib.loads((Path(__file__).resolve().parents[1] / 'configs/evidence_criteria.toml').read_text())


def prediction(offset=0.0):
    target = np.arange(1, 37, dtype=float).reshape(12, 1, 3)
    return {'targets': target, 'samples': np.repeat(target[:, None], 21, axis=1) + offset,
            'scales': np.ones((1, 3)), 'row_ids': np.array([f'r{i}' for i in range(12)]),
            'group_ids': np.array([f'g{i//2}' for i in range(12)])}


def test_identical_perfect_forecasts_have_zero_paired_differences():
    result = compare_predictions(prediction(), prediction(), CRITERIA, replicates=300)
    for key in ('macro_wis_relative_improvement', 'macro_r2_delta', 'macro_scaled_width_delta_80',
                'macro_scaled_width_delta_95', 'oil_wis_relative_degradation'):
        row = result['metrics'][key]
        assert row['estimate'] == row['lower'] == row['upper'] == 0
    assert result['status'] == 'development_only'
    assert result['confirmatory_conclusion'] is None
    assert not result['all_development_gates_pass']


def test_better_forecasts_and_core_metric_agreement():
    reference, candidate = prediction(4), prediction(1)
    result = compare_predictions(reference, candidate, CRITERIA, replicates=300)
    gain = result['metrics']['macro_wis_relative_improvement']
    assert gain['estimate'] == pytest.approx(.75)
    assert gain['lower'] > 0
    for name, data in [('reference', reference), ('candidate', candidate)]:
        expected = evaluate_forecasts(data['targets'], data['samples'], data['scales'], include_joint_scores=False)
        assert result['metrics'][f'{name}_macro_scaled_wis']['estimate'] == pytest.approx(expected['macro_scaled_wis'])
    assert result['metrics']['macro_r2_delta']['lower'] > 0


def test_group_multiplicity_preserves_every_member_and_group_draw_count():
    groups = np.array(['a', 'a', 'a', 'b', 'c', 'c'])
    weight = whole_group_weights(groups, 300, 7)
    assert np.array_equal(weight[:, 0], weight[:, 1])
    assert np.array_equal(weight[:, 0], weight[:, 2])
    assert np.array_equal(weight[:, 4], weight[:, 5])
    assert np.all(weight[:, [0, 3, 4]].sum(axis=1) == 3)
    assert np.any(weight.sum(axis=1) != len(groups))
    assert np.array_equal(weight, whole_group_weights(groups, 300, 7))


@pytest.mark.parametrize('field', ['targets', 'scales', 'group_ids', 'row_ids', 'samples', 'mask'])
def test_mismatches_rejected(field):
    a, b = prediction(), prediction()
    if field == 'targets': b[field][0, 0, 0] = np.nan
    elif field == 'scales': b[field][0, 0] = 2
    elif field in ('group_ids', 'row_ids'): b[field][0] = 'xx'
    elif field == 'samples': b[field] = b[field][:, :-1]
    else: b[field] = np.zeros_like(b['targets'], dtype=bool)
    with pytest.raises(ValueError): align_predictions(a, b)


def test_reordering_requires_explicit_opt_in_and_does_not_mutate_inputs():
    a, b = prediction(3), prediction(1)
    for key in ('targets', 'samples', 'row_ids', 'group_ids'): b[key] = b[key][::-1].copy()
    old = deepcopy(b)
    with pytest.raises(ValueError): align_predictions(a, b)
    _, aligned = align_predictions(a, b, reorder_by_ids=True)
    assert np.array_equal(a['row_ids'], aligned['row_ids'])
    assert np.array_equal(old['samples'], b['samples'])


def test_nonfinite_forecasts_at_observed_targets_fail():
    a, b = prediction(), prediction()
    b['samples'][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError): compare_predictions(a, b, CRITERIA, replicates=300)


def test_one_sided_noninferiority_uses_corresponding_bound():
    result = compare_predictions(prediction(1), prediction(1.1), CRITERIA, replicates=300)
    assert not result['gates']['oil_wis_uncertainty_noninferiority']
    row = result['metrics']['oil_wis_relative_degradation']
    assert row['one_sided_upper'] > CRITERIA['practical_gates']['phase_wis_relative_degradation_maximum']


def test_missing_targets_keep_equal_cell_phase_weighting_and_frozen_scales():
    a, b = prediction(4), prediction(1)
    for data in (a, b):
        data['targets'] = np.repeat(data['targets'], 2, axis=1)
        data['samples'] = np.repeat(data['samples'], 2, axis=2)
        data['scales'] = np.array([[1., 2., 3.], [4., 5., 6.]])
        data['targets'][::2, 1, 0] = np.nan
        data['samples'][::2, :, 1, 0] = np.nan
    original_scales = a['scales'].copy()
    result = compare_predictions(a, b, CRITERIA, replicates=300)
    for label, data in [('reference', a), ('candidate', b)]:
        standalone = evaluate_forecasts(data['targets'], data['samples'], data['scales'], include_joint_scores=False)
        assert result['metrics'][label+'_macro_scaled_wis']['estimate'] == pytest.approx(standalone['macro_scaled_wis'])
    assert np.array_equal(a['scales'], original_scales)


def test_zero_reference_with_worse_candidate_is_undefined_and_cannot_pass():
    result = compare_predictions(prediction(), prediction(1), CRITERIA, replicates=300)
    assert result['metrics']['macro_wis_relative_improvement']['estimate'] is None
    assert not result['gates']['macro_wis_practical_gain']
    assert not result['gates']['wis_improvement_interval_excludes_zero']


def history_prediction(offset=0.0):
    result = prediction(offset)
    wells, samples = result['samples'].shape[:2]
    history = np.zeros((1, wells, 4, 3), dtype=bool)
    history[:, :, :3] = True
    history[:, :, 1:3, 0] = False  # Oil hidden, day zero retained.
    history[:, ::2, 1, 1] = False  # Natural gas missingness varies by well.
    result.update(scenario=np.asarray('oil_hidden'), mask_seed=np.asarray(2042),
                  origins_days=np.array([60]), report_days=np.array([0, 30, 60, 180]),
                  cells_origin_endpoint_days=np.array([[60, 180]]), history_mask_by_origin=history,
                  origin_observed=history[:, :, 2].transpose(1, 0, 2).copy(),
                  prediction_anchors=np.broadcast_to(np.arange(wells)[:, None, None, None], (wells, samples, 1, 3)).copy(),
                  sampled_origin_reports=np.broadcast_to(np.arange(wells)[:, None, None, None], (wells, samples, 1, 3)).copy())
    return result


def test_matching_history_studies_preserve_metadata_and_record_scope():
    a, b = history_prediction(2), history_prediction(1)
    _, aligned = align_predictions(a, b)
    assert np.array_equal(aligned['history_mask_by_origin'], a['history_mask_by_origin'])
    result = compare_predictions(a, b, CRITERIA, replicates=20)
    assert result['conditioning']['scenario'] == 'oil_hidden'
    assert result['conditioning']['status'] == 'matched_explicit_history_study'
    assert result['conditioning']['report_days'] == [0, 30, 60, 180]


@pytest.mark.parametrize('field', ['scenario', 'mask_seed', 'history_mask_by_origin', 'origins_days', 'cells_origin_endpoint_days', 'origin_observed', 'report_days'])
def test_conditioning_mismatch_cannot_reach_research_gates(field):
    a, b = history_prediction(), history_prediction()
    if field == 'scenario': b[field] = np.asarray('gas_hidden')
    elif field == 'mask_seed': b[field] = np.asarray(2043)
    elif field == 'history_mask_by_origin': b[field][0, 1, 1, 1] = False
    elif field == 'origins_days': b[field] = np.array([30])
    elif field == 'cells_origin_endpoint_days': b[field][0, 1] = 90
    elif field == 'origin_observed': b[field][0, 0, 0] = True
    else: b[field][1] = 31  # Same slots, different actual historical timestamp.
    with pytest.raises(ValueError):
        compare_predictions(a, b, CRITERIA, replicates=20)


@pytest.mark.parametrize('field', ['scenario', 'mask_seed', 'history_mask_by_origin', 'origins_days', 'cells_origin_endpoint_days', 'origin_observed', 'report_days'])
def test_partial_or_asymmetric_history_metadata_rejected(field):
    a, b = history_prediction(), history_prediction()
    del b[field]
    with pytest.raises(ValueError):
        align_predictions(a, b)


def test_history_and_legacy_archives_cannot_be_mixed_but_legacy_pair_remains_valid():
    with pytest.raises(ValueError, match='Asymmetric conditioning'):
        align_predictions(prediction(), history_prediction())
    a, b = align_predictions(prediction(), prediction(1))
    assert 'scenario' not in a and 'scenario' not in b


@pytest.mark.parametrize('field,value', [
    ('scenario', np.array(['oil_hidden'])), ('scenario', np.array('')), ('scenario', np.array(1)),
    ('mask_seed', np.array(True)), ('mask_seed', np.array(2042.0)), ('mask_seed', np.array([2042])),
    ('origins_days', np.array([np.inf])), ('history_mask_by_origin', np.zeros((1, 12, 4, 3), dtype=int)),
])
def test_invalid_history_metadata_schema_rejected(field, value):
    a, b = history_prediction(), history_prediction()
    a[field] = value
    with pytest.raises(ValueError):
        align_predictions(a, b)


def test_history_row_alignment_reorders_correct_axes_and_model_specific_anchors():
    a, b = history_prediction(2), history_prediction(1)
    b['prediction_anchors'] += 100  # Anchor values need not match across models.
    for key in ('targets', 'samples', 'row_ids', 'group_ids', 'origin_observed', 'prediction_anchors', 'sampled_origin_reports'):
        b[key] = b[key][::-1].copy()
    b['history_mask_by_origin'] = b['history_mask_by_origin'][:, ::-1].copy()
    before = deepcopy(b)
    with pytest.raises(ValueError):
        align_predictions(a, b)
    _, aligned = align_predictions(a, b, reorder_by_ids=True)
    assert np.array_equal(aligned['history_mask_by_origin'], a['history_mask_by_origin'])
    assert np.array_equal(aligned['origin_observed'], a['origin_observed'])
    assert np.array_equal(aligned['prediction_anchors'], a['prediction_anchors'] + 100)
    assert np.array_equal(aligned['sampled_origin_reports'], a['sampled_origin_reports'])
    for key in before:
        assert np.array_equal(before[key], b[key])


def test_load_predictions_preserves_direct_history_metadata(tmp_path):
    data = history_prediction()
    path = tmp_path / 'predictions.npz'
    np.savez_compressed(path, **data)
    loaded = load_predictions(path)
    for key, value in data.items():
        assert np.array_equal(loaded[key], value)


def old_history_archive(tmp_path):
    data = history_prediction()
    days = data.pop('report_days')
    prediction_path = tmp_path / 'predictions.npz'
    shared_path = tmp_path / 'history_masks.npz'
    np.savez_compressed(prediction_path, **data)
    np.savez_compressed(shared_path, report_days=days, origins_days=data['origins_days'],
                        row_ids=data['row_ids'], group_ids=data['group_ids'], scenario_names=np.array(['oil_hidden']),
                        history_mask_by_scenario_origin=data['history_mask_by_origin'][None])
    digest = hashlib.sha256(shared_path.read_bytes()).hexdigest()
    (tmp_path / 'completion.json').write_text(json.dumps({'status': 'completed', 'artifacts': {'history_masks.npz': digest}}))
    return prediction_path, shared_path, data


def test_older_study_grid_recovered_only_from_verified_shared_history_archive(tmp_path):
    path, shared, data = old_history_archive(tmp_path)
    loaded = load_predictions(path)
    assert np.array_equal(loaded['report_days'], [0, 30, 60, 180])
    assert loaded['_report_days_source_sha256'].item() == hashlib.sha256(shared.read_bytes()).hexdigest()
    align_predictions(loaded, history_prediction())


def test_old_study_rejects_tampered_shared_archive(tmp_path):
    path, shared, data = old_history_archive(tmp_path)
    with shared.open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='completion hash mismatch'):
        load_predictions(path)


def test_old_study_rejects_different_actual_masks_even_with_valid_parent_hash(tmp_path):
    path, shared, data = old_history_archive(tmp_path)
    data['history_mask_by_origin'][0, 1, 1, 1] = False
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match='conditioning differs'):
        load_predictions(path)


def test_old_study_without_verifiable_time_grid_is_rejected(tmp_path):
    data = history_prediction()
    del data['report_days']
    path = tmp_path / 'predictions.npz'
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match='verified parent'):
        load_predictions(path)
