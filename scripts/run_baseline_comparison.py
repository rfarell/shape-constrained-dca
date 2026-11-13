"""Run a bounded, validation-only comparison with preserved source and predictions."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import time
import traceback
import tomllib

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from joint_production.artifacts import canonical_hash, capture_run, write_json, sha256_file
from joint_production.metrics import conditional_increments, fit_increment_scales, evaluate_forecasts
from joint_production.data import load_dataset
from joint_production.seed_roles import resolve_seed_roles


FINITE_FOREST_CONFIG_SHA256 = "5313b7871d36d0c52a5e5f896e1575b608b150acf10c4e338175e4efa9d1ed4a"
REPORT_INTEGRATED_CONFIG_SHA256 = "06755bad5c1d558a4251fca2f5b54442fbb259f6c4cc2dd057c0511c51d8b54c"


def finite_forest_contract(config, workers=None):
    """The implemented finite-prior settings must match the frozen experiment."""
    active = any(model.get('kind') == 'finite_forest_rate' for model in config['models'])
    if active and (canonical_hash(config) != FINITE_FOREST_CONFIG_SHA256
                   or workers not in (None, 2)):
        raise ValueError('Finite forest pilot differs from its frozen configuration')
    return active


def report_integrated_contract(config):
    """Keep the first integrated-score comparison within its frozen scope."""
    active = any(model.get('name') == 'basis_report_integrated'
                 or model.get('marginal_score_estimator', 'sampled') != 'sampled'
                 for model in config['models'])
    if active and canonical_hash(config) != REPORT_INTEGRATED_CONFIG_SHA256:
        raise ValueError('Integrated-score pilot differs from its frozen configuration')
    return active


def rate_score_metadata(model, model_config):
    """Bind the fitted estimator; default omission preserves existing artifacts."""
    expected = model_config.get('marginal_score_estimator', 'sampled')
    if expected not in {'sampled', 'gaussian_report_integrated'}:
        raise ValueError('Unknown conditional marginal score estimator')
    if (getattr(model, 'marginal_score_estimator', 'sampled') != expected
            or getattr(model, 'fit_diagnostics_', {}).get('marginal_score_estimator', 'sampled') != expected):
        raise ValueError('Fitted marginal score estimator differs from configuration')
    if expected == 'gaussian_report_integrated':
        if model.observation_law != 'conditional_gaussian_accumulated_bridge':
            raise ValueError('Integrated marginal scoring requires accumulated reports')
        return {'marginal_score_estimator': expected}
    return {}


def full_curve_metrics(target, pred, cumulative_target, cumulative_pred, scales,
                       cumulative_scales, cells, origins):
    """Score future cells only; never let pinned history inflate curve accuracy."""
    def score(y, draws, scaling, indices, *, cumulative=False):
        result = evaluate_forecasts(y[:, indices], draws[:, :, indices], scaling[indices],
                                    include_joint_scores=False)
        if cumulative:
            result['macro_cumulative_r2'] = result.pop('macro_increment_r2')
        result['cells_origin_endpoint_days'] = [cells[index] for index in indices]
        return result
    all_indices = list(range(len(cells)))
    result = {'cumulative': score(cumulative_target, cumulative_pred, cumulative_scales,
                                   all_indices, cumulative=True),
              'increment_by_origin': {}, 'cumulative_by_origin': {}}
    for origin in origins:
        indices = [i for i, cell in enumerate(cells) if cell[0] == origin]
        result['increment_by_origin'][str(origin)] = score(target, pred, scales, indices)
        result['cumulative_by_origin'][str(origin)] = score(
            cumulative_target, cumulative_pred, cumulative_scales, indices, cumulative=True)
    legacy = [i for i, cell in enumerate(cells) if cell[0] in (60, 90, 180, 360, 720)]
    if legacy:
        result['legacy_history_increment'] = score(target, pred, scales, legacy)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/bakken_v1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="Override forest CPU workers; recorded in effective model settings")
    parser.add_argument("--device", choices=['cpu', 'cuda'], help="Override Torch device; recorded in effective model settings")
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text())
    finite_contract = finite_forest_contract(config, args.workers)
    report_integrated_contract(config)
    static_mode = config['experiment'].get('static_features', 'none')
    if static_mode not in {'none', 'nonspatial_17'}:
        raise ValueError('Unknown static_features mode')
    tree_contract = None
    if any(model.get('tree_static_mode', 'none') != 'none' for model in config['models']):
        from joint_production.tree_static_prior import FOLD_COUNT, COEFFICIENT_FLOOR, FOREST_SETTINGS
        tree_contract = dict(folds=FOLD_COUNT, fold_seed=271828, coefficient_floor_normalized=COEFFICIENT_FLOOR,
            n_estimators=FOREST_SETTINGS['n_estimators'], min_samples_leaf=FOREST_SETTINGS['min_samples_leaf'], n_jobs=2)
        if config.get('tree_teacher') != tree_contract or (args.workers is not None and args.workers != tree_contract['n_jobs']):
            raise ValueError('Tree teacher contract differs from fixed module settings')
    for model_config in config['models']:
        if 'marginal_score_estimator' in model_config and model_config['kind'] != 'conditional_rate':
            raise ValueError('Marginal score estimator requires a conditional rate model')
        if model_config.get('tree_static_mode', 'none') != 'none' and (
                model_config['kind'] != 'conditional_rate' or static_mode != 'nonspatial_17'
                or model_config.get('feature_variant') != 'actual'):
            raise ValueError('Tree static conditioning requires actual nonspatial static features and conditional_rate')
        if static_mode != 'none':
            if (model_config['kind'] not in {'forest', 'neural', 'conditional_rate', 'finite_forest_rate'}
                    or model_config.get('feature_variant') not in {'actual', 'zero'}
                    or model_config.get('covariate_preprocessing') != 'external'):
                raise ValueError('Static study requires supported models, explicit actual/zero variants and external preprocessing')
        elif 'feature_variant' in model_config:
            raise ValueError('feature_variant requires an explicit static feature study')
    seed_roles = {str(seed): resolve_seed_roles(config["experiment"], seed)
                  for seed in config["experiment"]["seeds"]}
    manifest = capture_run(ROOT, args.output, config=args.config,
                           data=ROOT / "data/raw/bakken.csv", split=args.data / "split_manifest.json",
                           seeds=config["experiment"]["seeds"])
    manifest.update({"seed_roles_by_training_seed": seed_roles,
                     "manifest_seeds_role": "training_seeds; effective prediction and evaluation-bootstrap seeds recorded separately"})
    write_json(args.output / "manifest.json", manifest, exclusive=False)
    start = time.monotonic()
    try:
        dataset = load_dataset(args.data)
        if dataset['manifest']['dataset_sha256'] != manifest['data_sha256']:
            raise ValueError('Processed split does not match the verified raw dataset')
        records = dataset['records']
        train = [r for r in records if r["split"] == "train"]
        validation = [r for r in records if r["split"] == "validation"]
        days = np.array([0, 30, 60, 90, 180, 360, 720, 1080])
        y_train = np.asarray([r["reports"] for r in train], dtype=float)
        y_validation = np.asarray([r["reports"] for r in validation], dtype=float)
        groups = np.asarray([r["group_id"] for r in validation])
        # No calibration/test observations are materialized into model or metric arrays.
        del records, dataset
        feature_provenance = None
        if static_mode == 'nonspatial_17':
            from joint_production.static_features import StaticFeaturePreprocessor
            processor = StaticFeaturePreprocessor.fit(train)
            x_train, x_validation = processor.transform(train), processor.transform(validation)
            feature_provenance = processor.to_dict()
            feature_provenance['availability_assumption'] = config['experiment']['static_availability_assumption']
            write_json(args.output / 'static_features.json', feature_provenance)
            np.savez_compressed(args.output / 'static_features.npz', train_features=x_train,
                validation_features=x_validation,
                train_row_ids=np.asarray([r['row_id'] for r in train]),
                validation_row_ids=np.asarray([r['row_id'] for r in validation]))
        origins = config["experiment"]["origins_days"]
        cells = [(int(s), int(t)) for s in origins for t in days if t > s]
        train_targets = np.stack([y_train[:, np.flatnonzero(days == t)[0]] - y_train[:, np.flatnonzero(days == s)[0]] for s, t in cells], axis=1)
        scales, scale_provenance = fit_increment_scales(train_targets, return_details=True)
        curve_evaluation = config['experiment'].get('full_curve_evaluation', False)
        if curve_evaluation:
            train_cumulative = np.stack([y_train[:, np.flatnonzero(days == t)[0]] for _, t in cells], axis=1)
            cumulative_scales, cumulative_provenance = fit_increment_scales(train_cumulative, return_details=True)
            write_json(args.output / 'cumulative_scales.json', {
                'cells_origin_endpoint_days': cells, 'phase_order': ['oil', 'gas', 'water'],
                'scales': cumulative_scales.tolist(), 'fit_split': 'train',
                'scale_provenance': cumulative_provenance})
        write_json(args.output / "scales.json", {
            "cells_origin_endpoint_days": cells, "phase_order": ["oil", "gas", "water"],
            "scales": scales.tolist(), "fit_split": "train", "scale_provenance": scale_provenance,
            "train_row_ids_sha256": __import__('hashlib').sha256('\n'.join(r['row_id'] for r in train).encode()).hexdigest(),
        })
        all_results = []
        for seed in config["experiment"]["seeds"]:
            effective_seeds = seed_roles[str(seed)]
            reference_tree = None
            for model_config in config["models"]:
                model_config = dict(model_config)
                if args.workers is not None and model_config['kind'] == 'forest':
                    model_config['n_jobs'] = args.workers
                if args.device is not None and model_config['kind'] in {'hierarchical', 'neural', 'conditional_rate', 'conditional_report', 'finite_forest_rate'}:
                    model_config['device'] = args.device
                model_start = time.monotonic()
                label = model_config["name"]
                model_kwargs = {k:v for k,v in model_config.items() if k not in {'name', 'kind', 'feature_variant'}}
                fit_features, prediction_features = {}, {}
                tree_provenance = None
                forest_origin0_provenance = None
                finite_provenance = None
                if feature_provenance is not None:
                    actual = model_config['feature_variant'] == 'actual'
                    fit_features = {'covariates': x_train if actual else np.zeros_like(x_train)}
                    prediction_features = {'covariates': x_validation if actual else np.zeros_like(x_validation)}
                print(json.dumps({"event": "model_start", "model": label, "seed": seed}), flush=True)
                if model_config.get('tree_static_mode', 'none') != 'none':
                    from joint_production.tree_static_prior import TreeStaticPrior
                    if reference_tree is None:
                        raise ValueError('Fit the informed forest_reference before its tree-conditioned candidate')
                    teacher = TreeStaticPrior(seed=seed, fold_seed=tree_contract['fold_seed'], n_jobs=tree_contract['n_jobs']).fit(train, y_train, days)
                    if teacher.train_row_ids_ != [row['row_id'] for row in train]:
                        raise ValueError('Tree OOF bank TRAIN row order differs from core training order')
                    train_prior = teacher.oof_coefficients_native_
                    # Query API sees no held-out reports, row IDs, or histories.
                    validation_prior, validation_projected, validation_tree_mean = teacher.predict_components(
                        [{'covariates_raw':row['covariates_raw']} for row in validation])
                    teacher_metadata = teacher.metadata()
                    if (teacher_metadata['full_train']['origin0_forest_law_fingerprint'] != reference_tree['fingerprint']
                            or teacher_metadata['full_train']['preprocessing'] != reference_tree['preprocessing']
                            or not np.array_equal(validation_tree_mean, reference_tree['native_mean'])):
                        raise ValueError('Full-TRAIN teacher differs from the informed reference origin0 issued law')
                    tree_stem = f'{label}_seed{seed}'
                    teacher_file = f'{tree_stem}_tree_teacher.joblib'
                    metadata_file = f'{tree_stem}_tree_metadata.json'
                    bank_file = f'{tree_stem}_tree_priors.npz'
                    teacher.save(args.output / teacher_file)
                    write_json(args.output / metadata_file, teacher_metadata)
                    np.savez_compressed(args.output / bank_file,
                        oof_train_coefficients_native=train_prior,
                        oof_train_integrated_mean_native=teacher.oof_mean_cumulative_native_,
                        oof_train_tree_mean_native=teacher.oof_tree_mean_cumulative_native_,
                        oof_train_fold=teacher.folds_, train_row_ids=np.asarray([r['row_id'] for r in train]),
                        train_group_ids=np.asarray([r['group_id'] for r in train]),
                        full_train_validation_coefficients_native=validation_prior,
                        full_train_validation_integrated_mean_native=validation_projected,
                        full_train_validation_tree_mean_native=validation_tree_mean,
                        validation_row_ids=np.asarray([r['row_id'] for r in validation]), validation_group_ids=groups,
                        report_days=days, phase_order=np.asarray(['oil','gas','water']))
                    tree_provenance = {name:{'file':file,'sha256':sha256_file(args.output/file)} for name,file in (
                        ('teacher',teacher_file),('metadata',metadata_file),('prior_bank',bank_file))}
                    tree_provenance.update(training_input='group-cross-fitted TRAIN coefficient means only',
                        validation_input='full-TRAIN teacher; static attributes only', fold_seed=271828,
                        teacher_training_seed=seed, native_coefficient_order='oil[4],gas[4],water[4]',
                        reference_origin0_forest_law_fingerprint=reference_tree['fingerprint'],
                        reference_origin0_mean=reference_tree['mean_artifact'],
                        exact_reference_teacher_origin0_law_match=True)
                    fit_features['tree_prior'] = train_prior
                    prediction_features['tree_prior'] = validation_prior
                    del teacher
                if model_config["kind"] == "forest":
                    from joint_production.baselines import ProbabilisticTrajectoryForest
                    model = ProbabilisticTrajectoryForest(random_state=seed,
                        **model_kwargs)
                    model.fit(y_train, days, origins=origins, **fit_features)
                    if (tree_contract is not None or finite_contract) and label == 'forest_reference':
                        from joint_production.tree_static_prior import origin0_forest_law_fingerprint, exact_static_forest_mean
                        fingerprint = origin0_forest_law_fingerprint(model)
                        native_mean = exact_static_forest_mean(model, prediction_features['covariates'])
                        mean_file = f'{label}_seed{seed}_origin0_mean.npz'
                        np.savez_compressed(args.output / mean_file, native_mean_cumulative=native_mean,
                            row_ids=np.asarray([r['row_id'] for r in validation]), group_ids=groups,
                            report_days=days, phase_order=np.asarray(['oil','gas','water']))
                        forest_origin0_provenance = dict(origin0_forest_law_fingerprint=fingerprint,
                            origin0_mean_artifact={'file':mean_file,'sha256':sha256_file(args.output/mean_file)},
                            origin0_static_preprocessing=processor.to_dict())
                        reference_tree = dict(fingerprint=fingerprint, native_mean=native_mean,
                            preprocessing=processor.to_dict(), mean_artifact=forest_origin0_provenance['origin0_mean_artifact'])
                elif model_config["kind"] == "dca":
                    from joint_production.dca import HierarchicalDCABaseline
                    model = HierarchicalDCABaseline(seed=seed, **model_kwargs)
                    model.fit(y_train, days)
                elif model_config["kind"] == "hierarchical":
                    from joint_production.hierarchical import HierarchicalForecaster
                    model = HierarchicalForecaster(seed=seed,
                        **model_kwargs)
                    model.fit(y_train, days)
                elif model_config["kind"] == "neural":
                    from joint_production.neural import ContextualGaussianNeuralBaseline
                    model = ContextualGaussianNeuralBaseline(seed=seed,
                        **model_kwargs)
                    model.fit(y_train, days, origins=origins, **fit_features)
                elif model_config["kind"] == "conditional_rate":
                    from joint_production.conditional_rates import ConditionalRateForecaster
                    model = ConditionalRateForecaster(seed=seed,
                        **model_kwargs)
                    model.fit(y_train, days, origins=origins, **fit_features)
                elif model_config["kind"] == "finite_forest_rate":
                    from joint_production.finite_forest_rates import FiniteForestRateForecaster
                    from joint_production.tree_static_prior import origin0_forest_law_fingerprint, exact_static_forest_mean
                    if reference_tree is None:
                        raise ValueError('Fit the informed forest_reference before finite-prior candidates')
                    if model_kwargs.pop('covariate_preprocessing') != 'external':
                        raise ValueError('Finite prior requires the explicit external preprocessing contract')
                    model = FiniteForestRateForecaster(seed=seed, **model_kwargs)
                    model.fit(y_train, days, origins=origins, records=train, **fit_features)
                    finite_stem = f'{label}_seed{seed}'
                    finite_evidence = model.save_evidence(args.output, finite_stem)
                    if (model.fit_diagnostics_.get('observation_law') != 'conditional_gaussian_accumulated_bridge'
                            or model.fit_diagnostics_.get('noise_mode') != model_config['noise_mode']):
                        raise ValueError('Finite-prior fitted law metadata differs from the frozen configuration')
                    # Preserve an unresolved fitting attempt, and prohibit any
                    # candidate validation prediction or score from that fit.
                    if not model.fit_diagnostics_.get('converged', False):
                        model.save(args.output / f'{label}_seed{seed}_model.pt')
                        write_json(args.output / f'{label}_seed{seed}_fit_diagnostics.json', model.fit_diagnostics_)
                        raise RuntimeError('Finite-prior noise optimization unresolved; candidate validation prohibited')
                    bank = model.bank_
                    query_records = [{'covariates_raw': row['covariates_raw']} for row in validation]
                    fingerprint = origin0_forest_law_fingerprint(bank.forest)
                    native_mean = exact_static_forest_mean(bank.forest, bank.processor.transform(query_records))
                    if (fingerprint != reference_tree['fingerprint']
                            or bank.processor.to_dict() != reference_tree['preprocessing']
                            or not np.array_equal(native_mean, reference_tree['native_mean'])
                            or not np.array_equal(bank.forest.train_reports_, y_train)
                            or not np.array_equal(bank.forest.donor_row_indices_, np.arange(len(train)))):
                        raise ValueError('Finite full-TRAIN teacher differs from the informed forest origin-zero law')
                    prediction_features['records'] = query_records
                    support_file = f'{finite_stem}_donor_support.npz'
                    metadata_file = f'{finite_stem}_finite_metadata.json'
                    prior_file = f'{finite_stem}_static_prior.npz'
                    prior_weights, prior_receipt = bank.exact_weights(query_records, return_receipt=True)
                    np.savez_compressed(args.output / prior_file,
                        prior_probabilities=prior_weights, **prior_receipt,
                        query_features=bank.processor.transform(query_records),
                        row_ids=np.asarray([row['row_id'] for row in validation]), group_ids=groups,
                        phase_order=np.asarray(['oil', 'gas', 'water']), report_days=days)
                    np.savez_compressed(args.output / support_file,
                        coefficients_native=bank.coefficients_native, cumulative_native=bank.cumulative_native,
                        raw_reports_native=y_train, endpoint_scales_native=bank.endpoint_scales_native,
                        phase_scales_native=bank.phase_scales_native, report_days=days,
                        row_ids=np.asarray([row['row_id'] for row in train]),
                        group_ids=np.asarray([row['group_id'] for row in train]),
                        phase_order=np.asarray(['oil', 'gas', 'water']), units=np.asarray(['BBL', 'MCF', 'BBL']))
                    write_json(args.output / metadata_file, model.fit_diagnostics_)
                    finite_provenance = {
                        'support': {'file': support_file, 'sha256': sha256_file(args.output / support_file)},
                        'metadata': {'file': metadata_file, 'sha256': sha256_file(args.output / metadata_file)},
                        'static_prior': {'file': prior_file, 'sha256': sha256_file(args.output / prior_file)},
                        'origin0_forest_law_fingerprint': fingerprint,
                        'exact_reference_teacher_origin0_law_match': True,
                        'reference_origin0_mean': reference_tree['mean_artifact'],
                        'oof_noise_evidence': finite_evidence,
                    }
                elif model_config["kind"] == "conditional_report":
                    from joint_production.conditional_reports import ConditionalReportGaussianForecaster
                    model = ConditionalReportGaussianForecaster(seed=seed,
                        **model_kwargs)
                    model.fit(y_train, days, origins=origins)
                else:
                    raise ValueError(f"Unsupported model kind: {model_config['kind']}")
                score_metadata = (rate_score_metadata(model, model_config)
                                  if model_config['kind'] == 'conditional_rate' else {})
                if hasattr(model, 'save'):
                    model.save(args.output / f'{label}_seed{seed}_model.pt')
                else:
                    with (args.output / f'{label}_seed{seed}_model.pkl').open('xb') as handle:
                        pickle.dump(model, handle, protocol=5)
                rate_report_metadata = {}
                rate_history_metadata = {}
                rate_coefficient_metadata = {}
                if model_config['kind'] == 'conditional_rate':
                    half_life = model_config.get('report_half_life_days')
                    rate_report_metadata = {
                        'observation_law': ('conditional_gaussian_accumulated_bridge' if half_life is None
                                            else 'conditional_gaussian_fading_bridge'),
                        'report_half_life_days': half_life,
                    }
                    rate_history_metadata = {
                        'history_head_mode': model_config.get('history_head_mode', 'shared'),
                        'common_initial_parameter_sha256': getattr(model, 'common_initial_parameter_sha256_',
                            getattr(model, 'fit_diagnostics_', {}).get('initial_parameter_sha256', '')),
                        'initial_output_head_sha256': getattr(model, 'initial_output_head_sha256_', ''),
                        'no_history_head_initial_sha256': getattr(model, 'no_history_head_initial_sha256_', None),
                    }
                    reference = getattr(model, 'coefficient_initial_variance', None)
                    rate_coefficient_metadata = {
                        'coefficient_parameterization': model_config.get('coefficient_parameterization', 'log_location'),
                        'coefficient_initial_variance': None if reference is None else reference.detach().cpu().tolist(),
                    }
                predicted, targets, diagnostics = [], [], {}
                cumulative_targets, cumulative_predictions = [], []
                for origin in origins:
                    history = y_validation.copy()
                    history[:, days > origin, :] = np.nan
                    sample_arguments = dict(origin=origin, n_samples=config["experiment"]["samples"],
                                            rng=np.random.default_rng(effective_seeds["prediction_seed"] * 10000 + origin),
                                            **prediction_features)
                    path_metadata = {}
                    if curve_evaluation:
                        path_metadata = dict(history_mask=np.isfinite(history), origin_day=np.asarray(origin),
                            group_ids=groups, static_feature_variant=np.asarray(model_config.get('feature_variant', 'none')),
                            static_features_sha256=np.asarray(sha256_file(args.output / 'static_features.json')
                                if feature_provenance is not None else ''))
                    if model_config['kind'] in {'hierarchical', 'conditional_rate', 'finite_forest_rate'}:
                        distribution = model.predict_distribution(history, **sample_arguments)
                        draws = distribution['report_cumulative']
                        if model_config['kind'] == 'finite_forest_rate':
                            if (distribution.get('observation_law') != 'conditional_gaussian_accumulated_bridge'
                                    or distribution.get('noise_mode') != model_config['noise_mode']):
                                raise ValueError('Finite-prior prediction report law differs from fitting')
                            path_metadata.update(
                                observation_law=np.asarray('conditional_gaussian_accumulated_bridge'),
                                noise_mode=np.asarray(model_config['noise_mode']),
                                latent_law=np.asarray('exact_finite_joint_categorical_posterior'),
                                static_prior_sha256=np.asarray(finite_provenance['static_prior']['sha256']),
                                donor_support_sha256=np.asarray(finite_provenance['support']['sha256']))
                        if rate_report_metadata:
                            expected_estimator = model_config.get('marginal_score_estimator', 'sampled')
                            if distribution.get('marginal_score_estimator', 'sampled') != expected_estimator:
                                raise ValueError('Prediction marginal score estimator differs from fitting')
                            path_metadata.update({key: np.asarray(value) for key, value in score_metadata.items()})
                            for key, expected in rate_report_metadata.items():
                                if distribution.get(key, expected) != expected:
                                    raise ValueError('Conditional rate prediction report-law metadata differs from configuration')
                            if distribution.get('history_head_mode', rate_history_metadata['history_head_mode']) != rate_history_metadata['history_head_mode']:
                                raise ValueError('Conditional rate prediction history-head mode differs from configuration')
                            if distribution.get('coefficient_parameterization', rate_coefficient_metadata['coefficient_parameterization']) != rate_coefficient_metadata['coefficient_parameterization']:
                                raise ValueError('Conditional rate prediction coefficient parameterization differs from configuration')
                            reference = rate_coefficient_metadata['coefficient_initial_variance']
                            if reference is not None and not np.array_equal(distribution.get('coefficient_initial_variance'), reference):
                                raise ValueError('Conditional rate prediction coefficient variance reference differs from fit')
                            # Scalars otherwise disappear in the ndarray-only
                            # path filter. Infinity is a non-object NPZ sentinel
                            # for the legacy infinite-memory kernel.
                            path_metadata.update(
                                observation_law=np.asarray(rate_report_metadata['observation_law']),
                                report_half_life_days=np.asarray(np.inf if half_life is None else half_life))
                            path_metadata.update({key: np.asarray('' if value is None else value)
                                                  for key, value in rate_history_metadata.items()})
                            path_metadata.update(coefficient_parameterization=np.asarray(rate_coefficient_metadata['coefficient_parameterization']),
                                coefficient_initial_variance=np.asarray([] if reference is None else reference, dtype=np.float32))
                            if tree_provenance is not None:
                                mode = model_config['tree_static_mode']
                                if distribution.get('tree_static_mode') != mode or not np.array_equal(distribution.get('tree_prior_native'), prediction_features['tree_prior']):
                                    raise ValueError('Tree prediction mode or native prior differs from full-TRAIN teacher bank')
                                path_metadata.update(tree_static_mode=np.asarray(mode),
                                    tree_prior_bank_sha256=np.asarray(tree_provenance['prior_bank']['sha256']))
                        np.savez_compressed(args.output / f'{label}_seed{seed}_origin{origin}_paths.npz',
                            **{k:v for k,v in distribution.items() if isinstance(v, np.ndarray) and k not in path_metadata},
                            report_days=days, row_ids=np.array([r['row_id'] for r in validation]), **path_metadata)
                    else:
                        draws = model.predict_samples(history, **sample_arguments)
                        if curve_evaluation:
                            np.savez_compressed(args.output / f'{label}_seed{seed}_origin{origin}_paths.npz',
                                report_cumulative=draws, report_days=days,
                                row_ids=np.asarray([r['row_id'] for r in validation]), **path_metadata)
                    endpoint_idx = np.flatnonzero(days > origin)
                    origin_idx = np.flatnonzero(days == origin)[0]
                    target, pred = conditional_increments(y_validation[:, origin_idx], y_validation[:, endpoint_idx], draws[:, :, endpoint_idx])
                    targets.append(target)
                    predicted.append(pred)
                    if curve_evaluation:
                        cumulative_targets.append(y_validation[:, endpoint_idx])
                        cumulative_predictions.append(draws[:, :, endpoint_idx])
                    diagnostic = getattr(model, 'last_conditional_diagnostics_',
                                         getattr(model, 'last_posterior_diagnostics_',
                                                 getattr(model, 'last_diagnostics_', None)))
                    if diagnostic is not None:
                        diagnostics[str(origin)] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                                    for k, v in diagnostic.items()}
                target = np.concatenate(targets, axis=1)
                pred = np.concatenate(predicted, axis=2)
                joint_cell_groups = [[i for i, cell in enumerate(cells) if cell[0] == origin]
                                     for origin in origins]
                result = evaluate_forecasts(target, pred, scales, groups=groups,
                    joint_cell_groups=joint_cell_groups,
                    bootstrap_replicates=config["experiment"].get("bootstrap_replicates", 0),
                    seed=effective_seeds["evaluation_bootstrap_seed"])
                result.update({"model": label, "model_config": model_config, "seed": seed,
                               "seed_roles": effective_seeds,
                               "elapsed_seconds": time.monotonic() - model_start, "evaluation_split": "validation",
                               "cells_origin_endpoint_days": cells,
                               "fit_diagnostics": getattr(model, 'fit_diagnostics_',
                                                          getattr(model, 'training_diagnostics_', None)),
                               "posterior_diagnostics_by_origin": diagnostics})
                if curve_evaluation:
                    cumulative_target = np.concatenate(cumulative_targets, axis=1)
                    cumulative_pred = np.concatenate(cumulative_predictions, axis=2)
                    curve_metrics = full_curve_metrics(target, pred, cumulative_target, cumulative_pred,
                        scales, cumulative_scales, cells, origins)
                    write_json(args.output / f'{label}_seed{seed}_full_curve_metrics.json', curve_metrics)
                    np.savez_compressed(args.output / f'{label}_seed{seed}_cumulative_predictions.npz',
                        targets=cumulative_target, samples=cumulative_pred, scales=cumulative_scales,
                        group_ids=groups, row_ids=np.asarray([r['row_id'] for r in validation]),
                        cells_origin_endpoint_days=np.asarray(cells))
                    result['full_curve_metrics_file'] = f'{label}_seed{seed}_full_curve_metrics.json'
                if feature_provenance is not None:
                    result['static_feature_variant'] = model_config['feature_variant']
                    result['static_features_sha256'] = sha256_file(args.output / 'static_features.json')
                if forest_origin0_provenance is not None:
                    result.update(forest_origin0_provenance)
                if model_config['kind'] == 'conditional_rate':
                    result['conditional_diagnostics_by_origin'] = result.pop('posterior_diagnostics_by_origin')
                    result.update(rate_report_metadata)
                    result.update(rate_history_metadata)
                    result.update(rate_coefficient_metadata)
                    result.update(score_metadata)
                    if tree_provenance is not None:
                        result.update(tree_static_mode=model_config['tree_static_mode'], tree_prior_provenance=tree_provenance)
                    mixing_law = ('Gaussian' if model_config.get('mixture_mode', 'single') == 'single'
                                  else 'two-component Gaussian mixture with shared covariance')
                    report_description = ('accumulated-Gaussian conditional report discrepancies' if half_life is None
                                          else f'finite-memory Gaussian conditional report discrepancies (half-life {half_life:g} days)')
                    result['predictive_interpretation'] = (
                        f'Learned conditional {mixing_law} mixing law over joint rate parameters, '
                        f'with {report_description}; '
                        'not a Bayesian well posterior or a sequentially consistent population model.'
                    )
                elif model_config['kind'] == 'neural' and model_config.get('mixture_components', 1) == 2:
                    result['predictive_interpretation'] = (
                        'Conditional two-component joint Gaussian mixture in asinh-transformed reports; '
                        'shared within-component covariance and one component per complete report trajectory; '
                        'a signed reported-trajectory law without latent rate, regime or Bayesian posterior claims.'
                    )
                elif model_config['kind'] == 'finite_forest_rate':
                    result.update(finite_prior_provenance=finite_provenance,
                        observation_law='conditional_gaussian_accumulated_bridge',
                        noise_mode=model_config['noise_mode'])
                    result['predictive_interpretation'] = (
                        'Exact categorical well posterior under a static forest-weighted finite prior over '
                        'complete joint positive-rate trajectories, with the same accumulated Gaussian '
                        'likelihood and conditional report sampler. Fitted population quantities and donor '
                        'curves are treated as fixed; finite tail outputs depend on the assumed basis.'
                    )
                elif model_config['kind'] == 'conditional_report':
                    result['conditional_diagnostics_by_origin'] = result.pop('posterior_diagnostics_by_origin')
                    result['predictive_interpretation'] = (
                        'Directly scored conditional joint Gaussian in asinh-transformed reports; '
                        'a signed reported-trajectory law without latent rate or Bayesian posterior claims.'
                    )
                stem = f"{label}_seed{seed}"
                np.savez_compressed(args.output / f"{stem}_predictions.npz", targets=target, samples=pred,
                                    scales=scales, group_ids=groups, row_ids=np.array([r['row_id'] for r in validation]))
                write_json(args.output / f"{stem}_metrics.json", result)
                all_results.append(result)
                print(json.dumps({"event": "model_done", "model": label, "seed": seed,
                                  "metrics": {k:v for k,v in result.items() if k in {"macro_scaled_wis", "macro_increment_r2"}},
                                  "elapsed_seconds": result['elapsed_seconds']}), flush=True)
        write_json(args.output / "comparison.json", {
            "status": "development_only", "source_sha256": manifest["source_sha256"],
            "configuration_sha256": manifest["config_sha256"], "results": all_results,
            "limitations": [("Retrospective static-attribute availability assumption; original feature timestamps remain unverified."
                             if feature_provenance is not None else "History-only pilot; static attribute availability remains unverified."),
                            "Bakken previously inspected; this is not independent external confirmation.",
                            "No calibration/test evaluation; this pilot does not establish model superiority."],
        })
        write_json(args.output / "completion.json", {"status": "completed", "elapsed_seconds": time.monotonic() - start,
            "artifacts": {p.name: sha256_file(p) for p in args.output.iterdir() if p.is_file()}})
    except BaseException as error:
        write_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
