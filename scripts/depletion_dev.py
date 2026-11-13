"""Fit, forecast and score every model in the paper on one accepted split.

    python scripts/depletion_dev.py run --models bayes --split validation --set key=value ... --output <run>
    python scripts/depletion_dev.py repredict --model <run>/<label>_seed42_model.pt --split test \
        --protocol configs/depletion_arch_protocol.toml --label <label> --output <dir>
    python scripts/depletion_dev.py score --runs <run> [<run> ...] --mixtures --output <score>

``run`` fits on accepted TRAIN and forecasts the named split: ``validation``,
``calibration``, ``test`` (which requires a frozen protocol file), or ``internal``, the
TRAIN-internal development holdout (20% of TRAIN groups, seed 314159: 5,625 fitting and
1,392 holdout wells in 813 groups). Models are ``bayes`` (the shape-constrained model;
``--set`` overrides its settings), the comparators ``basis``, ``dca``, ``neural`` and
``forest``, and per-well decline curves ``pw_<arps|duong|sedm|scaling>_<bayes|map>``.
Every model receives the same 34 TRAIN-fitted static inputs, the same cutoff-masked
histories and forecast dates, and 256 draws per forecast. Saved draws are complete future
report endpoints after each forecast date; ``score`` uses ``joint_production.metrics``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from joint_production.artifacts import capture_run, write_json
from joint_production.metrics import _json_ready, evaluate_forecasts, fit_increment_scales

DAYS = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=float)
ORIGINS = [0, 30, 60, 90, 180, 360, 720]
CELLS = [(int(s), int(t)) for s in ORIGINS for t in DAYS if t > s]
SCOPES = {"original15": [i for i, (s, _) in enumerate(CELLS) if s >= 60],
          "origin0": [i for i, (s, _) in enumerate(CELLS) if s == 0],
          "all28": list(range(len(CELLS)))}
EXPECTED = dict(parent_wells=7017, parent_groups=4065, fit_wells=5625, holdout_wells=1392, holdout_groups=813)
CONFIG = ROOT / "configs" / "depletion_dev.toml"

COMPETITORS = {
    "basis": dict(kind="conditional_rate", rate_family="basis", basis_dictionary="fast",
                  history_rate_features="last_interval", epochs=60, hidden_width=128, batch_size=256,
                  train_samples=16, learning_rate=0.001, history_dropout=0.1, phase_dropout=0.1,
                  log_every=10),
    "dca": dict(kind="conditional_rate", rate_family="dca", basis_dictionary="fast",
                history_rate_features="last_interval", epochs=60, hidden_width=128, batch_size=256,
                train_samples=16, learning_rate=0.001, history_dropout=0.1, phase_dropout=0.1,
                log_every=10),
    "neural": dict(kind="neural", history_rate_features="last_interval", epochs=300, batch_size=1024,
                   rank=4, hidden_width=128, learning_rate=0.001, log_every=50),
    "forest": dict(kind="forest", min_samples_leaf=4, n_estimators=200, n_jobs=8),
}


def partition(records):
    """Hold out 20% of TRAIN groups, ordered by a salted hash of the group identifier."""
    groups = sorted({row["group_id"] for row in records}, key=lambda group: (
        hashlib.sha256(("earlier-history-v1|314159|" + group).encode("utf8")).digest(), group))
    held = set(groups[:len(groups) // 5])
    return np.asarray([row["group_id"] in held for row in records])


def load_internal():
    from joint_production.train_only import load_train_only
    from joint_production.static_features import StaticFeaturePreprocessor
    records, split, raw_hash, split_hash = load_train_only(
        ROOT / "data/processed/bakken_v1", ROOT / "data/raw/bakken.csv", ROOT / "data/manifest.json")
    reports = np.asarray([row["reports"] for row in records], dtype=np.float64)
    reports[:, 0] = 0.0
    holdout = partition(records)
    fit_rows = [row for row, held in zip(records, holdout) if not held]
    observed = dict(parent_wells=len(records), parent_groups=len({r["group_id"] for r in records}),
                    fit_wells=int((~holdout).sum()), holdout_wells=int(holdout.sum()),
                    holdout_groups=len({r["group_id"] for r, h in zip(records, holdout) if h}))
    if observed != EXPECTED:
        raise ValueError(f"TRAIN-internal partition differs: {observed}")
    processor = StaticFeaturePreprocessor.fit(fit_rows)
    features = processor.transform(records)
    fit_y = reports[~holdout]
    increments = np.stack([fit_y[:, list(DAYS).index(t)] - fit_y[:, list(DAYS).index(s)] for s, t in CELLS], axis=1)
    scales = fit_increment_scales(increments)
    return dict(fit_reports=fit_y, holdout_reports=reports[holdout], fit_features=features[~holdout],
                holdout_features=features[holdout], scales=scales,
                holdout_row_ids=np.asarray([r["row_id"] for r, h in zip(records, holdout) if h]),
                holdout_groups=np.asarray([r["group_id"] for r, h in zip(records, holdout) if h]),
                data_sha256=raw_hash, split_sha256=split_hash)


def load_split(split, protocol=None):
    """Fit on accepted TRAIN; evaluate one named accepted split (never decoded otherwise).

    ``internal`` is the TRAIN-internal development holdout. ``test`` additionally
    requires a protocol file whose [experiment] status is "frozen".
    """
    if split == "internal":
        return load_internal()
    if split not in {"validation", "calibration", "test"}:
        raise ValueError("split must be internal, validation, calibration or test")
    if split == "test":
        import tomllib
        if protocol is None or tomllib.loads(Path(protocol).read_text())["experiment"].get("status") != "frozen":
            raise ValueError("Test evaluation requires a frozen protocol file")
    from joint_production.data import load_dataset
    from joint_production.static_features import StaticFeaturePreprocessor
    dataset = load_dataset(ROOT / "data/processed/bakken_v1")
    train = [r for r in dataset["records"] if r["split"] == "train"]
    target = [r for r in dataset["records"] if r["split"] == split]
    del dataset
    train_y = np.asarray([r["reports"] for r in train], dtype=np.float64)
    target_y = np.asarray([r["reports"] for r in target], dtype=np.float64)
    train_y[:, 0] = 0.0
    target_y[:, 0] = 0.0
    processor = StaticFeaturePreprocessor.fit(train)
    increments = np.stack([train_y[:, list(DAYS).index(t)] - train_y[:, list(DAYS).index(s)] for s, t in CELLS], axis=1)
    from joint_production.artifacts import sha256_file
    return dict(fit_reports=train_y, holdout_reports=target_y, fit_features=processor.transform(train),
                holdout_features=processor.transform(target), scales=fit_increment_scales(increments),
                holdout_row_ids=np.asarray([r["row_id"] for r in target]),
                holdout_groups=np.asarray([r["group_id"] for r in target]),
                data_sha256=sha256_file(ROOT / "data/raw/bakken.csv"),
                split_sha256=sha256_file(ROOT / "data/processed/bakken_v1/split_manifest.json"))


def _value(text):
    if "," in text:  # comma-separated numbers, e.g. per-phase report_inflation=1.2,1.0,1.0
        return [_value(part) for part in text.split(",")]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(text.lower(), text)


def build(kind, seed, device, overrides):
    if kind == "bayes":
        from joint_production.depletion_bayes import DepletionBayesForecaster
        settings = dict(seed=seed, device=device, **overrides)
        return DepletionBayesForecaster(**settings), settings
    if kind.startswith("pw_"):
        # Industry per-well DCA benchmarks: pw_<arps|sedm|duong|scaling>_<bayes|map>; CPU float64.
        from joint_production.perwell_dca import PerWellDCAForecaster
        _, family, mode = kind.split("_")
        settings = dict(family=family, mode=mode, seed=seed, device="cpu", **overrides)
        return PerWellDCAForecaster(**settings), settings
    settings = dict(COMPETITORS[kind], **overrides)
    model_kind = settings.pop("kind")
    if model_kind == "conditional_rate":
        from joint_production.conditional_rates import ConditionalRateForecaster
        model = ConditionalRateForecaster(seed=seed, device=device, **settings)
    elif model_kind == "neural":
        from joint_production.neural import ContextualGaussianNeuralBaseline
        model = ContextualGaussianNeuralBaseline(seed=seed, device=device, **settings)
    else:
        from joint_production.baselines import ProbabilisticTrajectoryForest
        model = ProbabilisticTrajectoryForest(random_state=seed, **settings)
    return model, dict(settings, kind=model_kind)


def run(args):
    data = load_split(args.split, args.protocol)
    output = Path(args.output)
    seeds = [int(s) for s in args.seeds.split(",")]
    manifest = capture_run(ROOT, output, config=CONFIG, data=ROOT / "data/raw/bakken.csv",
                           split=ROOT / "data/processed/bakken_v1/split_manifest.json", seeds=seeds)
    overrides = {key: _value(value) for key, value in (item.split("=", 1) for item in args.set)}
    manifest.update(evaluation_split=("train_internal_holdout" if args.split == "internal" else args.split),
                    partition_expected=EXPECTED if args.split == "internal" else None, protocol=args.protocol,
                    overrides=overrides, models=args.models, samples=args.samples, label=args.label)
    write_json(output / f"manifest_{args.label or 'run'}.json", manifest, exclusive=False)
    np.savez(output / "holdout.npz", reports=data["holdout_reports"], row_ids=data["holdout_row_ids"],
             groups=data["holdout_groups"], scales=data["scales"], days=DAYS)
    for kind in args.models.split(","):
        for seed in seeds:
            label = f"{args.label or kind}_seed{seed}"
            start = time.monotonic()
            model, settings = build(kind, seed, args.device, overrides)
            print(json.dumps(dict(event="fit_start", model=label, settings=settings)), flush=True)
            if kind == "forest":
                model.fit(data["fit_reports"], DAYS, origins=ORIGINS, covariates=data["fit_features"])
            elif kind == "bayes":
                model.fit(data["fit_reports"], DAYS, covariates=data["fit_features"])
            else:
                model.fit(data["fit_reports"], DAYS, origins=ORIGINS, covariates=data["fit_features"])
            fit_seconds = time.monotonic() - start
            if hasattr(model, "save"):
                model.save(output / f"{label}_model.pt")
            else:  # the forest has no torch state; keep it for later forecasts
                with open(output / f"{label}_model.pkl", "wb") as handle:
                    pickle.dump(model, handle, protocol=5)
            arrays, diagnostics = {}, {}
            for origin in ORIGINS:
                history = data["holdout_reports"].copy()
                history[:, DAYS > origin] = np.nan
                rng = np.random.default_rng(2026 * 10000 + origin)
                if kind == "bayes":
                    result = model.predict_distribution(history, origin, args.samples, rng, data["holdout_features"])
                    draws = result["report_cumulative"]
                    diagnostics[str(origin)] = model.last_diagnostics_
                    if origin in (0, 90, 360):
                        arrays[f"latent_o{origin}"] = result["latent_parameters"].astype(np.float32)
                else:
                    draws = model.predict_samples(history, origin=origin, n_samples=args.samples, rng=rng,
                                                  covariates=data["holdout_features"])
                future = DAYS > origin
                if not np.isfinite(draws[..., future, :]).all():
                    raise ValueError(f"{label}: non-finite forecast draws at origin {origin}")
                arrays[f"draws_o{origin}"] = draws[:, :, future, :].astype(np.float32)
            np.savez(output / f"{label}_draws.npz", **arrays)
            write_json(output / f"{label}_fit.json", _json_ready(dict(
                model=label, kind=kind, seed=seed, settings=settings, fit_seconds=fit_seconds,
                total_seconds=time.monotonic() - start, prediction_diagnostics=diagnostics,
                fit_diagnostics=getattr(model, "fit_diagnostics_", None))),
                exclusive=False)
            print(json.dumps(dict(event="done", model=label, seconds=round(time.monotonic() - start, 1))), flush=True)


def repredict(args):
    """Reissue holdout forecasts from a saved Bayesian model with changed inference/calibration settings."""
    from joint_production.depletion_bayes import DepletionBayesForecaster
    data = load_split(args.split, args.protocol)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = DepletionBayesForecaster.load(args.model, device=args.device)
    overrides = {key: _value(value) for key, value in (item.split("=", 1) for item in args.set)}
    model.config.update(overrides)
    np.savez(output / "holdout.npz", reports=data["holdout_reports"], row_ids=data["holdout_row_ids"],
             groups=data["holdout_groups"], scales=data["scales"], days=DAYS)
    arrays, diagnostics = {}, {}
    for origin in ORIGINS:
        history = data["holdout_reports"].copy()
        history[:, DAYS > origin] = np.nan
        # Prediction stream: the original rule is shared by all fits (2026*10000 + origin); --per-seed-stream gives
        # each fitted seed its own stream so equal mixtures of fits pool independent Monte Carlo draws.
        stream = 2026 * 10000 + origin + (1000 * int(model.config["seed"]) if args.per_seed_stream else 0)
        result = model.predict_distribution(history, origin, args.samples, np.random.default_rng(stream),
                                            data["holdout_features"])
        diagnostics[str(origin)] = model.last_diagnostics_
        arrays[f"draws_o{origin}"] = result["report_cumulative"][:, :, DAYS > origin, :].astype(np.float32)
    np.savez(output / f"{args.label}_seed{model.config['seed']}_draws.npz", **arrays)
    write_json(output / f"{args.label}_repredict.json", _json_ready(dict(source_model=str(args.model), overrides=overrides,
               prediction_diagnostics=diagnostics,
               evaluation_split=("train_internal_holdout" if args.split == "internal" else args.split))), exclusive=False)
    print(json.dumps(dict(event="done", label=args.label)), flush=True)


# ----- scoring -----------------------------------------------------------------

def assemble(draw_file, holdout):
    """Targets [N,28,3] and draws [N,S,28,3] as increments from the observed origin."""
    reports = holdout["reports"]
    with np.load(draw_file) as bank:
        parts, targets = [], []
        for origin in ORIGINS:
            future = DAYS > origin
            origin_report = reports[:, list(DAYS).index(origin)]
            parts.append(bank[f"draws_o{origin}"].astype(np.float64) - origin_report[:, None, None, :])
            targets.append(reports[:, future] - origin_report[:, None, :])
    return np.concatenate(targets, axis=1), np.concatenate(parts, axis=2)


def coverage_table(targets, draws, cells):
    """Coverage and scaled width at 80/90/95 by origin and phase (history regimes)."""
    observed = np.isfinite(targets)
    out = {}
    for level in (0.8, 0.9, 0.95):
        lower, upper = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=1, method="linear")
        hit = np.where(observed, (targets >= lower) & (targets <= upper), np.nan)
        by_origin = {}
        for origin in ORIGINS:
            index = [i for i, (s, _) in enumerate(cells) if s == origin]
            by_origin[str(origin)] = np.nanmean(hit[:, index], axis=(0, 1)).tolist()
        out[str(level)] = dict(by_origin_phase=by_origin,
                               macro_original15=float(np.nanmean(np.nanmean(hit[:, SCOPES["original15"]], axis=0))),
                               macro_all28=float(np.nanmean(np.nanmean(hit, axis=0))),
                               phase_all28=np.nanmean(hit, axis=(0, 1)).tolist())
    return out


def macro_wis_cells(targets, draws, scales):
    from joint_production.metrics import weighted_interval_score
    return weighted_interval_score(targets, draws) / scales  # [N, cells, 3]


def bootstrap_relative(candidate, reference, groups, cells, replicates=1000, seed=42):
    """Paired whole-group bootstrap of 1 - macroWIS(candidate)/macroWIS(reference)."""
    from joint_production.comparison import whole_group_weights
    weights = whole_group_weights(groups, replicates, seed).astype(float)
    def macro(values, w):
        v = values[:, cells]
        mask = np.isfinite(v)
        num = np.einsum("bn,ncp->bcp", w, np.where(mask, v, 0))
        den = np.einsum("bn,ncp->bcp", w, mask.astype(float))
        return (num / den).mean(axis=(1, 2))
    point = 1 - np.nanmean(np.nanmean(candidate[:, cells], 0)) / np.nanmean(np.nanmean(reference[:, cells], 0))
    draws = 1 - macro(candidate, weights) / macro(reference, weights)
    return dict(point=float(point), lower95=float(np.quantile(draws, 0.025)), upper95=float(np.quantile(draws, 0.975)))


def _evaluation_split(runs):
    """The split named by the runs' own records (all runs share one holdout)."""
    found = set()
    for run in runs:
        for path in Path(run).glob("*.json"):
            try:
                value = json.loads(path.read_text()).get("evaluation_split")
            except (ValueError, AttributeError):
                continue
            if isinstance(value, str):
                found.add(value)
    return found.pop() if len(found) == 1 else "unrecorded"


def score(args):
    runs = [Path(run) for run in args.runs]
    holdout = dict(np.load(runs[0] / "holdout.npz", allow_pickle=False))
    for run in runs[1:]:
        other = np.load(run / "holdout.npz", allow_pickle=False)
        if not np.array_equal(other["row_ids"], holdout["row_ids"]):
            raise ValueError("Runs disagree on holdout identities")
    files = sorted(f for run in runs for f in run.glob("*_draws.npz"))
    scales = holdout["scales"]
    results, wis_cache = {}, {}
    mixtures = {}
    for file in files:
        label = file.name.removesuffix("_draws.npz")
        targets, draws = assemble(file, holdout)
        family = label.rsplit("_seed", 1)[0]
        mixtures.setdefault(family, []).append(file)
        results[label] = summarize(targets, draws, scales, holdout["groups"])
        wis_cache[label] = macro_wis_cells(targets, draws, scales)
        print(json.dumps(dict(model=label, **{k: results[label][k] for k in ("wis_original15", "r2_original15", "wis_origin0", "cov90_all28")})), flush=True)
    if args.mixtures:
        for family, members in mixtures.items():
            if len(members) < 2:
                continue
            parts = [assemble(f, holdout) for f in members]
            targets = parts[0][0]
            draws = np.concatenate([p[1] for p in parts], axis=1)
            label = f"{family}_mix{len(members)}"
            results[label] = summarize(targets, draws, scales, holdout["groups"])
            wis_cache[label] = macro_wis_cells(targets, draws, scales)
            print(json.dumps(dict(model=label, **{k: results[label][k] for k in ("wis_original15", "r2_original15", "wis_origin0", "cov90_all28")})), flush=True)
    comparisons = {}
    if args.reference:
        for label in results:
            for reference in args.reference.split(","):
                if label == reference or reference not in wis_cache:
                    continue
                comparisons[f"{label}_vs_{reference}"] = {
                    scope: bootstrap_relative(wis_cache[label], wis_cache[reference], holdout["groups"], cells)
                    for scope, cells in SCOPES.items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "scores.json", _json_ready(dict(results=results, comparisons=comparisons,
               runs=[str(r) for r in runs], evaluation_split=_evaluation_split(runs))), exclusive=False)
    print(json.dumps(comparisons, indent=1))


def summarize(targets, draws, scales, groups):
    out = {}
    for scope, cells in SCOPES.items():
        origins = sorted({CELLS[i][0] for i in cells})
        groups_by_origin = [[j for j, i in enumerate(cells) if CELLS[i][0] == o] for o in origins]
        metrics = evaluate_forecasts(targets[:, cells], draws[:, :, cells], scales[cells],
                                     joint_cell_groups=groups_by_origin)
        out[scope] = metrics
        out[f"wis_{scope}"] = metrics["macro_scaled_wis"]
        out[f"r2_{scope}"] = metrics["macro_increment_r2"]
    coverage = coverage_table(targets, draws, CELLS)
    out["coverage"] = coverage
    out["cov90_all28"] = coverage["0.9"]["macro_all28"]
    return out


def _cap_gpu_memory(device):
    """On unified-memory GPUs (GB10) CUDA allocations consume system RAM; cap each process."""
    if str(device).startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("JPF_GPU_MEMORY_FRACTION", "0.25")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--models", required=True)
    run_parser.add_argument("--seeds", default="42")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--samples", type=int, default=256)
    run_parser.add_argument("--label", default="")
    run_parser.add_argument("--set", action="append", default=[], help="bayes override key=value")
    run_parser.add_argument("--split", default="internal")
    run_parser.add_argument("--protocol", default=None)
    re_parser = sub.add_parser("repredict")
    re_parser.add_argument("--model", required=True)
    re_parser.add_argument("--label", required=True)
    re_parser.add_argument("--output", required=True)
    re_parser.add_argument("--device", default="cpu")
    re_parser.add_argument("--samples", type=int, default=256)
    re_parser.add_argument("--set", action="append", default=[])
    re_parser.add_argument("--split", default="internal")
    re_parser.add_argument("--protocol", default=None)
    re_parser.add_argument("--per-seed-stream", action="store_true")
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--runs", nargs="+", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--reference", default="")
    score_parser.add_argument("--mixtures", action="store_true")
    args = parser.parse_args(argv)
    _cap_gpu_memory(getattr(args, "device", "cpu"))
    {"run": run, "score": score, "repredict": repredict}[args.command](args)


if __name__ == "__main__":
    main()
