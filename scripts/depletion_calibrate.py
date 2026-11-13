"""Fit the two calibration scalars of the Bayesian forecaster on the CALIBRATION split.

Prior inflation nu (prior covariance x nu) and report inflation kappa (report
discrepancy clock x kappa^2) are chosen on a fixed grid to minimize the macro
scaled WIS over all 28 origin-endpoint cells of the equal mixture of the supplied
fits. Both preserve every structural property (rates, integrals, EUR, joint
sampling, exact conditioning given the tempered law). No validation or test
record is decoded.

    python scripts/depletion_calibrate.py --models <fit>/x_seed42_model.pt ... --output build/<cal>
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from depletion_dev import CELLS, DAYS, ORIGINS, SCOPES, _cap_gpu_memory, load_split
from joint_production.depletion_bayes import DepletionBayesForecaster
from joint_production.metrics import _json_ready, weighted_interval_score

NU_GRID = (1.0, 1.25, 1.5, 1.75, 2.0)
KAPPA_GRID = (1.0, 1.15, 1.3)
PHASE_KAPPA_GRID = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)  # --per-phase: report inflation chosen per phase


def mixture_scores(models, data, nu, kappa, samples):
    reports = data["holdout_reports"]
    targets, parts = [], []
    for origin in ORIGINS:
        history = reports.copy()
        history[:, DAYS > origin] = np.nan
        draws = []
        for model in models:
            model.config.update(prior_inflation=nu, report_inflation=kappa)
            out = model.predict_distribution(history, origin, samples, np.random.default_rng(2026 * 10000 + origin),
                                             data["holdout_features"])["report_cumulative"]
            draws.append(out[:, :, DAYS > origin])
        index = list(DAYS).index(origin)
        base = reports[:, index]
        parts.append(np.concatenate(draws, axis=1) - base[:, None, None, :])
        targets.append(reports[:, DAYS > origin] - base[:, None, :])
    target, draw = np.concatenate(targets, 1), np.concatenate(parts, 2)
    scaled = weighted_interval_score(target, draw) / data["scales"]
    result = {scope: float(np.nanmean(np.nanmean(scaled[:, cells], axis=0))) for scope, cells in SCOPES.items()}
    result["phase_all28"] = np.nanmean(np.nanmean(scaled, axis=0), axis=0).tolist()
    observed = np.isfinite(target)
    for level in (0.8, 0.9, 0.95):
        lo, hi = np.quantile(np.where(observed[:, None], draw, 0), [(1 - level) / 2, (1 + level) / 2], axis=1)
        hit = np.where(observed, (target >= lo) & (target <= hi), np.nan)
        result[f"coverage_{level}"] = float(np.nanmean(np.nanmean(hit, axis=0)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--per-phase", action="store_true",
                        help="fix prior inflation at --nu and choose report inflation per phase; report noise is "
                             "independent across phases given the latent draw, so per-phase selection is exact")
    parser.add_argument("--nu", type=float, default=1.0)
    args = parser.parse_args(argv)
    _cap_gpu_memory(args.device)
    data = load_split("calibration")
    models = [DepletionBayesForecaster.load(path, device=args.device) for path in args.models]
    if args.per_phase:
        grid = []
        for kappa in PHASE_KAPPA_GRID:
            scores = mixture_scores(models, data, args.nu, kappa, args.samples)
            grid.append(dict(nu=args.nu, kappa=kappa, **scores))
            print(json.dumps(grid[-1]), flush=True)
        selected = [min(grid, key=lambda row: row["phase_all28"][k])["kappa"] for k in range(3)]
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "calibration.json").write_text(json.dumps(_json_ready(dict(
            split="calibration", wells=int(len(data["holdout_reports"])), models=args.models, mode="per-phase report inflation",
            objective="macro scaled WIS per phase, all 28 cells, equal mixture", grid=grid,
            selected=dict(prior_inflation=args.nu, report_inflation=selected))), indent=1))
        print(json.dumps(dict(selected=dict(prior_inflation=args.nu, report_inflation=selected))), flush=True)
        return
    grid = []
    for nu, kappa in itertools.product(NU_GRID, KAPPA_GRID):
        scores = mixture_scores(models, data, nu, kappa, args.samples)
        grid.append(dict(nu=nu, kappa=kappa, **scores))
        print(json.dumps(grid[-1]), flush=True)
    best = min(grid, key=lambda row: row["all28"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "calibration.json").write_text(json.dumps(_json_ready(dict(
        split="calibration", wells=int(len(data["holdout_reports"])), models=args.models, objective="macro scaled WIS, all 28 cells, equal mixture",
        grid=grid, selected=dict(prior_inflation=best["nu"], report_inflation=best["kappa"]))), indent=1))
    print(json.dumps(dict(selected=dict(prior_inflation=best["nu"], report_inflation=best["kappa"]))), flush=True)


if __name__ == "__main__":
    main()
