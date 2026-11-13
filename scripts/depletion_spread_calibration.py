"""Equal post-hoc calibration opportunity for comparators: one spread scalar per family.

The Bayesian forecaster's calibration scalars (prior/report inflation) were chosen
on the CALIBRATION split by macro scaled WIS over all 28 cells of the equal
five-seed mixture. This gives every comparator family the same opportunity: each
seed's draws are scaled about their per-(well, endpoint, phase) median,
d' = m + c (d - m), with c on a fixed grid chosen by the same criterion on the
calibration split. The selected c is then applied unchanged to other splits. The
transform keeps each sample's cross-endpoint/phase coupling but, for c > 1, need not
keep sampled cumulatives monotone; it is a marginal recalibration, not a model.

    python scripts/depletion_spread_calibration.py fit --draws <cal_competitors> --output <dir>
    python scripts/depletion_spread_calibration.py apply --spread <dir>/spread.json --draws <split_competitors> --output <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from depletion_dev import ORIGINS, SCOPES, assemble
from joint_production.metrics import _json_ready, weighted_interval_score

SPREAD_GRID = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)


def spread(draws, c, axis=1):
    median = np.median(draws, axis=axis, keepdims=True)
    return median + c * (draws - median)


def families(directory, names):
    out = {}
    for file in sorted(Path(directory).glob("*_draws.npz")):
        family = file.name.removesuffix("_draws.npz").rsplit("_seed", 1)[0]
        if names is None or family in names:
            out.setdefault(family, []).append(file)
    return out


def fit(args):
    directory = Path(args.draws)
    holdout = dict(np.load(directory / "holdout.npz", allow_pickle=False))
    grid = {}
    for family, files in families(directory, args.families and args.families.split(",")).items():
        parts = [assemble(f, holdout) for f in files]
        targets = parts[0][0]
        rows = []
        for c in SPREAD_GRID:
            draws = np.concatenate([spread(p[1], c) for p in parts], axis=1)
            scaled = weighted_interval_score(targets, draws) / holdout["scales"]
            row = {scope: float(np.nanmean(np.nanmean(scaled[:, cells], axis=0))) for scope, cells in SCOPES.items()}
            observed = np.isfinite(targets)
            lo, hi = np.quantile(np.where(observed[:, None], draws, 0), [0.05, 0.95], axis=1)
            row["coverage_0.9"] = float(np.nanmean(np.nanmean(np.where(observed, (targets >= lo) & (targets <= hi), np.nan), axis=0)))
            row["phase_all28"] = np.nanmean(np.nanmean(scaled, axis=0), axis=0).tolist()
            rows.append(dict(c=c, **row))
            print(json.dumps(dict(family=family, **rows[-1])), flush=True)
        if args.per_phase:  # spread acts phase by phase, so per-phase selection is exact
            chosen = [min(rows, key=lambda r: r["phase_all28"][k])["c"] for k in range(3)]
        else:
            chosen = min(rows, key=lambda r: r["all28"])["c"]
        grid[family] = dict(seeds=len(files), rows=rows, selected=chosen)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "spread.json").write_text(json.dumps(_json_ready(dict(
        split="calibration", source=str(directory), wells=int(len(holdout["reports"])),
        objective="macro scaled WIS, all 28 cells, equal mixture" + (", per phase" if args.per_phase else ""),
        grid_values=SPREAD_GRID, families=grid)), indent=1))
    print(json.dumps({family: g["selected"] for family, g in grid.items()}), flush=True)


def apply(args):
    selected = {family: g["selected"] for family, g in json.loads(Path(args.spread).read_text())["families"].items()}
    directory, output = Path(args.draws), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(directory / "holdout.npz", output / "holdout.npz")
    for family, files in families(directory, set(selected)).items():
        for file in files:
            seed = file.name.removesuffix("_draws.npz").rsplit("_seed", 1)[1]
            with np.load(file) as bank:
                c = np.asarray(selected[family], dtype=np.float64)  # scalar or per phase (last axis)
                arrays = {f"draws_o{o}": spread(bank[f"draws_o{o}"].astype(np.float64), c).astype(np.float32)
                          for o in ORIGINS}
            np.savez(output / f"{family}_spread_seed{seed}_draws.npz", **arrays)
    (output / "spread_receipt.json").write_text(json.dumps(dict(source=str(directory), spread=str(args.spread),
                                                                selected=selected), indent=1))
    print(json.dumps(selected), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--draws", required=True)
    p.add_argument("--families", default="basis,dca,neural,forest")
    p.add_argument("--per-phase", action="store_true")
    p.add_argument("--output", required=True)
    p.set_defaults(func=fit)
    p = sub.add_parser("apply")
    p.add_argument("--spread", required=True)
    p.add_argument("--draws", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=apply)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
