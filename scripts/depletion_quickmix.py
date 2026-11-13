"""Fast development scoring of equal mixtures: WIS, R2, coverage and paired bootstrap only.

The full ``depletion_dev.py score`` also computes energy/variogram scores for every
file, which takes hours for many runs. For iteration this scores only named
mixtures (each a list of draw files, pooled equally), with the same frozen cells,
scales and whole-group paired bootstrap. Not a substitute for the full report.

    python scripts/depletion_quickmix.py --mix name=glob[,glob...] ... --reference name --output <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from depletion_dev import SCOPES, assemble, bootstrap_relative, coverage_table, macro_wis_cells
from joint_production.metrics import _json_ready


def r2_macro(targets, draws, cells):
    """Macro increment R2 of the sample mean over (cell, phase), as in the main scorer."""
    mean = draws.mean(axis=1)
    values = []
    for c in cells:
        for p in range(3):
            y, f = targets[:, c, p], mean[:, c, p]
            ok = np.isfinite(y)
            if ok.sum() < 2 or np.var(y[ok]) == 0:
                continue
            values.append(1 - np.sum((y[ok] - f[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
    return float(np.mean(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", action="append", required=True, help="name=glob[,glob...] of *_draws.npz files")
    parser.add_argument("--reference", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    mixtures = {}
    for spec in args.mix:
        name, patterns = spec.split("=", 1)
        files = sorted(f for pattern in patterns.split(",") for f in glob.glob(pattern))
        if not files:
            raise ValueError(f"{name}: no files")
        mixtures[name] = [Path(f) for f in files]
    holdout = dict(np.load(mixtures[next(iter(mixtures))][0].parent / "holdout.npz", allow_pickle=False))
    results, wis = {}, {}
    from depletion_dev import CELLS
    for name, files in mixtures.items():
        for f in files:
            other = np.load(f.parent / "holdout.npz", allow_pickle=False)
            if not np.array_equal(other["row_ids"], holdout["row_ids"]):
                raise ValueError(f"{f}: holdout identities differ")
        parts = [assemble(f, holdout) for f in files]
        targets = parts[0][0]
        draws = np.concatenate([p[1] for p in parts], axis=1)
        wis[name] = macro_wis_cells(targets, draws, holdout["scales"])
        coverage = coverage_table(targets, draws, CELLS)
        results[name] = dict(members=[str(f) for f in files],
                             **{f"wis_{s}": float(np.nanmean(np.nanmean(wis[name][:, c], axis=0))) for s, c in SCOPES.items()},
                             r2_original15=r2_macro(targets, draws, SCOPES["original15"]),
                             coverage_all28={lvl: coverage[lvl]["macro_all28"] for lvl in coverage},
                             phase_wis_original15=np.nanmean(wis[name][:, SCOPES["original15"]], axis=(0, 1)).tolist(),
                             wis_by_origin_phase={str(o): np.nanmean(np.nanmean(wis[name][:, [i for i, (s0, _) in enumerate(CELLS) if s0 == o]], axis=0), axis=0).tolist()
                                                  for o in sorted({s0 for s0, _ in CELLS})})
        r = results[name]
        print(json.dumps(dict(model=name, n=len(files), wis15=round(r["wis_original15"], 4), r2=round(r["r2_original15"], 3),
                              wis0=round(r["wis_origin0"], 4), cov=[round(v, 3) for v in r["coverage_all28"].values()],
                              phase=[round(v, 3) for v in r["phase_wis_original15"]])), flush=True)
    comparisons = {}
    for reference in filter(None, args.reference.split(",")):
        for name in mixtures:
            if name != reference:
                comparisons[f"{name}_vs_{reference}"] = {s: bootstrap_relative(wis[name], wis[reference], holdout["groups"], c)
                                                         for s, c in SCOPES.items()}
                c = comparisons[f"{name}_vs_{reference}"]
                print(name, "vs", reference, {s: tuple(round(100 * v, 2) for v in (x["point"], x["lower95"], x["upper95"]))
                                               for s, x in c.items()}, flush=True)
    print("WIS by origin (oil/gas/water):")
    for name in mixtures:
        print(name, {o: [round(v, 3) for v in vals] for o, vals in results[name]["wis_by_origin_phase"].items()})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "quickmix.json").write_text(json.dumps(_json_ready(dict(results=results, comparisons=comparisons)), indent=1))


if __name__ == "__main__":
    main()
