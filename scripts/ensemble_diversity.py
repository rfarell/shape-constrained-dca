"""Score ensemble variants of the Bayesian model: mixture accuracy, member accuracy and diversity.

Each variant is a set of member draw files (same holdout). For every variant this
reports the equal-mixture macro scaled WIS (original15 / origin0 / all28), R2 and
coverage; the mean and range of member WIS; the ensemble gain (mean member WIS minus
mixture WIS, relative); and a disagreement index: the across-member standard deviation
of member medians divided by the mixture's 80% interval width, averaged over observed
cells. Optional fixed references (e.g. comparator mixtures) get paired whole-group
bootstrap intervals against every variant.

    python scripts/ensemble_diversity.py --variant name=glob ... --reference basis5=glob --output <dir>
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

from depletion_dev import CELLS, SCOPES, assemble, bootstrap_relative, coverage_table, macro_wis_cells
from joint_production.metrics import _json_ready


def macro(wis, cells):
    return float(np.nanmean(np.nanmean(wis[:, cells], axis=0)))


def r2_macro(targets, draws, cells):
    mean = draws.mean(axis=1)
    values = []
    for c in cells:
        for p in range(3):
            y, f = targets[:, c, p], mean[:, c, p]
            ok = np.isfinite(y)
            if ok.sum() >= 2 and np.var(y[ok]) > 0:
                values.append(1 - np.sum((y[ok] - f[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
    return float(np.mean(values))


def load(specs):
    out = {}
    for spec in specs:
        name, patterns = spec.split("=", 1)
        files = sorted(Path(f) for pattern in patterns.split(",") for f in glob.glob(pattern))
        if not files:
            raise ValueError(f"{name}: no files match {patterns}")
        out[name] = files
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    variants, references = load(args.variant), load(args.reference)
    first = next(iter(variants.values()))[0]
    holdout = dict(np.load(first.parent / "holdout.npz", allow_pickle=False))
    scales = holdout["scales"]
    results, wis_cache = {}, {}
    for group, items in (("variant", variants), ("reference", references)):
        for name, files in items.items():
            parts = []
            for f in files:
                other = np.load(f.parent / "holdout.npz", allow_pickle=False)
                if not np.array_equal(other["row_ids"], holdout["row_ids"]):
                    raise ValueError(f"{f}: holdout identities differ")
                parts.append(assemble(f, holdout))
            targets = parts[0][0]
            mixture = np.concatenate([p[1] for p in parts], axis=1)
            wis = macro_wis_cells(targets, mixture, scales)
            wis_cache[name] = wis
            cov = coverage_table(targets, mixture, CELLS)
            row = dict(group=group, members=[str(f) for f in files],
                       **{f"wis_{s}": macro(wis, c) for s, c in SCOPES.items()},
                       r2_original15=r2_macro(targets, mixture, SCOPES["original15"]),
                       coverage_all28={lv: cov[lv]["macro_all28"] for lv in cov})
            if len(parts) > 1:
                member = [macro(macro_wis_cells(targets, p[1], scales), SCOPES["original15"]) for p in parts]
                medians = np.stack([np.median(p[1], axis=1) for p in parts])  # [M,N,C,3]
                lo, hi = np.quantile(mixture, [0.1, 0.9], axis=1)
                observed = np.isfinite(targets)
                disagreement = np.where(observed, medians.std(axis=0) / np.maximum(hi - lo, 1e-12), np.nan)
                row.update(member_wis_original15=member, member_wis_mean=float(np.mean(member)),
                           ensemble_gain_pct=float(100 * (1 - row["wis_original15"] / np.mean(member))),
                           disagreement_index=float(np.nanmean(disagreement[:, SCOPES["original15"]])))
            results[name] = row
            print(json.dumps(dict(name=name, group=group, wis15=round(row["wis_original15"], 4), wis0=round(row["wis_origin0"], 4),
                                  r2=round(row["r2_original15"], 3), cov=[round(v, 3) for v in row["coverage_all28"].values()],
                                  member_mean=round(row.get("member_wis_mean", float("nan")), 4),
                                  gain_pct=round(row.get("ensemble_gain_pct", float("nan")), 2),
                                  disagreement=round(row.get("disagreement_index", float("nan")), 4))), flush=True)
    comparisons = {}
    for ref in references:
        for name in variants:
            c = {s: bootstrap_relative(wis_cache[name], wis_cache[ref], holdout["groups"], cells) for s, cells in SCOPES.items()}
            comparisons[f"{name}_vs_{ref}"] = c
            print(name, "vs", ref, {s: tuple(round(100 * v, 2) for v in (x["point"], x["lower95"], x["upper95"])) for s, x in c.items()}, flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diversity.json").write_text(json.dumps(_json_ready(dict(results=results, comparisons=comparisons)), indent=1))


if __name__ == "__main__":
    main()
