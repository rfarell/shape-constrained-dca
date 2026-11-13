"""Hindcast reliability in industry P-notation and reported phase-ratio scores.

For each model (equal mixtures of all seeds in the supplied runs; single-seed
families as they are), forecast origin and phase, on the conditional increment
from the origin to the last endpoint (day 1080) and over all observed cells:

  * P10-P90 coverage (central 80%) and exceedance frequencies of P90/P50/P10 in
    the petroleum (exceedance) convention: P90 is the 10th percentile, so the
    actual should exceed it 90% of the time (Gong et al. 2014, SPE J 19(6)).
  * median relative width (P10 - P90) / actual over wells with positive actuals.

Reported ratios over the origin->1080 interval, from the same joint draw:
GOR = gas/oil (MCF/BBL) and water cut = water/(oil + water), with invalid ratios
(nonpositive denominators) excluded and counted. These are ratios of REPORTED
increments, not latent ratios. Scores: 80% coverage, WIS of log GOR and of
logit water cut (unit-free), and median absolute log error of the P50.

    python scripts/depletion_hindcast.py --runs <dir> ... --output <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from depletion_dev import CELLS, ORIGINS, assemble
from joint_production.metrics import _json_ready, interval_score, sample_derived_quantities

PHASES = ("oil", "gas", "water")
LAST = [i for i, (s, t) in enumerate(CELLS) if t == 1080]  # one cell per origin, in ORIGINS order


def reliability(targets, draws):
    """targets [N,C,3], draws [N,S,C,3] -> dict of arrays [C,3]."""
    observed = np.isfinite(targets)
    p90, p50, p10 = np.quantile(draws, [0.1, 0.5, 0.9], axis=1)  # exceedance convention
    def rate(hit):
        return np.nanmean(np.where(observed, hit, np.nan), axis=0)
    positive = observed & (targets > 0)
    relative_width = np.where(positive, (p10 - p90) / np.where(positive, targets, 1.0), np.nan)
    return dict(p10_p90_coverage=rate((targets >= p90) & (targets <= p10)),
                exceed_p90=rate(targets > p90), exceed_p50=rate(targets > p50), exceed_p10=rate(targets > p10),
                median_relative_width=np.nanmedian(relative_width, axis=0), wells=observed.sum(axis=0))


def ratio_scores(targets, draws):
    """Reported GOR and water cut over origin->1080. targets [N,O,3], draws [N,S,O,3]."""
    actual = sample_derived_quantities(targets)
    sampled = sample_derived_quantities(draws)
    out = {}
    for name, transform in (("gor_mcf_per_bbl", np.log), ("water_cut", lambda p: np.log(p) - np.log1p(-p))):
        y, x = actual[name], sampled[name]
        upper = 1.0 if name == "water_cut" else np.inf  # transforms need the open support
        y = np.where((y > 0) & (y < upper), y, np.nan)
        x = np.where((x > 0) & (x < upper), x, np.nan)
        valid_draws = np.isfinite(x).mean(axis=1)  # fraction of usable draws per well/origin
        usable = np.isfinite(y) & (valid_draws >= 0.5)
        x = np.where(usable[:, None], x, np.nan)
        lo, med, hi = np.nanquantile(x, [0.1, 0.5, 0.9], axis=1)
        ty = transform(np.where(usable, y, np.nan))
        q = np.nanquantile(transform(x), [0.025, 0.1, 0.5, 0.9, 0.975], axis=1)
        wis = (0.5 * np.abs(ty - q[2]) + 0.1 * interval_score(ty, q[1], q[3], 0.2)
               + 0.025 * interval_score(ty, q[0], q[4], 0.05)) / 2.5
        out[name] = dict(coverage80=np.nanmean(np.where(usable, (y >= lo) & (y <= hi), np.nan), axis=0),
                         wis_transformed=np.nanmean(np.where(usable, wis, np.nan), axis=0),
                         median_abs_log_error_p50=np.nanmedian(np.where(usable, np.abs(np.log(med) - np.log(y)), np.nan), axis=0),
                         usable_wells=usable.sum(axis=0), invalid_draw_fraction=float(1 - np.nanmean(valid_draws)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--families", default=None, help="comma-separated family labels to keep")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    runs = [Path(r) for r in args.runs]
    holdout = dict(np.load(runs[0] / "holdout.npz", allow_pickle=False))
    families = {}
    for run in runs:
        other = np.load(run / "holdout.npz", allow_pickle=False)
        if not np.array_equal(other["row_ids"], holdout["row_ids"]):
            raise ValueError("Runs disagree on holdout identities")
        for file in sorted(run.glob("*_draws.npz")):
            family = file.name.removesuffix("_draws.npz").rsplit("_seed", 1)[0]
            if args.families is None or family in args.families.split(","):
                families.setdefault(family, []).append(file)
    results, rows = {}, []
    for family, files in families.items():
        parts = [assemble(f, holdout) for f in files]
        targets = parts[0][0]
        draws = np.concatenate([p[1] for p in parts], axis=1)
        label = f"{family}_mix{len(files)}" if len(files) > 1 else family
        last = reliability(targets[:, LAST], draws[:, :, LAST])
        every = reliability(targets, draws)
        ratios = ratio_scores(targets[:, LAST], draws[:, :, LAST])
        results[label] = dict(to1080_by_origin=last, all_cells_mean={k: np.nanmean(v, axis=0) for k, v in every.items()},
                              ratios_to1080_by_origin=ratios, seeds=len(files), draws=int(draws.shape[1]))
        r = last
        rows.append(f"| {label} | " + " | ".join(
            f"{100 * np.nanmean(r[k]):.1f}" for k in ("p10_p90_coverage", "exceed_p90", "exceed_p50", "exceed_p10"))
            + f" | {np.nanmean(r['median_relative_width']):.2f} | "
            + f"{100 * np.nanmean(ratios['gor_mcf_per_bbl']['coverage80']):.1f} | {np.nanmean(ratios['gor_mcf_per_bbl']['wis_transformed']):.3f} | "
            + f"{100 * np.nanmean(ratios['water_cut']['coverage80']):.1f} | {np.nanmean(ratios['water_cut']['wis_transformed']):.3f} |")
        print(rows[-1], flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    header = ("| Model | P10-P90 cov | >P90 | >P50 | >P10 | rel. width | GOR cov80 | GOR WIS(log) | WC cov80 | WC WIS(logit) |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    (output / "hindcast.md").write_text(header + "\n" + "\n".join(rows) + "\n\nIncrement origin->1080, mean over origins "
                                        f"{ORIGINS} and phases {PHASES}; nominal 80 / 90 / 50 / 10.\n")
    (output / "hindcast.json").write_text(json.dumps(_json_ready(dict(runs=[str(r) for r in runs], origins=ORIGINS,
                                                                      phases=PHASES, results=results)), indent=1))
    print(header)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
