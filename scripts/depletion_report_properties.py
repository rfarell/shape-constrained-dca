"""Report-level properties of sampled forecast trajectories, for every model family.

Neural and forest comparators issue report trajectories only, so latent-rate
properties cannot be audited for them. This compares what every model's
sampled REPORTED cumulatives imply, against the same statistics of the observed
reports of the same wells (computed only where the needed reports exist):

  * negative interval volume: a sampled cumulative that decreases between
    successive report days (observed Bakken cumulatives come from nonnegative
    monthly volumes, so the observed frequency is a reference);
  * oil interval-average rate rising between successive intervals after day 180
    (180-360, 360-720, 720-1080);
  * gas interval rate rising while oil interval rate falls after day 180.

Fractions are over (well, draw) trajectories at a given origin; observed
fractions are over wells with all needed reports. Mixtures pool all seeds.

    python scripts/depletion_report_properties.py --runs <dir> ... --origins 0 180 --output <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from depletion_dev import DAYS
from joint_production.metrics import _json_ready

LATE = [180.0, 360.0, 720.0, 1080.0]


def statistics(paths, origin):
    """paths [N,S,T',3] cumulative on report days > origin, preceded by the origin value."""
    days = np.concatenate(([origin], DAYS[DAYS > origin]))
    increments = np.diff(paths, axis=2)
    negative = increments < 0
    out = dict(negative_interval_fraction=float(np.nanmean(negative)),
               trajectories_with_negative_interval=float(np.nanmean(negative.any(axis=(2, 3)))))
    if origin <= 180:
        index = [int(np.where(days == d)[0][0]) for d in LATE]
        volumes = np.diff(paths[:, :, index], axis=2)  # 180-360, 360-720, 720-1080
        rates = volumes / np.diff(LATE)[None, None, :, None]
        rise = np.diff(rates, axis=2) > 0
        out["oil_rate_rise_after_180"] = float(np.nanmean(rise[..., 0].any(axis=2)))
        out["gas_rises_while_oil_falls_after_180"] = float(np.nanmean((rise[..., 1] & ~rise[..., 0]).any(axis=2)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--origins", nargs="+", type=int, default=[0, 180])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    runs = [Path(r) for r in args.runs]
    holdout = dict(np.load(runs[0] / "holdout.npz", allow_pickle=False))
    reports = holdout["reports"]
    families = {}
    for run in runs:
        if not np.array_equal(np.load(run / "holdout.npz", allow_pickle=False)["row_ids"], holdout["row_ids"]):
            raise ValueError("Runs disagree on holdout identities")
        for file in sorted(run.glob("*_draws.npz")):
            families.setdefault(file.name.removesuffix("_draws.npz").rsplit("_seed", 1)[0], []).append(file)
    results = {}
    for origin in args.origins:
        future = DAYS > origin
        base = reports[:, list(DAYS).index(origin)]
        observed = np.concatenate((base[:, None], reports[:, future]), axis=1)[:, None]
        complete = np.isfinite(observed).all(axis=(1, 2, 3))
        results.setdefault("observed", {})[str(origin)] = dict(wells=int(complete.sum()), **statistics(observed[complete], origin))
        for family, files in families.items():
            parts = []
            for file in files:
                with np.load(file) as bank:
                    draws = bank[f"draws_o{origin}"].astype(np.float64)
                parts.append(np.concatenate((np.broadcast_to(base[:, None, None], draws[:, :, :1].shape), draws), axis=2))
            label = f"{family}_mix{len(files)}" if len(files) > 1 else family
            results.setdefault(label, {})[str(origin)] = dict(wells=int(len(reports)), **statistics(np.concatenate(parts, 1), origin))
            print(json.dumps(dict(model=label, origin=origin, **results[label][str(origin)])), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report_properties.json").write_text(json.dumps(_json_ready(dict(runs=[str(r) for r in runs], results=results)), indent=1))
    print(json.dumps(dict(model="observed", **results["observed"])), flush=True)


if __name__ == "__main__":
    main()
