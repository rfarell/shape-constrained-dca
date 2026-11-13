"""EUR sensitivity to the slowest decay time tau_max (reviewer request; validation wells only).

Loads fits of the same configuration that differ only in tau_max, draws posterior
latent parameters for validation wells at several origins, and compares, per well,
the posterior median EUR (all phases) and the median forecast cumulative at the last
report day (1,080 d) against the reference fit. WIS of the fits comes from their
saved score files. Writes taumax.json and a markdown table.

    python scripts/paper_taumax_sensitivity.py --model 1825=<pt> --model 3650=<pt> --model 7300=<pt> --reference 3650 --output <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch

from depletion_dev import DAYS, load_split
from joint_production.depletion_bayes import DepletionBayesForecaster
from joint_production.metrics import _json_ready

PHASES = ("oil", "gas", "water")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="tau_max_days=path")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--wells", type=int, default=200)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = load_split("validation")
    order = np.argsort(data["holdout_row_ids"])[:args.wells]
    reports, x = data["holdout_reports"][order], data["holdout_features"][order]
    summary = {}
    for spec in args.model:
        tau, path = spec.split("=", 1)
        model = DepletionBayesForecaster.load(path, device=args.device)
        law, scale = model.model.rate, model.model.phase_scales.double().cpu().numpy()
        summary[tau] = {"grid_days": [float(v) for v in law.decay_times]}
        for origin in (90, 360, 720):
            history = reports.copy()
            history[:, DAYS > origin] = np.nan
            out = model.predict_distribution(history, origin, args.draws, np.random.default_rng(7 + origin), x)
            z = torch.as_tensor(out["latent_parameters"], dtype=torch.float64)
            with torch.no_grad():
                eur = law.total_volume(z).numpy() * scale  # [N,S,3]
                cum1080 = law.cumulative(z, torch.tensor([1080.0], dtype=torch.float64)).numpy()[:, :, 0, :] * scale
            summary[tau][str(origin)] = dict(eur_median=np.median(eur, 1), eur_p10=np.percentile(eur, 10, 1),
                                             eur_p90=np.percentile(eur, 90, 1), cum1080_median=np.median(cum1080, 1))
    ref = summary[args.reference]
    table = {}
    rows = ["| tau_max (d) | origin | median EUR ratio vs reference o/g/w | EUR P90-P10 width ratio o/g/w | median 1080-d cumulative ratio o/g/w |",
            "|---:|---:|---:|---:|---:|"]
    for tau, values in summary.items():
        for origin in ("90", "360", "720"):
            v, r = values[origin], ref[origin]
            eur_ratio = np.median(v["eur_median"] / r["eur_median"], 0)
            width_ratio = np.median((v["eur_p90"] - v["eur_p10"]) / (r["eur_p90"] - r["eur_p10"]), 0)
            cum_ratio = np.median(v["cum1080_median"] / r["cum1080_median"], 0)
            table[f"{tau}/{origin}"] = dict(eur_ratio=eur_ratio, width_ratio=width_ratio, cum1080_ratio=cum_ratio,
                                            eur_median_oil_bbl=float(np.median(v["eur_median"][:, 0])))
            rows.append(f"| {tau} | {origin} | " + " / ".join(f"{u:.2f}" for u in eur_ratio) + " | "
                        + " / ".join(f"{u:.2f}" for u in width_ratio) + " | " + " / ".join(f"{u:.3f}" for u in cum_ratio) + " |")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "taumax.json").write_text(json.dumps(_json_ready(dict(wells=int(len(order)), reference=args.reference,
                                                                    table=table, grids={k: v["grid_days"] for k, v in summary.items()})), indent=1))
    (output / "taumax.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
