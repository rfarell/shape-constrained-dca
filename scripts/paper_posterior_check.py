"""Check the importance-sampling posterior against a much larger reference run (validation wells).

For a fixed identity-ordered subset of validation wells and origins, draws posterior
latent parameters with the fitted configuration (standard: its proposals and
adaptive rounds) and with a reference configuration (many more proposals and
rounds), then compares per-well quantiles (P10/P50/P90 in the exceedance sense of
the 10/50/90th percentiles) of derived quantities: latent cumulative at 1,080 d and
EUR, per phase, plus effective sample sizes. Also reports how often observed oil
interval decline rates increase after day 180 in the reports of all validation wells.

    python scripts/paper_posterior_check.py --model <v1 model.pt> --output <dir>
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

from depletion_dev import DAYS, _cap_gpu_memory, load_split
from joint_production.depletion_bayes import DepletionBayesForecaster
from joint_production.metrics import _json_ready

PHASES = ("oil", "gas", "water")


def derived(model, z):
    law, scale = model.model.rate, model.model.phase_scales.double().cpu().numpy()
    z = torch.as_tensor(z, dtype=torch.float64)
    with torch.no_grad():
        cum = law.cumulative(z, torch.tensor([1080.0], dtype=torch.float64)).numpy()[:, :, 0, :] * scale
        eur = law.total_volume(z).numpy() * scale
    return {"cum1080": cum, "eur": eur}


def observed_acceleration(reports):
    """Share of wells whose observed oil interval-average decline rate increases between successive
    intervals after day 180 (180-360 -> 360-720 -> 720-1080), using continuous decline between interval
    midpoints: D = ln(r_k / r_{k+1}) / (m_{k+1} - m_k)."""
    edges = [180.0, 360.0, 720.0, 1080.0]
    idx = [list(DAYS).index(e) for e in edges]
    vol = np.diff(reports[:, idx, 0], axis=1)
    rate = vol / np.diff(edges)
    ok = np.isfinite(rate).all(1) & (rate > 0).all(1)
    mids = 0.5 * (np.asarray(edges[1:]) + np.asarray(edges[:-1]))
    d = np.log(rate[ok, :-1] / rate[ok, 1:]) / np.diff(mids)
    return dict(wells=int(ok.sum()), share_decline_rate_increases=float(np.mean(d[:, 1] > d[:, 0])),
                share_rate_increases=float(np.mean((rate[ok, 1:] > rate[ok, :-1]).any(1))),
                median_decline_per_year=(np.median(d, 0) * 365.25).tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wells", type=int, default=40)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--ref-proposals", type=int, default=65536)
    parser.add_argument("--ref-rounds", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    _cap_gpu_memory(args.device)
    data = load_split("validation")
    reports_all = data["holdout_reports"]
    order = np.argsort(data["holdout_row_ids"])[:args.wells]
    reports, x = reports_all[order], data["holdout_features"][order]
    model = DepletionBayesForecaster.load(args.model, device=args.device)
    standard = dict(proposals=model.config["proposals"], adaptive_rounds=model.config["adaptive_rounds"])
    result = dict(model=args.model, wells=int(len(order)), draws=args.draws, standard=standard,
                  reference=dict(proposals=args.ref_proposals, adaptive_rounds=args.ref_rounds), origins={})
    for origin in (90, 360, 720):
        history = reports.copy()
        history[:, DAYS > origin] = np.nan
        out = {}
        for name, cfg in (("standard", standard), ("reference", dict(proposals=args.ref_proposals, adaptive_rounds=args.ref_rounds))):
            model.config.update(cfg)
            chunk = max(1, int(4 * 65536 // cfg["proposals"]))  # bound memory: wells x proposals per call
            zs, ess = [], []
            for start in range(0, len(history), chunk):
                sl = slice(start, start + chunk)
                drawn = model.posterior_draws(history[sl], origin, args.draws, np.random.default_rng(31 + origin + start), x[sl])
                zs.append(drawn["z"].cpu().numpy()); ess.append(drawn["ess"].cpu().numpy())
                del drawn
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            out[name] = dict(values=derived(model, np.concatenate(zs)), ess=np.concatenate(ess))
        model.config.update(standard)
        row = {}
        for key in ("cum1080", "eur"):
            s_q = np.percentile(out["standard"]["values"][key], [10, 50, 90], axis=1)   # [3, N, 3]
            r_q = np.percentile(out["reference"]["values"][key], [10, 50, 90], axis=1)
            rel = np.abs(s_q - r_q) / np.maximum(np.abs(r_q), 1e-9)
            width_ratio = (s_q[2] - s_q[0]) / np.maximum(r_q[2] - r_q[0], 1e-9)
            row[key] = dict(median_abs_rel_diff_p10_p50_p90=np.median(rel, axis=1).tolist(),  # [3 quantiles][3 phases]
                            p90_abs_rel_diff_p50=np.percentile(rel[1], 90, axis=0).tolist(),
                            median_width_ratio=np.median(width_ratio, axis=0).tolist())
        row["ess_standard_median_p10"] = [float(np.median(out["standard"]["ess"])), float(np.percentile(out["standard"]["ess"], 10))]
        row["ess_reference_median_p10"] = [float(np.median(out["reference"]["ess"])), float(np.percentile(out["reference"]["ess"], 10))]
        result["origins"][str(origin)] = row
        print(json.dumps({"origin": origin, **{k: (v if not isinstance(v, dict) else {kk: np.round(vv, 3).tolist() for kk, vv in v.items()}) for k, v in row.items()}}), flush=True)
    result["observed_acceleration_validation"] = observed_acceleration(reports_all)
    print(json.dumps(result["observed_acceleration_validation"]), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "posterior_check.json").write_text(json.dumps(_json_ready(result), indent=1))


if __name__ == "__main__":
    main()
