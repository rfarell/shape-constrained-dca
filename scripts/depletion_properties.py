"""Audit structural properties of sampled trajectories, by model family.

For a fixed identity-ordered subset of TRAIN-internal holdout wells, issue
forecasts at origins 0, 90 and 360, take every model's own sampled latent rate
parameters and evaluate rates on a dense daily grid to 3,650 days. Reports the
fraction of sampled trajectories violating or exhibiting each property. Neural
and forest comparators issue reports only, so latent-rate properties are
recorded as unavailable for them. No outcome after an origin is used.

    python scripts/depletion_properties.py --bayes <run>/<label>_seed42_model.pt \
        --basis <dir>/basis_seed42_model.pt --dca <dir>/dca_seed42_model.pt --output build/<audit>
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
from joint_production.artifacts import write_json
from joint_production.metrics import _json_ready

GRID = np.arange(0.0, 3651.0, 1.0)


def rate_checks(rate, cumulative, peak=None):
    """rate/cumulative [N,S,G,3] native units on GRID (days). Fractions of trajectories."""
    t = GRID
    after = t >= 180.0
    d_rate = np.diff(rate, axis=2)
    oil = rate[..., 0]
    out = {}
    out["negative_rate_any_phase"] = float((rate < -1e-9 * np.abs(rate).max()).any(axis=(2, 3)).mean())
    rise_after = (d_rate[:, :, after[1:], :] > 1e-9 * np.maximum(rate[:, :, :-1][:, :, after[1:], :], 1e-12))
    for k, name in enumerate(("oil", "gas", "water")):
        out[f"{name}_any_rise_after_day180"] = float(rise_after[..., k].any(axis=2).mean())
    early = (t[1:] <= 90.0)
    rise_early = (d_rate[:, :, early, 0] > 1e-9 * np.maximum(rate[:, :, :-1, 0][:, :, early], 1e-12))
    out["oil_any_rise_first_90_days"] = float(rise_early.any(axis=2).mean())
    gas_up_oil_down = rise_after[..., 1] & (d_rate[:, :, after[1:], 0] < 0)
    out["gas_rises_while_oil_declines_after_day180"] = float(gas_up_oil_down.any(axis=2).mean())
    log_oil = np.log(np.maximum(oil[:, :, after], 1e-300))
    curvature = np.diff(log_oil, n=2, axis=2)
    out["oil_decline_rate_increases_after_day180"] = float((curvature < -1e-9).any(axis=2).mean())
    integral = np.concatenate([np.zeros_like(rate[:, :, :1]), np.cumsum(0.5 * (rate[:, :, 1:] + rate[:, :, :-1]), axis=2)], axis=2)
    scale = np.maximum(np.abs(cumulative[:, :, -1:]), 1e-12)
    out["max_relative_integral_error"] = float(np.max(np.abs(integral - cumulative) / scale))
    out["cumulative_decreasing_any"] = float((np.diff(cumulative, axis=2) < -1e-9 * scale).any(axis=(2, 3)).mean())
    # Interval-average statistics on 180-360, 360-720, 720-1080 (comparable with report-level audits).
    late = [180, 360, 720, 1080]
    rates = np.diff(cumulative[:, :, late], axis=2) / np.diff(late)[None, None, :, None]
    rise = np.diff(rates, axis=2) > 1e-12 * np.abs(rates[:, :, :-1])
    out["interval_oil_rate_rise_after_180"] = float(rise[..., 0].any(axis=2).mean())
    out["interval_gas_rises_while_oil_falls_after_180"] = float((rise[..., 1] & ~rise[..., 0]).any(axis=2).mean())
    if peak is not None:
        out["peak_day_max"] = float(np.max(peak))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayes", required=True)
    parser.add_argument("--basis")
    parser.add_argument("--dca")
    parser.add_argument("--wells", type=int, default=96)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--protocol", default=None, help="frozen protocol (required for the test split)")
    args = parser.parse_args(argv)
    data = load_split(args.split, args.protocol)
    order = np.argsort(data["holdout_row_ids"])[:args.wells]
    reports = data["holdout_reports"][order]
    x = data["holdout_features"][order]
    results = {}
    from joint_production.depletion_bayes import DepletionBayesForecaster
    bayes = DepletionBayesForecaster.load(args.bayes, device=args.device)
    models = {"bayes": bayes}
    if args.basis:
        from joint_production.conditional_rates import ConditionalRateForecaster
        models["basis"] = ConditionalRateForecaster.load(args.basis, device=args.device)
    if args.dca:
        from joint_production.conditional_rates import ConditionalRateForecaster
        models["dca"] = ConditionalRateForecaster.load(args.dca, device=args.device)
    grid = torch.as_tensor(GRID, dtype=torch.float64)
    for name, model in models.items():
        results[name] = {}
        for origin in (0, 90, 360):
            history = reports.copy()
            history[:, DAYS > origin] = np.nan
            rng = np.random.default_rng(99 + origin)
            out = model.predict_distribution(history, origin, args.draws, rng, x)
            z = torch.as_tensor(out["latent_parameters"], dtype=torch.float64)
            with torch.no_grad():
                if name == "bayes":
                    law = model.model.rate
                    scale = model.model.phase_scales.double().numpy()
                    rate = law.rate(z, grid).numpy() * scale
                    cumulative = law.cumulative(z, grid).numpy() * scale
                    peak = law.unpack(z)["peak"].numpy()
                    eur = law.total_volume(z).numpy() * scale
                else:
                    law = model.rate_model.double()
                    scale = model.volume_scale.double().cpu().numpy()
                    rate = law.rate(z, grid).numpy() * scale
                    cumulative = law.cumulative(z, grid).numpy() * scale
                    peak = None
                    eur = law.total_volume(z).numpy() * scale
            checks = rate_checks(rate, cumulative, peak)
            checks["eur_finite_fraction"] = float(np.isfinite(eur).all(axis=-1).mean())
            checks["eur_over_cumulative3650_median"] = np.median(eur / np.maximum(cumulative[:, :, -1], 1e-12), axis=(0, 1)).tolist()
            results[name][str(origin)] = checks
            print(json.dumps({"model": name, "origin": origin, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in checks.items()}}), flush=True)
    results["neural"] = results["forest"] = "Issues report trajectories only: no continuous latent rate, decline structure, or total-volume law."
    results["split"] = args.split
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "properties.json", _json_ready(dict(wells=int(len(order)), draws=args.draws, grid_days=[0, 3650],
                                                             results=results)), exclusive=False)


if __name__ == "__main__":
    main()
