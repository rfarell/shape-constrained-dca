"""Synthetic inference check for a fitted physics-structured Bayesian model.

Synthetic wells use real holdout ATTRIBUTES only (no production outcomes):
z* ~ p(z | x), latent Q* = Q(z*), reports Y* from the fitted accumulated
Gaussian law. For each origin, the issued inference (amortized Gaussian
with importance reweighting) is applied to Y* through the origin. If
inference is correct, normalized ranks of z* and of future reports among the
posterior/predictive draws are uniform, interval coverage is nominal, and
posterior spread contracts in expectation as history grows.

    python scripts/depletion_sbc.py --model <run>/bayes_seed42_model.pt --output build/<sbc>
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

from depletion_dev import DAYS, ORIGINS, load_internal
from joint_production.artifacts import write_json
from joint_production.depletion_bayes import DepletionBayesForecaster, LATENT_DIM, sample_bridge_reports
from joint_production.metrics import _json_ready

LEVELS = (0.5, 0.8, 0.9, 0.95)


def ks_uniform(u):
    u = np.sort(np.asarray(u).ravel())
    n = len(u)
    grid = np.arange(1, n + 1) / n
    return float(max(np.max(grid - u), np.max(u - (grid - 1 / n))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wells", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--proposals", type=int, default=0)
    args = parser.parse_args(argv)
    data = load_internal()
    forecaster = DepletionBayesForecaster.load(args.model, device=args.device)
    if args.proposals:
        forecaster.config["proposals"] = args.proposals
    model = forecaster.model
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(data["holdout_features"]), size=min(args.wells, len(data["holdout_features"])), replace=False)
    x = data["holdout_features"][pick]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.no_grad():
        log_pi, mean_p, tril_p = model.prior(torch.as_tensor(x, dtype=torch.float32, device=args.device))
        label = torch.multinomial(log_pi.exp(), 1, generator=generator).squeeze(-1)
        rows = torch.arange(len(label), device=args.device)
        mean_k, tril_k = mean_p[rows, label], tril_p[rows, label]
        eps = torch.randn(mean_k.shape, dtype=torch.float64, device=args.device, generator=generator)
        z_true = mean_k + (tril_k @ eps[..., None]).squeeze(-1)
        scales = model.phase_scales.to(torch.float64)
        days = model.report_days.to(torch.float64)
        latent = model.rate.cumulative(z_true, days) * scales
        a, r = model.noise_for(z_true[:, None])
        empty = torch.full_like(latent, float("nan"))
        empty[:, 0] = 0
        reports = sample_bridge_reports(latent[:, None], empty, days, a * scales, r,
                                        cutoff=0.0, generator=generator)[:, 0].cpu().numpy()
    z_true = z_true.cpu().numpy()
    latent = latent.cpu().numpy()
    summary = {}
    for origin in ORIGINS:
        history = reports.copy()
        history[:, DAYS > origin] = np.nan
        out = forecaster.predict_distribution(history, origin, args.draws, np.random.default_rng(1000 + origin), x)
        z = out["latent_parameters"]
        ranks_z = (z < z_true[:, None, :]).mean(axis=1)  # [N, D]
        future = DAYS > origin
        idx = list(DAYS).index(origin)
        targets = reports[:, future] - reports[:, idx][:, None]
        draws = out["report_cumulative"][:, :, future] - reports[:, None, idx][:, :, None]
        pit = (draws < targets[:, None]).mean(axis=1) + 0.5 * (draws == targets[:, None]).mean(axis=1)
        coverage = {}
        for level in LEVELS:
            lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=1)
            hit = (targets >= lo) & (targets <= hi)
            coverage[str(level)] = dict(all=float(hit.mean()), by_phase=hit.mean(axis=(0, 1)).tolist())
        latent_draws = out["latent_cumulative"][:, :, -1]  # latent cumulative at 1080 [N,S,3]
        log_q1080 = np.log(np.maximum(latent_draws, 1e-9))
        rank_q1080 = (latent_draws < latent[:, None, -1]).mean(axis=1)
        summary[str(origin)] = dict(
            coverage_future_reports=coverage,
            pit_ks=ks_uniform(pit), pit_mean=float(pit.mean()),
            z_rank_ks_by_dim=[ks_uniform(ranks_z[:, d]) for d in range(LATENT_DIM)],
            z_rank_mean_by_dim=ranks_z.mean(axis=0).tolist(),
            latent1080_rank_ks_by_phase=[ks_uniform(rank_q1080[:, k]) for k in range(3)],
            posterior_sd_z_mean=z.std(axis=1).mean(axis=0).tolist(),
            posterior_sd_log_latent1080=log_q1080.std(axis=1).mean(axis=0).tolist(),
            ess_median=float(np.median(out["ess"])), ess_p10=float(np.quantile(out["ess"], 0.1)),
            unique_draws_p10=float(np.quantile(out["unique_draws"], 0.1)))
        s = summary[str(origin)]
        print(json.dumps(dict(origin=origin, cov90=round(coverage["0.9"]["all"], 4),
                              cov90_phase=[round(v, 3) for v in coverage["0.9"]["by_phase"]],
                              pit_ks=round(s["pit_ks"], 4), z_ks_max=round(max(s["z_rank_ks_by_dim"]), 4),
                              sd_log_q1080=[round(v, 3) for v in s["posterior_sd_log_latent1080"]],
                              ess_med=round(s["ess_median"], 1))), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    critical = 1.36 / np.sqrt(len(pick))
    write_json(output / "sbc.json", _json_ready(dict(
        model=str(args.model), wells=len(pick), draws=args.draws, importance=True,
        ks_critical_95=float(critical), by_origin=summary,
        interpretation="Synthetic wells from the fitted prior and report law; uniform ranks and nominal coverage test inference, not model fit to field data.")),
        exclusive=False)


if __name__ == "__main__":
    main()
