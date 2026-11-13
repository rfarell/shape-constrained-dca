"""Write the ensemble-diversity sweep plan (plan.tsv) used in development.

The sweeps compare ways of diversifying the five ensemble members on the TRAIN-internal
holdout; sweep 1 selected prior-network width diversity ("arch"). Each variant is a list
of extra --set overrides applied to five members (seeds 42-46) on top of a base
configuration; a variant may instead give per-seed overrides. One plan line per member:
name<TAB>output_dir<TAB>depletion_dev arguments.

    python scripts/ensemble_sweep.py --sweep outputs/sweep1 --variants sweep1
    while IFS=$'\\t' read -r name out margs; do
        python scripts/depletion_dev.py $margs --label "$name" --output "$out"
    done < outputs/sweep1/plan.tsv
    python scripts/ensemble_diversity.py --variant arch='outputs/sweep1/arch_s*/*_draws.npz' \\
        --variant base='outputs/sweep1/base_s*/*_draws.npz' ... --output outputs/diversity_dev
"""
from __future__ import annotations

import argparse
from pathlib import Path

V2_BASE = ("run --models bayes --device cuda --split internal --set width=256 --set rank=8 --set train_samples=16 "
           "--set objective=predictive --set log_every=50 --set epochs=200 --set predictive_samples=32 "
           "--set rate_clock=true --set discrepancy=ou_interval")
SEEDS = (42, 43, 44, 45, 46)

SWEEPS = {
    "sweep1": {
        "base": [],
        "boot": ["bootstrap=true"],
        "rp03": ["random_prior_scale=0.3"],
        "rp10": ["random_prior_scale=1.0"],
        "fb60": ["feature_keep=0.6"],
        "arch": {42: ["prior_width=32"], 43: ["prior_width=64"], 44: ["prior_width=128"], 45: ["prior_width=256"], 46: ["prior_width=512"]},
        "cap": ["prior_width=256", "prior_weight_decay=0.0001"],
        "combo": ["bootstrap=true", "random_prior_scale=0.3", "feature_keep=0.75"],
    },
    # Sweep 2: build on prior-width diversity (arch), the sweep-1 winner on the internal holdout.
    "sweep2": {
        "arch2": {47: ["prior_width=32"], 48: ["prior_width=64"], 49: ["prior_width=128"], 50: ["prior_width=256"], 51: ["prior_width=512"]},
        "archwd": {42: ["prior_width=32", "prior_weight_decay=0.0001"], 43: ["prior_width=64", "prior_weight_decay=0.0001"],
                   44: ["prior_width=128", "prior_weight_decay=0.0001"], 45: ["prior_width=256", "prior_weight_decay=0.0001"],
                   46: ["prior_width=512", "prior_weight_decay=0.0001"]},
        "archwide": {42: ["prior_width=64"], 43: ["prior_width=128"], 44: ["prior_width=256"], 45: ["prior_width=512"], 46: ["prior_width=1024"]},
        "archboot": {42: ["prior_width=32", "bootstrap=true"], 43: ["prior_width=64", "bootstrap=true"], 44: ["prior_width=128", "bootstrap=true"],
                     45: ["prior_width=256", "bootstrap=true"], 46: ["prior_width=512", "bootstrap=true"]},
        "archrp": {42: ["prior_width=32", "random_prior_scale=0.3"], 43: ["prior_width=64", "random_prior_scale=0.3"],
                   44: ["prior_width=128", "random_prior_scale=0.3"], 45: ["prior_width=256", "random_prior_scale=0.3"],
                   46: ["prior_width=512", "random_prior_scale=0.3"]},
        "archenc": {42: ["prior_width=32", "width=128"], 43: ["prior_width=64", "width=192"], 44: ["prior_width=128", "width=256"],
                    45: ["prior_width=256", "width=384"], 46: ["prior_width=512", "width=512"]},
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="sweep directory (plan.tsv and member outputs go here)")
    parser.add_argument("--variants", required=True, choices=sorted(SWEEPS))
    parser.add_argument("--base", default=V2_BASE)
    args = parser.parse_args(argv)
    sweep = Path(args.sweep)
    sweep.mkdir(parents=True, exist_ok=True)
    lines = []
    for variant, overrides in SWEEPS[args.variants].items():
        for seed in (overrides if isinstance(overrides, dict) else SEEDS):  # per-seed variants define their own seeds
            extra = overrides.get(seed, []) if isinstance(overrides, dict) else overrides
            sets = " ".join(f"--set {o}" for o in extra)
            lines.append(f"{variant}\t{sweep}/{variant}_s{seed}\t{args.base} {sets} --seeds {seed}".replace("  ", " "))
    (sweep / "plan.tsv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} members -> {sweep / 'plan.tsv'}")


if __name__ == "__main__":
    main()
