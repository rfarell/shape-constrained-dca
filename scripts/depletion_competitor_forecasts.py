"""Issue forecasts from the retained competitor fits for a named accepted split (no refitting).

Loads the saved five-seed basis/DCA/neural checkpoints and forest pickles of the
secondary-attribute study, encodes the split's static attributes with the same
TRAIN-fitted preprocessing, and draws 256 forecasts per origin with the original
prediction seed rule (2026 * 10000 + origin). Writes depletion_dev draw files and a
holdout.npz for scoring. The test split requires a frozen protocol.

    python scripts/depletion_competitor_forecasts.py --split test --protocol configs/depletion_final_protocol.toml \
        --output <dir> --checkpoint basis=<path>:42 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from depletion_dev import DAYS, ORIGINS, _cap_gpu_memory, load_split


def load_model(family, path, device):
    if family in {"basis", "dca"}:
        from joint_production.conditional_rates import ConditionalRateForecaster
        return ConditionalRateForecaster.load(path, device=device)
    if family == "neural":
        from joint_production.neural import ContextualGaussianNeuralBaseline
        return ContextualGaussianNeuralBaseline.load(path, device=device)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["validation", "calibration", "test"])
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", action="append", required=True, help="family=path:seed")
    parser.add_argument("--per-seed-stream", action="store_true", help="independent prediction stream per fitted seed")
    args = parser.parse_args(argv)
    _cap_gpu_memory(args.device)
    data = load_split(args.split, args.protocol)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "holdout.npz", reports=data["holdout_reports"], row_ids=data["holdout_row_ids"],
             groups=data["holdout_groups"], scales=data["scales"], days=DAYS)
    receipt = []
    for spec in args.checkpoint:
        family, rest = spec.split("=", 1)
        path, seed = rest.rsplit(":", 1)
        model = load_model(family, path, args.device)
        arrays = {}
        for origin in ORIGINS:
            history = data["holdout_reports"].copy()
            history[:, DAYS > origin] = np.nan
            draws = model.predict_samples(history, origin=origin, n_samples=256,
                                          rng=np.random.default_rng(2026 * 10000 + origin + (1000 * int(seed) if args.per_seed_stream else 0)),
                                          covariates=data["holdout_features"])
            if not np.isfinite(draws[:, :, DAYS > origin]).all():
                raise ValueError(f"{family} seed {seed}: non-finite draws at origin {origin}")
            arrays[f"draws_o{origin}"] = draws[:, :, DAYS > origin, :].astype(np.float32)
        np.savez(output / f"{family}_seed{seed}_draws.npz", **arrays)
        receipt.append(dict(family=family, seed=int(seed), checkpoint=path))
        print(json.dumps(receipt[-1]), flush=True)
    (output / "competitor_receipt.json").write_text(json.dumps(dict(split=args.split, sources=receipt), indent=1))


if __name__ == "__main__":
    main()
