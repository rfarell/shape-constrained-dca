"""Convert archived per-origin forecast banks into depletion_dev draw files.

Reads ``{label}_seed{seed}_origin{origin}_paths.npz`` files (report_cumulative
[N,S,T,3] and row_ids) from retained runs, verifies row identities against the
requested accepted split, and writes ``{family}_seed{seed}_draws.npz`` plus a
``holdout.npz`` (reports, row_ids, groups, TRAIN scales) so that
``depletion_dev.py score`` scores every family on identical wells and scales.
No model is refit and no new draws are made.

    python scripts/depletion_import_paths.py --split validation --output <dir> \
        --source basis_secondary=<run>:42 --source basis_secondary=<run>:43,44,45,46 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np

from depletion_dev import DAYS, ORIGINS, load_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["validation", "calibration", "test"])
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="label=run_dir:seed[,seed...]; output family name is the label prefix before '_'")
    args = parser.parse_args(argv)
    data = load_split(args.split, args.protocol)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ids = data["holdout_row_ids"]
    np.savez(output / "holdout.npz", reports=data["holdout_reports"], row_ids=ids,
             groups=data["holdout_groups"], scales=data["scales"], days=DAYS)
    receipt = []
    for spec in args.source:
        label, rest = spec.split("=", 1)
        run, seeds = rest.rsplit(":", 1)
        family = label.split("_")[0]
        for seed in (int(s) for s in seeds.split(",")):
            arrays = {}
            for origin in ORIGINS:
                path = Path(run) / f"{label}_seed{seed}_origin{origin}_paths.npz"
                with np.load(path, allow_pickle=False) as bank:
                    if not np.array_equal(bank["row_ids"].astype(str), ids.astype(str)):
                        raise ValueError(f"{path}: row identities differ from the {args.split} split order")
                    if not np.array_equal(bank["report_days"], DAYS):
                        raise ValueError(f"{path}: report grid differs")
                    draws = bank["report_cumulative"]
                arrays[f"draws_o{origin}"] = draws[:, :, DAYS > origin, :].astype(np.float32)
            np.savez(output / f"{family}_seed{seed}_draws.npz", **arrays)
            receipt.append(dict(family=family, label=label, seed=seed, run=str(run)))
            print(json.dumps(receipt[-1]), flush=True)
    (output / "import_receipt.json").write_text(json.dumps(dict(split=args.split, sources=receipt), indent=1))


if __name__ == "__main__":
    main()
