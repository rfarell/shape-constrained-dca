"""Load the accepted TRAIN records after verifying the raw data and split hashes.

Validation, calibration and test lines are never decoded. Used for the
TRAIN-internal development holdout (``depletion_dev.py --split internal``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from joint_production.artifacts import sha256_file

PHASES = ("oil", "gas", "water")


def _read_json(path):
    return json.loads(Path(path).read_text())


def load_train_only(directory, raw_path, raw_manifest_path):
    """Verify accepted bytes, then decode only manifest-selected TRAIN lines."""
    directory = Path(directory)
    raw_manifest = _read_json(raw_manifest_path)
    raw_hash = sha256_file(raw_path)
    if raw_hash != raw_manifest["sha256"]:
        raise ValueError("Immutable raw dataset hash disagrees with its manifest")
    split_path = directory / "split_manifest.json"
    split = _read_json(split_path)
    if (split.get("status") != "accepted_immutable" or split.get("schema_version") != 1
            or split.get("dataset_sha256") != raw_hash or split.get("phase_order") != list(PHASES)):
        raise ValueError("Unsupported accepted split or raw-data/phase mismatch")
    policy_hash = hashlib.sha256((json.dumps(split["curation_policy"], sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
    if policy_hash != split["curation_policy_sha256"]:
        raise ValueError("Accepted curation policy hash mismatch")
    required = {"records.jsonl", "quarantine.jsonl", "curation_audit.json"}
    if set(split["artifact_sha256"]) != required:
        raise ValueError("Accepted artifact hash inventory is incomplete")
    for name, expected in split["artifact_sha256"].items():
        if Path(name).name != name or sha256_file(directory / name) != expected:
            raise ValueError(f"Accepted artifact hash mismatch: {name}")
    membership = {}
    for group in split["groups"]:
        for row_id in group["row_ids"]:
            if row_id in membership:
                raise ValueError("Repeated row ID in accepted grouping")
            membership[row_id] = (group["group_id"], group["split"])
    row_ids = split["retained_row_ids"]
    quarantined_ids = split["quarantined_row_ids"]
    if (len(set(row_ids)) != len(row_ids) or len(set(quarantined_ids)) != len(quarantined_ids)
            or set(row_ids) & set(quarantined_ids)
            or set(row_ids) | set(quarantined_ids) != set(membership)):
        raise ValueError("Accepted row and grouping inventories disagree")
    train = []
    with (directory / "records.jsonl").open() as handle:
        for row_id, line in zip(row_ids, handle, strict=True):
            if membership[row_id][1] != "train":
                continue  # Never decode nontraining outcome/covariate fields.
            record = json.loads(line)
            if (record["row_id"] != row_id or record["split"] != "train"
                    or record["group_id"] != membership[row_id][0]):
                raise ValueError("Training record disagrees with accepted identity/group/split")
            train.append(record)
    return train, split, raw_hash, sha256_file(split_path)
