"""Conservative, reversible curation of the private Bakken cumulative export.

No outcome is imputed, clipped, monotonicized, or used as a model covariate by
this module. Row identities identify semantic exports, not verified well APIs.
All array boundaries use [time, phase], with phases oil, gas, water.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .data_audit import MISSING, PHASE_COLUMNS, REPORT_DAYS

PHASE_ORDER = ("oil", "gas", "water")
VOLUME_UNITS = ("BBL", "MCF", "BBL")
SPLIT_NAMES = ("train", "validation", "calibration", "test")
SPLIT_PROBABILITIES = (0.80, 0.05, 0.05, 0.10)
EARTH_RADIUS_M = 6_371_008.8
SCHEMA_VERSION = 1


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _numeric_token(value):
    """An exact Decimal identity token: no rounding via Decimal.normalize()."""
    value = value.strip()
    if value.lower() in MISSING:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return "invalid:" + value
    if not number.is_finite():
        return "invalid:" + value.lower()
    if number == 0:
        return "0"
    sign, digits, exponent = number.as_tuple()
    digits = list(digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    return ("-" if sign else "") + "".join(map(str, digits)) + "e" + str(exponent)


def _number(value):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _semantic_row(row, columns):
    return {
        key: row[key] if key in {"API", "FirstProdDate"} else _numeric_token(row[key])
        for key in sorted(columns)
    }


class _Components:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        i, j = self.find(i), self.find(j)
        if i != j:
            self.parent[max(i, j)] = min(i, j)


def _coordinate(row):
    lat, lon = _number(row["Latitude"]), _number(row["Longitude"])
    if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None
    return lat, lon


def _distance_m(a, b):
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    delta_lat, delta_lon = lat2 - lat1, math.radians(b[1] - a[1])
    hav = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, hav))))


def _spatial_edges(coordinates, radius_m):
    """ECEF bins find every spherical-distance neighbor, including date lines."""
    bins = defaultdict(list)
    for i, coordinate in enumerate(coordinates):
        if coordinate is None:
            continue
        lat, lon = map(math.radians, coordinate)
        xyz = (EARTH_RADIUS_M * math.cos(lat) * math.cos(lon),
               EARTH_RADIUS_M * math.cos(lat) * math.sin(lon),
               EARTH_RADIUS_M * math.sin(lat))
        cell = tuple(math.floor(v / radius_m) for v in xyz)
        for offset in itertools.product((-1, 0, 1), repeat=3):
            neighbor = tuple(c + d for c, d in zip(cell, offset))
            for j in bins.get(neighbor, ()):
                distance = _distance_m(coordinate, coordinates[j])
                if distance <= radius_m:
                    yield j, i, distance
        bins[cell].append(i)


def assign_split(group_id, seed=42):
    """Content-stable independent group assignment; proportions are expected."""
    # Integer boundaries avoid platform-dependent rounding at thresholds.
    value = int(hashlib.sha256(f"bakken-split-v1:{seed}:{group_id}".encode()).hexdigest(), 16)
    total = 1 << 256
    for name, boundary in zip(SPLIT_NAMES, (80, 85, 90, 100)):
        if value * 100 < boundary * total:
            return name
    raise AssertionError("Unreachable split boundary")


def curate_rows(rows, *, radius_m=50.0, seed=42):
    """Return retained/quarantined records and a complete curation audit.

    Exact and numeric-semantic duplicate exports are collapsed. Every ambiguous
    shared history/location/date or shared location/date record is quarantined;
    all matching complete histories and 50 m surface components are grouped
    before quarantine, so an excluded export cannot erase a leakage bridge.
    Missing surface coordinates are quarantined because spatial safety cannot
    be established. API and dates remain exact text, never inferred identities.
    """
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    if not rows:
        raise ValueError("No data rows")
    columns = tuple(rows[0])
    production_columns = tuple(c for p in PHASE_ORDER for c in PHASE_COLUMNS[p])
    required = {"API", "Latitude", "Longitude", "FirstProdDate", *production_columns}
    if not required.issubset(columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(columns))}")
    if any(set(row) != set(columns) or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Inconsistent CSV fields")
    numeric_columns = [c for c in columns if c not in {"API", "FirstProdDate"}]
    covariate_columns = [c for c in columns if c not in {*production_columns, "API", "FirstProdDate"}]

    exports = defaultdict(list)
    exact_counts = Counter()
    for raw_number, row in enumerate(rows, 1):
        exact_counts[tuple(row[c] for c in columns)] += 1
        semantic = _semantic_row(row, columns)
        exports[_digest(semantic)].append((raw_number, row, semantic))
    row_ids = sorted(exports)
    canonical_rows, semantic_rows = [], []
    for row_id in row_ids:
        # A canonical raw representative independent of input order.
        _, row, semantic = min(exports[row_id], key=lambda item: tuple(item[1][c] for c in sorted(columns)))
        canonical_rows.append(row)
        semantic_rows.append(semantic)
    n = len(row_ids)
    components = _Components(n)
    spatial_components = _Components(n)
    reasons = [set() for _ in row_ids]
    coordinates = [_coordinate(row) for row in canonical_rows]
    trajectories = defaultdict(list)
    observation_groups = defaultdict(list)
    location_date_groups = defaultdict(list)
    for i, (row, semantic, coordinate) in enumerate(zip(canonical_rows, semantic_rows, coordinates)):
        trajectory = tuple(semantic[c] for c in production_columns)
        if any(token is not None for token in trajectory):
            trajectories[trajectory].append(i)
        if not any(_number(row[c]) is not None for p in PHASE_ORDER for c in PHASE_COLUMNS[p][1:]):
            reasons[i].add("no_finite_positive_time_reports")
        if any(semantic[c] is not None and (_number(row[c]) is None or str(semantic[c]).startswith("invalid:"))
               for c in production_columns):
            reasons[i].add("invalid_production_token")
        if coordinate is None:
            reasons[i].add("missing_or_invalid_surface_coordinates")
        elif row["FirstProdDate"].strip().lower() not in MISSING:
            location_date = (semantic["Latitude"], semantic["Longitude"], row["FirstProdDate"])
            location_date_groups[location_date].append(i)
            observation_groups[location_date + trajectory].append(i)

    repeated_trajectories = [indices for indices in trajectories.values() if len(indices) > 1]
    for indices in repeated_trajectories:
        for i in indices[1:]:
            components.union(indices[0], i)
    observation_conflicts = [indices for indices in observation_groups.values() if len(indices) > 1]
    for indices in observation_conflicts:
        for i in indices:
            reasons[i].add("conflicting_exports_same_observations")
    location_conflicts = [indices for indices in location_date_groups.values() if len(indices) > 1]
    for indices in location_conflicts:
        for i in indices:
            reasons[i].add("ambiguous_identity_same_surface_and_first_production_date")

    edge_count, max_edge_m = 0, 0.0
    for i, j, distance in _spatial_edges(coordinates, radius_m):
        components.union(i, j)
        spatial_components.union(i, j)
        edge_count += 1
        max_edge_m = max(max_edge_m, distance)
    groups, spatial_groups = defaultdict(list), defaultdict(list)
    for i in range(n):
        groups[components.find(i)].append(i)
        spatial_groups[spatial_components.find(i)].append(i)
    group_ids = {}
    group_details = []
    for indices in groups.values():
        group_id = _digest({"group_policy": "trajectory-or-surface-components-v1", "members": sorted(row_ids[i] for i in indices)})
        for i in indices:
            group_ids[i] = group_id
        group_details.append({"group_id": group_id, "split": assign_split(group_id, seed),
                              "row_ids": sorted(row_ids[i] for i in indices)})

    # Explicit diameter audit detects chains beyond the nominal edge radius.
    chain_audit = []
    for indices in spatial_groups.values():
        valid = [i for i in indices if coordinates[i] is not None]
        diameter = max((_distance_m(coordinates[i], coordinates[j])
                        for i, j in itertools.combinations(valid, 2)), default=0.0)
        if diameter > radius_m:
            chain_audit.append({"member_row_ids": sorted(row_ids[i] for i in indices),
                                "size": len(indices), "diameter_m": diameter})

    retained, quarantined = [], []
    for i, (row_id, row) in enumerate(zip(row_ids, canonical_rows)):
        reports = [[_number(row[PHASE_COLUMNS[p][t]]) for p in PHASE_ORDER] for t in range(len(REPORT_DAYS))]
        record = {
            "row_id": row_id, "group_id": group_ids[i], "split": assign_split(group_ids[i], seed),
            "api_text": row["API"], "first_prod_date_raw": row["FirstProdDate"],
            "reports": reports, "report_mask": [[v is not None for v in values] for values in reports],
            "covariates_raw": {c: row[c] for c in covariate_columns},
            "raw_row_numbers": sorted(item[0] for item in exports[row_id]),
            "duplicate_count": len(exports[row_id]) - 1,
            "quarantine_reasons": sorted(reasons[i]),
        }
        (quarantined if reasons[i] else retained).append(record)

    production_audit = {}
    for p, phase in enumerate(PHASE_ORDER):
        values = [[row["reports"][t][p] for t in range(len(REPORT_DAYS))] for row in retained]
        production_audit[phase] = {
            "missing_cells": sum(v is None for row in values for v in row),
            "negative_cells": sum(v is not None and v < 0 for row in values for v in row),
            "decreasing_adjacent_pairs": sum(a is not None and b is not None and b < a
                                            for row in values for a, b in zip(row, row[1:])),
            "negative_cells_by_day": {str(day): sum(row[t] is not None and row[t] < 0 for row in values)
                                      for t, day in enumerate(REPORT_DAYS)},
        }
    audit = {
        "schema_version": SCHEMA_VERSION, "raw_rows": len(rows), "semantic_unique_rows": n,
        "exact_duplicate_exports_removed": sum(count - 1 for count in exact_counts.values()),
        "additional_numeric_semantic_duplicates_removed": len(exact_counts) - n,
        "retained_rows": len(retained), "quarantined_rows": len(quarantined),
        "quarantine_reason_counts": dict(sorted(Counter(reason for record in quarantined for reason in record["quarantine_reasons"]).items())),
        "conflicting_same_observation_groups": len(observation_conflicts),
        "ambiguous_same_location_date_groups": len(location_conflicts),
        "matching_complete_trajectory_groups": len(repeated_trajectories),
        "missing_or_invalid_coordinates": sum(c is None for c in coordinates),
        "spatial_radius_m": radius_m, "spatial_neighbor_edges": edge_count,
        "maximum_neighbor_edge_m": max_edge_m, "spatial_components": len(spatial_groups),
        "spatial_chains_over_radius": sorted(chain_audit, key=lambda entry: entry["member_row_ids"]),
        "leakage_groups_including_quarantine": len(groups),
        "maximum_leakage_group_rows": max(map(len, groups.values())),
        "split_retained_rows": {name: sum(r["split"] == name for r in retained) for name in SPLIT_NAMES},
        "split_retained_groups": {name: len({r["group_id"] for r in retained if r["split"] == name}) for name in SPLIT_NAMES},
        "retained_production": production_audit,
        "covariate_policy": {"primary_verified_columns": [], "preserved_raw_columns": covariate_columns,
                             "reason": "Units and availability at cutoff have not been verified; no unverified attributes enter primary features."},
        "identity_policy": "API remains exact text; semantic export hashes are not verified physical well identifiers.",
        "numeric_identity_columns": numeric_columns,
        "observation_policy": "Preserve signed cumulative reports and missingness; no cleaning or imputation; day 0 is the export baseline.",
    }
    return {"records": retained, "quarantine": quarantined, "audit": audit,
            "groups": sorted(group_details, key=lambda entry: entry["group_id"])}


def history_at_cutoff(record, cutoff_day):
    """Expose only permitted report values, timestamps and masks; no future keys."""
    if not math.isfinite(cutoff_day) or cutoff_day < 0:
        raise ValueError("cutoff_day must be nonnegative and finite")
    indices = [t for t, day in enumerate(REPORT_DAYS) if day <= cutoff_day]
    return {"report_days": [REPORT_DAYS[t] for t in indices],
            "phase_order": list(PHASE_ORDER), "cutoff_day": cutoff_day,
            "reports": [list(record["reports"][t]) for t in indices],
            "report_mask": [list(record["report_mask"][t]) for t in indices]}


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Missing or repeated CSV column names")
        rows = list(reader)
    return columns, rows


def prepare_dataset(raw_path, output_dir, *, manifest_path=None, radius_m=50.0, seed=42):
    """Verify, curate and atomically accept a never-overwritten split directory."""
    raw_path, output_dir = Path(raw_path), Path(output_dir)
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    columns, rows = _read_rows(raw_path)
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text())
        actual = {"sha256": source_hash, "size_bytes": raw_path.stat().st_size,
                  "raw_rows": len(rows), "columns": columns}
        for name, value in actual.items():
            if manifest[name] != value:
                raise ValueError(f"Raw manifest mismatch: {name}")
    result = curate_rows(rows, radius_m=radius_m, seed=seed)
    payloads = {
        "records.jsonl": b"".join(_json_bytes(row) for row in result["records"]),
        "quarantine.jsonl": b"".join(_json_bytes(row) for row in result["quarantine"]),
        "curation_audit.json": _json_bytes(result["audit"]),
    }
    policy = {
        "version": "bakken-conservative-v1", "split_seed": seed, "spatial_radius_m": radius_m,
        "deduplication": "All fields equal after exact numeric normalization except API/date kept as text",
        "quarantine": ["ambiguous same location and first-production date", "invalid production token",
                       "no finite positive-time reports", "missing or invalid surface coordinates"],
        "grouping": "Identical complete joint reports including missingness OR 50m connected surface neighbors, before quarantine",
        "observation_transforms": [], "primary_verified_covariates": [],
        "phase_order": list(PHASE_ORDER), "report_days": list(REPORT_DAYS),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "accepted_immutable", "dataset_sha256": source_hash,
        "phase_order": list(PHASE_ORDER), "report_days": list(REPORT_DAYS),
        "volume_units": dict(zip(PHASE_ORDER, VOLUME_UNITS)), "time_unit": "day",
        "seed": seed, "split_probabilities": dict(zip(SPLIT_NAMES, SPLIT_PROBABILITIES)),
        "assignment": "SHA256(bakken-split-v1:seed:group_id), integer 80/85/90/100 percent boundaries",
        "spatial_radius_m": radius_m, "spatial_distance": "great-circle spherical Earth radius 6371008.8 m",
        "grouping": "Connected components of identical joint report trajectories (including masks) and surface neighbors; build before quarantine.",
        "curation_policy_version": "bakken-conservative-v1",
        "curation_policy": policy, "curation_policy_sha256": _digest(policy),
        "curation_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ("data.py", "data_audit.py")},
        "prior_bakken_outcome_exposure": True, "groups": result["groups"],
        "retained_row_ids": [row["row_id"] for row in result["records"]],
        "quarantined_row_ids": [row["row_id"] for row in result["quarantine"]],
        "artifact_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
    }
    payloads["split_manifest.json"] = _json_bytes(manifest)
    if output_dir.exists():
        if all((output_dir / name).is_file() and (output_dir / name).read_bytes() == content
               for name, content in payloads.items()):
            return load_dataset(output_dir)
        raise FileExistsError(f"Accepted output exists and differs; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bakken-curation-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "accepted"
        stage.mkdir()
        for name, content in payloads.items():
            (stage / name).write_bytes(content)
        # The destination must remain absent: rename cannot replace nonempty dirs.
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during curation: {output_dir}")
        os.rename(stage, output_dir)
    return load_dataset(output_dir)


def load_dataset(directory):
    """Load retained rows only, after verifying accepted artifact hashes."""
    directory = Path(directory)
    manifest = json.loads((directory / "split_manifest.json").read_text())
    if manifest.get("status") != "accepted_immutable" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or unaccepted split manifest")
    if manifest.get("phase_order") != list(PHASE_ORDER) or manifest.get("report_days") != list(REPORT_DAYS):
        raise ValueError("Unexpected phase or time ordering")
    if set(manifest.get("artifact_sha256", {})) != {"records.jsonl", "quarantine.jsonl", "curation_audit.json"}:
        raise ValueError("Accepted artifact hash list is incomplete")
    if manifest.get("curation_policy_sha256") != _digest(manifest.get("curation_policy")):
        raise ValueError("Curation policy hash mismatch")
    for name, expected in manifest["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Artifact names must be local file names")
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Accepted artifact hash mismatch: {name}")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    if [row["row_id"] for row in records] != manifest["retained_row_ids"]:
        raise ValueError("Retained row identities disagree with manifest")
    membership = {row_id: (group["group_id"], group["split"])
                  for group in manifest["groups"] for row_id in group["row_ids"]}
    for row in records:
        if membership.get(row["row_id"]) != (row["group_id"], row["split"]):
            raise ValueError("Record split/group disagrees with accepted manifest")
    return {"records": records, "manifest": manifest,
            "audit": json.loads((directory / "curation_audit.json").read_text())}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/raw/bakken.csv")
    parser.add_argument("--manifest", type=Path, default=root / "data/manifest.json")
    parser.add_argument("--output", type=Path, default=root / "data/processed/bakken_v1")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare_dataset(args.data, args.output, manifest_path=args.manifest,
                                 radius_m=args.radius_m, seed=args.seed)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Curation failed: {error}\n")
    print(json.dumps({"output": str(args.output), "retained_rows": len(result["records"]),
                      "quarantined_rows": result["audit"]["quarantined_rows"],
                      "split_rows": result["audit"]["split_retained_rows"]}, indent=2))


if __name__ == "__main__":
    main()
