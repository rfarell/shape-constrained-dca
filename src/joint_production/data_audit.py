"""Read-only integrity/schema audit; no cleaning, splitting, or modeling."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPORT_DAYS = (0, 30, 60, 90, 180, 360, 720, 1080)
PHASE_COLUMNS = {
    "oil": tuple(f"Prod{day}DaysOil_BBL_rs" for day in REPORT_DAYS),
    "gas": tuple(f"Prod{day}DaysGas_MCF_rs" for day in REPORT_DAYS),
    "water": tuple(f"Prod{day}DaysWtr_BBL_rs" for day in REPORT_DAYS),
}
MISSING = {"", "na", "nan", "null", "none", "n/a"}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def audit(path):
    """Summarize exact parsed-field matches; preserve all source values."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Missing or repeated CSV column names")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("CSV rows have inconsistent field counts")
    required = ["API", "Latitude", "Longitude", "FirstProdDate"]
    required += [c for group in PHASE_COLUMNS.values() for c in group]
    absent = sorted(set(required) - set(columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}")

    record_counts = Counter(tuple(row[c] for c in columns) for row in rows)
    observation_columns = ["Latitude", "Longitude", "FirstProdDate"] + required[4:]
    observations = defaultdict(list)
    for row in rows:
        observations[tuple(row[c] for c in observation_columns)].append(
            tuple(row[c] for c in columns)
        )
    conflicts = [group for group in observations.values() if len(set(group)) > 1]
    missing = {
        column: sum(row[column].strip().lower() in MISSING for row in rows)
        for column in columns
    }
    production = {}
    for phase, phase_columns in PHASE_COLUMNS.items():
        values = [[_number(row[c]) for c in phase_columns] for row in rows]
        negative = [[v for v in row if v is not None and v < 0] for row in values]
        decreases = [
            sum(a is not None and b is not None and b < a
                for a, b in zip(row, row[1:]))
            for row in values
        ]
        production[phase] = {
            "missing_or_nonfinite_or_nonnumeric_cells": sum(
                v is None for row in values for v in row
            ),
            "negative_cumulative_cells": sum(map(len, negative)),
            "rows_with_negative_cumulative": sum(bool(v) for v in negative),
            "decreasing_adjacent_pairs": sum(decreases),
            "rows_with_decreasing_cumulative": sum(n > 0 for n in decreases),
            "zero_cumulative_cells": sum(v == 0 for row in values for v in row),
        }
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "raw_rows": len(rows),
        "column_count": len(columns),
        "columns": columns,
        "unique_api_strings": len({row["API"] for row in rows}),
        "exact_field_unique_rows": len(record_counts),
        "exact_field_duplicate_rows": sum(n - 1 for n in record_counts.values()),
        "conflicting_observation_groups_exact_fields": len(conflicts),
        "rows_in_conflicting_observation_groups": sum(map(len, conflicts)),
        "missing_tokens_by_column": missing,
        "production": production,
        "notes": [
            "Raw rows are not assumed to be unique wells.",
            "Equality is exact parsed-field equality, not numeric normalization.",
            "Missing, invalid, nonfinite production are counted together here.",
            "No cleaning, grouping, split, or model evaluation is performed.",
        ],
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/raw/bakken.csv")
    parser.add_argument("--manifest", type=Path, default=root / "data/manifest.json")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.data)
        if args.verify:
            manifest = json.loads(args.manifest.read_text())
            for key in ("sha256", "size_bytes", "raw_rows", "columns"):
                if report[key] != manifest[key]:
                    raise ValueError(f"Manifest mismatch for {key}")
            print(f"Verified Bakken copy: {report['raw_rows']} raw rows, "
                  f"{report['column_count']} columns, SHA-256 matches.")
        if args.output:
            output = args.output.resolve()
            if output in {args.data.resolve(), args.manifest.resolve()}:
                raise ValueError("Output must not overwrite data or manifest")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Audit written to {output}")
        elif not args.verify:
            print(json.dumps(report, indent=2))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Audit failed: {error}\n")


if __name__ == "__main__":
    main()
