"""Consequential identity, leakage, raw-report and accepted-split invariants."""
import copy
import csv
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from joint_production.data import (
    EARTH_RADIUS_M, PHASE_ORDER, REPORT_DAYS, _numeric_token, _spatial_edges,
    curate_rows, history_at_cutoff, load_dataset, prepare_dataset,
)
from joint_production.data_audit import PHASE_COLUMNS


def row(index=0, *, offset_m=0, date=None):
    record = {"API": "3.31E+13", "FirstProdDate": date or f"{index + 1}/1/14",
              "Latitude": "48", "Longitude": str(-103 + offset_m / (EARTH_RADIUS_M * math.cos(math.radians(48))) * 180 / math.pi),
              "LateralLength_FT": str(8000 + index), "EffectivePorosity_PCT": "0.04"}
    for p, phase in enumerate(PHASE_ORDER):
        for t, column in enumerate(PHASE_COLUMNS[phase]):
            record[column] = str((index + 1) * (p + 1) * REPORT_DAYS[t])
    return record


class CurationTests(unittest.TestCase):
    def test_duplicate_exports_normalize_numbers_but_preserve_api_text(self):
        original = row()
        duplicate = copy.deepcopy(original)
        duplicate["LateralLength_FT"] = "8.000e3"
        duplicate["Prod0DaysOil_BBL_rs"] = "-0.000"
        out = curate_rows([original, original, duplicate])
        self.assertEqual(len(out["records"]), 1)
        self.assertEqual(out["records"][0]["duplicate_count"], 2)
        self.assertEqual(out["records"][0]["api_text"], "3.31E+13")
        self.assertEqual(out["audit"]["exact_duplicate_exports_removed"], 1)
        self.assertEqual(out["audit"]["additional_numeric_semantic_duplicates_removed"], 1)
        # Numeric identity must not round long decimal strings into equality.
        self.assertNotEqual(_numeric_token("1.12345678901234567890123456781"),
                            _numeric_token("1.12345678901234567890123456782"))

    def test_conflicting_covariates_and_histories_are_quarantined(self):
        first = row()
        covariate_conflict = dict(first, LateralLength_FT="9000")
        history_conflict = dict(first, Prod360DaysOil_BBL_rs="999")
        out = curate_rows([first, covariate_conflict, history_conflict])
        self.assertEqual(out["records"], [])
        self.assertEqual(len(out["quarantine"]), 3)
        self.assertEqual(out["audit"]["conflicting_same_observation_groups"], 1)
        self.assertEqual(out["audit"]["ambiguous_same_location_date_groups"], 1)

    def test_spatial_chain_and_quarantined_bridge_prevent_leakage(self):
        first, bridge, last = row(0, offset_m=0), row(1, offset_m=40), row(2, offset_m=80)
        bridge_conflict = dict(bridge, LateralLength_FT="9500")
        out = curate_rows([first, bridge, bridge_conflict, last])
        self.assertEqual(len(out["records"]), 2)
        self.assertEqual(len(out["quarantine"]), 2)
        self.assertEqual(len({r["group_id"] for r in out["records"]}), 1)
        self.assertEqual(len({r["split"] for r in out["records"]}), 1)
        self.assertEqual(len(out["audit"]["spatial_chains_over_radius"]), 1)
        self.assertGreater(out["audit"]["spatial_chains_over_radius"][0]["diameter_m"], 75)

    def test_exact_trajectory_links_distant_records(self):
        first = row()
        remote = dict(first, Longitude="-104", FirstProdDate="2/1/14")
        out = curate_rows([first, remote])
        self.assertEqual(len(out["records"]), 2)
        self.assertEqual(out["records"][0]["group_id"], out["records"][1]["group_id"])

    def test_order_does_not_change_identity_group_or_split(self):
        records = [row(i, offset_m=i * 70) for i in range(10)]
        records.append(dict(records[2]))
        forward = curate_rows(records)
        random.Random(721).shuffle(records)
        reverse = curate_rows(records)
        key = lambda out: {(r["row_id"], r["group_id"], r["split"]) for r in out["records"]}
        self.assertEqual(key(forward), key(reverse))
        self.assertEqual(forward["groups"], reverse["groups"])

    def test_missing_coordinates_quarantined_and_api_not_group_key(self):
        missing = dict(row(2), Latitude="nan")
        out = curate_rows([row(0, offset_m=0), row(1, offset_m=500), missing])
        self.assertEqual(len(out["records"]), 2)
        self.assertEqual(len({r["group_id"] for r in out["records"]}), 2)
        self.assertEqual(out["quarantine"][0]["quarantine_reasons"], ["missing_or_invalid_surface_coordinates"])

    def test_signed_reports_missingness_and_cutoff_do_not_use_future(self):
        source = row()
        source["Prod30DaysGas_MCF_rs"] = "-17"
        source["Prod60DaysOil_BBL_rs"] = ""
        out = curate_rows([source])["records"][0]
        self.assertEqual(out["reports"][1], [30.0, -17.0, 90.0])
        self.assertIsNone(out["reports"][2][0])
        self.assertFalse(out["report_mask"][2][0])
        history = history_at_cutoff(out, 60)
        changed = copy.deepcopy(out)
        changed["reports"][3:] = [[999.0] * 3 for _ in range(5)]
        changed["covariates_raw"]["LateralLength_FT"] = "10000000"
        self.assertEqual(history, history_at_cutoff(changed, 60))
        self.assertEqual(history["report_days"], [0, 30, 60])
        self.assertNotIn("covariates_raw", history)
        self.assertNotIn("row_id", history)
        history["reports"][1][1] = 100
        self.assertEqual(out["reports"][1][1], -17)

    def test_invalid_production_is_not_silently_missing(self):
        for value in ("garbage", "inf", "1e1000"):
            source = dict(row(), Prod30DaysOil_BBL_rs=value)
            out = curate_rows([source])
            self.assertEqual(out["records"], [])
            self.assertIn("invalid_production_token", out["quarantine"][0]["quarantine_reasons"])

    def test_spatial_bin_search_matches_brute_force_including_date_line(self):
        from joint_production.data import _distance_m
        rng = random.Random(7)
        coordinates = [(48 + rng.uniform(-0.002, 0.002), -103 + rng.uniform(-0.002, 0.002)) for _ in range(80)]
        coordinates += [(0, 179.9999), (0, -179.9999), None]
        expected = {(i, j) for i in range(len(coordinates)) for j in range(i + 1, len(coordinates))
                    if coordinates[i] is not None and coordinates[j] is not None
                    and _distance_m(coordinates[i], coordinates[j]) <= 50}
        self.assertEqual({(i, j) for i, j, _ in _spatial_edges(coordinates, 50)}, expected)

    def test_accepted_output_is_idempotent_tamper_evident_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / "raw.csv", root / "accepted"
            records = [row(0), row(1, offset_m=300)]
            with raw.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            data_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
            manifest_path = root / "raw_manifest.json"
            manifest = {"sha256": data_hash, "size_bytes": raw.stat().st_size,
                        "raw_rows": len(records), "columns": list(records[0])}
            manifest_path.write_text(json.dumps(manifest))
            manifest["sha256"] = "invalid"
            incorrect_manifest = root / "incorrect_manifest.json"
            incorrect_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Raw manifest mismatch"):
                prepare_dataset(raw, output, manifest_path=incorrect_manifest)
            self.assertFalse(output.exists())
            first = prepare_dataset(raw, output, manifest_path=manifest_path)
            split_bytes = (output / "split_manifest.json").read_bytes()
            self.assertEqual(first, prepare_dataset(raw, output))
            with self.assertRaises(FileExistsError):
                prepare_dataset(raw, output, seed=43)
            self.assertEqual(split_bytes, (output / "split_manifest.json").read_bytes())
            self.assertEqual(data_hash, hashlib.sha256(raw.read_bytes()).hexdigest())
            (output / "records.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_dataset(output)


if __name__ == "__main__":
    unittest.main()
