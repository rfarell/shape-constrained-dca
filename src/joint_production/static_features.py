"""Training-only preprocessing of the Bakken secondary-attribute experiment.

The selected 17 raw fields exclude location, identifiers, dates, and reports.
Their availability is an experiment assumption, not established by this parser.
Raw records remain unchanged. Output order is standardized values followed by
presence indicators; callers use all-zero output for the matched control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Real

import numpy as np

from .data_audit import MISSING


STATIC_FEATURE_NAMES = (
    "LateralLength_FT", "FracStages", "AverageStageSpacing_FT",
    "ProppantIntensity_LBSPerFT", "FluidIntensity_BBLPerFT", "Isopach_FT",
    "EffectivePorosity_PCT", "TopOfZone_FT", "GammaRay_API", "Resistivity_OHMSM",
    "ClayVolume_PCT", "WaterSaturation_PCT", "PhiH_FT", "HCPV_PCT",
    "TotalOrganicCarbon_WTPCT", "BottomOfZone_FT", "TrueVerticalDepth_FT",
)
STATIC_OUTPUT_NAMES = tuple(f"standardized:{name}" for name in STATIC_FEATURE_NAMES) + tuple(
    f"present:{name}" for name in STATIC_FEATURE_NAMES
)
SCALE_FLOOR = 1e-8
PARSER_POLICY = {
    "missing_strings_case_insensitive_after_strip": sorted(MISSING),
    "null_and_numeric_nan": "missing",
    "malformed_nonempty_token_boolean_or_infinity": "raise ValueError",
    "absent_selected_field": "raise ValueError",
    "unit_conversion": "none; standardization uses training values in raw export units",
}


def _digest(value) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
               + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse(value, *, row: int, name: str) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in MISSING:
            return np.nan
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Invalid secondary attribute type at row {row}, feature {name}")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid secondary attribute at row {row}, feature {name}") from error
    if np.isnan(number) and not isinstance(value, str):
        return np.nan
    if not np.isfinite(number):
        raise ValueError(f"Nonfinite secondary attribute at row {row}, feature {name}")
    return number


def _matrix(records: Sequence[Mapping]) -> np.ndarray:
    matrix = np.empty((len(records), len(STATIC_FEATURE_NAMES)), dtype=np.float64)
    for index, record in enumerate(records):
        raw = record.get("covariates_raw")
        if not isinstance(raw, Mapping):
            raise ValueError(f"Record {index} lacks a covariates_raw mapping")
        for column, name in enumerate(STATIC_FEATURE_NAMES):
            if name not in raw:
                raise ValueError(f"Record {index} lacks selected secondary attribute {name}")
            matrix[index, column] = _parse(raw[name], row=index, name=name)
    return matrix


@dataclass(frozen=True)
class StaticFeaturePreprocessor:
    """Immutable median centering and population-SD scaling fitted on TRAIN.

    Missing values become zero standardized values with presence indicator 0.
    Observed values equal to the median retain indicator 1. An entirely missing
    training column has center 0 and scale 1; finite constant columns use the
    fixed scale floor. Fit rejects held-out rows and duplicate row identities.
    """

    center: tuple[float, ...]
    scale: tuple[float, ...]
    finite_counts: tuple[int, ...]
    n_training_rows: int
    training_row_sha256: str

    @classmethod
    def fit(cls, train_records: Sequence[Mapping], *, split: str = "train") -> "StaticFeaturePreprocessor":
        if split != "train" or not train_records:
            raise ValueError("Secondary preprocessing requires nonempty TRAIN records")
        if any(record.get("split") != "train" for record in train_records):
            raise ValueError("Secondary preprocessing must be fitted on TRAIN records only")
        row_ids = [record.get("row_id") for record in train_records]
        if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
            raise ValueError("Training records require nonempty text row_id values")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Training records must have distinct row_id values")
        matrix = _matrix(train_records)
        centers, scales, counts = [], [], []
        for column in matrix.T:
            observed = column[np.isfinite(column)]
            counts.append(int(len(observed)))
            if len(observed):
                with np.errstate(over="ignore", invalid="ignore"):
                    center = float(np.median(observed))
                    scale = max(float(np.std(observed, ddof=0)), SCALE_FLOOR)
                if not np.isfinite(center) or not np.isfinite(scale):
                    raise ValueError("Secondary training values overflow centering or scaling")
            else:
                center, scale = 0.0, 1.0
            centers.append(center)
            scales.append(scale)
        return cls(tuple(centers), tuple(scales), tuple(counts), len(row_ids), _digest(row_ids))

    def transform(self, records: Sequence[Mapping], *, zero_control: bool = False) -> np.ndarray:
        """Transform without fitting; controls zero both values and masks.

        The control still parses and validates its raw inputs. This gives both
        arms the same eligibility and prevents malformed data being hidden by
        a control flag. No report or nonselected covariate is read.
        """
        matrix = _matrix(records)
        present = np.isfinite(matrix)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.where(present, (matrix - self.center) / self.scale, 0.0)
        if not np.isfinite(values).all():
            raise ValueError("Secondary values overflow the frozen training transform")
        features = np.column_stack((values, present.astype(np.float64)))
        if zero_control:
            features.fill(0.0)
        return features

    def to_dict(self) -> dict:
        """Return a JSON-safe description including ordered training identity."""
        return {
            "schema_version": 1, "kind": "bakken_static_17_median_std_mask",
            "fit_split": "train", "feature_names": list(STATIC_FEATURE_NAMES),
            "output_names": list(STATIC_OUTPUT_NAMES), "output_width": len(STATIC_OUTPUT_NAMES),
            "center": list(self.center), "scale": list(self.scale),
            "finite_counts": list(self.finite_counts), "n_training_rows": self.n_training_rows,
            "training_row_sha256": self.training_row_sha256,
            "training_row_hash_encoding": "SHA256 of ordered row_id JSON list, sorted keys, compact separators, trailing newline",
            "centering": "median of finite training values; all-missing fallback 0",
            "scaling": "population standard deviation of finite training values; all-missing fallback 1",
            "scale_floor": SCALE_FLOOR, "missing_standardized_value": 0.0,
            "zero_control": "all 34 values and presence indicators zero",
            "parser_policy": deepcopy(PARSER_POLICY),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "StaticFeaturePreprocessor":
        """Restore only this exact feature/transform schema; reject corruption."""
        if not isinstance(value, Mapping):
            raise ValueError("Secondary preprocessing metadata must be a mapping")
        try:
            center = tuple(float(x) for x in value["center"])
            scale = tuple(float(x) for x in value["scale"])
            counts = tuple(value["finite_counts"])
            rows = value["n_training_rows"]
            row_hash = value["training_row_sha256"]
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Malformed secondary preprocessing metadata") from error
        if (len(center) != 17 or len(scale) != 17 or len(counts) != 17
                or not np.isfinite(center).all() or not np.isfinite(scale).all()
                or any(x < SCALE_FLOOR for x in scale)
                or type(rows) is not int or rows <= 0
                or any(type(x) is not int or not 0 <= x <= rows for x in counts)
                or not isinstance(row_hash, str) or len(row_hash) != 64
                or any(x not in "0123456789abcdef" for x in row_hash)):
            raise ValueError("Invalid secondary preprocessing metadata values")
        restored = cls(center, scale, counts, rows, row_hash)
        if restored.to_dict() != dict(value):
            raise ValueError("Secondary preprocessing schema or policy does not match")
        return restored
