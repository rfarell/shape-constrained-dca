"""Training-only preprocessing of the 17 Bakken well attributes.

The selected 17 raw fields exclude location, identifiers, dates, and reports.
Their availability is an experiment assumption, not established by this parser.
Raw records remain unchanged. Output order is standardized values followed by
presence indicators; callers use all-zero output for the matched control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
SCALE_FLOOR = 1e-8


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
