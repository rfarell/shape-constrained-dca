"""Leakage, missingness, units and metadata invariants of secondary inputs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from joint_production.static_features import STATIC_FEATURE_NAMES, StaticFeaturePreprocessor


def records(values=(1.0, 2.0, 4.0), split="train"):
    return [
        {"row_id": f"row-{i}", "split": split,
         "covariates_raw": {name: str(value) for name in STATIC_FEATURE_NAMES},
         "reports": [[0, 0, 0], [1e15, 2e15, 3e15]]}
        for i, value in enumerate(values)
    ]


def test_feature_order_is_exact_manifest_order_without_location_or_other_metadata():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/manifest.json").read_text())
    assert list(STATIC_FEATURE_NAMES) == manifest["columns"][1:18]
    assert len(STATIC_FEATURE_NAMES) == 17


@pytest.mark.parametrize("split", ["validation", "calibration", "test"])
def test_fit_rejects_any_held_out_row_even_when_caller_claims_train(split):
    training = records()
    training[-1]["split"] = split
    with pytest.raises(ValueError, match="TRAIN"):
        StaticFeaturePreprocessor.fit(training, split="train")
    with pytest.raises(ValueError, match="TRAIN"):
        StaticFeaturePreprocessor.fit(records(), split=split)


def test_fit_rejects_duplicate_or_missing_training_ids():
    training = records()
    training[-1]["row_id"] = training[0]["row_id"]
    with pytest.raises(ValueError, match="distinct"):
        StaticFeaturePreprocessor.fit(training)
    training[-1].pop("row_id")
    with pytest.raises(ValueError, match="row_id"):
        StaticFeaturePreprocessor.fit(training)


def test_frozen_train_transform_and_raw_records_are_unmodified_by_query_values():
    training = records()
    original = deepcopy(training)
    preprocessing = StaticFeaturePreprocessor.fit(training)
    metadata = preprocessing.to_dict()
    query = records((1e9, -1e9), "validation")
    first = preprocessing.transform(query)
    query[0]["reports"] = [[-1e100] * 3 for _ in range(8)]
    query[0]["covariates_raw"]["Latitude"] = "nonsense-not-a-feature"
    query[0]["covariates_raw"]["Longitude"] = "Infinity"
    query[0]["api_text"] = "unknown"
    second = preprocessing.transform(query)
    np.testing.assert_array_equal(first, second)
    assert preprocessing.to_dict() == metadata
    assert training == original
    np.testing.assert_allclose(preprocessing.center, 2.0)
    np.testing.assert_allclose(preprocessing.scale, np.std([1, 2, 4], ddof=0))


def test_missing_values_remain_distinct_from_observed_median_and_zero_control():
    training = records((-1, 0, 1))
    preprocessor = StaticFeaturePreprocessor.fit(training)
    query = records((0, 0, 0), "validation")
    query[1]["covariates_raw"][STATIC_FEATURE_NAMES[0]] = "n/a"
    query[2]["covariates_raw"][STATIC_FEATURE_NAMES[0]] = None
    actual = preprocessor.transform(query)
    assert actual.shape == (3, 34)
    np.testing.assert_array_equal(actual[:, 0], 0)
    np.testing.assert_array_equal(actual[:, 17], [1, 0, 0])
    np.testing.assert_array_equal(preprocessor.transform(query, zero_control=True), np.zeros((3, 34)))
    assert preprocessor.transform([]).shape == (0, 34)


def test_all_missing_and_constant_training_columns_have_explicit_fallbacks():
    training = records((5, 5, 5))
    for row in training:
        row["covariates_raw"][STATIC_FEATURE_NAMES[0]] = "NaN"
    preprocessing = StaticFeaturePreprocessor.fit(training)
    assert preprocessing.center[0] == 0
    assert preprocessing.scale[0] == 1
    assert preprocessing.scale[1] == 1e-8
    assert preprocessing.finite_counts[:2] == (0, 3)
    assert np.isfinite(preprocessing.transform(training)).all()


def test_consistent_positive_unit_rescaling_is_removed_by_training_standardization():
    training = records()
    training[1]["covariates_raw"][STATIC_FEATURE_NAMES[0]] = ""
    query = records((0.5, 5), "validation")
    factors = np.geomspace(0.001, 1000, 17)
    changed_train, changed_query = deepcopy(training), deepcopy(query)
    for collection in (changed_train, changed_query):
        for row in collection:
            for name, factor in zip(STATIC_FEATURE_NAMES, factors):
                value = row["covariates_raw"][name]
                if value:
                    row["covariates_raw"][name] = str(float(value) * factor)
    original = StaticFeaturePreprocessor.fit(training).transform(query)
    changed = StaticFeaturePreprocessor.fit(changed_train).transform(changed_query)
    np.testing.assert_allclose(original, changed, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("token", ["bad", "1,000", "infinity", "-inf", "1e999", "+nan", True, {}, float("inf")])
def test_invalid_tokens_fail_deterministically_without_coercion(token):
    training = records()
    preprocessing = StaticFeaturePreprocessor.fit(training)
    training[0]["covariates_raw"][STATIC_FEATURE_NAMES[0]] = token
    with pytest.raises(ValueError, match="attribute"):
        preprocessing.transform(training)
    with pytest.raises(ValueError, match="attribute"):
        preprocessing.transform(training, zero_control=True)


@pytest.mark.parametrize("token", ["", " NA ", "NaN", "nUlL", "none", "N/A", None, float("nan")])
def test_declared_missing_tokens_map_to_missing_indicator(token):
    training = records()
    preprocessing = StaticFeaturePreprocessor.fit(training)
    training[0]["covariates_raw"][STATIC_FEATURE_NAMES[0]] = token
    values = preprocessing.transform(training)
    assert values[0, 0] == values[0, 17] == 0


def test_absent_feature_does_not_silently_become_missing():
    training = records()
    training[0]["covariates_raw"].pop(STATIC_FEATURE_NAMES[0])
    with pytest.raises(ValueError, match="lacks selected"):
        StaticFeaturePreprocessor.fit(training)


def test_json_roundtrip_has_ordered_training_hash_and_rejects_schema_changes():
    training = records()
    preprocessing = StaticFeaturePreprocessor.fit(training)
    metadata = json.loads(json.dumps(preprocessing.to_dict(), allow_nan=False))
    restored = StaticFeaturePreprocessor.from_dict(metadata)
    expected_hash = hashlib.sha256((json.dumps([row["row_id"] for row in training],
                                              separators=(",", ":")) + "\n").encode()).hexdigest()
    assert metadata["training_row_sha256"] == expected_hash
    assert StaticFeaturePreprocessor.fit(training[::-1]).training_row_sha256 != expected_hash
    np.testing.assert_array_equal(preprocessing.transform(training), restored.transform(training))
    for key, value in [("feature_names", list(reversed(STATIC_FEATURE_NAMES))),
                       ("scale", [0] * 17), ("fit_split", "test"),
                       ("finite_counts", [4] * 17), ("training_row_sha256", "unknown")]:
        changed = deepcopy(metadata)
        changed[key] = value
        with pytest.raises(ValueError):
            StaticFeaturePreprocessor.from_dict(changed)


def test_transform_rejects_overflow_even_if_raw_values_are_finite():
    preprocessing = StaticFeaturePreprocessor.fit(records((1, 1, 1)))
    with pytest.raises(ValueError, match="overflow"):
        preprocessing.transform(records((1e308,), "validation"))
