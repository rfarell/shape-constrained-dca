"""Independent evaluation streams retain the historical default seed mapping."""
import numpy as np
import pytest

from joint_production.seed_roles import resolve_seed_roles


def test_legacy_forecast_and_bootstrap_streams_are_exact():
    for training_seed in (0, 42, 43):
        roles = resolve_seed_roles({}, training_seed)
        assert roles["evaluation_bootstrap_seed"] == training_seed
        assert roles["prediction_seed_source"] == "training_seed_legacy_default"
        for origin in (60, 90, 180, 360, 720):
            before = np.random.default_rng(training_seed * 10000 + origin).normal(size=(3, 8))
            after = np.random.default_rng(roles["prediction_seed"] * 10000 + origin).normal(size=(3, 8))
            np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("name", ["prediction_seed", "evaluation_bootstrap_seed"])
def test_each_override_is_independent_and_zero_is_a_valid_explicit_seed(name):
    roles = resolve_seed_roles({name: 0}, 43)
    other = "evaluation_bootstrap_seed" if name == "prediction_seed" else "prediction_seed"
    assert roles[name] == 0 and roles[other] == 43 and roles["training_seed"] == 43
    assert roles[name + "_source"] == "experiment." + name


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "42", None])
@pytest.mark.parametrize("name", ["prediction_seed", "evaluation_bootstrap_seed"])
def test_invalid_explicit_seed_cannot_silently_fall_back(name, value):
    with pytest.raises(ValueError, match=name):
        resolve_seed_roles({name: value}, 42)
