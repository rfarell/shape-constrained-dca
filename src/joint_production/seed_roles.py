"""Resolve experiment randomness without changing model training settings.

Legacy configurations use the training seed for all three roles. Explicit
prediction and metric-bootstrap seeds let future runs vary training randomness
while holding evaluation streams fixed. Equal prediction seeds specify equal
RNG initialization, not identical draws across different model algorithms.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral


def resolve_seed_roles(experiment: Mapping, training_seed: int) -> dict:
    """Return validated effective seeds and their configuration provenance."""
    def seed_value(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)

    training_seed = seed_value(training_seed, "training_seed")
    roles = {"training_seed": training_seed}
    for name in ("prediction_seed", "evaluation_bootstrap_seed"):
        explicit = name in experiment
        roles[name] = seed_value(experiment[name] if explicit else training_seed, name)
        roles[name + "_source"] = ("experiment." + name if explicit else "training_seed_legacy_default")
    roles["prediction_draw_seed_rule"] = "prediction_seed * 10000 + origin_day"
    return roles
