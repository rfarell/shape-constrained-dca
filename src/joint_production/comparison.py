"""Whole-group bootstrap weights for paired comparisons.

Spatial groups, not wells, are the resampling units. The bootstrap uncertainty is
conditional on the two supplied fitted models.
"""
from __future__ import annotations

import numpy as np


def whole_group_weights(groups, replicates, seed):
    """Return well multiplicities: every well in a drawn group has equal weight."""
    groups = np.asarray(groups)
    labels, inverse = np.unique(groups, return_inverse=True)
    if groups.ndim != 1 or len(labels) < 2:
        raise ValueError("At least two distinct spatial groups are required")
    if not isinstance(replicates, (int, np.integer)) or replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    group_weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=replicates)
    return group_weights[:, inverse]
