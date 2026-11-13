import numpy as np
import pytest

from joint_production.comparison import whole_group_weights


def test_group_multiplicity_preserves_every_member_and_group_draw_count():
    groups = np.array(['a', 'a', 'a', 'b', 'c', 'c'])
    weight = whole_group_weights(groups, 300, 7)
    assert np.array_equal(weight[:, 0], weight[:, 1])
    assert np.array_equal(weight[:, 0], weight[:, 2])
    assert np.array_equal(weight[:, 4], weight[:, 5])
    assert np.all(weight[:, [0, 3, 4]].sum(axis=1) == 3)
    assert np.any(weight.sum(axis=1) != len(groups))
    assert np.array_equal(weight, whole_group_weights(groups, 300, 7))


@pytest.mark.parametrize('groups, replicates', [(['a', 'a'], 10), (['a', 'b'], 1)])
def test_degenerate_bootstrap_rejected(groups, replicates):
    with pytest.raises(ValueError):
        whole_group_weights(groups, replicates, 0)
