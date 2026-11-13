"""Last-interval rate summaries in the contextual neural comparator."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from joint_production import neural


def data():
    rng = np.random.default_rng(191)
    days = np.array([0., 30., 60., 90., 180.])
    reports = rng.lognormal(3, .3, (17, 1, 3)) * (1 - np.exp(-days[None, :, None] / 90))
    reports[0, 1, 0] = -.2
    reports[1, 2, 2] = np.nan
    return reports, days


def test_training_features_receive_native_float64_reports_once_per_batch():
    reports, days = data()
    inputs = []
    original_features = neural.last_interval_rate_features

    def features(raw, observed, grid, cutoff, scales, **kwargs):
        assert raw.dtype == torch.float64
        inputs.append((raw.clone(), observed.clone(), cutoff.clone()))
        return original_features(raw, observed, grid, cutoff, scales, **kwargs)

    with patch.object(neural, "last_interval_rate_features", side_effect=features):
        model = neural.ContextualGaussianNeuralBaseline(
            hidden_width=8, batch_size=7, epochs=2, log_every=0, seed=19).fit(reports, days, origins=[60, 90])
    assert model.fit_diagnostics_["optimizer_updates"] == 10  # Two epochs, five mixed-origin batches each.
    assert len(inputs) == 10
    for raw, observed, cutoff in inputs:
        assert not observed[days[None, :] > cutoff[:, None].numpy()].any()


def test_future_values_and_checkpoint_do_not_change_draws():
    reports, days = data()
    model = neural.ContextualGaussianNeuralBaseline(
        hidden_width=8, epochs=1, batch_size=9, seed=11, log_every=0).fit(reports, days, origins=[60, 90])
    history = reports[:3].copy()
    history[:, days > 60] = np.nan
    contaminated = history.copy()
    contaminated[:, days > 60] = np.inf
    a = model.predict_samples(history, 60, 16, np.random.default_rng(5))
    b = model.predict_samples(contaminated, 60, 16, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    known = np.isfinite(history)
    np.testing.assert_array_equal(a.transpose(0, 2, 3, 1)[known], np.repeat(history[known, None], 16, axis=1))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        model.save(path)
        loaded = neural.ContextualGaussianNeuralBaseline.load(path)
    np.testing.assert_array_equal(a, loaded.predict_samples(history, 60, 16, np.random.default_rng(5)))
