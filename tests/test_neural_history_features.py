"""Controls for adding deterministic interval-rate summaries to the NLL model."""
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


def test_feature_control_has_identical_initial_parameters_and_actual_contexts():
    reports, days = data()
    initial_states, inputs, fitted = [], {}, {}
    original_network = neural._JointReportNetwork
    original_features = neural.last_interval_rate_features

    def network(*args, **kwargs):
        model = original_network(*args, **kwargs)
        initial_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return model

    for mode in ("zero", "last_interval"):
        inputs[mode] = []

        def features(raw, observed, grid, cutoff, scales, **kwargs):
            assert raw.dtype == torch.float64
            inputs[mode].append((raw.clone(), observed.clone(), cutoff.clone()))
            return original_features(raw, observed, grid, cutoff, scales, **kwargs)

        with patch.object(neural, "_JointReportNetwork", side_effect=network), \
                patch.object(neural, "last_interval_rate_features", side_effect=features):
            fitted[mode] = neural.ContextualGaussianNeuralBaseline(
                history_rate_features=mode, hidden_width=8, batch_size=7,
                epochs=2, log_every=0, seed=19).fit(reports, days, origins=[60, 90])
    for key in initial_states[0]:
        torch.testing.assert_close(initial_states[0][key], initial_states[1][key], rtol=0, atol=0)
    a, b = (fitted[mode].fit_diagnostics_ for mode in ("zero", "last_interval"))
    for key in ("initial_parameter_sha256", "context_mask_sha256_by_epoch", "optimizer_updates",
                "total_training_contexts", "phase_scales"):
        assert a[key] == b[key]
    assert a["optimizer_updates"] == 10  # Two epochs, five mixed-origin batches each.
    assert len(inputs["zero"]) == len(inputs["last_interval"]) == 10
    for left, right in zip(inputs["zero"], inputs["last_interval"]):
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)


def test_feature_checkpoint_and_default_ignore_future_values():
    reports, days = data()
    for mode in ("none", "zero", "last_interval"):
        model = neural.ContextualGaussianNeuralBaseline(history_rate_features=mode,
            hidden_width=8, epochs=1, batch_size=9, seed=11, log_every=0).fit(reports, days, origins=[60, 90])
        history = reports[:3].copy()
        history[:, days > 60] = np.nan
        contaminated = history.copy()
        contaminated[:, days > 60] = np.inf
        a = model.predict_samples(history, 60, 16, np.random.default_rng(5))
        b = model.predict_samples(contaminated, 60, 16, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)
        known = np.isfinite(history)
        np.testing.assert_array_equal(a.transpose(0, 2, 3, 1)[known],
                                      np.repeat(history[known, None], 16, axis=1))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            loaded = neural.ContextualGaussianNeuralBaseline.load(path)
            assert loaded.history_rate_features == mode
            np.testing.assert_array_equal(a, loaded.predict_samples(history, 60, 16, np.random.default_rng(5)))
            if mode == "none":
                checkpoint = torch.load(path, weights_only=True)
                del checkpoint["configuration"]["history_rate_features"]
                torch.save(checkpoint, path)
                legacy = neural.ContextualGaussianNeuralBaseline.load(path)
                np.testing.assert_array_equal(a, legacy.predict_samples(history, 60, 16, np.random.default_rng(5)))
