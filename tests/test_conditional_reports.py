"""Report-law sampling, matched proper-score training, and cutoff regressions."""

import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from joint_production.conditional_rates import ConditionalRateForecaster, _u_scores
from joint_production.conditional_reports import ConditionalReportGaussianForecaster, _draw_reports
from joint_production.metrics import fit_increment_scales


DAYS = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=float)
ORIGINS = [60, 90, 180, 360, 720]


def synthetic_reports(n=17):
    rng = np.random.default_rng(29)
    amplitude = rng.lognormal(0, .35, (n, 1, 3)) * [150, 700, 120]
    reports = amplitude * (1-np.exp(-DAYS[None, :, None]/300))
    reports += rng.normal(0, .1, reports.shape)
    reports[:, 0] = 0
    return reports


class TestConditionalReportDraws(unittest.TestCase):
    def test_transformed_joint_moments_and_signed_report_support(self):
        location = torch.linspace(-.25, .25, 9, dtype=torch.float64)[None]
        variance = torch.linspace(.03, .11, 9, dtype=torch.float64)[None]
        factor = torch.tensor([[.1, -.2], [.2, .1], [.1, .1], [.3, -.1],
                               [.3, .2], [.2, .3], [.4, -.2], [.2, .3],
                               [.1, .4]], dtype=torch.float64)[None]
        actual = torch.full((1, 3, 3), float("nan"), dtype=torch.float64)
        actual[:, 0] = 0
        observed = torch.isfinite(actual)
        generator = torch.Generator().manual_seed(71)
        count = 30000
        reports = _draw_reports(location, variance, factor, actual, observed,
            torch.randn((1, count, 2), generator=generator, dtype=torch.float64),
            torch.randn((1, count, 9), generator=generator, dtype=torch.float64))
        z = torch.asinh(reports[0, :, 1:].flatten(start_dim=1)).numpy()
        expected_covariance = (factor[0] @ factor[0].T + torch.diag(variance[0])).numpy()[3:, 3:]
        np.testing.assert_allclose(z.mean(axis=0), location.numpy()[0, 3:], atol=.009, rtol=0)
        np.testing.assert_allclose(np.cov(z.T), expected_covariance, atol=.009, rtol=0)
        self.assertGreater(expected_covariance[0, 3], .1)
        self.assertTrue((reports[:, :, 1:] < 0).any())
        self.assertTrue((reports[:, :, 1:] > 0).any())
        torch.testing.assert_close(reports[:, :, 0], torch.zeros((1, count, 3), dtype=torch.float64))

    def test_report_domain_scores_include_hidden_origin_and_reach_all_heads(self):
        days = torch.tensor([0., 30, 60, 90])
        actual = torch.tensor([[[0., 0, 0], [.25, float("nan"), -.2],
                                [.7, float("nan"), .1], [1.2, .4, -.1]]])
        observed = torch.tensor([[[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]]], dtype=torch.bool)
        location = torch.linspace(-.2, .5, 12)[None].requires_grad_()
        variance = torch.full((1, 12), .12, requires_grad=True)
        factor = torch.linspace(-.2, .3, 24).reshape(1, 12, 2).requires_grad_()
        generator = torch.Generator().manual_seed(113)
        samples = _draw_reports(location, variance, factor, actual, observed,
            torch.randn((1, 4, 2), generator=generator),
            torch.randn((1, 4, 12), generator=generator))
        scale = torch.tensor([[.8, .4, .2]])
        marginal, energy, dimension = _u_scores(actual, samples, observed, days, 2, scale, torch.ones_like(scale))
        prediction = samples.detach().numpy()[0]
        target = actual.numpy()[0]
        increments = (prediction[:, 3]-prediction[:, 2])/scale.numpy()[0]
        target_increment = (target[3]-target[2])/scale.numpy()[0]
        valid = np.isfinite(target_increment)
        cell_scores = np.abs(increments[:, valid]-target_increment[valid]).mean(axis=0)
        cell_scores -= sum(np.abs(increments[i, valid]-increments[j, valid])
                           for i in range(4) for j in range(i+1, 4))/12
        self.assertAlmostEqual(float(marginal.detach()), cell_scores.sum()/3, places=6)
        scored = np.isfinite(target) & ~observed.numpy()[0] & (days.numpy() > 0)[:, None]
        residual = prediction[:, scored]-target[scored]
        expected_energy = np.linalg.norm(residual, axis=1).mean()
        expected_energy -= sum(np.linalg.norm(residual[i]-residual[j])
                               for i in range(4) for j in range(i+1, 4))/12
        self.assertEqual(int(dimension), 5)
        self.assertAlmostEqual(float(energy.detach()), expected_energy/np.sqrt(5), places=6)
        self.assertGreater(float(samples[0, :, 2, 2].std().detach()), 0)
        (marginal+energy).mean().backward()
        for parameter in (location, variance, factor):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_unused_observed_transform_overflow_is_masked_before_sinh(self):
        actual = torch.tensor([[[0., 0, 0], [1.23456789012345, float("nan"), -.7]]], dtype=torch.float64)
        observed = torch.isfinite(actual)
        location = torch.zeros((1, 6), dtype=torch.float64)
        location[observed.flatten(start_dim=1)] = 1000
        variance = torch.full_like(location, .1)
        factor = torch.zeros((1, 6, 1), dtype=torch.float64)
        reports = _draw_reports(location, variance, factor, actual, observed,
            torch.zeros((1, 2, 1), dtype=torch.float64), torch.zeros((1, 2, 6), dtype=torch.float64))
        self.assertTrue(torch.isfinite(reports).all())
        torch.testing.assert_close(reports[:, :, observed[0]], actual[:, None, observed[0]].expand(1, 2, -1), rtol=0, atol=0)
        location[0, 4] = 1000
        with self.assertRaises(FloatingPointError):
            _draw_reports(location, variance, factor, actual, observed,
                torch.zeros((1, 2, 1), dtype=torch.float64), torch.zeros((1, 2, 6), dtype=torch.float64))


class TestConditionalReportForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = synthetic_reports()
        cls.reports[0, 4, 1] = np.nan
        cls.reports[1, 2, 2] = np.nan
        cls.reports[2, 1, 0] = np.nan
        cls.settings = dict(epochs=2, hidden_width=8, batch_size=6, train_samples=4,
                            log_every=0, seed=17)
        def fit_with_score_trace(model, module):
            trace = []
            def record(*args):
                if args[0].dtype != torch.float32 or args[1].dtype != torch.float32:
                    raise AssertionError("Matched proper scores must receive float32 targets and samples")
                trace.append(tuple(args[i].detach().clone() if torch.is_tensor(args[i]) else args[i]
                                   for i in (0, 2, 3, 4, 5, 6)))
                return _u_scores(*args)
            with patch(module+"._u_scores", side_effect=record):
                model.fit(cls.reports, DAYS, origins=ORIGINS)
            return model, trace
        cls.model, cls.report_score_inputs = fit_with_score_trace(
            ConditionalReportGaussianForecaster(rank=2, **cls.settings), "joint_production.conditional_reports")
        cls.rate, cls.rate_score_inputs = fit_with_score_trace(
            ConditionalRateForecaster(**cls.settings), "joint_production.conditional_rates")

    def test_matched_scales_masks_missing_target_counts_and_remainder_schedule(self):
        for name in ("volume_scale", "increment_scales"):
            np.testing.assert_array_equal(getattr(self.model, name).cpu(), getattr(self.rate, name).cpu())
        for key in ("origins_days", "cells_origin_endpoint_days", "increment_observation_counts",
                    "phase_scales_native", "phase_order", "epoch_semantics", "contexts_per_epoch",
                    "optimizer_updates", "total_training_contexts", "context_mask_sha256_by_epoch",
                    "objective", "score_coefficients", "train_samples_iid", "uniform_origin_marginal_weights",
                    "missing_target_weighting"):
            self.assertEqual(self.model.fit_diagnostics_[key], self.rate.fit_diagnostics_[key], key)
        self.assertEqual(self.model.fit_diagnostics_["optimizer_updates"], 2*5*3)
        self.assertEqual(self.model.fit_diagnostics_["total_training_contexts"], 2*5*17)
        self.assertEqual(len(self.report_score_inputs), 30)
        self.assertEqual(len(self.report_score_inputs), len(self.rate_score_inputs))
        for report_call, rate_call in zip(self.report_score_inputs, self.rate_score_inputs):
            for report_input, rate_input in zip(report_call, rate_call):
                if torch.is_tensor(report_input):
                    torch.testing.assert_close(report_input, rate_input, rtol=0, atol=0, equal_nan=True)
                else:
                    self.assertEqual(report_input, rate_input)
        increments = np.stack([self.reports[:, DAYS == t, :][:, 0]-self.reports[:, DAYS == s, :][:, 0]
                               for s in ORIGINS for t in DAYS if t > s], axis=1)
        np.testing.assert_array_equal(self.model.increment_scales.cpu(), fit_increment_scales(increments))
        np.testing.assert_array_equal(self.model.fit_diagnostics_["increment_observation_counts"],
                                      np.isfinite(increments).sum(axis=0))
        self.assertLess(np.isfinite(increments).sum(axis=0).min(), len(self.reports))
        for entry in self.model.fit_diagnostics_["optimization"]:
            self.assertTrue(np.isfinite([entry["loss"], entry["macro_increment_crps"], entry["joint_energy"]]).all())
            self.assertGreater(entry["max_gradient_norm"], 0)

    def test_cutoff_values_are_ignored_and_known_native_history_is_exact(self):
        history = self.reports[:2].copy()
        history[:, DAYS > 90] = np.nan
        history[0, 1, 0] = 17.123456789012345
        history[1, 0] = np.nan
        changed = history.copy()
        changed[:, DAYS > 90] = np.inf
        a = self.model.predict_distribution(history, 90, 11, np.random.default_rng(13))
        b = self.model.predict_distribution(changed, 90, 11, np.random.default_rng(13))
        np.testing.assert_array_equal(a["report_cumulative"], b["report_cumulative"])
        for p, q in zip(self.model.conditional_parameters(history, 90), self.model.conditional_parameters(changed, 90)):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        known = np.isfinite(history) & (DAYS <= 90)[None, :, None]
        expected = np.broadcast_to(history[:, None], a["report_cumulative"].shape)
        np.testing.assert_array_equal(a["report_cumulative"][np.broadcast_to(known[:, None], expected.shape)],
                                      expected[np.broadcast_to(known[:, None], expected.shape)])
        np.testing.assert_array_equal(a["report_cumulative"][:, :, 0], 0)
        self.assertEqual(a["interpretation"], "conditional_joint_report_law_no_latent_rate")
        self.assertEqual(a["phase_order"], ["oil", "gas", "water"])
        self.assertNotIn("latent_cumulative", a)
        self.assertNotIn("latent_rate", a)
        self.assertNotIn("posterior_diagnostics", a)
        missing = history.copy()
        missing[0, 1, 0] = np.nan
        zero = history.copy()
        zero[0, 1, 0] = 0
        self.assertFalse(torch.equal(self.model.conditional_parameters(missing, 90)[0],
                                     self.model.conditional_parameters(zero, 90)[0]))
        invalid = history.copy()
        invalid[0, 1, 0] = np.inf
        with self.assertRaises(ValueError):
            self.model.predict_samples(invalid, 90, 2, np.random.default_rng(1))

    def test_missing_phase_origin_draws_are_joint_and_reused_in_derived_increments(self):
        history = self.reports[:2].copy()
        history[:, :, 1] = np.nan
        report = self.model.predict_samples(history, 90, 500, np.random.default_rng(42))
        self.assertTrue(np.isfinite(report).all())
        self.assertGreater(report[0, :, 3, 1].std(), 0)
        np.testing.assert_array_equal(report[:, :, 0], 0)
        np.testing.assert_allclose(report[:, :, 7]-report[:, :, 3],
            (report[:, :, 7]-report[:, :, 5])+(report[:, :, 5]-report[:, :, 3]), rtol=1e-12, atol=1e-9)

    def test_checkpoint_reproduces_parameters_draws_and_training_protocol(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"conditional_reports.pt"
            self.model.save(path)
            restored = ConditionalReportGaussianForecaster.load(path)
        self.assertEqual(self.model.configuration, restored.configuration)
        self.assertEqual(self.model.fit_diagnostics_, restored.fit_diagnostics_)
        self.assertEqual(self.model.scale_provenance_, restored.scale_provenance_)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        a = self.model.predict_samples(self.reports[:2], 60, 12, np.random.default_rng(6))
        b = restored.predict_samples(self.reports[:2], 60, 12, np.random.default_rng(6))
        np.testing.assert_array_equal(a, b)

    def test_repeated_seed_reproduces_training(self):
        duplicate = ConditionalReportGaussianForecaster(rank=2, **self.settings).fit(self.reports, DAYS, origins=ORIGINS)
        self.assertEqual(self.model.fit_diagnostics_, duplicate.fit_diagnostics_)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, duplicate.state_dict()[key], rtol=0, atol=0)

    def test_finite_training_and_history_float32_overflow_is_rejected(self):
        reports = self.reports.copy()
        reports[0, 1, 0] = 1e100
        with self.assertRaisesRegex(ValueError, "float32"):
            ConditionalReportGaussianForecaster(epochs=1, hidden_width=8, log_every=0).fit(reports, DAYS, origins=[60])
        with self.assertRaisesRegex(ValueError, "float32"):
            self.model.predict_samples(reports[:1], 60, 2, np.random.default_rng(3))

    def test_native_report_overflow_is_rejected_after_unit_conversion(self):
        def extreme_draw(location, diagonal_variance, factor, actual, observed, factor_normals, diagonal_normals):
            return actual.new_full((len(actual), factor_normals.shape[1], len(DAYS), 3), 1e308)
        with patch("joint_production.conditional_reports._draw_reports", side_effect=extreme_draw):
            with self.assertRaises(FloatingPointError):
                self.model.predict_samples(self.reports[:1], 60, 2, np.random.default_rng(20))


if __name__ == "__main__":
    unittest.main()
