"""Consequential distribution, conditioning, and training tests for the NN."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from joint_production.neural import ContextualGaussianNeuralBaseline, _masked_gaussian_nll


def example_reports(n=32):
    rng = np.random.default_rng(13)
    days = np.array([0, 30, 60, 90, 180], dtype=float)
    amplitude = np.exp(rng.normal(2, 0.4, size=(n, 1, 3)))
    reports = amplitude * (1 - np.exp(-days[None, :, None] / 70))
    reports += rng.normal(0, 0.25, size=reports.shape)
    reports[:, 0] = 0
    reports[0, 1, 2] = -0.25
    reports[1, 2, 1] = np.nan
    return reports, days


class TestMaskedGaussianLikelihood(unittest.TestCase):
    def test_dense_gaussian_agreement_and_missing_target_gradients(self):
        generator = np.random.default_rng(21)
        n, d, rank = 4, 7, 3
        y = generator.normal(size=(n, d))
        mu = generator.normal(size=(n, d))
        variance = np.exp(generator.normal(size=(n, d)))
        factors = generator.normal(size=(n, d, rank))
        masks = np.array([[True] * d, [False, True, False, True, False, True, False],
                          [False, False, True, False, False, False, False], [False] * d])
        dense_nll = []
        for i in range(n):
            m = masks[i]
            covariance = np.diag(variance[i, m]) + factors[i, m] @ factors[i, m].T
            residual = y[i, m] - mu[i, m]
            dense_nll.append(0.5 * (residual @ np.linalg.solve(covariance, residual)
                                    + np.linalg.slogdet(covariance)[1]
                                    + int(m.sum()) * np.log(2 * np.pi)))
        location = torch.tensor(mu, dtype=torch.float64, requires_grad=True)
        actual = _masked_gaussian_nll(torch.tensor(y), location, torch.tensor(variance),
                                      torch.tensor(factors), torch.tensor(masks))
        np.testing.assert_allclose(actual.detach(), dense_nll, rtol=1e-11, atol=1e-11)
        actual.sum().backward()
        np.testing.assert_array_equal(location.grad.numpy()[~masks], 0)
        changed = y.copy()
        changed[~masks] = 1e8
        other = _masked_gaussian_nll(torch.tensor(changed), location, torch.tensor(variance),
                                     torch.tensor(factors), torch.tensor(masks))
        np.testing.assert_allclose(other.detach(), actual.detach(), rtol=1e-11, atol=1e-11)

    def test_near_floor_diagonal_and_large_factor_remain_finite(self):
        dtype = torch.float64
        target = torch.ones((2, 24), dtype=dtype)
        value = _masked_gaussian_nll(target, torch.zeros_like(target),
                                     torch.full_like(target, 0.025 ** 2),
                                     torch.ones((2, 24, 8), dtype=dtype),
                                     torch.ones_like(target, dtype=torch.bool))
        self.assertTrue(torch.isfinite(value).all())

    def test_training_dtype_covariance_loss_and_gradients_match_dense_reference(self):
        target = torch.full((1, 24), 3.0, dtype=torch.float32)
        location = torch.zeros_like(target, requires_grad=True)
        variance = torch.full_like(target, 0.025 ** 2, requires_grad=True)
        factor = torch.ones((1, 24, 8), dtype=torch.float32, requires_grad=True)
        actual = _masked_gaussian_nll(target, location, variance, factor,
                                      torch.ones_like(target, dtype=torch.bool))
        actual.sum().backward()
        reference_location = location.detach().double().requires_grad_()
        reference_variance = variance.detach().double().requires_grad_()
        reference_factor = factor.detach().double().requires_grad_()
        covariance = torch.diag_embed(reference_variance) + reference_factor @ reference_factor.transpose(-2, -1)
        reference = -torch.distributions.MultivariateNormal(reference_location,
                                                            covariance_matrix=covariance).log_prob(target.double())
        reference.sum().backward()
        torch.testing.assert_close(actual.detach(), reference.detach(), atol=1e-8, rtol=1e-8)
        for parameter, expected in ((location, reference_location), (variance, reference_variance),
                                     (factor, reference_factor)):
            torch.testing.assert_close(parameter.grad.double(), expected.grad, atol=1e-7, rtol=2e-6)


class TestContextualNeural(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports, cls.days = example_reports()
        cls.model = ContextualGaussianNeuralBaseline(
            epochs=8, hidden_width=20, rank=4, batch_size=32, seed=11)
        cls.model.fit(cls.reports, cls.days, origins=[60, 90])

    def test_cutoff_ignores_supplied_future_and_preserves_available_history(self):
        history = self.reports[:3].copy()
        history[:, self.days > 60] = np.nan
        altered_future = history.copy()
        altered_future[:, self.days > 60] = 1e100
        scales_before = self.model.phase_scales_.copy()
        first = self.model.predict_samples(history, 60, 20, np.random.default_rng(2))
        second = self.model.predict_samples(altered_future, 60, 20, np.random.default_rng(2))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, 20, 5, 3))
        self.assertTrue(np.isfinite(first).all())
        observed = np.isfinite(history) & (self.days[None, :, None] <= 60)
        np.testing.assert_array_equal(first.transpose(0, 2, 3, 1)[observed],
                                      np.repeat(history[observed, None], 20, axis=1))
        np.testing.assert_array_equal(self.model.phase_scales_, scales_before)
        # A naturally missing historical phase is jointly predicted, not filled
        # with its held-out value or independently re-anchored for each target.
        self.assertGreater(float(first[1, :, 2, 1].std()), 0)

    def test_zero_mask_and_early_phase_identity_are_encoded(self):
        history = self.reports[:1].copy()
        history[:, self.days > 60] = np.nan
        zero = history.copy()
        zero[0, 1, 0] = 0
        missing = zero.copy()
        missing[0, 1, 0] = np.nan
        zero_parameters = self.model.predict_transformed_parameters(zero, 60)
        missing_parameters = self.model.predict_transformed_parameters(missing, 60)
        self.assertTrue(zero_parameters["observed_mask"][0, 3])
        self.assertFalse(missing_parameters["observed_mask"][0, 3])
        self.assertFalse(np.array_equal(zero_parameters["location"], missing_parameters["location"]))
        altered_early = history.copy()
        altered_early[0, 1, 1] += 5
        self.assertFalse(np.array_equal(
            self.model.predict_transformed_parameters(history, 60)["location"],
            self.model.predict_transformed_parameters(altered_early, 60)["location"]))

    def test_sampler_matches_joint_gaussian_including_missing_origin(self):
        model = ContextualGaussianNeuralBaseline(epochs=1, hidden_width=8, rank=4, batch_size=64)
        model.fit(self.reports, self.days, origins=[60])
        with torch.no_grad():
            model.network_.mean.weight.zero_()
            model.network_.mean.bias.zero_()
            model.network_.factor.weight.zero_()
            model.network_.factor.bias.fill_(0.6)
            model.network_.std.weight.zero_()
        history = self.reports[:1].copy()
        history[:, self.days > 60] = np.nan
        history[:, :, 1] = np.nan  # Constructed missing phase, including origin.
        parameters = model.predict_transformed_parameters(history, 60)
        samples = model.predict_samples(history, 60, 60000, np.random.default_rng(73))
        flattened = np.arcsinh(samples / model.phase_scales_).reshape(1, 60000, -1)[0]
        missing = ~parameters["observed_mask"][0]
        empirical = flattened[:, missing]
        factor = parameters["factor"][0, missing]
        covariance = np.diag(parameters["diagonal_variance"][0, missing]) + factor @ factor.T
        np.testing.assert_allclose(empirical.mean(0), parameters["location"][0, missing], atol=0.025)
        np.testing.assert_allclose(np.cov(empirical.T), covariance, atol=0.035, rtol=0.025)
        self.assertGreater(np.corrcoef(samples[0, :, 2, 1], samples[0, :, 4, 0])[0, 1], 0.4)
        self.assertTrue((samples[0, :, 4, 0] < 0).any())
        self.assertTrue(np.isfinite(samples).all())
        np.testing.assert_array_equal(samples[:, :, 0], 0)
        # Derived increments telescope because every interval uses this draw.
        np.testing.assert_allclose(samples[:, :, 4] - samples[:, :, 2],
                                   (samples[:, :, 4] - samples[:, :, 3])
                                   + (samples[:, :, 3] - samples[:, :, 2]), atol=1e-10)

    def test_seed_reproducibility_and_fit_diagnostics(self):
        kwargs = dict(epochs=2, hidden_width=8, rank=4, batch_size=17, seed=9)
        a = ContextualGaussianNeuralBaseline(**kwargs).fit(self.reports, self.days, origins=[60])
        b = ContextualGaussianNeuralBaseline(**kwargs).fit(self.reports, self.days, origins=[60])
        for name, value in a.network_.state_dict().items():
            torch.testing.assert_close(value, b.network_.state_dict()[name], atol=0, rtol=0)
        self.assertEqual(a.fit_diagnostics_, b.fit_diagnostics_)
        self.assertTrue(np.isfinite(a.fit_diagnostics_["conditional_report_nll_by_epoch"]).all())
        np.testing.assert_array_equal(a.predict_samples(self.reports[:2], 60, 4, np.random.default_rng(3)),
                                      b.predict_samples(self.reports[:2], 60, 4, np.random.default_rng(3)))

    def test_all_zero_phase_and_missing_covariates_have_train_only_fallback(self):
        reports = self.reports.copy()
        reports[:, :, 2] = 0
        covariates = np.column_stack((np.arange(len(reports)), np.full(len(reports), np.nan)))
        model = ContextualGaussianNeuralBaseline(epochs=1, hidden_width=8).fit(
            reports, self.days, origins=[60], covariates=covariates)
        self.assertEqual(model.phase_scales_[2], 1.0)
        centers = model.covariate_center_.copy()
        draws = model.predict_samples(reports[:2], 60, 3, np.random.default_rng(1), covariates[:2])
        self.assertTrue(np.isfinite(draws).all())
        np.testing.assert_array_equal(model.covariate_center_, centers)
        with self.assertRaisesRegex(ValueError, "requires cutoff"):
            model.predict_samples(reports[:2], 60, 3, np.random.default_rng(1))

    def test_invalid_queries_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "not included"):
            self.model.predict_samples(self.reports[:1], 30, 3, np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.model.predict_samples(self.reports[:1], 60, 0, np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "Every endpoint/phase"):
            invalid = self.reports.copy()
            invalid[:, 3, 1] = np.nan
            ContextualGaussianNeuralBaseline(epochs=1).fit(invalid, self.days, origins=[60])
        invalid_zero = self.reports[:1].copy()
        invalid_zero[0, 0, 1] = -1
        with self.assertRaisesRegex(ValueError, "deterministic zero"):
            self.model.predict_samples(invalid_zero, 60, 3, np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "deterministic zero"):
            ContextualGaussianNeuralBaseline(epochs=1).fit(invalid_zero, self.days, origins=[60])

    def test_checkpoint_preserves_conditional_law_and_training_metadata(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "neural.pt"
            self.model.save(checkpoint)
            restored = ContextualGaussianNeuralBaseline.load(checkpoint)
        self.assertEqual(restored.fit_diagnostics_, self.model.fit_diagnostics_)
        np.testing.assert_array_equal(restored.phase_scales_, self.model.phase_scales_)
        np.testing.assert_array_equal(
            restored.predict_samples(self.reports[:2], 60, 20, np.random.default_rng(14)),
            self.model.predict_samples(self.reports[:2], 60, 20, np.random.default_rng(14)))

    def test_extreme_inverse_transform_fails_without_silent_clipping(self):
        model = ContextualGaussianNeuralBaseline(epochs=1, hidden_width=8, log_every=0).fit(
            self.reports, self.days, origins=[60])
        with torch.no_grad():
            model.network_.mean.weight.zero_()
            model.network_.mean.bias.fill_(1000)
        with self.assertRaisesRegex(FloatingPointError, "no clipping"):
            model.predict_samples(self.reports[:1], 60, 3, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
