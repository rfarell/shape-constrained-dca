"""Consequential checks for endpoint likelihood and history-conditioned inference."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, StudentT

from joint_production.hierarchical import (
    HierarchicalForecaster,
    _inverse_softplus,
    _raw_tril,
    endpoint_log_likelihood,
    likelihood_mask,
)


class FixedShapeAmplitudeRate(nn.Module):
    """Controlled inference case with a one-dimensional observed log amplitude."""

    def cumulative(self, z, times):
        fractions = -torch.expm1(-times / 360)
        return z[..., [0, 4, 8]].exp().unsqueeze(-2) * fractions[:, None]


class HierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.days = np.array([0., 30., 90., 360., 1080.])
        generator = np.random.default_rng(73)
        amplitude = np.exp(generator.normal(0, .3, size=(8, 1, 3)))
        fraction = -np.expm1(-cls.days[None, :, None] / 360)
        cls.reports = amplitude * fraction * np.array([100., 400., 200.])
        cls.model = HierarchicalForecaster(
            epochs=12, inference_steps=15, train_mc_samples=2,
            inference_mc_samples=4, batch_size=8, dtype="float64",
        ).fit(cls.reports, cls.days)

    def test_endpoint_likelihood_support_and_mask_count(self):
        reports = torch.tensor([[[0., 0., 0.], [0., -2., float("nan")], [-1., 5., 6.]]], dtype=torch.float64)
        days = torch.tensor([0., 30., 90.], dtype=torch.float64)
        prediction = torch.ones((4, 1, 3, 3), dtype=torch.float64)
        prediction[:, :, 0] = 0
        noise_abs = torch.tensor([.1, .2, .3], dtype=torch.float64)
        noise_rel = torch.tensor([.05, .08, .1], dtype=torch.float64)
        mask = likelihood_mask(reports, days, cutoff=30)
        self.assertEqual(int(mask.sum()), 2)
        result = endpoint_log_likelihood(reports, prediction, noise_abs, noise_rel, mask)
        expected = StudentT(5., 1., .15).log_prob(torch.tensor(0.)) + StudentT(5., 1., .28).log_prob(torch.tensor(-2.))
        torch.testing.assert_close(result, expected.double().expand(4, 1), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(result).all())

    def test_all_future_values_excluded_and_generative_parameters_frozen(self):
        history = self.reports[:2].copy()
        modified = history.copy()
        modified[:, self.days > 90] = np.inf
        # A missing observed phase is distinct from a reported zero.
        history[0, 1, 2] = np.nan
        modified[0, 1, 2] = np.nan
        before = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        first = self.model.infer_posterior(history, 90)
        second = self.model.infer_posterior(modified, 90)
        torch.testing.assert_close(first.mean, second.mean, rtol=0, atol=0)
        torch.testing.assert_close(first.scale_tril, second.scale_tril, rtol=0, atol=0)
        self.assertEqual(first.diagnostics["n_observed_endpoints"], 11)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(torch.linalg.eigvalsh(first.scale_tril @ first.scale_tril.mT).min() > 0)

    def test_empty_history_returns_exact_prior(self):
        history = np.full_like(self.reports[:2], np.nan)
        history[:, 0] = 0
        posterior = self.model.infer_posterior(history, 90)
        torch.testing.assert_close(posterior.mean, self.model.prior_mean.expand(2, -1), rtol=0, atol=0)
        torch.testing.assert_close(posterior.scale_tril, self.model.prior_scale_tril.expand(2, -1, -1), rtol=0, atol=0)
        self.assertEqual(posterior.diagnostics["steps"], 0)

    def test_report_sampler_anchors_only_observed_history_and_preserves_latents(self):
        history = self.reports[:2].copy()
        history[0, 2, 2] = np.nan
        history[1, 1, 0] = -1.0
        result = self.model.predict_distribution(history, 90, 30, np.random.default_rng(92))
        latent, report = result["latent_cumulative"], result["report_cumulative"]
        self.assertEqual(report.shape, (2, 30, 5, 3))
        self.assertTrue(np.all(np.diff(latent, axis=2) >= 0))
        self.assertTrue(np.all(latent[:, :, 0] == 0))
        self.assertTrue(np.all(report[:, :, 0] == 0))
        observed = np.isfinite(history) & (self.days[None, :, None] <= 90)
        np.testing.assert_array_equal(report.transpose(0, 2, 3, 1)[observed], np.repeat(history[observed, None], 30, axis=1))
        self.assertGreater(np.std(report[0, :, 2, 2]), 0)
        self.assertGreater(np.std(report[:, :, -1]), 0)
        # Historical negative reports are never forced into latent production.
        self.assertTrue(np.all(report[1, :, 1, 0] == -1))
        self.assertTrue(np.all(latent >= 0))
        dense = self.model.evaluate_trajectories(result["latent_parameters"], [0., 30., 77., 90., 360., 511., 1080.])
        np.testing.assert_allclose(dense["latent_cumulative"][:, :, [0, 1, 3, 4, 6]], latent, rtol=1e-12)
        np.testing.assert_allclose(np.diff(dense["latent_cumulative"], axis=2).sum(axis=2), latent[:, :, -1], rtol=1e-12)

    def test_checkpoint_restores_sampler_and_training_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.model.save(path)
            loaded = HierarchicalForecaster.load(path)
        first = self.model.predict_samples(self.reports[:1], 90, 4, np.random.default_rng(10))
        second = loaded.predict_samples(self.reports[:1], 90, 4, np.random.default_rng(10))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(self.model.training_diagnostics_, loaded.training_diagnostics_)

    def test_report_draws_use_the_fitted_student_t_scale(self):
        empty = np.full_like(self.reports[:1], np.nan)
        empty[:, 0] = 0
        draws = self.model.predict_distribution(empty, 90, 5000, np.random.default_rng(271))
        latent = draws["latent_cumulative"][:, :, 1:]
        report = draws["report_cumulative"][:, :, 1:]
        abs_noise = (self.model.noise_abs * self.model.volume_scale).detach().numpy()
        rel_noise = self.model.noise_rel.detach().numpy()
        residual = (report - latent) / (abs_noise + rel_noise * latent)
        self.assertLess(abs(float(residual.mean())), .025)
        self.assertLess(abs(float(residual.var()) - 5 / 3), .12)

    def test_float32_inference_preserves_exact_float64_historical_reports(self):
        model = HierarchicalForecaster(epochs=0, inference_steps=0, dtype="float32").fit(self.reports, self.days)
        history = self.reports[:1].copy()
        history[0, 1, 0] = .123456789012345
        self.assertNotEqual(float(np.float32(history[0, 1, 0])), history[0, 1, 0])
        samples = model.predict_samples(history, 90, 5, np.random.default_rng(876))
        expected = np.repeat(history[:, None, :3, :], 5, axis=1)
        np.testing.assert_array_equal(samples[:, :, :3, :], expected)

    def test_positive_basis_fit_and_posterior_smoke(self):
        model = HierarchicalForecaster(rate_family="basis", epochs=3, inference_steps=4, dtype="float64")
        model.fit(self.reports, self.days)
        draws = model.predict_distribution(self.reports[:1], 90, 8, np.random.default_rng(314))
        self.assertTrue(np.isfinite(draws["report_cumulative"]).all())
        self.assertTrue(np.all(np.diff(draws["latent_cumulative"], axis=2) >= 0))
        self.assertTrue(np.all(draws["latent_total_volume"] >= draws["latent_cumulative"][:, :, -1]))

    def test_fast_basis_configuration_checkpoint_and_old_default(self):
        model = HierarchicalForecaster(rate_family="basis", basis_dictionary="fast", epochs=3, inference_steps=4, dtype="float64")
        model.fit(self.reports, self.days)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fast.pt"
            model.save(path)
            loaded = HierarchicalForecaster.load(path)
        self.assertEqual(loaded.configuration["basis_dictionary"], "fast")
        self.assertEqual(loaded.rate_model.preset, "fast")
        expected = model.predict_samples(self.reports[:1], 90, 4, np.random.default_rng(27))
        actual = loaded.predict_samples(self.reports[:1], 90, 4, np.random.default_rng(27))
        np.testing.assert_array_equal(actual, expected)
        original = HierarchicalForecaster(rate_family="basis", epochs=0).fit(self.reports, self.days)
        original.configuration.pop("basis_dictionary")  # Pre-preset checkpoint schema.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            original.save(path)
            old_loaded = HierarchicalForecaster.load(path)
        self.assertEqual(old_loaded.rate_model.preset, "original")
        torch.testing.assert_close(original.prior_mean, old_loaded.prior_mean, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            HierarchicalForecaster(rate_family="envelope", basis_dictionary="fast")
        with self.assertRaises(ValueError):
            HierarchicalForecaster(rate_family="basis", basis_dictionary="unknown")

    def test_gaussian_model_fit_cutoff_frozen_parameters_and_checkpoint(self):
        for law in ("gaussian_independent", "gaussian_accumulated"):
            with self.subTest(law=law):
                model = HierarchicalForecaster(observation_law=law, epochs=3, inference_steps=4, dtype="float64")
                model.fit(self.reports, self.days)
                history = self.reports[:2].copy()
                history[0, 1, 2] = np.nan
                history[1, 2, 0] = .123456789012345
                changed = history.copy()
                changed[:, self.days > 90] = np.inf
                before = {name: value.clone() for name, value in model.state_dict().items()}
                first = model.predict_samples(history, 90, 6, np.random.default_rng(72))
                second = model.predict_samples(changed, 90, 6, np.random.default_rng(72))
                np.testing.assert_array_equal(first, second)
                self.assertTrue(np.isfinite(first).all())
                self.assertTrue(np.all(first[1, :, 2, 0] == history[1, 2, 0]))
                self.assertGreater(np.std(first[0, :, 1, 2]), 0)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                self.assertEqual(model.training_diagnostics_["observation_law"], law)
                self.assertEqual(model.last_diagnostics_["observation_law"], law)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "gaussian.pt"
                    model.save(path)
                    loaded = HierarchicalForecaster.load(path)
                self.assertEqual(loaded.observation_law, law)
                restored = loaded.predict_samples(history, 90, 6, np.random.default_rng(72))
                np.testing.assert_array_equal(first, restored)

    def test_gaussian_model_likelihood_matches_dense_normal_and_backpropagates(self):
        for law in ("gaussian_independent", "gaussian_accumulated"):
            model = HierarchicalForecaster(observation_law=law, epochs=0, dtype="float64").fit(self.reports, self.days)
            reports = model._tensor(self.reports[:1]) / model.volume_scale
            mask = likelihood_mask(reports, model.report_days, cutoff=360)
            mask[0, 1, 1] = False
            latent = torch.full((2, 1, 12), .1, dtype=model.dtype, requires_grad=True)
            cumulative = model.rate_model.cumulative(latent, model.report_days)
            actual = model._report_log_likelihood(reports, cumulative, model.noise_abs, model.noise_rel, mask)
            expected = torch.zeros_like(actual)
            for draw in range(2):
                for phase in range(3):
                    index = torch.where(mask[0, :, phase])[0]
                    curve = cumulative[draw, 0, index, phase]
                    variance = model.noise_abs[phase].square() * model.report_days[index] / 360 + model.noise_rel[phase].square() * curve.square()
                    covariance = torch.diag(variance) if law == "gaussian_independent" else torch.minimum(variance[:, None], variance[None])
                    expected[draw, 0] += torch.distributions.MultivariateNormal(curve, covariance_matrix=covariance).log_prob(reports[0, index, phase])
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            actual.sum().backward()
            for gradient in (latent.grad, model.noise_abs_raw.grad, model.noise_rel_raw.grad):
                self.assertTrue(torch.isfinite(gradient).all())

    def test_gaussian_model_sampler_matches_native_conditional_covariance(self):
        for law in ("gaussian_independent", "gaussian_accumulated"):
            model = HierarchicalForecaster(observation_law=law, epochs=0, inference_steps=0, dtype="float64").fit(self.reports, self.days)
            # Fix latent draws to isolate report-law conditioning and unit conversion.
            model.sample_latents = lambda posterior, count, rng: model.reference_mean.expand(1, count, -1).clone()
            history = np.full_like(self.reports[:1], np.nan)
            history[:, 0] = 0
            history[0, 2, 0] = 20.123456789012345
            paths = model.predict_distribution(history, 90, 20000, np.random.default_rng(812))
            curve = paths["latent_cumulative"][0, 0, :, 0]
            absolute = float((model.noise_abs * model.volume_scale)[0].detach())
            relative = float(model.noise_rel[0].detach())
            variance = absolute ** 2 * self.days / 360 + relative ** 2 * curve ** 2
            covariance = np.diag(variance) if law == "gaussian_independent" else np.minimum.outer(variance, variance)
            unknown = np.array([1, 3, 4])
            gain = covariance[unknown, 2] / covariance[2, 2]
            expected_mean = curve[unknown] + gain * (history[0, 2, 0] - curve[2])
            expected_covariance = covariance[np.ix_(unknown, unknown)] - gain[:, None] * covariance[2, unknown][None]
            samples = paths["report_cumulative"][0, :, :, 0][:, unknown]
            mean_error = np.abs(samples.mean(axis=0) - expected_mean)
            self.assertTrue(np.all(mean_error < 6 * np.sqrt(np.diag(expected_covariance) / len(samples))))
            empirical = np.cov(samples, rowvar=False)
            standardized_error = np.abs(empirical - expected_covariance) / np.sqrt(np.diag(expected_covariance)[:, None] * np.diag(expected_covariance)[None])
            self.assertLess(float(standardized_error.max()), .045)
            self.assertTrue(np.all(paths["report_cumulative"][0, :, 2, 0] == history[0, 2, 0]))

    def test_observation_law_change_requires_refit_and_reinference(self):
        with self.assertRaises(ValueError):
            HierarchicalForecaster(observation_law="unknown")
        model = HierarchicalForecaster(epochs=0, inference_steps=0).fit(self.reports, self.days)
        original_posterior = model.infer_posterior(self.reports[:1], 90)
        model.observation_law = "gaussian_accumulated"
        with self.assertRaisesRegex(RuntimeError, "refitting"):
            model.infer_posterior(self.reports[:1], 90)
        model.fit(self.reports, self.days)
        with self.assertRaisesRegex(ValueError, "different observation law"):
            model.sample_latents(original_posterior, 4, np.random.default_rng(9))
        self.assertEqual(model.configuration["observation_law"], "gaussian_accumulated")
        original = HierarchicalForecaster(epochs=0).fit(self.reports, self.days)
        original.configuration.pop("observation_law")
        original.training_diagnostics_.pop("observation_law")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_student_t.pt"
            original.save(path)
            old_loaded = HierarchicalForecaster.load(path)
        self.assertEqual(old_loaded.observation_law, "student_t_independent")

    def test_old_family_amplitude_initialization_preserved_numerically(self):
        for family, preset in (("envelope", "original"), ("basis", "original"), ("basis", "fast")):
            model = HierarchicalForecaster(rate_family=family, basis_dictionary=preset, epochs=0, dtype="float64").fit(self.reports, self.days)
            self.assertEqual(model.latent_size, 12)
            self.assertEqual(model.parameters_per_phase, 4)
            scaled = model._tensor(self.reports) / model.volume_scale
            mask = likelihood_mask(scaled, model.report_days)
            actual = model._initialize_locations(scaled, mask, training=True)
            reference = model.reference_mean.expand(len(scaled), -1).clone()
            baseline = model.rate_model.cumulative(model.reference_mean, model.report_days)
            for phase in range(3):
                shift = torch.log(scaled[:, -1, phase] / baseline[-1, phase])
                if family == "envelope":
                    reference[:, 4 * phase] += shift
                else:
                    reference[:, 4 * phase:4 * phase + 4] += shift[:, None]
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_nine_dimensional_dca_fit_posterior_and_checkpoint(self):
        for law in ("student_t_independent", "gaussian_independent", "gaussian_accumulated"):
            with self.subTest(law=law):
                model = HierarchicalForecaster(rate_family="dca", observation_law=law, epochs=3, inference_steps=4, dtype="float64")
                model.fit(self.reports, self.days)
                self.assertEqual(model.latent_size, 9)
                self.assertEqual(model.parameters_per_phase, 3)
                self.assertEqual(model.amplitude_indices, (0, 3, 6))
                self.assertEqual(model.prior_mean.shape, (9,))
                self.assertEqual(model.prior_scale_tril.shape, (9, 9))
                scaled = model._tensor(self.reports) / model.volume_scale
                mask = likelihood_mask(scaled, model.report_days)
                initialized = model._initialize_locations(scaled, mask, training=True)
                unchanged_indices = [1, 2, 4, 5, 7, 8]
                torch.testing.assert_close(initialized[:, unchanged_indices], model.reference_mean[unchanged_indices].expand(len(scaled), -1), rtol=0, atol=0)
                history = self.reports[:2].copy()
                history[0, 2, 2] = np.nan
                posterior = model.infer_posterior(history, 90)
                self.assertEqual(posterior.mean.shape, (2, 9))
                self.assertEqual(posterior.scale_tril.shape, (2, 9, 9))
                self.assertTrue(torch.linalg.eigvalsh(posterior.scale_tril @ posterior.scale_tril.mT).min() > 0)
                first = model.predict_distribution(history, 90, 6, np.random.default_rng(487))
                self.assertEqual(first["latent_parameters"].shape, (2, 6, 9))
                self.assertEqual(first["report_cumulative"].shape, (2, 6, len(self.days), 3))
                self.assertTrue(np.isfinite(first["report_cumulative"]).all())
                self.assertTrue(np.all(np.diff(first["latent_cumulative"], axis=2) >= 0))
                self.assertTrue(np.all(first["latent_total_volume"] >= first["latent_cumulative"][:, :, -1]))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "dca.pt"
                    model.save(path)
                    loaded = HierarchicalForecaster.load(path)
                self.assertEqual(loaded.rate_family, "dca")
                self.assertEqual(loaded.latent_size, 9)
                restored = loaded.predict_samples(history, 90, 6, np.random.default_rng(487))
                np.testing.assert_array_equal(first["report_cumulative"], restored)
                with self.assertRaisesRegex(ValueError, "9"):
                    model.evaluate_trajectories(np.zeros((1, 12)), self.days)

    def test_phase_specific_scale_fallback_and_day_zero_contract(self):
        reports = np.zeros((2, 3, 3))
        reports[:, 2, 0] = [8, 12]
        reports[:, 1, 1] = [100, 200]
        reports[:, 2, 1] = np.nan
        model = HierarchicalForecaster(epochs=0).fit(reports, [0, 30, 1080])
        np.testing.assert_allclose(model.volume_scale.detach().numpy(), [10., 150., 1.])
        bad = reports.copy()
        bad[0, 0, 1] = 1
        with self.assertRaisesRegex(ValueError, "day-zero"):
            HierarchicalForecaster(epochs=0).fit(bad, [0, 30, 1080])

    def test_local_variational_posterior_against_one_dimensional_quadrature(self):
        # Only log amplitude z0 is observed; all other parameters integrate out.
        # This checks actual history conditioning under the fitted Student-t law,
        # independently of the general rate-family implementation.
        model = HierarchicalForecaster(
            epochs=0, inference_steps=300, inference_mc_samples=64,
            inference_learning_rate=.02, dtype="float64", seed=219,
        ).fit(np.array([[[0., 0., 0.], [1., 1., 1.]]]), [0., 360.])
        model.rate_model = FixedShapeAmplitudeRate()
        with torch.no_grad():
            model.prior_mean.zero_()
            covariance = .7 ** 2 * torch.eye(12, dtype=model.dtype)
            covariance[0, 4] = covariance[4, 0] = .8 * .7 ** 2
            model.prior_raw_tril.copy_(_raw_tril(torch.linalg.cholesky(covariance), model.cholesky_floor))
            model.noise_abs_raw.copy_(_inverse_softplus(torch.full((3,), .02 - model.noise_floor, dtype=model.dtype)))
            model.noise_rel_raw.copy_(_inverse_softplus(torch.full((3,), .08 - model.noise_floor, dtype=model.dtype)))
        history = np.array([[[0., 0., 0.], [.8, np.nan, np.nan]]])
        posterior = model.infer_posterior(history, 360)
        grid = torch.linspace(-4., 4., 50001, dtype=model.dtype)
        cumulative = grid.exp() * (-np.expm1(-1.))
        log_weight = Normal(0., .7).log_prob(grid) + StudentT(5., cumulative, .02 + .08 * cumulative).log_prob(torch.tensor(.8))
        weight = (log_weight - log_weight.max()).exp()
        weight /= torch.trapezoid(weight, grid)
        exact_mean = torch.trapezoid(grid * weight, grid)
        exact_variance = torch.trapezoid((grid - exact_mean).square() * weight, grid)
        variance = (posterior.scale_tril @ posterior.scale_tril.mT)[0, 0, 0]
        self.assertLess(abs(float(posterior.mean[0, 0] - exact_mean)), .04)
        self.assertGreater(float(variance / exact_variance), .45)
        self.assertLess(float(variance / exact_variance), 1.5)
        self.assertLess(float(variance), .7 ** 2 / 3)
        # A missing gas phase is updated through a correlated population prior.
        self.assertLess(abs(float(posterior.mean[0, 4] - .8 * exact_mean)), .08)
        gas_variance = (posterior.scale_tril @ posterior.scale_tril.mT)[0, 4, 4]
        exact_gas_variance = .7 ** 2 * (1 - .8 ** 2) + .8 ** 2 * exact_variance
        self.assertLess(abs(float(gas_variance - exact_gas_variance)), .07)
        self.assertGreater(posterior.diagnostics["final_mean_elbo"], posterior.diagnostics["initial_mean_elbo"])


if __name__ == "__main__":
    unittest.main()
