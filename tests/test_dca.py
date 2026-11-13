"""Consequential DCA integration, conditioning, and observation-law tests."""

import unittest

import numpy as np

from joint_production.dca import ConditionalDCABaseline, DCATrajectorySamples, HierarchicalDCABaseline


DAYS = np.array([0.0, 30.0, 60.0, 90.0, 180.0, 360.0, 720.0, 1080.0])


def synthetic_training(n=50):
    rng = np.random.default_rng(123)
    shared = rng.normal(0, 0.5, n)
    parameters = np.empty((n, 3, 3))
    parameters[:, :, 0] = np.exp(shared[:, None] + rng.normal(0, 0.15, (n, 3))) * [100, 200, 70]
    parameters[:, :, 1] = 120
    parameters[:, :, 2] = [0.8, 0.5, 1.0]
    latent = DCATrajectorySamples(parameters).cumulative(DAYS)
    reports = latent + rng.normal(size=latent.shape) * (0.02 * latent + np.sqrt(DAYS[None, :, None]) * 2)
    return reports


class TestDCATrajectories(unittest.TestCase):
    def setUp(self):
        parameters = np.array([[[2.0, 120.0, 1.0], [3.0, 90.0, 0.4], [4.0, 200.0, 1.7]]])
        self.paths = DCATrajectorySamples(parameters)

    def test_integrals_and_nonnegativity(self):
        grid = np.linspace(0, 5000, 100001)
        q = self.paths.rates(grid)
        self.assertTrue(np.all(q >= 0))
        numeric = np.trapezoid(q, grid, axis=-2)
        np.testing.assert_allclose(numeric, self.paths.cumulative([5000])[..., 0, :], rtol=1e-7)
        cumulative = self.paths.cumulative(grid)
        self.assertTrue(np.all(np.diff(cumulative, axis=-2) >= 0))
        np.testing.assert_allclose(self.paths.cumulative([0]), 0)

    def test_interval_additivity_and_query_consistency(self):
        combined = self.paths.interval(20, 2300)
        pieces = self.paths.interval(20, 800) + self.paths.interval(800, 2300)
        np.testing.assert_allclose(combined, pieces, rtol=1e-13)
        coarse = self.paths.cumulative([30, 180, 1080])
        dense = self.paths.cumulative([0, 30, 60, 180, 360, 1080, 2000])
        np.testing.assert_array_equal(coarse, dense[..., [1, 3, 5], :])

    def test_terminal_continuity_and_finite_tail(self):
        rates = self.paths.rates([1080 - 1e-5, 1080, 1080 + 1e-5])
        np.testing.assert_allclose(rates[..., 0, :], rates[..., 2, :], rtol=1e-6)
        np.testing.assert_allclose(self.paths.cumulative([1e8])[..., 0, :], self.paths.ultimate(), rtol=1e-12)
        self.assertTrue(np.isfinite(self.paths.ultimate()).all())
        # p=1 has the analytic log integral, not a singularity.
        self.assertAlmostEqual(self.paths.cumulative([120])[0, 0, 0], 240 * np.log(2), places=10)

    def test_invalid_paths_fail(self):
        with self.assertRaises(ValueError):
            DCATrajectorySamples(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            self.paths.interval(100, 99)
        with self.assertRaises(ValueError):
            self.paths.rates([-1])


class TestDCAPosterior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = synthetic_training()
        cls.model = HierarchicalDCABaseline(n_particles=128, seed=7).fit(cls.training, DAYS)

    def test_exact_finite_posterior_with_signed_report(self):
        history = np.full((1, len(DAYS), 3), np.nan)
        history[0, 1, 0] = -200.0
        weights = self.model.posterior_weights(history, 30)[0]
        mean = self.model.prior_cumulative_[:, 1, 0]
        sigma = self.model.prior_sigma_[:, 1, 0]
        log_density = -0.5 * ((-200.0 - mean) / sigma) ** 2 - np.log(sigma)
        expected = np.exp(log_density - np.max(log_density))
        expected /= expected.sum()
        np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-15)
        self.assertTrue(np.all(np.isfinite(weights)))
        # No observed positive-day history means precisely the prior.
        empty = np.full_like(history, np.nan)
        np.testing.assert_allclose(self.model.posterior_weights(empty, 30), 1 / 128)

    def test_cutoff_and_missingness_prevent_future_leakage(self):
        history = self.training[:3].copy()
        contaminated = history.copy()
        contaminated[:, DAYS > 60] = 1e12
        original = self.model.predict_samples(history, 60, 15, np.random.default_rng(4))
        changed = self.model.predict_samples(contaminated, 60, 15, np.random.default_rng(4))
        np.testing.assert_array_equal(original, changed)
        missing = history.copy()
        missing[:, DAYS > 60] = np.nan
        np.testing.assert_array_equal(original, self.model.predict_samples(missing, 60, 15, np.random.default_rng(4)))
        fixed = np.broadcast_to(history[:, None, DAYS <= 60], original[:, :, DAYS <= 60].shape)
        np.testing.assert_array_equal(original[:, :, DAYS <= 60], fixed)
        self.assertEqual(self.model.last_posterior_diagnostics_["effective_sample_size"].shape, (3,))

    def test_fit_repeatability_and_no_input_mutation(self):
        before = self.training.copy()
        other = HierarchicalDCABaseline(n_particles=128, seed=7).fit(self.training, DAYS)
        np.testing.assert_array_equal(before, self.training)
        np.testing.assert_array_equal(self.model.prior_parameters_, other.prior_parameters_)

    def test_noise_sampler_matches_likelihood_and_increment_dependence(self):
        # Fix one latent function, leaving only report noise. Differences of
        # independent cumulative errors share a negative covariance.
        parameters = np.broadcast_to(self.model.prior_parameters_[0], (1, 40000, 3, 3))
        latent = DCATrajectorySamples(parameters)
        reports = self.model.sample_reports(latent, np.random.default_rng(11))
        means = latent.cumulative(DAYS)
        sigma = self.model.observation_sigma(means, DAYS)
        normalized = (reports - means)[:, :, 1:] / sigma[:, :, 1:]
        self.assertLess(abs(normalized.mean()), 0.01)
        self.assertLess(abs(normalized.std() - 1), 0.01)
        errors = reports[0, :, :, 0] - means[0, :, :, 0]
        left, right = errors[:, 2] - errors[:, 1], errors[:, 3] - errors[:, 2]
        covariance = np.cov(left, right)[0, 1]
        expected = -sigma[0, 0, 2, 0] ** 2
        self.assertAlmostEqual(covariance / expected, 1.0, delta=0.06)

    def test_joint_posterior_updates_unobserved_phase(self):
        # A declared two-component finite prior makes cross-phase conditioning
        # transparent and tests that no phase identity has been swapped.
        model = HierarchicalDCABaseline(n_particles=2, seed=1).fit(self.training, DAYS)
        low = np.array([[10, 120, 0.8], [20, 120, 0.5], [30, 120, 1.0]], dtype=float)
        model.prior_parameters_ = np.stack([low, low * np.array([4.0, 1.0, 1.0])])
        model.prior_cumulative_ = DCATrajectorySamples(model.prior_parameters_).cumulative(DAYS)
        model.prior_sigma_ = model.observation_sigma(model.prior_cumulative_, DAYS)
        history = np.full((1, len(DAYS), 3), np.nan)
        history[0, 2, 1] = model.prior_cumulative_[1, 2, 1]
        self.assertGreater(model.posterior_weights(history, 60)[0, 1], 0.99)
        samples = model.sample_latent(history, 60, 100, np.random.default_rng(2))
        self.assertGreater(samples.parameters[0, :, 0, 0].mean(), 39)

    def test_factorized_ablation_has_phase_specific_weights(self):
        model = ConditionalDCABaseline(n_particles=32, seed=1).fit(self.training, DAYS)
        history = np.full((1, len(DAYS), 3), np.nan)
        history[0, 2, 0] = self.training[0, 2, 0]
        weights = model.posterior_weights(history, 60)
        self.assertEqual(weights.shape, (1, 32, 3))
        np.testing.assert_allclose(weights[0, :, 1:], 1 / 32)
        self.assertEqual(model.predict_samples(history, 60, 10).shape, (1, 10, len(DAYS), 3))

    def test_missing_entire_phase_is_not_zero(self):
        training = self.training.copy()
        training[:3, :, 2] = np.nan
        model = HierarchicalDCABaseline(n_particles=32).fit(training, DAYS)
        self.assertEqual(model.fit_diagnostics_["wells_excluded_from_joint_prior_missing_entire_phase"], 3)
        self.assertTrue(np.isnan(model.train_parameters_[:3, 2]).all())

    def test_deterministic_zero_support_and_finite_grids(self):
        invalid = self.training[:3].copy()
        invalid[0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "Day zero"):
            HierarchicalDCABaseline(n_particles=32).fit(invalid, DAYS)
        with self.assertRaisesRegex(ValueError, "Day zero"):
            self.model.predict_samples(invalid, 60, 5)
        for kwargs in ({"tau_grid": (float("nan"),)}, {"p_grid": (float("inf"),)}):
            with self.assertRaisesRegex(ValueError, "finite positive"):
                HierarchicalDCABaseline(**kwargs)


if __name__ == "__main__":
    unittest.main()
