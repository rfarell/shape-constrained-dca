"""Consequential invariants of the joint reported-trajectory forest baseline."""

import unittest

import numpy as np

from joint_production.baselines import ProbabilisticTrajectoryForest


class TrajectoryForestTests(unittest.TestCase):
    def setUp(self):
        self.days = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=float)
        rng = np.random.default_rng(4)
        amplitude = rng.uniform(0.5, 3, size=(80, 1, 1))
        shapes = (self.days / (90 + self.days))[None, :, None]
        phase = np.array([10000, 75000, 18000])[None, None, :]
        self.reports = amplitude * shapes * phase
        self.reports += rng.normal(size=(80, 8, 3)) * phase * 0.02
        self.reports[:, 0] = 0

    def fit(self, reports=None, **kwargs):
        options = dict(n_estimators=12, min_samples_leaf=4, random_state=8, n_jobs=1)
        options.update(kwargs)
        return ProbabilisticTrajectoryForest(**options).fit(
            self.reports if reports is None else reports, self.days, origins=[60, 180]
        )

    def test_known_history_is_conditioned_exactly_and_signed_reports_survive(self):
        forest = self.fit()
        history = self.reports[:3].copy()
        history[0, 2, 1] = -11.0
        draws = forest.predict_samples(history, 60, 24, 17)
        self.assertEqual(draws.shape, (3, 24, 8, 3))
        np.testing.assert_array_equal(draws[:, :, :3], np.broadcast_to(history[:, None, :3], (3, 24, 3, 3)))
        self.assertTrue(np.isfinite(draws).all())

    def test_prediction_is_invariant_to_every_post_cutoff_report(self):
        forest = self.fit()
        first = self.reports[:4].copy()
        second = first.copy()
        first[:, 3:] = np.nan
        second[:, 3:] = -1e12
        np.testing.assert_array_equal(
            forest.predict_samples(first, 60, 40, 3),
            forest.predict_samples(second, 60, 40, 3),
        )

    def test_unobserved_cutoff_and_entire_phase_are_sampled_without_hidden_anchor(self):
        forest = self.fit()
        history = self.reports[:2].copy()
        history[:, :, 1] = np.nan
        history[:, 2, 0] = np.nan
        draws = forest.predict_samples(history, 60, 64, 5)
        self.assertTrue(np.isfinite(draws).all())
        self.assertTrue(np.any(np.std(draws[:, :, 2, 1], axis=1) > 0))
        self.assertTrue(np.any(np.std(draws[:, :, 2, 0], axis=1) > 0))
        np.testing.assert_array_equal(draws[:, :, 1, 0], np.broadcast_to(history[:, None, 1, 0], (2, 64)))

    def test_no_history_samples_one_whole_donor_without_independent_outputs(self):
        forest = self.fit(mask_augmentation=0)
        history = np.full((3, 8, 3), np.nan)
        draws = forest.predict_samples(history, 60, 50, 9)
        for trajectory in draws.reshape(-1, 8, 3):
            self.assertTrue(np.any(np.all(trajectory == self.reports, axis=(1, 2))))

    def test_shared_donor_preserves_cross_phase_and_time_relationship(self):
        # Exact relationships are deliberately absent from any marginal-only law.
        x = np.linspace(1, 9, 50)
        reports = np.zeros((50, 8, 3))
        reports[:, 3:, 0] = x[:, None] * np.arange(1, 6)
        reports[:, 3:, 1] = reports[:, 3:, 0] * 7
        reports[:, 3:, 2] = reports[:, 3:, 0] * 3
        forest = self.fit(reports, mask_augmentation=0, scale_adjustment=0)
        history = reports[:1].copy()
        history[:, 3:] = np.nan
        draws = forest.predict_samples(history, 60, 200, 10)
        np.testing.assert_array_equal(draws[:, :, :, 1], 7 * draws[:, :, :, 0])
        np.testing.assert_array_equal(draws[:, :, :, 2], 3 * draws[:, :, :, 0])
        np.testing.assert_array_equal(draws[:, :, 7, 0], 5 * draws[:, :, 3, 0])

    def test_phase_unit_conversion_does_not_change_learned_conditioning(self):
        forest = self.fit()
        factors = np.array([1, 1000, 0.1])
        converted = self.fit(self.reports * factors)
        original_draws = forest.predict_samples(self.reports[:3], 60, 50, 12)
        converted_draws = converted.predict_samples(self.reports[:3] * factors, 60, 50, 12)
        np.testing.assert_allclose(converted_draws / factors, original_draws, rtol=1e-10, atol=1e-8)

    def test_history_conditioning_improves_population_forecast_on_known_process(self):
        # Test on distinct synthetic amplitudes without examining project data.
        amplitude = np.linspace(0.1, 4, 160)
        shape = (self.days / (60 + self.days))[None, :, None]
        phase = np.array([100, 200, 300])[None, None, :]
        training = amplitude[:, None, None] * shape * phase
        forest = self.fit(training, n_estimators=20, min_samples_leaf=3, mask_augmentation=0)
        query = np.array([0.4, 1.7, 3.2])[:, None, None] * shape * phase
        draws = forest.predict_samples(query, 60, 400, 21)
        error = np.mean((draws.mean(axis=1)[:, 3:] - query[:, 3:]) ** 2)
        unconditional_error = np.mean((training.mean(axis=0)[None, 3:] - query[:, 3:]) ** 2)
        self.assertLess(error, 0.05 * unconditional_error)

    def test_missing_phase_conditions_on_other_phases_even_with_constant_tails(self):
        # Increment-only split targets are constant here and cannot learn the
        # missing gas level, although observed oil identifies it exactly.
        amplitude = np.linspace(1, 80, 160)
        training = np.zeros((160, 8, 3))
        training[:, 1:] = amplitude[:, None, None] * np.array([1, 7, 3])
        forest = self.fit(training, n_estimators=24, min_samples_leaf=2,
                          mask_augmentation=2, phase_mask_probability=0.3)
        query = training[[10, 140]].copy()
        query[:, :, 1] = np.nan
        draws = forest.predict_samples(query, 60, 800, 73)
        expected = training[[10, 140], 2, 1]
        predicted = draws[:, :, 2, 1].mean(axis=1)
        unconditional = training[:, 2, 1].mean()
        self.assertLess(np.mean((predicted - expected) ** 2),
                        0.05 * np.mean((unconditional - expected) ** 2))

    def test_covariates_and_missing_covariates_are_supported_and_required_consistently(self):
        cov = np.column_stack([np.arange(80), np.full(80, np.nan)])
        model = ProbabilisticTrajectoryForest(n_estimators=8, min_samples_leaf=3).fit(
            self.reports, self.days, cov, origins=[60]
        )
        prediction = model.predict_samples(self.reports[:2], 60, 12, 1, covariates=cov[:2])
        self.assertTrue(np.isfinite(prediction).all())
        with self.assertRaises(ValueError):
            model.predict_samples(self.reports[:2], 60, 12, 1)
        # Historical pickles predate the optional preprocessing-mode field.
        del model.covariate_preprocessing
        np.testing.assert_array_equal(
            model.predict_samples(self.reports[:2], 60, 12, 1, covariates=cov[:2]), prediction)

    def test_incomplete_training_donors_are_counted_and_never_imputed(self):
        reports = self.reports.copy()
        reports[3, 7, 1] = np.nan
        reports[11, 1, 0] = np.nan
        forest = self.fit(reports)
        self.assertEqual(forest.complete_donor_count_, 78)
        self.assertEqual(forest.dropped_incomplete_count_, 2)
        self.assertNotIn(3, forest.donor_row_indices_)
        self.assertNotIn(11, forest.donor_row_indices_)
        with self.assertRaises(ValueError):
            self.fit(np.full_like(reports, np.nan))

    def test_external_covariates_are_not_transformed_or_given_another_mask(self):
        cov = np.arange(80 * 34, dtype=float).reshape(80, 34) / 9 - 11
        model = ProbabilisticTrajectoryForest(
            n_estimators=4, covariate_preprocessing="external", mask_augmentation=0,
        ).fit(self.reports, self.days, cov, origins=[0])
        self.assertEqual(model.n_covariates_, 34)
        np.testing.assert_array_equal(model._covariate_features(cov[:2], 2), cov[:2])
        history_width = model._history_features(self.reports, 0, model.origin_models_[0].feature_scale).shape[1]
        self.assertEqual(model.origin_models_[0].forest.n_features_in_, history_width + 34)
        with self.assertRaises(ValueError):
            model.predict_samples(self.reports[:2], 0, 3, 2, covariates=cov[:2, :-1])
        with self.assertRaises(ValueError):
            model.predict_samples(self.reports[:2], 0, 3, 2, covariates=np.full((2, 34), np.nan))
        with self.assertRaises(ValueError):
            ProbabilisticTrajectoryForest(covariate_preprocessing="unknown")
        reports = self.reports.copy()
        reports[0, -1, 0] = np.nan
        cov[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ProbabilisticTrajectoryForest(covariate_preprocessing="external").fit(reports, self.days, cov)

    def test_origin_zero_conditions_on_secondary_features_with_joint_donor_and_no_future_leakage(self):
        amplitude = np.linspace(1, 30, 160)
        phase = np.array([1, 7, 3])
        training = amplitude[:, None, None] * self.days[None, :, None] * phase
        cov = np.zeros((160, 34))
        cov[:, 0] = amplitude
        cov[:, 17] = 1
        options = dict(n_estimators=20, min_samples_leaf=2, mask_augmentation=0,
                       scale_adjustment=0, covariate_preprocessing="external", random_state=42)
        informed = ProbabilisticTrajectoryForest(**options).fit(training, self.days, cov, origins=[0])
        control = ProbabilisticTrajectoryForest(**options).fit(training, self.days, np.zeros_like(cov), origins=[0])
        indices = [8, 145]
        history = training[indices].copy()
        history[:, 1:] = np.nan
        draws = informed.predict_samples(history, 0, 400, 52, covariates=cov[indices])
        baseline = control.predict_samples(history, 0, 400, 52, covariates=np.zeros((2, 34)))
        observed = training[indices]
        self.assertLess(np.mean((draws.mean(axis=1) - observed) ** 2),
                        0.02 * np.mean((baseline.mean(axis=1) - observed) ** 2))
        future_changed = history.copy()
        future_changed[:, 1:] = -1e15
        np.testing.assert_array_equal(
            informed.predict_samples(future_changed, 0, 400, 52, covariates=cov[indices]), draws)
        np.testing.assert_allclose(draws[:, :, :, 1], 7 * draws[:, :, :, 0])
        np.testing.assert_allclose(draws[:, :, :, 2], 3 * draws[:, :, :, 0])
        for trajectory in draws.reshape(-1, 8, 3):
            self.assertTrue(np.any(np.all(trajectory == training, axis=(1, 2))))

    def test_reproducible_rng_and_empty_queries(self):
        forest = self.fit()
        np.testing.assert_array_equal(
            forest.predict_samples(self.reports[:2], 60, 16, np.random.default_rng(1)),
            forest.predict_samples(self.reports[:2], 60, 16, np.random.default_rng(1)),
        )
        self.assertEqual(forest.predict_samples(self.reports[:0], 60, 16, 0).shape, (0, 16, 8, 3))

    def test_invalid_array_boundary_and_origin_fail_explicitly(self):
        forest = self.fit()
        with self.assertRaises(ValueError):
            forest.predict_samples(self.reports[:2], 3, 10)
        with self.assertRaises(ValueError):
            forest.predict_samples(self.reports[:2], 90, 10)
        with self.assertRaises(ValueError):
            forest.predict_samples(self.reports[:2], 60, 0)
        with self.assertRaises(ValueError):
            ProbabilisticTrajectoryForest().fit(self.reports[:, :, :2], self.days)
        with self.assertRaises(ValueError):
            ProbabilisticTrajectoryForest().fit(self.reports, self.days[::-1])


if __name__ == "__main__":
    unittest.main()
