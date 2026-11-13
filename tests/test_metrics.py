"""Consequential checks of scoring targets, normalization, and joint dependence."""

import json
import unittest

import numpy as np

from joint_production.metrics import (
    energy_score,
    evaluate_forecasts,
    fit_increment_scales,
    interval_score,
    safe_ratio,
    sample_derived_quantities,
    variogram_score,
    weighted_interval_score,
)


class MetricTests(unittest.TestCase):
    def test_wis_analytic_intervals_and_normalization(self):
        # Exactly median 1, central 80% [0,2], and central 95% [-1,3].
        draws = np.interp(np.arange(41), [0, 1, 4, 20, 36, 39, 40], [-2, -1, 0, 1, 2, 3, 4])
        samples = np.broadcast_to(draws[None, :, None, None], (2, 41, 1, 3))
        targets = np.broadcast_to(np.array([1., 4.])[:, None, None], (2, 1, 3))
        score = weighted_interval_score(targets, samples)
        np.testing.assert_allclose(score[:, 0, 0], [0.12, 1.92])
        deterministic = np.full((2, 1, 1, 3), 2.)
        np.testing.assert_allclose(weighted_interval_score(targets, deterministic), np.abs(targets - 2))
        self.assertEqual(float(interval_score(4., 0., 2., 0.2)), 22.)

    def test_missing_targets_and_invalid_predictions_are_distinct(self):
        targets = np.array([[[1., np.nan, 2.]]])
        samples = np.array([[[[1., np.nan, 2.]], [[1., np.inf, 2.]]]])
        scores = weighted_interval_score(targets, samples)
        np.testing.assert_array_equal(scores[..., [0, 2]], 0.)
        self.assertTrue(np.isnan(scores[..., 1]).all())
        bad_samples = samples.copy()
        bad_samples[0, 1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite forecast"):
            weighted_interval_score(targets, bad_samples)
        with self.assertRaisesRegex(ValueError, "Nonfinite forecast"):
            energy_score(targets, bad_samples, np.ones((1, 3)))

    def test_finite_exploded_predictions_cannot_disappear_from_score_means(self):
        targets = np.broadcast_to(np.array([0., 1.])[:, None, None], (2, 1, 3))
        samples = np.zeros((2, 4, 1, 3))
        samples[1] = 1e308
        with self.assertRaisesRegex(ValueError, "Numerical failure"):
            evaluate_forecasts(targets, samples, np.ones((1, 3)))
        # This smaller explosion leaves WIS finite but overflows squared error.
        samples[1] = 1e160
        with self.assertRaisesRegex(ValueError, "Numerical failure"):
            evaluate_forecasts(targets, samples, np.ones((1, 3)))
        with self.assertRaisesRegex(ValueError, "Numerical failure"):
            energy_score(targets, samples, np.ones((1, 3)))

    def test_training_scales_preserve_zero_negative_and_phase_units(self):
        training = np.array([
            [[0., -10., np.nan], [2., np.nan, 0.]],
            [[0., 30., np.nan], [6., np.nan, 0.]],
        ])
        scales, details = fit_increment_scales(training, phase_fallback=[1., 1., 7.], return_details=True)
        np.testing.assert_allclose(scales, [[4., 20., 7.], [4., 20., 7.]])
        self.assertEqual(details["scale_source"][0][0], "phase_pooled_positive_median")
        self.assertEqual(details["scale_source"][0][2], "explicit_native_unit_fallback")
        self.assertEqual(details["training_observation_count"][0][2], 0)
        with self.assertRaises(ValueError):
            fit_increment_scales(training, phase_fallback=[0., 1., 1.])

    def test_positive_unit_conversion_preserves_all_scaled_scores(self):
        rng = np.random.default_rng(123)
        training = rng.normal(10, 2, size=(20, 3, 3))
        targets = rng.normal(10, 2, size=(7, 3, 3))
        samples = rng.normal(10, 2, size=(7, 16, 3, 3))
        conversion = np.array([6.28981, 1000., 6.28981])
        scales = fit_increment_scales(training)
        converted_scales = fit_increment_scales(training * conversion)
        np.testing.assert_allclose(converted_scales, scales * conversion)
        score = evaluate_forecasts(targets, samples, scales)
        converted = evaluate_forecasts(targets * conversion, samples * conversion, converted_scales)
        for key in ("macro_scaled_wis", "macro_increment_r2", "macro_scaled_width_80", "macro_coverage_95"):
            self.assertAlmostEqual(score[key], converted[key])
        for key in ("mean_energy_score", "mean_variogram_score_p05"):
            self.assertAlmostEqual(score["joint"][key], converted["joint"][key])
        for phase, factor in zip(("oil", "gas", "water"), conversion):
            self.assertAlmostEqual(score["per_phase"][phase]["mae"] * factor, converted["per_phase"][phase]["mae"])

    def test_joint_scores_detect_wrong_dependence_with_equal_marginals(self):
        targets = np.zeros((1, 1, 2))
        correlated = np.array([[[[0., 0.]], [[2., 2.]]]])
        anticorrelated = np.array([[[[0., 2.]], [[2., 0.]]]])
        scales = np.ones((1, 2))
        np.testing.assert_allclose(weighted_interval_score(targets, correlated), weighted_interval_score(targets, anticorrelated))
        np.testing.assert_allclose(energy_score(targets, correlated, scales), np.sqrt(2) / 2)
        np.testing.assert_allclose(energy_score(targets, anticorrelated, scales), 2 - np.sqrt(2) / 2)
        np.testing.assert_allclose(variogram_score(targets, correlated, scales), 0.)
        np.testing.assert_allclose(variogram_score(targets, anticorrelated, scales), 2.)
        np.testing.assert_allclose(energy_score(targets, correlated[:, ::-1], scales), energy_score(targets, correlated, scales))

    def test_joint_scores_do_not_join_differently_conditioned_origins(self):
        targets = np.zeros((1, 2, 3))
        samples = np.zeros((1, 2, 2, 3))
        samples[:, 1] = 2.
        permuted = samples.copy()
        permuted[:, :, 1] = samples[:, ::-1, 1]
        scales = np.ones((2, 3))
        separate = evaluate_forecasts(targets, samples, scales, joint_cell_groups=[[0], [1]])
        separate_permuted = evaluate_forecasts(targets, permuted, scales, joint_cell_groups=[[0], [1]])
        self.assertEqual(separate["joint"], separate_permuted["joint"])
        joined = evaluate_forecasts(targets, samples, scales)
        joined_permuted = evaluate_forecasts(targets, permuted, scales)
        self.assertNotEqual(joined["joint"]["mean_energy_score"], joined_permuted["joint"]["mean_energy_score"])
        with self.assertRaisesRegex(ValueError, "partition"):
            evaluate_forecasts(targets, samples, scales, joint_cell_groups=[[0], [0]])

    def test_macro_uses_equal_cells_and_equal_phases(self):
        targets = np.zeros((3, 2, 3))
        targets[1:, 1, 0] = np.nan
        targets[:, 1, 1] = np.nan
        deterministic_errors = np.array([[1., 6., 10.], [9., 999., 14.]])
        samples = np.broadcast_to(deterministic_errors[None, None], (3, 1, 2, 3))
        score = evaluate_forecasts(targets, samples, np.ones((2, 3)))
        self.assertAlmostEqual(score["macro_scaled_wis"], (5 + 6 + 12) / 3)
        self.assertEqual(score["per_phase"]["oil"]["scored_cells"], 2)
        self.assertEqual(score["per_phase"]["gas"]["scored_cells"], 1)
        self.assertIsNone(score["macro_increment_r2"])
        self.assertEqual(score["per_phase"]["oil"]["r2_undefined_fewer_than_two_targets"], 1)
        self.assertEqual(score["per_phase"]["oil"]["r2_undefined_zero_target_variance"], 1)
        json.dumps(score, allow_nan=False)

    def test_point_scores_use_mean_for_r2_rmse_and_median_for_mae(self):
        targets = np.broadcast_to(np.array([0., 2.])[:, None, None], (2, 1, 3))
        samples = targets[:, None] + np.array([0., 0., 9.])[None, :, None, None]
        result = evaluate_forecasts(targets, samples, np.ones((1, 3)))
        self.assertAlmostEqual(result["macro_increment_r2"], -8.)
        self.assertAlmostEqual(result["per_phase"]["oil"]["rmse"], 3.)
        self.assertEqual(result["per_phase"]["oil"]["mae"], 0.)

    def test_ratio_is_samplewise_and_zero_denominator_stays_undefined(self):
        volumes = np.array([[1., 2., 1.], [2., 8., 0.], [0., 3., 0.], [-1., 3., 2.]])
        result = sample_derived_quantities(volumes)
        np.testing.assert_allclose(result["gor_mcf_per_bbl"][:2], [2., 4.])
        np.testing.assert_allclose(result["water_cut"][:2], [.5, 0.])
        self.assertTrue(np.isnan(result["gor_mcf_per_bbl"][2:]).all())
        self.assertTrue(np.isnan(result["water_cut"][2:]).all())
        self.assertNotEqual(result["gor_mcf_per_bbl"][:2].mean(), volumes[:2, 1].mean() / volumes[:2, 0].mean())
        self.assertTrue(np.isnan(safe_ratio([1., 1., np.nan], [0., -1., 1.])).all())
        self.assertTrue(np.isnan(safe_ratio(1., .01, minimum_denominator=.01)))

    def test_unobserved_joint_vector_and_single_coordinate_variogram(self):
        targets = np.array([[[np.nan, np.nan, np.nan]], [[1., np.nan, np.nan]]])
        samples = np.zeros((2, 3, 1, 3))
        es = energy_score(targets, samples, np.ones((1, 3)))
        vs = variogram_score(targets, samples, np.ones((1, 3)))
        self.assertTrue(np.isnan(es[0]))
        self.assertEqual(es[1], 1.)
        self.assertTrue(np.isnan(vs).all())


if __name__ == "__main__":
    unittest.main()
