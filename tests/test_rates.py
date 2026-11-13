"""Consequential trajectory accounting and differentiability checks."""

import math
import unittest

import torch

from joint_production.rates import PositiveBasisRate


class RateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.z = torch.randn(2, 3, 12, dtype=torch.float64) * .6
        self.families = (PositiveBasisRate(),)

    def test_shapes_positivity_monotonicity_and_finite_tail(self):
        times = [0., 30., 60., 90., 180., 360., 720., 1080., 100000., math.inf]
        for family in self.families:
            with self.subTest(family=type(family).__name__):
                cumulative = family.cumulative(self.z, times)
                rate = family.rate(self.z, times)
                self.assertEqual(cumulative.shape, (2, 3, len(times), 3))
                self.assertTrue(torch.all(torch.isfinite(cumulative)))
                self.assertTrue(torch.all(rate >= 0))
                self.assertTrue(torch.all(torch.diff(cumulative, dim=-2) >= -1e-12))
                torch.testing.assert_close(cumulative[..., 0, :], torch.zeros_like(cumulative[..., 0, :]))
                torch.testing.assert_close(cumulative[..., -1, :], family.total_volume(self.z))
                torch.testing.assert_close(rate[..., -1, :], torch.zeros_like(rate[..., -1, :]))

    def test_query_coherence_phase_order_and_additivity(self):
        for family in self.families:
            dense = family.cumulative(self.z, [0., 43., 90., 171., 360., 902., 1080.])
            sparse = family.cumulative(self.z, [43., 902., 1080.])
            torch.testing.assert_close(dense[..., [1, 5, 6], :], sparse)
            pieces = torch.diff(dense, dim=-2).sum(-2)
            torch.testing.assert_close(pieces, dense[..., -1, :] - dense[..., 0, :])
            z = family.initial_latent(dtype=torch.float64)
            shifted = z.clone()
            indices = [index for index in family.amplitude_indices if index < 4]
            shifted[indices] += math.log(2.)
            original = family.cumulative(z, [70., 500.])
            changed = family.cumulative(shifted, [70., 500.])
            torch.testing.assert_close(changed[:, 0], 2 * original[:, 0])
            torch.testing.assert_close(changed[:, 1:], original[:, 1:])

    def test_integrals_against_independent_midpoint_quadrature(self):
        z = self.z[0, 0]
        for family in self.families:
            for start, end in [(0., 90.), (90., 360.), (360., 1493.)]:
                count = 8192
                step = (end - start) / count
                midpoints = start + (torch.arange(count, dtype=z.dtype) + .5) * step
                numerical = family.rate(z, midpoints).sum(0) * step
                endpoints = family.cumulative(z, [start, end])
                torch.testing.assert_close(numerical, endpoints[1] - endpoints[0], atol=1e-8, rtol=2e-7)

    def test_cumulative_derivative_matches_daily_rate(self):
        for family in self.families:
            times = torch.tensor([0., 13., 88., 140., 650., 20000.], dtype=torch.float64, requires_grad=True)
            z = self.z[0, 0].clone().requires_grad_(True)
            cumulative = family.cumulative(z, times)
            time_gradient, latent_gradient = torch.autograd.grad(cumulative.sum(), (times, z))
            torch.testing.assert_close(time_gradient, family.rate(z, times).sum(-1), atol=1e-10, rtol=1e-8)
            self.assertTrue(torch.all(torch.isfinite(latent_gradient)))

    def test_float32_latent_gradients_zero_and_large_time(self):
        for family in self.families:
            z = family.initial_latent(dtype=torch.float32).requires_grad_(True)
            result = family.cumulative(z, [0., 1e-6, 1080., 1e8, math.inf])
            result.sum().backward()
            self.assertTrue(torch.all(torch.isfinite(z.grad)))
            self.assertTrue(torch.all(result >= 0))

    def test_initialization_and_validation(self):
        family = PositiveBasisRate()
        torch.testing.assert_close(family.cumulative(family.initial_latent(), [1080.]), torch.ones(1, 3))
        for invalid in ([-1.], [math.nan], [-math.inf]):
            with self.assertRaises(ValueError):
                family.cumulative(self.z, invalid)
        with self.assertRaises(ValueError):
            family.rate(torch.zeros(11), [0.])

    def test_basis_dictionary_against_independent_gamma_cdf(self):
        # Analytic integer-gamma CDF, independently evaluated using SciPy.
        from scipy.special import gammainc
        import numpy as np
        times = np.array([0., 1e-6, 30., 60., 90., 180., 360., 720., 1080., math.inf])
        matrix = gammainc(np.array([1, 1, 2, 2])[None], times[:, None] / np.array([30., 180., 360., 720.])[None])
        weights = self.z.numpy().reshape(2, 3, 3, 4)
        expected = np.einsum("...pk,tk->...tp", np.exp(weights), matrix)
        actual = PositiveBasisRate().cumulative(self.z, times).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
