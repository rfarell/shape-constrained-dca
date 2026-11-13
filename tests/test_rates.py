"""Consequential trajectory accounting and differentiability checks."""

import math
import unittest

import torch

from joint_production.rates import EnvelopeRate, PositiveBasisRate


class RateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.z = torch.randn(2, 3, 12, dtype=torch.float64) * .6
        self.families = (EnvelopeRate(), PositiveBasisRate(), PositiveBasisRate(preset="fast"))

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
        # Split at envelope knots so a numerical grid never straddles a jump.
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
        for preset in ("original", "fast"):
            family = PositiveBasisRate(preset=preset)
            torch.testing.assert_close(family.cumulative(family.initial_latent(), [1080.]), torch.ones(1, 3))
        with self.assertRaises(ValueError):
            PositiveBasisRate(preset="unknown")
        for family in self.families:
            for invalid in ([-1.], [math.nan], [-math.inf]):
                with self.assertRaises(ValueError):
                    family.cumulative(self.z, invalid)
            with self.assertRaises(ValueError):
                family.rate(torch.zeros(11), [0.])

    def test_prespecified_basis_dictionaries_against_independent_gamma_cdf(self):
        # Analytic integer-gamma CDF, independently evaluated using SciPy.
        from scipy.special import gammainc
        import numpy as np
        times = np.array([0., 1e-6, 30., 60., 90., 180., 360., 720., 1080., math.inf])
        specifications = {
            "original": ([1, 2, 2, 3], [360., 90., 360., 720.]),
            "fast": ([1, 1, 2, 2], [30., 180., 360., 720.]),
        }
        for preset, (shapes, scales) in specifications.items():
            matrix = gammainc(np.asarray(shapes)[None], times[:, None] / np.asarray(scales)[None])
            weights = self.z.numpy().reshape(2, 3, 3, 4)
            expected = np.einsum("...pk,tk->...tp", np.exp(weights), matrix)
            actual = PositiveBasisRate(preset=preset).cumulative(self.z, times).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(PositiveBasisRate().cumulative(self.z, times), PositiveBasisRate(preset="original").cumulative(self.z, times), rtol=0, atol=0)

    def test_basis_rate_derivatives_match_autograd_and_boundary_limits(self):
        for preset in ("original", "fast"):
            family = PositiveBasisRate(preset=preset)
            z = self.z[0, 0]
            times = torch.tensor([0., 1e-6, 13., 88., 360., 720., 3000.],
                                 dtype=torch.float64, requires_grad=True)
            rate = family.rate(z, times)
            for phase in range(3):
                first, = torch.autograd.grad(rate[:, phase].sum(), times, create_graph=True)
                second, = torch.autograd.grad(first.sum(), times, retain_graph=True)
                torch.testing.assert_close(family.rate_derivative(z, times)[:, phase], first, rtol=1e-12, atol=1e-16)
                torch.testing.assert_close(family.rate_derivative(z, times, order=2)[:, phase], second, rtol=1e-12, atol=1e-16)
            scales = z.new_tensor(family.scales_days)
            weights = z.reshape(3, 4).exp()
            # Independent right limits of integer-gamma densities at zero.
            first_at_zero = z.new_tensor([-1 if n == 1 else 1 if n == 2 else 0 for n in family.shapes]) / scales**2
            second_at_zero = z.new_tensor([1 if n == 1 else -2 if n == 2 else 1 for n in family.shapes]) / scales**3
            torch.testing.assert_close(family.rate_derivative(z, [0.])[0], weights @ first_at_zero)
            torch.testing.assert_close(family.rate_derivative(z, [0.], order=2)[0], weights @ second_at_zero)

    def test_basis_derivative_draw_coherence_phase_order_and_latent_gradients(self):
        for preset in ("original", "fast"):
            family = PositiveBasisRate(preset=preset)
            for order in (1, 2):
                dense = family.rate_derivative(self.z, [0., 43., 902., math.inf], order=order)
                self.assertEqual(dense.shape, (2, 3, 4, 3))
                torch.testing.assert_close(dense[..., [1, 2], :], family.rate_derivative(self.z, [43., 902.], order=order))
                torch.testing.assert_close(dense[..., -1, :], torch.zeros_like(dense[..., -1, :]))
                shifted = self.z.clone()
                shifted[..., :4] += math.log(2.)
                changed = family.rate_derivative(shifted, [0., 43., 902., math.inf], order=order)
                torch.testing.assert_close(changed[..., 0], 2 * dense[..., 0])
                torch.testing.assert_close(changed[..., 1:], dense[..., 1:])
                z = family.initial_latent(dtype=torch.float32).requires_grad_(True)
                family.rate_derivative(z, [0., 1e-6, 1e8, math.inf], order=order).sum().backward()
                self.assertTrue(torch.isfinite(z.grad).all())
            for order in (0, 3, True, 1.0):
                with self.assertRaises(ValueError):
                    family.rate_derivative(self.z, [0.], order=order)


if __name__ == "__main__":
    unittest.main()
