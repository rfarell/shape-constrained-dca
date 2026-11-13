"""Independent integration and gradient checks for Torch conventional DCA."""

import math
import unittest

import numpy as np
from scipy.integrate import quad
import torch

from joint_production.dca import DCATrajectorySamples
from joint_production.dca_rates import DCARate


class TestDCARate(unittest.TestCase):
    def setUp(self):
        self.family = DCARate()
        rng = np.random.default_rng(24)
        parameters = np.empty((2, 4, 3, 3))
        parameters[..., 0] = np.exp(rng.normal(-5, 0.6, (2, 4, 3)))
        parameters[..., 1] = np.exp(rng.normal(5, 0.8, (2, 4, 3)))
        parameters[..., 2] = np.exp(rng.normal(-0.3, 0.8, (2, 4, 3)))
        self.parameters = parameters
        self.z = torch.tensor(np.log(parameters).reshape((2, 4, 9)), dtype=torch.float64)

    def test_agrees_with_native_day_numpy_dca_and_units(self):
        times = [0., 1e-6, 30., 60., 180., 360., 720., 1080., 1493., 10000.]
        numpy_paths = DCATrajectorySamples(self.parameters)
        np.testing.assert_allclose(self.family.cumulative(self.z, times).numpy(), numpy_paths.cumulative(times), rtol=2e-12, atol=1e-13)
        np.testing.assert_allclose(self.family.rate(self.z, times).numpy(), numpy_paths.rates(times), rtol=2e-12, atol=1e-13)
        np.testing.assert_allclose(self.family.total_volume(self.z).numpy(), numpy_paths.ultimate(), rtol=2e-12, atol=1e-13)
        # Native phase scales multiply both integrated volumes and daily rates.
        scale = np.array([10000., 30000., 5000.])
        native = self.parameters.copy()
        native[..., 0] *= scale
        np.testing.assert_allclose(self.family.cumulative(self.z, times).numpy() * scale, DCATrajectorySamples(native).cumulative(times), rtol=2e-12)

    def test_independent_quadrature_and_p_one_limit(self):
        for p in (0.02, 0.999999999, 1.0, 1.000000001, 2.5, 30.0):
            amplitude, tau = 0.03, 140.0
            z = torch.tensor([math.log(amplitude), math.log(tau), math.log(p)] * 3, dtype=torch.float64)

            def direct_rate(t):
                if t <= 1080:
                    return amplitude * (1 + t / tau) ** (-p)
                return amplitude * (1 + 1080 / tau) ** (-p) * math.exp(-p * (t - 1080) / (tau + 1080))

            for start, end in ((0., 42.), (42., 1080.), (1080., 4397.)):
                expected = quad(direct_rate, start, end, epsabs=1e-11, epsrel=1e-11)[0]
                endpoints = self.family.cumulative(z, [start, end])
                self.assertAlmostEqual(float(endpoints[1, 0] - endpoints[0, 0]), expected, delta=1e-10)
            if p == 1:
                self.assertAlmostEqual(float(self.family.cumulative(z, [tau])[0, 0]), amplitude * tau * math.log(2), places=12)

    def test_query_consistency_additivity_and_phase_identity(self):
        dense = self.family.cumulative(self.z, [0., 15., 90., 360., 1080., 1500., math.inf])
        sparse = self.family.cumulative(self.z, [15., 1080., math.inf])
        torch.testing.assert_close(dense[..., [1, 4, 6], :], sparse, rtol=0, atol=0)
        torch.testing.assert_close(torch.diff(dense, dim=-2).sum(-2), self.family.total_volume(self.z))
        shifted = self.z.clone()
        shifted[..., 0] += math.log(2)
        changed = self.family.cumulative(shifted, [30., 720.])
        original = self.family.cumulative(self.z, [30., 720.])
        torch.testing.assert_close(changed[..., 0], original[..., 0] * 2)
        torch.testing.assert_close(changed[..., 1:], original[..., 1:])
        self.assertEqual(self.family.amplitude_indices, (0, 3, 6))
        self.assertEqual(self.family.latent_dimension, 9)
        self.assertEqual(self.family.parameters_per_phase, 3)

    def test_time_derivatives_at_zero_and_terminal_junction(self):
        for p in (0.1, 1.0, 8.0):
            z = torch.tensor([math.log(.02), math.log(120), math.log(p)] * 3, dtype=torch.float64, requires_grad=True)
            times = torch.tensor([0., 0.01, 1079.999, 1080., 1080.001, 3000.], dtype=torch.float64, requires_grad=True)
            volumes = self.family.cumulative(z, times)
            time_derivative, latent_gradient = torch.autograd.grad(volumes.sum(), (times, z))
            torch.testing.assert_close(time_derivative, self.family.rate(z, times).sum(-1), rtol=2e-10, atol=1e-12)
            self.assertTrue(torch.isfinite(latent_gradient).all())
            left = self.family.rate(z, [1080 - 1e-5])
            right = self.family.rate(z, [1080 + 1e-5])
            torch.testing.assert_close(left, right, rtol=2e-6, atol=1e-15)

    def test_latent_gradcheck_including_p_one_and_tail(self):
        z = torch.tensor([math.log(.02), math.log(120), 0., math.log(.03), math.log(300), math.log(1 + 1e-8), math.log(.01), math.log(50), math.log(1 - 1e-8)], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda value: self.family.cumulative(value, [0., 1e-6, 180., 1080., 2500.]), (z,), eps=1e-6, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.autograd.gradcheck(self.family.total_volume, (z,), eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_float32_extreme_shapes_have_finite_values_and_gradients(self):
        for p in (1e-5, 0.999999, 1.0, 1.000001, 1e3):
            z = torch.tensor([math.log(.01), math.log(120), math.log(p)] * 3, dtype=torch.float32, requires_grad=True)
            cumulative = self.family.cumulative(z, [0., 1e-6, 1080., 1e8, math.inf])
            rates = self.family.rate(z, [0., 1e-6, 1080., 1e8, math.inf])
            loss = torch.log1p(cumulative).sum() + rates.sum()
            loss.backward()
            self.assertTrue(torch.isfinite(cumulative).all())
            self.assertTrue(torch.isfinite(rates).all())
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertTrue(torch.all(cumulative >= 0))
            self.assertTrue(torch.all(torch.diff(cumulative, dim=-2) >= -1e-5))
            self.assertTrue(torch.all(rates >= 0))

    def test_initialization_and_invalid_queries(self):
        initial = self.family.initial_latent(dtype=torch.float64)
        torch.testing.assert_close(self.family.cumulative(initial, [1080.]), torch.ones(1, 3, dtype=torch.float64), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(self.family.rate(initial, [math.inf]), torch.zeros(1, 3, dtype=torch.float64))
        for times in ([-1.], [math.nan], [-math.inf]):
            with self.assertRaises(ValueError):
                self.family.cumulative(initial, times)
        for latent in (torch.zeros(12), torch.zeros(9, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                self.family.rate(latent, [30.])
        with self.assertRaises(ValueError):
            DCARate(tail_start_days=0)

    def test_unrequested_overflowing_total_does_not_corrupt_finite_forecasts(self):
        # The finite-time nearly constant curve is representable in float32;
        # its imposed lifetime total exceeds that dtype. Training endpoints
        # must not acquire NaN gradients from an unused infinity branch.
        z = torch.tensor([math.log(.01), math.log(120), math.log(1e-40)] * 3, dtype=torch.float32, requires_grad=True)
        cumulative = self.family.cumulative(z, [0., 30., 1080., 2000.])
        cumulative.sum().backward()
        self.assertTrue(torch.isfinite(cumulative).all())
        self.assertTrue(torch.isfinite(z.grad).all())
        torch.testing.assert_close(cumulative[-1], torch.full((3,), 20.), rtol=1e-5, atol=1e-5)

    def test_saturated_hyperbolic_integrals_are_strictly_monotone_in_float32(self):
        # Synthetic counterexamples to the former log1p*exprel factorization:
        # with A=.01,tau=1,p=10, Q(60) rounded one ULP below Q(30).
        # These are arithmetic examples, not retained field-well parameters.
        days = [0., 30., 60., 90., 180., 360., 720., 1080.]
        specifications = [(tau, p) for tau in (1., 10., 30., 100., 120., 300.) for p in (10., 20., 30., 50., 100., 1000.)]
        z = torch.tensor([[math.log(.01), math.log(tau), math.log(p)] * 3 for tau, p in specifications], dtype=torch.float32, requires_grad=True)
        cumulative = self.family.cumulative(z, days)
        self.assertTrue(torch.isfinite(cumulative).all())
        self.assertTrue(torch.all(torch.diff(cumulative, dim=-2) >= 0))
        # Saturated cells must remain equal without applying cummax/clipping.
        torch.testing.assert_close(cumulative[0, 1:, :], cumulative[0, 1, :].expand(7, 3), rtol=0, atol=0)
        cumulative.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())


if __name__ == "__main__":
    unittest.main()
