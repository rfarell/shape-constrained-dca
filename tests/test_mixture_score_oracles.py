"""Independent finite-law and gradient oracles for stratified mixture scores.

The population oracle integrates over all ordered draws from a discrete law.
It does not use the stratified estimator's within/between-component formula.
All stochastic checks use fixed seeds; no fitted models or field data are used.
"""

import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import itertools
import math
import unittest

import numpy as np
import torch

from joint_production.conditional_rates import _stratified_scores, _u_scores


DTYPE = torch.float64


def _population_scores(paths, probabilities, target, observed, days, origin,
                       scales, target_weights):
    """Exactly integrate a finite predictive law using the score definition."""
    future_indices = torch.nonzero(days > days[origin], as_tuple=False).flatten()
    zero = paths.sum() * 0 + probabilities.sum() * 0
    marginal = zero
    for cell, time_index in enumerate(future_indices):
        for phase in range(3):
            if not bool(torch.isfinite(target[time_index, phase])
                        & torch.isfinite(target[origin, phase])):
                continue
            prediction = (paths[:, time_index, phase] - paths[:, origin, phase]) / scales[cell, phase]
            truth = (target[time_index, phase] - target[origin, phase]) / scales[cell, phase]
            accuracy = sum(probabilities[i] * abs(prediction[i] - truth)
                           for i in range(len(paths)))
            pair_distance = sum(probabilities[i] * probabilities[j]
                                * abs(prediction[i] - prediction[j])
                                for i in range(len(paths)) for j in range(len(paths)))
            marginal = marginal + target_weights[cell, phase] * (accuracy - pair_distance / 2)
    marginal = marginal / (len(future_indices) * 3)

    scored = torch.isfinite(target) & ~observed & (days > 0)[:, None]
    dimension = int(scored.sum())
    if not dimension:
        return marginal, zero, dimension
    prediction = paths[:, scored]
    truth = target[scored]
    accuracy = sum(probabilities[i] * torch.linalg.vector_norm(prediction[i] - truth)
                   for i in range(len(paths)))
    pair_distance = sum(probabilities[i] * probabilities[j]
                        * torch.linalg.vector_norm(prediction[i] - prediction[j])
                        for i in range(len(paths)) for j in range(len(paths)))
    return marginal, (accuracy - pair_distance / 2) / math.sqrt(dimension), dimension


def _fixture():
    days = torch.tensor([0., 30., 60., 90., 180.], dtype=DTYPE)
    target = torch.tensor([[0., 0., 0.], [2., 5., 3.], [3., 7., 4.],
                           [5., 11., float("nan")], [8., 14., 9.]], dtype=DTYPE)
    observed = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 0, 1],
                             [0, 0, 0], [0, 0, 0]], dtype=torch.bool)
    offsets = torch.tensor([
        [[0., 0., 0.], [0., -1.7, -.8], [0., -1.1, 0.], [.8, -2.1, .3], [1.4, -.9, .6]],
        [[0., 0., 0.], [0., .4, 1.3], [0., .7, 0.], [-.5, 1.9, -.4], [-1.8, 2.4, -.7]],
        [[0., 0., 0.], [0., 2.2, -.4], [0., 1.8, 0.], [2.4, 3.2, 1.4], [3.7, 4.1, 2.1]],
        [[0., 0., 0.], [0., 3.6, 2.1], [0., 4.1, 0.], [4.3, 5.7, 3.1], [5.8, 7.4, 4.2]],
    ], dtype=DTYPE).reshape(2, 2, 5, 3)
    paths = torch.nan_to_num(target)[None, None] + offsets
    scales = torch.tensor([[2., 4., 3.], [5., 6., 8.]], dtype=DTYPE)
    target_weights = torch.tensor([[1.25, 2., 3.], [1., 1.5, 1.75]], dtype=DTYPE)
    support_probability = torch.tensor([[.3, .7], [.65, .35]], dtype=DTYPE)
    return paths, target, observed, days, 2, scales, target_weights, support_probability


def _enumerated_estimator_expectation(paths, probabilities, support_probability,
                                      target, observed, days, origin, scales,
                                      target_weights):
    """Enumerate all independent two-draw-per-component experiments."""
    assignments = list(itertools.product(range(2), repeat=4))
    experiments = torch.stack([
        torch.stack([paths[0, list(assignment[:2])], paths[1, list(assignment[2:])]])
        for assignment in assignments
    ])
    experiment_probability = torch.stack([
        support_probability[0, assignment[0]] * support_probability[0, assignment[1]]
        * support_probability[1, assignment[2]] * support_probability[1, assignment[3]]
        for assignment in assignments
    ])
    marginal, energy, dimension = _stratified_scores(
        target[None].expand(len(assignments), -1, -1), experiments,
        probabilities[None].expand(len(assignments), -1),
        observed[None].expand(len(assignments), -1, -1), days, origin,
        scales, target_weights)
    return ((marginal * experiment_probability).sum(),
            (energy * experiment_probability).sum(), dimension)


class TestMixtureScoreOracles(unittest.TestCase):
    def test_scalar_hand_enumeration_values_and_closed_form_gradients(self):
        # Every phase repeats the same scalar. Dimension normalization then
        # makes the joint energy exactly equal to the marginal CRPS.
        scalar_paths = torch.tensor([[0., 2.], [4., 8.]], dtype=DTYPE, requires_grad=True)
        logits = torch.tensor([math.log(.25), math.log(.75)], dtype=DTYPE, requires_grad=True)
        future = scalar_paths[None, :, :, None, None].expand(1, 2, 2, 1, 3)
        samples = torch.cat((torch.zeros_like(future), future), dim=3)
        target = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]], dtype=DTYPE)
        observed = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.bool)
        m, e, d = _stratified_scores(target, samples, logits.softmax(-1)[None],
                                     observed, torch.tensor([0., 30.], dtype=DTYPE), 0,
                                     torch.ones((1, 3), dtype=DTYPE), torch.ones((1, 3), dtype=DTYPE))
        # Accuracy 4; within penalties 1/16 + 9/8; cross penalty 15/16.
        torch.testing.assert_close(m, torch.tensor([1.875], dtype=DTYPE), rtol=0, atol=1e-14)
        torch.testing.assert_close(e, m, rtol=0, atol=1e-14)
        self.assertEqual(d.tolist(), [3])
        sample_gradient, logit_gradient = torch.autograd.grad((m + e).sum(), (scalar_paths, logits))
        torch.testing.assert_close(sample_gradient, torch.tensor([[0., .375], [1.125, 0.]], dtype=DTYPE),
                                   rtol=0, atol=1e-14)
        torch.testing.assert_close(logit_gradient, torch.tensor([-1.5, 1.5], dtype=DTYPE),
                                   rtol=0, atol=1e-14)

    def test_exact_finite_law_expectation_and_parameter_gradients(self):
        paths, target, observed, days, origin, scales, weights, support_probability = _fixture()
        paths.requires_grad_()
        logits = torch.tensor([.35, -.55], dtype=DTYPE, requires_grad=True)
        probabilities = logits.softmax(-1)
        estimated = _enumerated_estimator_expectation(paths, probabilities, support_probability,
                                                       target, observed, days, origin, scales, weights)
        actual = _population_scores(paths.flatten(0, 1),
                                    (probabilities[:, None] * support_probability).flatten(),
                                    target, observed, days, origin, scales, weights)
        torch.testing.assert_close(estimated[0], actual[0], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(estimated[1], actual[1], rtol=1e-13, atol=1e-13)
        self.assertTrue(torch.all(estimated[2] == actual[2]))
        self.assertEqual(actual[2], 8)  # Hidden history/origin plus five finite future values.
        estimated_gradient = torch.autograd.grad(estimated[0] + .7 * estimated[1], (paths, logits), retain_graph=True)
        actual_gradient = torch.autograd.grad(actual[0] + .7 * actual[1], (paths, logits))
        for got, expected in zip(estimated_gradient, actual_gradient):
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-13)
        self.assertGreater(float(estimated_gradient[1].abs().sum()), 1e-3)

    def test_tied_components_have_the_same_expected_score_for_every_mixing_weight(self):
        paths, target, observed, days, origin, scales, weights, support_probability = _fixture()
        paths = paths[0:1].repeat(2, 1, 1, 1)
        support_probability = support_probability[0:1].repeat(2, 1)
        actual = _population_scores(paths[0], support_probability[0], target,
                                    observed, days, origin, scales, weights)
        for probability in (.03, .25, .5, .91):
            with self.subTest(probability=probability):
                logits = torch.tensor([math.log(probability), math.log(1 - probability)],
                                      dtype=DTYPE, requires_grad=True)
                estimated = _enumerated_estimator_expectation(paths, logits.softmax(-1), support_probability,
                                                               target, observed, days, origin, scales, weights)
                torch.testing.assert_close(estimated[0], actual[0], rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(estimated[1], actual[1], rtol=1e-13, atol=1e-13)
                gradient, = torch.autograd.grad(estimated[0] + estimated[1], logits)
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-13)

    def test_zero_weight_component_reduces_exactly_to_iid_u_scores(self):
        paths, target, observed, days, origin, scales, weights, _ = _fixture()
        for component in (0, 1):
            mixture_weights = torch.tensor([[1 - component, component]], dtype=DTYPE)
            got = _stratified_scores(target[None], paths[None], mixture_weights,
                                     observed[None], days, origin, scales, weights)
            expected = _u_scores(target[None], paths[component][None], observed[None],
                                 days, origin, scales, weights)
            for actual, reference in zip(got, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=1e-14)

    def test_fixed_seed_monte_carlo_converges_to_exact_finite_law_scores(self):
        paths, target, observed, days, origin, scales, weights, support_probability = _fixture()
        rng = np.random.default_rng(240913)
        repetitions, draws_per_component = 20000, 3
        samples = torch.stack([
            paths[k, torch.tensor(rng.choice(2, size=(repetitions, draws_per_component),
                                              p=support_probability[k].numpy()))]
            for k in range(2)
        ], dim=1)
        probability = torch.tensor([.27, .73], dtype=DTYPE)
        got = _stratified_scores(target[None].expand(repetitions, -1, -1), samples,
                                 probability[None].expand(repetitions, -1),
                                 observed[None].expand(repetitions, -1, -1), days,
                                 origin, scales, weights)
        expected = _population_scores(paths.flatten(0, 1),
                                      (probability[:, None] * support_probability).flatten(),
                                      target, observed, days, origin, scales, weights)
        for estimates, reference in zip(got[:2], expected[:2]):
            standard_error = estimates.std(unbiased=True) / math.sqrt(repetitions)
            self.assertLess(float(abs(estimates.mean() - reference)), float(6 * standard_error))
            self.assertLess(float(standard_error), .015)

    def test_observed_reports_and_nonpositive_days_do_not_enter_energy(self):
        paths, target, observed, days, origin, scales, weights, _ = _fixture()
        # Change only known past coordinates, and day zero even when marked
        # unobserved. Neither belongs to the energy vector or a future increment.
        observed = observed.clone()
        observed[0] = False
        probability = torch.tensor([[.2, .8]], dtype=DTYPE)
        original = _stratified_scores(target[None], paths[None], probability,
                                      observed[None], days, origin, scales, weights)
        changed = paths.clone()
        changed[:, :, 0] += 5000
        changed[:, :, 1, 0] -= 3000
        modified = _stratified_scores(target[None], changed[None], probability,
                                      observed[None], days, origin, scales, weights)
        for actual, expected in zip(modified, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_coincident_samples_have_finite_sample_and_logit_gradients(self):
        samples = torch.zeros((2, 2, 3, 3, 3), dtype=DTYPE, requires_grad=True)
        logits = torch.tensor([[.4, -.3], [-.2, .8]], dtype=DTYPE, requires_grad=True)
        target = torch.zeros((2, 3, 3), dtype=DTYPE)
        observed = torch.ones_like(target, dtype=torch.bool)
        observed[:, 2] = False
        m, e, _ = _stratified_scores(target, samples, logits.softmax(-1), observed,
                                     torch.tensor([0., 30., 60.], dtype=DTYPE), 1,
                                     torch.ones((1, 3), dtype=DTYPE), torch.ones((1, 3), dtype=DTYPE))
        gradients = torch.autograd.grad((m + e).sum(), (samples, logits))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
        torch.testing.assert_close(m + e, torch.zeros(2, dtype=DTYPE), rtol=0, atol=0)

    def test_fewer_than_two_draws_per_component_is_rejected(self):
        paths, target, observed, days, origin, scales, weights, _ = _fixture()
        with self.assertRaises(ValueError):
            _stratified_scores(target[None], paths[:, :1][None],
                               torch.tensor([[.4, .6]], dtype=DTYPE), observed[None],
                               days, origin, scales, weights)


if __name__ == "__main__":
    unittest.main()
