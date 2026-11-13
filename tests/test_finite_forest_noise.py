"""Independent synthetic finite-mixture likelihood and bounded noise-fit checks."""

from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pytest
from scipy.special import logsumexp
import torch

import joint_production.finite_forest_noise as noise
from joint_production.static_features import STATIC_FEATURE_NAMES, StaticFeaturePreprocessor
from joint_production.tree_static_prior import (
    assign_group_folds, fast_basis_cumulative_matrix, fit_projection_scales,
    project_positive_coefficients,
)

DAYS = np.array([0., 30., 60., 90., 180., 360., 720., 1080.])
THETA = np.log([.2, .15, .25, .18, .12, .2])


@pytest.fixture
def mixture_folds():
    rng = np.random.default_rng(617)
    folds = []
    for number, indices, atoms in ((0, [4, 0, 2], 4), (1, [3, 1], 3)):
        scales = np.array([120., 700., 45.]) * (number + 1)
        coefficients = rng.uniform(.05, .8, (atoms, 3, 4)) * scales[None, :, None]
        cumulative = np.einsum("mpk,tk->mtp", coefficients, fast_basis_cumulative_matrix(DAYS))
        reports = cumulative[np.arange(len(indices)) % atoms].copy()
        increments = rng.normal(size=(len(indices), 7, 3)) * scales * .11
        reports[:, 1:] += np.cumsum(increments, axis=1)
        reports[0, 1, 0] = -3.  # Signed reports remain in the Gaussian support.
        raw = rng.uniform(size=(len(indices), atoms))
        raw[:, -1] = 0.
        raw /= raw.sum(axis=1, keepdims=True)
        folds.append(noise.NoiseFoldData(number, np.array(indices), reports, cumulative,
                                        .95 * raw + .05 / atoms, scales))
    return folds


def dense_oracle(theta, folds):
    """Use dense accumulated endpoint covariances, independently of innovations."""
    count = sum(len(fold.query_indices) for fold in folds)
    evidence = np.empty(count)
    normalized = np.empty(count)
    phase_independent = np.empty(count)
    for fold in folds:
        a = np.exp(theta[:3]) * fold.phase_scales_native
        r = np.exp(theta[3:])
        for row, index in enumerate(fold.query_indices):
            phase_ll = np.zeros((len(fold.cumulative_native), 3))
            for atom, q in enumerate(fold.cumulative_native):
                for phase in range(3):
                    clock = a[phase] ** 2 * DAYS[1:] / 360 + (r[phase] * q[1:, phase]) ** 2
                    covariance = np.minimum(clock[:, None], clock[None, :])
                    residual = fold.reports_native[row, 1:, phase] - q[1:, phase]
                    sign, logdet = np.linalg.slogdet(covariance)
                    assert sign == 1
                    phase_ll[atom, phase] = -.5 * (
                        7 * np.log(2 * np.pi) + logdet
                        + residual @ np.linalg.solve(covariance, residual))
            prior = np.log(fold.prior_probabilities[row])
            evidence[index] = logsumexp(prior + phase_ll.sum(axis=1))
            normalized[index] = evidence[index] + 7 * np.log(fold.phase_scales_native).sum()
            phase_independent[index] = logsumexp(prior[:, None] + phase_ll, axis=0).sum()
    return evidence, normalized, phase_independent


def dense_autograd(theta, folds):
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    total = torch.zeros((), dtype=torch.float64)
    for fold in folds:
        q = torch.tensor(fold.cumulative_native, dtype=torch.float64)
        y = torch.tensor(fold.reports_native, dtype=torch.float64)
        a = parameter[:3].exp() * torch.tensor(fold.phase_scales_native, dtype=torch.float64)
        r = parameter[3:].exp()
        clock = a.square() * torch.tensor(DAYS[1:, None] / 360) + (r * q[:, 1:]).square()
        atom_log_density = torch.zeros((len(y), len(q)), dtype=torch.float64)
        for phase in range(3):
            times = clock[:, :, phase]
            covariance = torch.minimum(times[:, :, None], times[:, None, :])
            law = torch.distributions.MultivariateNormal(q[:, 1:, phase], covariance_matrix=covariance)
            atom_log_density = atom_log_density + law.log_prob(y[:, None, 1:, phase])
        prior = torch.tensor(fold.prior_probabilities, dtype=torch.float64).log()
        total = total - torch.logsumexp(prior + atom_log_density, dim=1).sum()
    total = total / sum(len(fold.query_indices) for fold in folds)
    total.backward()
    return total.item(), parameter.grad.numpy()


def test_native_joint_marginal_matches_dense_gaussian_and_per_well_denominator(mixture_folds):
    actual = noise.marginal_nll_and_gradient(THETA, mixture_folds, DAYS)
    native, normalized, independent = dense_oracle(THETA, mixture_folds)
    np.testing.assert_allclose(actual.native_log_evidence, native, rtol=2e-14, atol=2e-11)
    np.testing.assert_allclose(actual.normalized_log_evidence, normalized, rtol=2e-13, atol=2e-11)
    assert actual.native_nll == pytest.approx(-native.mean(), rel=2e-14, abs=2e-11)
    assert actual.normalized_nll == pytest.approx(-normalized.mean(), rel=2e-13, abs=2e-11)
    # One complete joint atom must explain all phases; each phase cannot choose its own donor.
    assert abs(actual.native_nll + independent.mean()) > .01
    assert not np.isclose(actual.native_nll, -native.sum() / (21 * len(native)))
    assert actual.gradient.shape == (6,) and actual.gradient.dtype == np.float64


def test_analytic_log_sd_gradient_matches_finite_differences_and_dense_autograd(mixture_folds):
    actual = noise.marginal_nll_and_gradient(THETA, mixture_folds, DAYS)
    finite_difference = np.empty(6)
    for index in range(6):
        step = np.eye(6)[index] * 1e-5
        upper = -dense_oracle(THETA + step, mixture_folds)[0].mean()
        lower = -dense_oracle(THETA - step, mixture_folds)[0].mean()
        finite_difference[index] = (upper - lower) / 2e-5
    value, autograd = dense_autograd(THETA, mixture_folds)
    assert actual.native_nll == pytest.approx(value, abs=2e-11)
    np.testing.assert_allclose(actual.gradient, finite_difference, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(actual.gradient, autograd, rtol=5e-12, atol=5e-11)


def test_native_phase_units_have_only_the_exact_endpoint_jacobian(mixture_folds):
    conversion = np.array([8., 1024., .125])
    changed = [noise.NoiseFoldData(f.fold, f.query_indices, f.reports_native * conversion,
               f.cumulative_native * conversion, f.prior_probabilities,
               f.phase_scales_native * conversion) for f in mixture_folds]
    first = noise.marginal_nll_and_gradient(THETA, mixture_folds, DAYS)
    second = noise.marginal_nll_and_gradient(THETA, changed, DAYS)
    jacobian = 7 * np.log(conversion).sum()
    assert second.native_nll == pytest.approx(first.native_nll + jacobian, abs=3e-11)
    assert second.normalized_nll == pytest.approx(first.normalized_nll, abs=3e-11)
    np.testing.assert_allclose(second.native_log_evidence, first.native_log_evidence - jacobian, atol=3e-11)
    np.testing.assert_allclose(second.gradient, first.gradient, rtol=2e-12, atol=3e-11)


@pytest.mark.parametrize("batch_size", [1, 2, 99])
def test_objective_is_invariant_to_batching_and_fold_order(mixture_folds, batch_size):
    expected = noise.marginal_nll_and_gradient(THETA, mixture_folds, DAYS, batch_size=99)
    actual = noise.marginal_nll_and_gradient(THETA, mixture_folds[::-1], DAYS, batch_size=batch_size)
    assert actual.native_nll == pytest.approx(expected.native_nll, abs=2e-12)
    np.testing.assert_allclose(actual.gradient, expected.gradient, rtol=2e-13, atol=2e-12)
    np.testing.assert_array_equal(actual.native_log_evidence, expected.native_log_evidence)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; CPU dense oracle is mandatory")
def test_cuda_float64_matches_cpu_value_gradient_and_all_evidence(mixture_folds):
    cpu = noise.marginal_nll_and_gradient(THETA, mixture_folds, DAYS, device="cpu", batch_size=2)
    gpu = noise.marginal_nll_and_gradient(THETA, mixture_folds, DAYS, device="cuda", batch_size=2)
    assert gpu.native_nll == pytest.approx(cpu.native_nll, rel=1e-11, abs=1e-10)
    np.testing.assert_allclose(gpu.gradient, cpu.gradient, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(gpu.native_log_evidence, cpu.native_log_evidence, rtol=1e-11, atol=1e-10)


def test_projected_gradient_respects_kkt_signs_at_both_box_bounds():
    lower, upper = np.log(1e-4), np.log(2.)
    parameter = np.array([lower, lower, upper, upper, 0., 0.])
    gradient = np.array([3., -4., -5., 6., -7., 8.])
    np.testing.assert_array_equal(noise.projected_gradient(parameter, gradient), [0., -4., 0., 6., -7., 8.])


@pytest.fixture
def population():
    count = 90
    records = [{"row_id": f"row-{i}", "group_id": f"group-{i // 2}", "split": "train",
                "covariates_raw": {name: str(1 + .1 * i + .03 * j)
                                   for j, name in enumerate(STATIC_FEATURE_NAMES)},
                "reports": "must not enter query metadata", "future_secret": object()}
               for i in range(count)]
    rng = np.random.default_rng(922)
    coefficients = rng.uniform(10., 90., (count, 3, 4))
    reports = np.einsum("npk,tk->ntp", coefficients, fast_basis_cumulative_matrix(DAYS))
    return records, reports


@pytest.fixture
def fake_banks():
    calls = []

    def factory(records, reports, days, *, seed, n_jobs=2):
        processor = StaticFeaturePreprocessor.fit(records)
        endpoint, phase, _ = fit_projection_scales(reports)
        coefficients, cumulative = project_positive_coefficients(reports, days, endpoint, phase)
        receipt = {"records": deepcopy(records), "reports": reports.copy(), "seed": seed,
                   "n_jobs": n_jobs, "processor": processor, "queries": []}
        calls.append(receipt)

        def exact_weights(queries, *, return_receipt=False):
            assert all(set(record) <= {"row_id", "group_id", "split", "covariates_raw"} for record in queries)
            assert {row["group_id"] for row in records}.isdisjoint(row["group_id"] for row in queries)
            processor.transform(queries)
            receipt["queries"].extend(deepcopy(queries))
            weights = np.arange(1., len(records) + 1)
            weights /= weights.sum()
            raw = np.broadcast_to(weights, (len(queries), len(records))).copy()
            final = .95 * raw + .05 / len(records)
            return (final, {"forest_weights": raw, "synthetic_donor_indices": np.arange(len(records))}) if return_receipt else final

        return SimpleNamespace(processor=processor, coefficients_native=coefficients,
            cumulative_native=cumulative, phase_scales_native=phase, endpoint_scales_native=endpoint,
            exact_weights=exact_weights, metadata=lambda: {"preprocessing": processor.to_dict()})

    return factory, calls


def quadratic_objective(monkeypatch, *, target=None, clock=None):
    """Isolate optimizer stop/acceptance behavior from the independently tested law."""
    optimum = np.log([.07, .08, .09, .12, .13, .14]) if target is None else np.asarray(target)
    calls = []

    def evaluate(log_sds, folds, report_days, **kwargs):
        theta = np.asarray(log_sds)
        calls.append(theta.copy())
        residual = theta - optimum
        native = 100. + .5 * float(residual @ residual)
        count = sum(len(fold.query_indices) for fold in folds)
        evidence = -native + np.linspace(-1., 1., count)
        if clock is not None:
            clock[0] += 10.
        return noise.NoiseObjective(native_nll=native, normalized_nll=native - 25.,
            gradient=residual, native_log_evidence=evidence,
            normalized_log_evidence=evidence + 25.)

    monkeypatch.setattr(noise, "marginal_nll_and_gradient", evaluate)
    return optimum, calls


def optimizer_result(theta, value, gradient, *, success=True, message="synthetic solver result"):
    return SimpleNamespace(x=np.asarray(theta), fun=float(value), jac=np.asarray(gradient),
        success=success, message=message, status=0 if success else 2, nit=1, nfev=2, njev=2)


def test_five_fold_factories_and_preprocessing_exclude_every_held_group(population, fake_banks, monkeypatch):
    records, reports = population
    factory, calls = fake_banks
    target, _ = quadratic_objective(monkeypatch)

    def minimize(function, x0, **kwargs):
        np.testing.assert_array_equal(x0, np.log([.05] * 3 + [.1] * 3))
        assert kwargs["method"] == "L-BFGS-B" and kwargs["jac"] is True
        np.testing.assert_array_equal(kwargs["bounds"], [[np.log(1e-4), np.log(2.)]] * 6)
        assert kwargs["options"] == {"gtol": 1e-5, "ftol": 1e-12, "maxiter": 60, "maxfun": 60}
        function(x0)
        value, gradient = function(target)
        return optimizer_result(target, value, gradient)

    monkeypatch.setattr(noise, "minimize", minimize)
    result = noise.fit_oof_forest_noise(records, reports, DAYS, seed=42, bank_factory=factory)
    folds = assign_group_folds(records)
    assert len(calls) == 5 and len(result.fold_data) == 5
    assert result.converged
    np.testing.assert_allclose(result.noise_abs, np.exp(target[:3]), rtol=1e-15)
    np.testing.assert_allclose(result.noise_rel, np.exp(target[3:]), rtol=1e-15)
    for fold, receipt in zip(result.fold_data, calls, strict=True):
        fit_indices = np.flatnonzero(folds != fold.fold)
        query_indices = np.flatnonzero(folds == fold.fold)
        allowed = [records[index] for index in fit_indices]
        assert [row["row_id"] for row in receipt["records"]] == [row["row_id"] for row in allowed]
        assert [row["row_id"] for row in receipt["queries"]] == [records[index]["row_id"] for index in query_indices]
        assert receipt["processor"].to_dict() == StaticFeaturePreprocessor.fit(allowed).to_dict()
        assert receipt["seed"] == 42 and receipt["n_jobs"] == 2
        np.testing.assert_array_equal(receipt["reports"], reports[fit_indices])
        np.testing.assert_array_equal(fold.query_indices, query_indices)
        np.testing.assert_array_equal(fold.reports_native, reports[query_indices])
        np.testing.assert_array_equal(fold.phase_scales_native, fit_projection_scales(reports[fit_indices])[1])
        assert fold.cumulative_native.shape == (len(fit_indices), 8, 3)
        assert fold.prior_probabilities.shape == (len(query_indices), len(fit_indices))
        assert np.all(fold.prior_probabilities >= .05 / len(fit_indices))
        np.testing.assert_allclose(fold.prior_probabilities.sum(axis=1), 1., atol=1e-15)
        raw = np.arange(1., len(fit_indices) + 1)
        raw /= raw.sum()
        expected = np.broadcast_to(.95 * raw + .05 / len(fit_indices), fold.prior_probabilities.shape)
        np.testing.assert_array_equal(fold.prior_probabilities, expected)
        np.testing.assert_array_equal(fold.forest_probabilities, np.broadcast_to(raw, expected.shape))
    assert result.final_native_log_evidence.shape == (len(records),)
    np.testing.assert_array_equal(result.final_native_log_evidence, -100. + np.linspace(-1., 1., len(records)))
    diagnostics = result.diagnostics
    assert diagnostics["status"] == "converged" and diagnostics["validation_authorized"]
    assert diagnostics["dtype"] == "float64" and diagnostics["device"] == "cpu"
    assert diagnostics["phase_order"] == ["oil", "gas", "water"]
    assert diagnostics["units"] == ["BBL", "MCF", "BBL"]
    assert diagnostics["completed_objective_evaluations"] == 2
    assert diagnostics["attempted_objective_evaluations"] == 2
    assert diagnostics["maximum_objective_evaluations"] == 60
    assert diagnostics["maximum_elapsed_seconds"] == 1800.
    assert diagnostics["required_projected_gradient_infinity"] == 1e-4
    assert diagnostics["training_row_ids"] == [record["row_id"] for record in records]
    assert diagnostics["training_group_ids"] == [record["group_id"] for record in records]
    np.testing.assert_array_equal(diagnostics["fold_assignments"], folds)
    assert diagnostics["fold_seed"] == 271828 and diagnostics["fold_count"] == 5
    for index, call in enumerate(diagnostics["objective_calls"], start=1):
        assert call["evaluation"] == index and call["status"] == "completed"
        assert {"log_standard_deviations", "standard_deviations", "native_nll", "normalized_nll",
                "gradient", "projected_gradient", "projected_gradient_infinity", "elapsed_seconds"} <= call.keys()
        assert len(call["gradient"]) == 6 and len(call["standard_deviations"]) == 6
    assert diagnostics["final"]["native_nll"] == 100.
    assert diagnostics["final"]["normalized_nll"] == 75.
    assert diagnostics["final"]["projected_gradient_infinity"] == 0.
    assert diagnostics["optimizer"]["success"] and diagnostics["failure"] is None
    assert diagnostics["initial"]["native_nll"] == diagnostics["objective_calls"][0]["native_nll"]
    for fold, metadata in zip(result.fold_data, diagnostics["folds"], strict=True):
        assert metadata == fold.metadata
        assert set(metadata["fit_group_ids"]).isdisjoint(metadata["query_group_ids"])
        assert metadata["whole_group_exclusion_verified"]
        assert len(metadata["fit_row_sha256"]) == len(metadata["query_row_sha256"]) == 64


@pytest.mark.parametrize("fault", ["validation", "calibration", "test", "duplicate_id", "missing_group",
                                   "missing_endpoint", "infinite_endpoint", "nonzero_origin", "wrong_days"])
def test_invalid_training_population_is_rejected_before_any_factory(population, fake_banks, fault):
    records, reports = deepcopy(population)
    days = DAYS.copy()
    if fault in {"validation", "calibration", "test"}:
        records[0]["split"] = fault
    elif fault == "duplicate_id":
        records[0]["row_id"] = records[1]["row_id"]
    elif fault == "missing_group":
        records[0].pop("group_id")
    elif fault == "missing_endpoint":
        reports[0, 3, 1] = np.nan
    elif fault == "infinite_endpoint":
        reports[0, 3, 1] = np.inf
    elif fault == "nonzero_origin":
        reports[0, 0, 1] = 1.
    else:
        days[1] = 31.
    with pytest.raises(ValueError):
        noise.fit_oof_forest_noise(records, reports, days, seed=42, bank_factory=fake_banks[0])
    assert not fake_banks[1]


@pytest.mark.parametrize("override", [{"max_evaluations": 61}, {"max_elapsed_seconds": 1801.}])
def test_frozen_optimizer_resource_limits_cannot_be_increased(population, fake_banks, override):
    with pytest.raises(ValueError):
        noise.fit_oof_forest_noise(*population, DAYS, seed=42, bank_factory=fake_banks[0], **override)
    assert not fake_banks[1]


@pytest.mark.parametrize("optimizer_success,at_optimum,expected_status", [
    (False, True, "unresolved_optimizer"),
    (True, False, "unresolved_kkt"),
])
def test_optimizer_success_and_projected_gradient_are_both_required(
        population, fake_banks, monkeypatch, optimizer_success, at_optimum, expected_status):
    target, _ = quadratic_objective(monkeypatch)

    def minimize(function, x0, **kwargs):
        candidate = target if at_optimum else x0
        value, gradient = function(candidate)
        return optimizer_result(candidate, value, gradient, success=optimizer_success)

    monkeypatch.setattr(noise, "minimize", minimize)
    result = noise.fit_oof_forest_noise(*population, DAYS, seed=42, bank_factory=fake_banks[0])
    assert result.status == expected_status and not result.converged
    assert not result.diagnostics["validation_authorized"]
    assert result.diagnostics["optimizer"]["success"] == optimizer_success
    assert result.diagnostics["completed_objective_evaluations"] == 1
    assert np.isfinite(result.final_native_log_evidence).all()


def test_bound_optimum_accepts_nonzero_gradient_when_projected_gradient_is_zero(
        population, fake_banks, monkeypatch):
    lower, upper = np.log(1e-4), np.log(2.)
    candidate = np.log([.07, .08, .09, .12, .13, .14])
    candidate[:2] = [lower, upper]
    target = candidate + np.array([-1., 1., 0., 0., 0., 0.])
    quadratic_objective(monkeypatch, target=target)

    def minimize(function, x0, **kwargs):
        value, gradient = function(candidate)
        return optimizer_result(candidate, value, gradient)

    monkeypatch.setattr(noise, "minimize", minimize)
    result = noise.fit_oof_forest_noise(*population, DAYS, seed=42, bank_factory=fake_banks[0])
    assert result.converged and result.status == "converged"
    np.testing.assert_array_equal(result.diagnostics["final"]["gradient"], [1., -1., 0., 0., 0., 0.])
    assert result.diagnostics["final"]["projected_gradient_infinity"] == 0.
    assert result.diagnostics["final"]["lower_bound_indices"] == [0]
    assert result.diagnostics["final"]["upper_bound_indices"] == [1]
    assert result.noise_abs[0] == pytest.approx(1e-4)
    assert result.noise_abs[1] == pytest.approx(2.)


def test_sixty_objective_calls_are_a_hard_ceiling_and_partial_fit_is_retained(
        population, fake_banks, monkeypatch):
    _, calls = quadratic_objective(monkeypatch)

    def minimize(function, x0, **kwargs):
        for index in range(61):
            function(x0 + np.eye(6)[0] * index * 1e-4)
        pytest.fail("The 61st objective evaluation must never execute")

    monkeypatch.setattr(noise, "minimize", minimize)
    result = noise.fit_oof_forest_noise(*population, DAYS, seed=42, bank_factory=fake_banks[0])
    assert len(calls) == 60
    assert result.status == "unresolved_evaluation_limit" and not result.converged
    assert result.diagnostics["completed_objective_evaluations"] == 60
    assert result.diagnostics["attempted_objective_evaluations"] == 60
    assert len(result.diagnostics["objective_calls"]) == 60
    assert not result.diagnostics["validation_authorized"]
    assert "evaluation_limit" in result.diagnostics["failure"]["message"]
    np.testing.assert_allclose(np.r_[result.noise_abs, result.noise_rel], np.exp(calls[-1]), rtol=1e-15)
    assert len(result.fold_data) == 5 and np.isfinite(result.final_native_log_evidence).all()


def test_deadline_crossed_during_objective_cannot_authorize_a_successful_optimizer(
        population, fake_banks, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(noise.time, "monotonic", lambda: clock[0])
    target, calls = quadratic_objective(monkeypatch, clock=clock)

    def minimize(function, x0, **kwargs):
        function(x0)
        value, gradient = function(target)
        return optimizer_result(target, value, gradient)

    monkeypatch.setattr(noise, "minimize", minimize)
    result = noise.fit_oof_forest_noise(*population, DAYS, seed=42,
        bank_factory=fake_banks[0], max_elapsed_seconds=15.)
    assert len(calls) == 2
    assert result.status == "unresolved_deadline" and not result.converged
    assert not result.diagnostics["validation_authorized"]
    assert result.diagnostics["elapsed_seconds"] == 20.
    assert result.diagnostics["failure"]["stage"]
    assert len(result.fold_data) == 5


def test_objective_failure_retains_prior_completed_call_without_acceptance(
        population, fake_banks, monkeypatch):
    _, calls = quadratic_objective(monkeypatch)
    evaluate = noise.marginal_nll_and_gradient

    def fail_second(*args, **kwargs):
        if calls:
            raise FloatingPointError("synthetic overflow in one donor")
        return evaluate(*args, **kwargs)

    def minimize(function, x0, **kwargs):
        function(x0)
        function(x0 + .01)
        pytest.fail("Nonfinite objective must stop the optimizer")

    monkeypatch.setattr(noise, "marginal_nll_and_gradient", fail_second)
    monkeypatch.setattr(noise, "minimize", minimize)
    result = noise.fit_oof_forest_noise(*population, DAYS, seed=42, bank_factory=fake_banks[0])
    assert result.status == "unresolved_numerical_or_bank" and not result.converged
    assert result.diagnostics["completed_objective_evaluations"] == 1
    assert result.diagnostics["attempted_objective_evaluations"] == 2
    assert result.diagnostics["objective_calls"][1]["status"] != "completed"
    assert "synthetic overflow" in result.diagnostics["failure"]["message"]
    assert np.isfinite(result.final_native_log_evidence).all()
    np.testing.assert_array_equal(np.r_[result.noise_abs, result.noise_rel], np.exp(calls[0]))


@pytest.mark.parametrize("fault", ["wrong_forest_receipt", "background_applied_twice"])
def test_weight_receipt_requires_the_single_fixed_background_mixture(population, fake_banks, monkeypatch, fault):
    valid_factory, _ = fake_banks

    def factory(*args, **kwargs):
        bank = valid_factory(*args, **kwargs)
        exact = bank.exact_weights

        def corrupted(queries, *, return_receipt=False):
            prior, receipt = exact(queries, return_receipt=True)
            if fault == "wrong_forest_receipt":
                receipt["forest_weights"] = receipt["forest_weights"][:, ::-1].copy()
            else:
                prior = .95 * prior + .05 / prior.shape[1]
            return prior, receipt

        bank.exact_weights = corrupted
        return bank

    monkeypatch.setattr(noise, "minimize", lambda *args, **kwargs: pytest.fail("Invalid donor law reached optimizer"))
    result = noise.fit_oof_forest_noise(*population, DAYS, seed=42, bank_factory=factory)
    assert result.status == "unresolved_numerical_or_bank" and not result.converged
    assert "probabilities disagree" in result.diagnostics["failure"]["message"]
    assert result.diagnostics["attempted_objective_evaluations"] == 0
    assert not result.fold_data


def test_saved_fold_arrays_replay_likelihood_without_forest_and_never_overwrite(
        population, fake_banks, monkeypatch, tmp_path):
    def minimize(function, x0, **kwargs):
        value, gradient = function(x0)
        return optimizer_result(x0, value, gradient, success=False)

    monkeypatch.setattr(noise, "minimize", minimize)
    result = noise.fit_oof_forest_noise(*population, DAYS, seed=42, bank_factory=fake_banks[0])
    assert result.status == "unresolved_optimizer"
    files = result.save_artifacts(tmp_path, "synthetic")
    assert len(files) == 6
    replay_folds = []
    for fold in result.fold_data:
        filename = f"synthetic_noise_fold{fold.fold}.npz"
        assert files[filename] == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
        with np.load(tmp_path / filename, allow_pickle=False) as saved:
            assert {"reports_native", "cumulative_native", "prior_probabilities", "forest_probabilities",
                    "coefficients_native", "phase_scales_native", "query_indices", "report_days",
                    "query_row_ids", "query_group_ids", "donor_row_ids", "donor_group_ids",
                    "final_native_log_evidence", "noise_abs_normalized", "noise_rel", "phase_order",
                    "units", "synthetic_donor_indices"} <= set(saved.files)
            assert all(not saved[key].dtype.hasobject for key in saved.files)
            np.testing.assert_array_equal(saved["report_days"], DAYS)
            np.testing.assert_array_equal(saved["phase_order"], ["oil", "gas", "water"])
            np.testing.assert_array_equal(saved["units"], ["BBL", "MCF", "BBL"])
            np.testing.assert_array_equal(saved["noise_abs_normalized"], result.noise_abs)
            np.testing.assert_array_equal(saved["noise_rel"], result.noise_rel)
            np.testing.assert_array_equal(saved["query_row_ids"], fold.metadata["query_row_ids"])
            np.testing.assert_array_equal(saved["donor_row_ids"], fold.metadata["fit_row_ids"])
            np.testing.assert_array_equal(saved["query_group_ids"], fold.metadata["query_group_ids"])
            np.testing.assert_array_equal(saved["donor_group_ids"], fold.metadata["fit_group_ids"])
            np.testing.assert_array_equal(saved["final_native_log_evidence"],
                                          result.final_native_log_evidence[fold.query_indices])
            for name in ("reports_native", "cumulative_native", "prior_probabilities", "phase_scales_native",
                         "coefficients_native", "forest_probabilities"):
                np.testing.assert_array_equal(saved[name], getattr(fold, name))
            replay_folds.append(noise.NoiseFoldData(fold.fold, saved["query_indices"], saved["reports_native"],
                saved["cumulative_native"], saved["prior_probabilities"], saved["phase_scales_native"]))
    replay = noise.marginal_nll_and_gradient(np.log(np.r_[result.noise_abs, result.noise_rel]), replay_folds, DAYS)
    np.testing.assert_array_equal(replay.native_log_evidence, result.final_native_log_evidence)
    np.testing.assert_array_equal(replay.gradient, result.diagnostics["final"]["gradient"])
    assert replay.native_nll == result.diagnostics["final"]["native_nll"]
    assert json.loads((tmp_path / "synthetic_noise_fit.json").read_text()) == result.diagnostics
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        result.save_artifacts(tmp_path, "synthetic")
    assert original == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
