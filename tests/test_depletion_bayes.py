import math

import numpy as np
import pytest
import torch

from joint_production.depletion_bayes import (
    DECAY_TIMES_DAYS, GAS, LATENT_DIM, PEAK, PEAK_MAX_DAYS, WATER, DepletionBayesForecaster,
    DepletionBayesModel, DepletionRate, bernstein, gaussian_log_prob_precision, gaussian_log_prob_tril,
    interval_ou_covariance,
)
from joint_production.depletion_bayes import NOISE, RATE_DIM, sample_bridge_reports
from observation_reference import gaussian_endpoint_log_likelihood
from joint_production.torch_observation import sample_accumulated_reports

DAYS = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=float)
LOG_2PI = math.log(2 * math.pi)


def accumulated_log_likelihood(reports, cumulative, noise_abs, noise_rel, mask, days):
    """Reference masked accumulated-Gaussian log likelihood [N, S] (successive observed endpoints)."""
    n, s, t, _ = cumulative.shape
    if noise_abs.ndim == 2:
        a2, r = noise_abs.square()[:, None, :], noise_rel[:, None, :]
    else:
        a2, r = noise_abs.square(), noise_rel
    safe_reports = torch.where(mask, reports, torch.zeros_like(reports))
    prev_q = cumulative.new_zeros((n, s, 3))
    prev_err = cumulative.new_zeros((n, s, 3))
    prev_t = cumulative.new_zeros((n, 1, 3))
    total = cumulative.new_zeros((n, s, 3))
    for j in range(t):
        if float(days[j]) <= 0:
            continue
        q = cumulative[:, :, j]
        err = safe_reports[:, None, j] - q
        include = mask[:, None, j]
        var = a2 * ((days[j] - prev_t) / 360) + (r * (q - prev_q)) * (r * q + r * prev_q)
        safe_var = torch.where(include, var.clamp_min(1e-30), torch.ones_like(var))
        innovation = torch.where(include, err - prev_err, torch.zeros_like(err))
        log_p = -0.5 * (LOG_2PI + safe_var.log() + innovation.square() / safe_var)
        total = total + torch.where(include, log_p, torch.zeros_like(log_p))
        prev_q = torch.where(include, q, prev_q)
        prev_err = torch.where(include, err, prev_err)
        prev_t = torch.where(include, days[j].expand_as(prev_t), prev_t)
    return total.sum(-1)


def ou_covariance(cumulative_fn, z, days, a, r, sigma, tau):
    """Interval-OU persistent term plus accumulated Brownian reporting noise [..., 3, T', T']."""
    positive = days > 0
    q = cumulative_fn(z, days)
    v = a[..., None, :].square() * (days[:, None] / 360) + (r[..., None, :] * q).square()
    v = v[..., positive, :].transpose(-1, -2)
    index = torch.arange(int(positive.sum()))
    return interval_ou_covariance(q, days, sigma, tau) + v[..., torch.minimum(index[:, None], index[None, :])]


def random_latents(count=256, seed=0, spread=1.5):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn((count, LATENT_DIM), generator=generator, dtype=torch.float64) * spread
    z[:, :8] += math.log(0.15)
    return z


def test_oil_rate_is_ramp_then_completely_monotone_decline():
    rate = DepletionRate()
    z = random_latents()
    p = rate.unpack(z)
    t = torch.linspace(0, 4000, 8001, dtype=torch.float64)
    q = rate.oil_rate(p, t)
    assert torch.all(q > 0)
    peak = p["peak"][:, None]
    assert torch.all(peak <= PEAK_MAX_DAYS)
    dq = torch.diff(q, dim=-1)
    before = (t[1:] <= peak)
    after = (t[:-1] >= peak)
    assert torch.all(dq[before] >= -1e-12 * q[:, 1:][before])
    assert torch.all(dq[after] <= 1e-12 * q[:, :-1][after])
    # Complete monotonicity implies log-convexity: nondecreasing log-slope after the peak.
    log_slope = torch.diff(q.log(), dim=-1)
    curvature = torch.diff(log_slope, dim=-1)
    after2 = t[:-2] >= peak
    assert torch.all(curvature[after2] >= -1e-9)


def test_cumulatives_integrate_rates_and_reach_finite_totals():
    rate = DepletionRate()
    z = random_latents(64, seed=1)
    t = torch.linspace(0, 2000, 40001, dtype=torch.float64)
    q = rate.rate(z, t)
    integral = torch.cumulative_trapezoid(q, t, dim=-2)
    closed = rate.cumulative(z, t)[:, 1:]
    scale = closed[:, -1:].abs() + 1e-12
    assert torch.allclose(integral / scale, closed / scale, atol=2e-4)
    late = rate.cumulative(z, torch.tensor([1e7], dtype=torch.float64))[:, 0]
    assert torch.allclose(late, rate.total_volume(z), rtol=1e-9)
    assert torch.all(torch.isfinite(rate.total_volume(z)))


def test_cumulative_is_zero_at_origin_and_nondecreasing_with_positive_rates():
    rate = DepletionRate()
    z = random_latents(512, seed=2, spread=2.5)
    t = torch.linspace(0, 1e4, 4001, dtype=torch.float64)
    cumulative = rate.cumulative(z, t)
    assert torch.all(cumulative[:, 0] == 0)
    assert torch.all(torch.diff(cumulative, dim=1) >= -1e-12 * (1 + cumulative[:, 1:].abs()))
    assert torch.all(rate.rate(z, t) >= 0)


def ratios(law, z, t):
    """Normalized instantaneous gas/oil and water/oil ratios [..., T, 2] (WOR includes flowback)."""
    rates = law.rate(z, t)
    return torch.stack((rates[..., 1] / rates[..., 0], rates[..., 2] / rates[..., 0]), dim=-1)


def test_phase_ratios_are_bounded_by_bernstein_coefficients_and_define_rates():
    rate = DepletionRate()
    z = random_latents(128, seed=3)
    t = torch.linspace(0, 3000, 301, dtype=torch.float64)
    phase_ratios = ratios(rate, z, t)
    rates = rate.rate(z, t)
    coefficients = z[:, GAS].exp()
    low, high = coefficients.min(-1).values[:, None], coefficients.max(-1).values[:, None]
    assert torch.all(phase_ratios[..., 0] >= low * (1 - 1e-12))
    assert torch.all(phase_ratios[..., 0] <= high * (1 + 1e-12))
    water_floor = z[:, WATER].exp().min(-1).values[:, None]
    assert torch.all(phase_ratios[..., 1] >= water_floor * (1 - 1e-12))  # flowback only adds water
    for column in (0, 1):
        assert torch.allclose(rates[..., column + 1], phase_ratios[..., column] * rates[..., 0])


def test_ratio_reparameterization_permits_rising_gas_while_oil_declines():
    rate = DepletionRate()
    z = random_latents(1, seed=4, spread=0.0)
    z[:, :8] = math.log(0.01)
    z[:, 5:8] = math.log(0.5)  # slow-mode oil: early depletion over the window
    z[:, PEAK] = -10.0
    z[0, GAS] = torch.tensor([-3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    t = torch.linspace(200, 1080, 50, dtype=torch.float64)
    rates = rate.rate(z, t)[0]
    assert torch.all(torch.diff(rates[:, 0]) < 0)
    assert torch.any(torch.diff(rates[:, 1]) > 0)


def test_bernstein_gradient_is_finite_at_depletion_origin():
    f = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    bernstein(f, 4).sum().backward()
    assert torch.all(torch.isfinite(f.grad))


def test_likelihood_matches_reference_accumulated_law_under_masks():
    rate = DepletionRate()
    z = random_latents(6, seed=5)
    cumulative = rate.cumulative(z, torch.as_tensor(DAYS))
    generator = torch.Generator().manual_seed(9)
    reports = cumulative * (1 + 0.1 * torch.randn(cumulative.shape, generator=generator, dtype=torch.float64))
    reports[:, 0] = 0
    mask = torch.rand(reports.shape, generator=generator) > 0.3
    mask[:, 0] = True
    a = torch.tensor([0.03, 0.05, 0.04], dtype=torch.float64)
    r = torch.tensor([0.08, 0.1, 0.12], dtype=torch.float64)
    reference = gaussian_endpoint_log_likelihood(reports, cumulative, a, r, mask, torch.as_tensor(DAYS))
    ours = accumulated_log_likelihood(reports, cumulative[:, None], a, r, mask & (torch.as_tensor(DAYS) > 0)[None, :, None],
                                      torch.as_tensor(DAYS))[:, 0]
    assert torch.allclose(ours, reference, rtol=1e-10, atol=1e-8)


def _model(covariates=3):
    torch.manual_seed(0)
    return DepletionBayesModel(covariates, DAYS, [1e5, 1.5e5, 4e4])


def _pseudo_likelihood(model, context):
    out = model.encoder(context).double()
    d = LATENT_DIM
    diag = torch.nn.functional.softplus(out[:, d:2 * d])
    factor = out[:, 2 * d:].reshape(-1, d, model.rank)
    return out[:, :d], torch.diag_embed(diag) + factor @ factor.transpose(-1, -2)


def test_empty_history_posterior_equals_prior_exactly_and_data_only_adds_precision():
    model = _model()
    x = torch.randn(5, 3)
    context = torch.randn(5, model.context_dim)
    with torch.no_grad():
        model.encoder[-1].weight.normal_(0, 0.5)
    has_data = torch.tensor([False, True, False, True, True])
    post = model.posterior(x, context, has_data)
    precision_p = torch.cholesky_inverse(post["tril_p"])
    precision_q = post["tril_q"] @ post["tril_q"].transpose(-1, -2)
    empty = ~has_data
    assert torch.allclose(post["mean_q"][empty], post["mean_p"][empty], atol=1e-9)
    assert torch.allclose(precision_q[empty], precision_p[empty], atol=1e-7, rtol=1e-7)
    assert torch.allclose(post["log_w"][empty], post["log_pi"][empty], atol=1e-9)
    gain = torch.linalg.eigvalsh(precision_q - precision_p)
    assert torch.all(gain >= -1e-7)


def test_posterior_is_prior_times_pseudo_likelihood():
    model = _model()
    x = torch.randn(3, 3)
    context = torch.randn(3, model.context_dim)
    with torch.no_grad():
        model.encoder[-1].weight.normal_(0, 0.3)
    post = model.posterior(x, context, torch.ones(3, dtype=torch.bool))
    h, lam = _pseudo_likelihood(model, context)
    z = post["mean_p"][:, :1] + torch.randn(3, 7, LATENT_DIM, dtype=torch.float64)
    log_l = (z * h[:, None]).sum(-1) - 0.5 * torch.einsum("nsd,nde,nse->ns", z, lam, z)
    ratio = model.log_q_mix(z, post) - model.log_prior_mix(z, post) - log_l
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-6)


def test_gaussian_helpers_agree_with_torch_distributions():
    model = _model()
    x = torch.randn(4, 3)
    context = torch.randn(4, model.context_dim)
    with torch.no_grad():
        model.encoder[-1].weight.normal_(0, 0.3)
    post = model.posterior(x, context, torch.ones(4, dtype=torch.bool))
    mean_q, tril_q = post["mean_q"][:, 0], post["tril_q"][:, 0]
    mean_p, tril_p = post["mean_p"][:, 0], post["tril_p"][:, 0]
    cov_q = torch.cholesky_inverse(tril_q)
    q_dist = torch.distributions.MultivariateNormal(mean_q, covariance_matrix=cov_q)
    p_dist = torch.distributions.MultivariateNormal(mean_p, scale_tril=tril_p)
    z = model.sample_posterior(mean_q, tril_q, 5)
    assert torch.allclose(gaussian_log_prob_precision(z, mean_q, tril_q), q_dist.log_prob(z.transpose(0, 1)).T, atol=1e-6)
    assert torch.allclose(gaussian_log_prob_tril(z, mean_p, tril_p), p_dist.log_prob(z.transpose(0, 1)).T, atol=1e-6)
    assert torch.allclose(model.log_prior_mix(z, post), p_dist.log_prob(z.transpose(0, 1)).T, atol=1e-6)


def _synthetic(n=96, seed=0):
    rng = np.random.default_rng(seed)
    rate = DepletionRate()
    z = torch.as_tensor(rng.normal(0, 0.4, (n, LATENT_DIM)))
    z[:, :8] += math.log(0.15)
    x = rng.normal(size=(n, 3))
    z[:, 0] += 0.5 * torch.as_tensor(x[:, 0])
    scales = np.array([1e5, 1.5e5, 4e4])
    cumulative = rate.cumulative(z, torch.as_tensor(DAYS)).numpy() * scales
    reports = cumulative * (1 + 0.05 * rng.standard_normal(cumulative.shape))
    reports[:, 0] = 0
    reports[rng.random(reports.shape) < 0.05] = np.nan
    reports[:, 0] = 0
    return reports, x


def test_fit_and_prediction_reproduce_history_and_respect_cutoff():
    reports, x = _synthetic()
    forecaster = DepletionBayesForecaster(epochs=3, batch_size=32, proposals=256, log_every=0, seed=3)
    forecaster.fit(reports, DAYS, covariates=x)
    history = reports[:8].copy()
    history[:, DAYS > 90] = np.nan
    future_poison = reports[:8].copy()
    future_poison[:, DAYS > 90] = 1e12
    first = forecaster.predict_distribution(history, 90, 32, np.random.default_rng(1), x[:8])
    second = forecaster.predict_distribution(future_poison, 90, 32, np.random.default_rng(1), x[:8])
    assert np.array_equal(first["report_cumulative"], second["report_cumulative"])
    draws = first["report_cumulative"]
    assert draws.shape == (8, 32, len(DAYS), 3) and np.isfinite(draws).all()
    known = np.isfinite(history) & (DAYS <= 90)[None, :, None]
    repeated = np.broadcast_to(history[:, None], draws.shape)
    assert np.array_equal(draws[np.broadcast_to(known[:, None], draws.shape)],
                          repeated[np.broadcast_to(known[:, None], draws.shape)])
    assert np.all(np.diff(first["latent_cumulative"], axis=2) >= 0)


def test_origin_zero_prediction_samples_the_prior():
    reports, x = _synthetic(48, seed=2)
    forecaster = DepletionBayesForecaster(epochs=1, batch_size=16, proposals=128, log_every=0)
    forecaster.fit(reports, DAYS, covariates=x)
    empty = np.full_like(reports[:4], np.nan)
    empty[:, 0] = 0
    result = forecaster.predict_distribution(empty, 0, 64, np.random.default_rng(0), x[:4])
    assert np.allclose(result["ess"], 128)
    assert np.all(result["unique_draws"] == 64)


def test_bridge_sampler_matches_reference_for_constant_noise_and_supports_per_draw_noise():
    rate = DepletionRate()
    z = random_latents(5, seed=11)
    days = torch.as_tensor(DAYS)
    latent = rate.cumulative(z[:, None].expand(5, 7, LATENT_DIM), days) * 1000
    history = latent[:, 0] * 1.05
    history[1, 2] = float("nan")
    a = torch.tensor([30.0, 50.0, 40.0], dtype=torch.float64)
    r = torch.tensor([0.05, 0.08, 0.1], dtype=torch.float64)
    reference = sample_accumulated_reports(latent, history, days, a, r, cutoff=180.0,
                                           generator=torch.Generator().manual_seed(4))
    ours = sample_bridge_reports(latent, history, days, a.expand(5, 7, 3), r.expand(5, 7, 3), cutoff=180.0,
                                 generator=torch.Generator().manual_seed(4))
    assert torch.allclose(ours, reference, rtol=1e-12, atol=1e-9)
    scaled = sample_bridge_reports(latent, history, days, a * torch.rand(5, 7, 1, dtype=torch.float64) + 1,
                                   r.expand(5, 7, 3), cutoff=180.0, generator=torch.Generator().manual_seed(4))
    known = torch.isfinite(history) & (days <= 180)[None, :, None]
    assert torch.equal(scaled[known[:, None].expand_as(scaled)], history[:, None].expand_as(scaled)[known[:, None].expand_as(scaled)])


def test_well_noise_multiplier_scales_likelihood_noise():
    model = _model()
    z = random_latents(4, seed=12)[:, None]
    a0, r0 = model.noise()
    z0 = z.clone()
    z0[..., NOISE] = 0.0
    a, r = model.noise_for(z0)
    assert torch.allclose(a, a0.double().expand_as(a)) and torch.allclose(r, r0.double().expand_as(r))
    z1 = z0.clone()
    z1[..., NOISE] = math.log(2.0)
    a2, r2 = model.noise_for(z1)
    assert torch.allclose(a2, 2 * a) and torch.allclose(r2, 2 * r)
    assert RATE_DIM + 3 == LATENT_DIM


def test_ou_likelihood_reduces_to_accumulated_law_without_persistent_deviation():
    from joint_production.depletion_bayes import gaussian_discrepancy_log_likelihood
    rate = DepletionRate()
    z = random_latents(4, seed=21)[:, None].expand(4, 3, LATENT_DIM).clone()
    days = torch.as_tensor(DAYS)
    q = rate.cumulative(z, days)
    generator = torch.Generator().manual_seed(2)
    reports = q[:, 0] * (1 + 0.05 * torch.randn(q[:, 0].shape, generator=generator, dtype=torch.float64))
    reports[:, 0] = 0
    mask = (torch.rand(reports.shape, generator=generator) > 0.3) & (days > 0)[None, :, None]
    a = torch.full((4, 3, 3), 0.02, dtype=torch.float64)
    r = torch.full((4, 3, 3), 0.05, dtype=torch.float64)
    cov = ou_covariance(rate.cumulative, z, days, a, r, torch.full((4, 3, 3), 1e-9, dtype=torch.float64),
                        torch.full((3,), 120.0, dtype=torch.float64))
    ours = gaussian_discrepancy_log_likelihood(reports, q, cov, mask, days)
    reference = accumulated_log_likelihood(reports, q, a, r, mask, days)
    assert torch.allclose(ours, reference, rtol=1e-8, atol=1e-6)


def test_matheron_sampler_conditions_exactly():
    from joint_production.depletion_bayes import sample_gaussian_discrepancy_reports
    rate = DepletionRate()
    days = torch.as_tensor(DAYS)
    z = random_latents(1, seed=22)[:, None].expand(1, 20000, LATENT_DIM).clone()
    q = rate.cumulative(z, days)
    sigma = torch.full((1, 20000, 3), 0.2, dtype=torch.float64)
    a = torch.full((1, 20000, 3), 0.02, dtype=torch.float64)
    cov = ou_covariance(rate.cumulative, z, days, a, 0.5 * a, sigma, torch.full((3,), 200.0, dtype=torch.float64))
    history = q[:, 0].clone() * 1.1
    history[:, 2] = float("nan")  # hidden day-60 report
    draws = sample_gaussian_discrepancy_reports(q, history, days, cov, cutoff=180.0,
                                                generator=torch.Generator().manual_seed(5))
    known = torch.isfinite(history) & (days <= 180)[None, :, None]
    assert torch.equal(draws[0][known[0].expand(20000, -1, -1)], history[0][None].expand(20000, -1, -1)[known[0].expand(20000, -1, -1)])
    # Analytic conditional moments of the oil day-1080 report given observed oil endpoints.
    c = cov[0, 0, 0]
    pos = days[days > 0]
    obs = torch.tensor([d in (30.0, 90.0, 180.0) for d in pos.tolist()])
    tgt = int(torch.nonzero(pos == 1080.0))
    d_o = (history[0, 1:, 0] - q[0, 0, 1:, 0])[obs]
    gain = torch.linalg.solve(c[obs][:, obs], c[obs][:, tgt])
    mean = q[0, 0, -1, 0] + gain @ d_o
    var = c[tgt, tgt] - c[tgt, obs] @ gain
    sample = draws[0, :, -1, 0]
    assert abs(float(sample.mean() - mean)) < 4 * float(var.sqrt()) / math.sqrt(20000) + 1e-9
    assert abs(float(sample.var() / var) - 1) < 0.05


def test_gaussian_mixture_crps_matches_monte_carlo():
    from joint_production.depletion_bayes import gaussian_mixture_crps
    generator = torch.Generator().manual_seed(8)
    weights = torch.softmax(torch.randn(2, 5, generator=generator, dtype=torch.float64), dim=-1)
    mean = torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)
    var = torch.rand(2, 5, 3, generator=generator, dtype=torch.float64) + 0.1
    target = torch.randn(2, 3, generator=generator, dtype=torch.float64)
    exact = gaussian_mixture_crps(weights, mean, var, target)
    count = 400000
    labels = torch.multinomial(weights, count, replacement=True, generator=generator)
    rows = torch.arange(2)[:, None]
    x = mean[rows, labels] + var[rows, labels].sqrt() * torch.randn(2, count, 3, generator=generator, dtype=torch.float64)
    x2 = mean[rows, labels.roll(1, 1)] + var[rows, labels.roll(1, 1)].sqrt() * torch.randn(2, count, 3, generator=generator, dtype=torch.float64)
    mc = (x - target[:, None]).abs().mean(1) - 0.5 * (x - x2).abs().mean(1)
    assert torch.allclose(exact, mc, atol=0.01)


def test_brownian_covariance_shortcut_matches_general_form():
    model = _model()
    z = random_latents(3, seed=31)[:, None].expand(3, 4, LATENT_DIM).clone()
    a, r = model.noise_for(z)
    general = ou_covariance(model.rate.cumulative, z, torch.as_tensor(DAYS), a, r,
                            torch.zeros_like(a), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(model.covariance(z), general, rtol=1e-12, atol=1e-14)


def test_brownian_future_moments_match_general_conditioning():
    from joint_production.depletion_bayes import brownian_future_moments, conditional_report_moments
    model = _model()
    days = torch.as_tensor(DAYS)
    z = random_latents(3, seed=41)[:, None].expand(3, 5, LATENT_DIM).clone()
    q = model.cumulative(z)
    reports = q[:, 0] * 1.07
    known = (torch.isfinite(reports) & (days <= 180)[None, :, None] & (days > 0)[None, :, None]).clone()
    known[1, 3, 1] = False  # hidden day-90 gas report for one well
    a, r = model.noise_for(z)
    fast_mean, fast_var = brownian_future_moments(q, a, r, reports, known, days)
    mean, var = conditional_report_moments(model.covariance(z), q, reports, known, days)
    future = (days[days > 0] > 180)
    assert torch.allclose(fast_mean[..., future], mean[..., future], rtol=1e-9, atol=1e-12)
    assert torch.allclose(fast_var[..., future], var[..., future], rtol=1e-7, atol=1e-14)


def test_float32_ou_conditioning_is_close_to_float64():
    from joint_production.depletion_bayes import conditional_report_moments
    model = _model()
    days = torch.as_tensor(DAYS)
    z = random_latents(4, seed=51)[:, None].expand(4, 6, LATENT_DIM).clone()
    q = model.cumulative(z)
    a, r = model.noise_for(z)
    cov = ou_covariance(model.rate.cumulative, z, days, a, r, torch.full_like(a, 0.12),
                        torch.full((3,), 100.0, dtype=torch.float64))
    reports = q[:, 0] * 1.05
    known = (days <= 180)[None, :, None].expand_as(reports) & (days > 0)[None, :, None]
    m64, v64 = conditional_report_moments(cov, q, reports, known, days)
    m32, v32 = conditional_report_moments(cov.float() + 1e-8 * torch.eye(7), q.float(), reports.float(), known, days.float())
    future = days[days > 0] > 180
    assert torch.allclose(m32[..., future].double(), m64[..., future], rtol=1e-3, atol=1e-5)
    assert torch.allclose(v32[..., future].double(), v64[..., future], rtol=2e-2, atol=1e-7)


def test_fast_ou_likelihood_close_to_exact():
    from joint_production.depletion_bayes import gaussian_discrepancy_log_likelihood
    model = _model()
    days = torch.as_tensor(DAYS)
    z = random_latents(5, seed=61)[:, None].expand(5, 4, LATENT_DIM).clone()
    q = model.cumulative(z)
    a, r = model.noise_for(z)
    cov = ou_covariance(model.rate.cumulative, z, days, a, r, torch.full_like(a, 0.12),
                        torch.full((3,), 100.0, dtype=torch.float64))
    reports = q[:, 0] * 1.03
    mask = (days > 0)[None, :, None].expand_as(reports)
    exact = gaussian_discrepancy_log_likelihood(reports, q, cov, mask, days)
    fast = gaussian_discrepancy_log_likelihood(reports, q, cov, mask, days, fast=True)
    assert torch.allclose(fast, exact, atol=2e-2, rtol=1e-4)


def test_clock_likelihood_and_sampler_reduce_to_level_law():
    from joint_production.depletion_bayes import clock_log_likelihood, sample_clock_reports
    model = _model()
    days = torch.as_tensor(DAYS)
    z = random_latents(4, seed=71)[:, None].expand(4, 6, LATENT_DIM).clone()
    q = model.cumulative(z)
    a, r = model.noise_for(z)
    reports = q[:, 0] * 1.04
    mask = (torch.rand(reports.shape, generator=torch.Generator().manual_seed(3)) > 0.3) & (days > 0)[None, :, None]
    clock = model.clock(z)
    assert torch.allclose(clock_log_likelihood(reports, q, clock, mask, days),
                          accumulated_log_likelihood(reports, q, a, r, mask, days), rtol=1e-10, atol=1e-8)
    history = reports.clone()
    history[0, 2, 1] = float("nan")
    ref = sample_bridge_reports(q, history, days, a, r, cutoff=180.0, generator=torch.Generator().manual_seed(9))
    ours = sample_clock_reports(q, history, days, clock, cutoff=180.0, generator=torch.Generator().manual_seed(9))
    assert torch.allclose(ours, ref, rtol=1e-10, atol=1e-10)


def test_rate_clock_is_increasing_and_shrinks_late_relative_noise():
    model = DepletionBayesModel(3, DAYS, [1e5, 1.5e5, 4e4], rate_clock=True)
    z = random_latents(8, seed=72)[:, None]
    v = model.clock(z)
    assert torch.all(torch.diff(v, dim=-2) > 0)
    with torch.no_grad():
        model.log_noise_abs.fill_(-30.0)
        model.log_noise_rel.fill_(-30.0)
    v = model.clock(z)[:, 0]
    q = model.cumulative(z)[:, 0]
    early = (v[:, 2] - v[:, 1]) / (q[:, 2] - q[:, 1]).square()
    late = (v[:, -1] - v[:, -2]) / (q[:, -1] - q[:, -2]).square()
    assert torch.all(late < early)  # relative innovation variance falls as rates decline over longer intervals


def test_small_linear_algebra_matches_torch():
    from joint_production.depletion_bayes import (conditional_report_moments, small_cholesky, small_solve_lower,
                                                  small_solve_upper_t)
    generator = torch.Generator().manual_seed(5)
    root = torch.randn((10, 4, 7, 7), generator=generator, dtype=torch.float64)
    a = root @ root.transpose(-1, -2) + 0.5 * torch.eye(7, dtype=torch.float64)
    b = torch.randn((10, 4, 7), generator=generator, dtype=torch.float64)
    chol = small_cholesky(a)
    assert torch.allclose(chol, torch.linalg.cholesky(a), atol=1e-10)
    y = small_solve_lower(chol, b)
    assert torch.allclose(y, torch.linalg.solve_triangular(chol, b[..., None], upper=False).squeeze(-1), atol=1e-10)
    x = small_solve_upper_t(chol, y)
    assert torch.allclose(x, torch.linalg.solve(a, b[..., None]).squeeze(-1), atol=1e-8)
    model = _model()
    days = torch.as_tensor(DAYS)
    z = random_latents(3, seed=81)[:, None].expand(3, 4, LATENT_DIM).clone()
    q = model.cumulative(z)
    cov = model.covariance(z)
    reports = q[:, 0] * 1.02
    known = (days <= 90)[None, :, None].expand_as(reports) & (days > 0)[None, :, None]
    m1, v1 = conditional_report_moments(cov, q, reports, known, days)
    m2, v2 = conditional_report_moments(cov, q, reports, known, days, small=True)
    assert torch.allclose(m1, m2, atol=1e-10) and torch.allclose(v1, v2, rtol=1e-8, atol=1e-14)


def test_time_indexed_ratio_ablation_integrates_its_rate_and_has_finite_totals():
    from joint_production.depletion_bayes import DepletionRate, RATE_DIM
    z = 0.6 * torch.randn((5, RATE_DIM), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    t = torch.linspace(0.0, 4000.0, 40001, dtype=torch.float64)
    for index in ("depletion", "time"):
        law = DepletionRate(ratio_index=index, ratio_time_scale=730.0)
        rate, cumulative = law.rate(z, t), law.cumulative(z, t)
        trapezoid = torch.cat((torch.zeros_like(rate[:, :1]), torch.cumsum(0.05 * (rate[:, 1:] + rate[:, :-1]), dim=1)), 1)
        assert torch.all(rate >= 0)
        assert torch.allclose(trapezoid, cumulative, rtol=1e-4, atol=1e-6 * float(cumulative.abs().max()))
        far = law.cumulative(z, torch.tensor([1e7], dtype=torch.float64))[:, 0]
        assert torch.allclose(law.total_volume(z), far, rtol=1e-9)
    # Time indexing: the ratio depends on t only, not on the well's depletion.
    law = DepletionRate(ratio_index="time")
    z2 = z.clone()
    z2[:, :8] += 1.0  # different deliverability, same ratio coefficients
    assert torch.allclose(ratios(law, z, t[1::500])[..., 0], ratios(law, z2, t[1::500])[..., 0])


def test_interval_ou_kernel_matches_numerical_double_integral_and_is_positive_definite():
    from joint_production.depletion_bayes import interval_ou_covariance, interval_ou_kernel
    days = torch.tensor([0, 30, 60, 90, 180, 360, 720, 1080], dtype=torch.float64)
    tau = torch.tensor([40.0, 120.0, 400.0], dtype=torch.float64)
    kernel = interval_ou_kernel(days, tau)
    grid = torch.arange(0.5, 1080.0, 1.0, dtype=torch.float64)  # 1-day midpoint rule
    which = torch.bucketize(grid, days[1:], right=False)
    for p in range(3):
        k = torch.exp(-(grid[:, None] - grid[None, :]).abs() / tau[p])
        onehot = torch.nn.functional.one_hot(which, 7).double()
        numeric = onehot.T @ k @ onehot / (onehot.sum(0)[:, None] * onehot.sum(0)[None, :])
        assert torch.allclose(kernel[p], numeric, rtol=2e-3)
    assert torch.allclose(kernel, kernel.transpose(-1, -2))
    q = torch.cumsum(torch.rand((4, 8, 3), generator=torch.Generator().manual_seed(0), dtype=torch.float64), dim=1)
    q[:, 0] = 0
    cov = interval_ou_covariance(q, days, torch.full((4, 3), 0.2, dtype=torch.float64), tau)
    assert torch.allclose(cov, cov.transpose(-1, -2))
    assert torch.linalg.eigvalsh(cov).min() > -1e-10


def test_interval_ou_law_fits_predicts_and_respects_cutoff():
    reports, x = _synthetic()
    forecaster = DepletionBayesForecaster(epochs=2, batch_size=32, proposals=128, log_every=0, seed=5,
                                          discrepancy="ou_interval", objective="predictive", rate_clock=True,
                                          predictive_samples=8, train_samples=4)
    forecaster.fit(reports, DAYS, covariates=x)
    history = reports[:6].copy()
    history[:, DAYS > 90] = np.nan
    poisoned = reports[:6].copy()
    poisoned[:, DAYS > 90] = 1e12
    first = forecaster.predict_distribution(history, 90, 16, np.random.default_rng(2), x[:6])
    second = forecaster.predict_distribution(poisoned, 90, 16, np.random.default_rng(2), x[:6])
    assert np.array_equal(first["report_cumulative"], second["report_cumulative"])
    assert np.isfinite(first["report_cumulative"]).all()


def test_lognormal_law_likelihood_matches_reference_and_samples_are_positive():
    from joint_production.depletion_bayes import (interval_log_moments, lognormal_log_likelihood, observed_intervals,
                                                  sample_log_intervals, weighted_sample_crps)
    days = torch.tensor([0, 30, 60, 90, 180, 360, 720, 1080], dtype=torch.float64)
    g = torch.Generator().manual_seed(4)
    dq = torch.rand((2, 3, 3, 7), generator=g, dtype=torch.float64) + 0.1  # [N,S,3,T']
    dv = 0.01 * torch.rand((2, 3, 3, 7), generator=g, dtype=torch.float64)
    mean, cov = interval_log_moments(dq, dv, days, torch.full((2, 3, 3), 0.1, dtype=torch.float64),
                                     torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64))
    assert torch.allclose(torch.exp(mean + 0.5 * torch.diagonal(cov, dim1=-2, dim2=-1)), dq)  # mean-preserving
    reports = torch.cumsum(torch.rand((2, 8, 3), generator=g, dtype=torch.float64), 1)
    reports[:, 0] = 0
    reports[1, 3, 1] = reports[1, 2, 1]  # a reported zero interval (atom)
    observed = (days > 0)[None, :, None] & (days <= 180)[None, :, None] & torch.ones((2, 8, 3), dtype=torch.bool)
    volume, known, positive = observed_intervals(reports, observed)
    assert known[:, :, :4].all() and not known[:, :, 4:].any() and not positive[1, 1, 2]
    zero_prob = torch.full((7, 3), 0.01, dtype=torch.float64)
    ours = lognormal_log_likelihood(volume, known, positive, mean, cov, zero_prob)
    for n in range(2):
        for s_ in range(3):
            total = 0.0
            for p in range(3):
                idx = positive[n, p].nonzero().squeeze(-1)
                x = volume[n, p, idx].log()
                mvn = torch.distributions.MultivariateNormal(mean[n, s_, p, idx], cov[n, s_, p][idx][:, idx])
                total += float(mvn.log_prob(x) - x.sum())
                total += float((known[n, p] & ~positive[n, p]).sum()) * math.log(0.01) + float(positive[n, p].sum()) * math.log(0.99)
            assert abs(float(ours[n, s_]) - total) < 1e-8
    logs = sample_log_intervals(mean, cov, torch.where(positive, volume.clamp_min(1e-30).log(), torch.zeros_like(volume)),
                                positive, 4, torch.Generator().manual_seed(5))
    obs_mask = positive[None, :, None].expand_as(logs)
    target = volume.clamp_min(1e-30).log()[None, :, None].expand_as(logs)
    assert torch.allclose(logs[obs_mask], target[obs_mask])  # known intervals are reproduced exactly
    assert torch.all(torch.isfinite(logs))
    pts = torch.randn((5, 64), generator=g, dtype=torch.float64)
    w = torch.full((1, 64), 1 / 64, dtype=torch.float64)
    y = torch.zeros(5, dtype=torch.float64)
    brute = (pts - y[:, None]).abs().mean(-1) - 0.5 * (pts[:, :, None] - pts[:, None, :]).abs().mean((-1, -2))
    assert torch.allclose(weighted_sample_crps(pts, w, y), brute)


def test_lognormal_law_fits_and_forecasts_nonnegative_volumes():
    reports, x = _synthetic()
    forecaster = DepletionBayesForecaster(epochs=2, batch_size=32, proposals=128, log_every=0, seed=6,
                                          discrepancy="lognormal_ou", objective="predictive", rate_clock=True,
                                          predictive_samples=8, train_samples=4)
    forecaster.fit(reports, DAYS, covariates=x)
    history = reports[:6].copy()
    history[:, DAYS > 90] = np.nan
    poisoned = reports[:6].copy()
    poisoned[:, DAYS > 90] = 1e12
    first = forecaster.predict_distribution(history, 90, 32, np.random.default_rng(3), x[:6])
    second = forecaster.predict_distribution(poisoned, 90, 32, np.random.default_rng(3), x[:6])
    assert np.array_equal(first["report_cumulative"], second["report_cumulative"])
    draws = first["report_cumulative"]
    assert np.isfinite(draws).all() and np.all(np.diff(draws[:, :, DAYS >= 90], axis=2) >= 0)  # nonnegative future volumes
    known = np.isfinite(history) & ((DAYS <= 90) & (DAYS > 0))[None, :, None]
    full = np.broadcast_to(known[:, None], draws.shape)
    assert np.array_equal(draws[full], np.broadcast_to(history[:, None], draws.shape)[full])
