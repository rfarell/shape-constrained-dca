import math

import numpy as np
import torch
from scipy.special import gamma, gammainc

from joint_production.perwell_dca import FAMILIES, PerWellDCAForecaster, dca_cumulative

DAYS = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=float)


def test_sedm_quadrature_matches_incomplete_gamma():
    theta = torch.tensor([[math.log(0.02), math.log(150.0), -0.5]], dtype=torch.float64)
    t = torch.tensor([30.0, 180.0, 1080.0], dtype=torch.float64)
    ours = dca_cumulative("sedm", theta, t)[0].numpy()
    n = 0.05 + 0.95 / (1 + math.exp(0.5))
    exact = 0.02 * 150.0 / n * gamma(1 / n) * gammainc(1 / n, (t.numpy() / 150.0) ** n)
    assert np.allclose(ours, exact, rtol=1e-6)


def test_families_start_at_zero_increase_and_stay_finite():
    t = torch.linspace(0, 3650, 731, dtype=torch.float64)
    theta = torch.randn((64, 3), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    for family in FAMILIES:
        q = dca_cumulative(family, theta + torch.tensor([math.log(0.01), math.log(0.01), 0.0], dtype=torch.float64), t)
        assert torch.all(torch.isfinite(q)) and torch.all(q[:, 0].abs() < 1e-9)
        assert torch.all(torch.diff(q, dim=-1) >= -1e-12)


def test_arps_is_continuous_through_harmonic_b():
    t = torch.tensor([400.0], dtype=torch.float64)
    near = [math.log((1.0 - 0.05) / 1.9 / (1 - (1.0 - 0.05) / 1.9)) + eps for eps in (-1e-5, 0.0, 1e-5)]
    values = [dca_cumulative("arps", torch.tensor([[math.log(0.02), math.log(0.01), u]], dtype=torch.float64), t).item()
              for u in near]
    assert abs(values[0] + values[2] - 2 * values[1]) < 1e-9 * abs(values[1])  # smooth through b = 1


def _synthetic(n=120, seed=0):
    rng = np.random.default_rng(seed)
    theta = torch.tensor(rng.normal([math.log(0.01), math.log(0.01), 0.0], 0.3, (n, 3, 3)))
    q = torch.stack([dca_cumulative("arps", theta[:, k], torch.as_tensor(DAYS)) for k in range(3)], -1).numpy()
    reports = q * 1e5 * (1 + 0.03 * rng.standard_normal(q.shape))
    reports[:, 0] = 0
    return reports, rng.normal(size=(n, 5))


def test_fit_and_predict_reproduce_history_for_both_modes():
    reports, x = _synthetic()
    for mode in ("bayes", "map"):
        model = PerWellDCAForecaster(family="arps", mode=mode, analogs=30, proposals=256, fit_steps=50).fit(reports, DAYS, covariates=x)
        history = reports[:6].copy()
        history[:, DAYS > 90] = np.nan
        draws = model.predict_samples(history, 90, 16, np.random.default_rng(0), x[:6])
        assert draws.shape == (6, 16, 8, 3) and np.isfinite(draws).all()
        known = (DAYS <= 90)
        assert np.allclose(draws[:, :, known], np.broadcast_to(history[:, None, known], draws[:, :, known].shape))


def test_scaling_curve_matches_slab_series_and_has_finite_recovery():
    tau, total = 900.0, 1.7
    theta = torch.tensor([[math.log(total), math.log(tau), 0.0]], dtype=torch.float64)
    t = np.array([1.0, 30.0, 89.0, 90.0, 91.0, 360.0, 1080.0, 20000.0])
    ours = dca_cumulative("scaling", theta, torch.as_tensor(t))[0].numpy()
    x = t / tau
    odd = np.arange(1, 4001, 2)[:, None]
    exact = total * (1 - 8 / math.pi ** 2 * (np.exp(-odd ** 2 * math.pi ** 2 * x / 4) / odd ** 2).sum(0))
    assert np.allclose(ours, exact, rtol=1e-6)  # includes both sides of the short/long-time switch (x = 0.1)
    assert abs(ours[-1] - total) < 1e-9  # EUR = M
    early = total * 2 * np.sqrt(x[:2] / math.pi)  # linear flow: Q grows as sqrt(t)
    assert np.allclose(ours[:2], early, rtol=1e-6)
