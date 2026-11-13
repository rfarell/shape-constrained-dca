"""Which structural properties each rate family guarantees, excludes or permits.

These are statements about the deterministic rate laws themselves (no fitting):
the depletion family guarantees ramp-then-completely-monotone oil and permits
physically documented gas rises; the fast positive basis permits oil rises long
after cleanup; conventional DCA forces every phase, including gas, to decline
from day zero and cannot represent an early ramp.
"""
import math

import torch

from joint_production.dca_rates import DCARate
from joint_production.depletion_bayes import GAS, LATENT_DIM, PEAK, PEAK_MAX_DAYS, DepletionRate
from joint_production.rates import PositiveBasisRate

GRID = torch.linspace(0.0, 3650.0, 3651, dtype=torch.float64)


def _random(count, dim, seed, spread=1.5):
    return torch.randn((count, dim), generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * spread


def test_depletion_oil_declines_after_bounded_ramp_for_every_parameter():
    rate = DepletionRate()
    z = _random(512, LATENT_DIM, 1)
    oil = rate.rate(z, GRID)[..., 0]
    after = GRID >= PEAK_MAX_DAYS
    assert torch.all(torch.diff(oil[:, after], dim=-1) <= 1e-12 * oil[:, after][:, :-1])


def test_depletion_permits_rising_gas_with_declining_oil():
    rate = DepletionRate()
    z = torch.zeros(1, LATENT_DIM, dtype=torch.float64)
    z[:, :8] = math.log(0.01)
    z[:, 5:8] = math.log(0.5)
    z[:, PEAK] = -10.0
    z[0, GAS] = torch.tensor([-3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    r = rate.rate(z, torch.linspace(200, 1080, 50, dtype=torch.float64))[0]
    assert torch.all(torch.diff(r[:, 0]) < 0) and torch.any(torch.diff(r[:, 1]) > 0)


def test_fast_basis_permits_oil_rise_long_after_cleanup():
    basis = PositiveBasisRate()
    z = torch.full((1, 12), -8.0, dtype=torch.float64)
    z[0, 3] = 0.0  # all oil volume on the shape-2, 720-day Erlang term
    oil = basis.rate(z, torch.tensor([300.0, 700.0], dtype=torch.float64))[0, :, 0]
    assert oil[1] > oil[0]  # rising oil rate between days 300 and 700


def test_dca_forces_every_phase_to_decline_from_day_zero():
    dca = DCARate()
    z = _random(512, 9, 2, spread=1.0)
    z[:, [1, 4, 7]] += math.log(120.0)
    rates = dca.rate(z, GRID)
    assert torch.all(torch.diff(rates, dim=1) <= 1e-12 * rates[:, :-1])  # no early ramp, no gas rise
    assert torch.all(rates.argmax(dim=1) == 0)
