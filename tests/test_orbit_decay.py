import numpy as np
import pytest
from scipy.integrate import solve_ivp

from atmosphere import ExponentialAtmosphere
from orbit_decay import R_EARTH, circular_decay_rate, exponential_lifetime

ATM = ExponentialAtmosphere(rho0=3.7e-12, h0=400.0, scale_height=60.0)
B = 0.01  # m^2/kg, typical of a small satellite


def days_to_sink(h_start, h_end, rate):
    """Integrate da/dt numerically until the orbit reaches h_end."""
    reached = lambda t, a: a[0] - (R_EARTH + h_end)
    reached.terminal = True
    sol = solve_ivp(lambda t, a: rate(a), (0, 1e6), [R_EARTH + h_start], events=reached, rtol=1e-10, atol=1e-9)
    return sol.t_events[0][0]


@pytest.mark.parametrize("h_start", [250.0, 400.0, 600.0])
def test_numerical_lifetime_matches_exact_formula(h_start):
    exact = exponential_lifetime(h_start, 120.0, B, ATM.rho0, ATM.h0, ATM.scale_height)
    numerical = days_to_sink(h_start, 120.0, lambda a: circular_decay_rate(a, B, ATM.density))
    assert numerical == pytest.approx(exact, rel=1e-6)


def test_lifetime_is_about_one_scale_height_over_the_rate():
    h = 500.0
    rate = circular_decay_rate(R_EARTH + h, B, ATM.density)
    exact = exponential_lifetime(h, 120.0, B, ATM.rho0, ATM.h0, ATM.scale_height)
    assert exact == pytest.approx(ATM.scale_height / -rate, rel=0.02)


def test_lifetime_scales_as_one_over_B():
    t1 = exponential_lifetime(400.0, 120.0, B, ATM.rho0, ATM.h0, ATM.scale_height)
    t2 = exponential_lifetime(400.0, 120.0, 2 * B, ATM.rho0, ATM.h0, ATM.scale_height)
    assert t2 == pytest.approx(t1 / 2)
