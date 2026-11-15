import numpy as np
import pytest
from scipy.integrate import solve_ivp

from atmosphere import ExponentialAtmosphere
from orbit_decay import (R_EARTH, circular_decay_rate, drag_rates, exponential_lifetime, orbit_averaged_rates,
                         points_needed, propagate)

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


def test_orbit_average_matches_circular_rate_when_circular():
    a = R_EARTH + 400.0
    da, de = orbit_averaged_rates(a, 0.0, B, ATM.density)
    assert da == pytest.approx(circular_decay_rate(a, B, ATM.density), rel=1e-9)
    assert de == pytest.approx(0.0, abs=1e-15)


def test_eccentric_orbit_loses_apogee_not_perigee():
    # Perigee 250 km, apogee 1,500 km, with a thin atmosphere (H = 30 km) so drag acts near perigee.
    atm = ExponentialAtmosphere(rho0=5e-11, h0=250.0, scale_height=30.0)
    rp, ra = R_EARTH + 250.0, R_EARTH + 1500.0
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    da, de = orbit_averaged_rates(a, e, B, atm.density, n_points=512)
    perigee_rate = (1 - e) * da - a * de
    apogee_rate = (1 + e) * da + a * de
    assert apogee_rate < 0
    assert abs(perigee_rate) < 0.05 * abs(apogee_rate)


def test_orbit_average_converges():
    a, e = R_EARTH + 800.0, 0.05
    coarse = orbit_averaged_rates(a, e, B, ATM.density, n_points=32)
    fine = orbit_averaged_rates(a, e, B, ATM.density, n_points=1024)
    assert coarse[0] == pytest.approx(fine[0], rel=1e-3)
    assert coarse[1] == pytest.approx(fine[1], rel=1e-3)


def exponential(atm):
    return lambda h, idx, t: atm.density(h)


def test_propagate_matches_exact_lifetime():
    starts = np.array([200.0, 300.0, 400.0, 550.0])
    days = propagate(R_EARTH + starts, 0.0, B, exponential(ATM), max_days=1e6)
    exact = exponential_lifetime(starts, 120.0, B, ATM.rho0, ATM.h0, ATM.scale_height)
    np.testing.assert_allclose(days, exact, rtol=2e-3)


def test_propagate_lifetime_scales_as_one_over_B():
    days = propagate(np.full(3, R_EARTH + 400.0), 0.001, [0.005, 0.01, 0.02], exponential(ATM), max_days=1e6)
    np.testing.assert_allclose(days * [0.005, 0.01, 0.02], days[1] * 0.01, rtol=5e-3)


def test_propagate_gives_nan_past_max_days():
    days = propagate([R_EARTH + 400.0, R_EARTH + 900.0], 0.0, B, exponential(ATM), max_days=1000)
    assert np.isfinite(days[0]) and np.isnan(days[1])


def test_eccentric_orbit_lasts_longer_than_circular_at_same_perigee():
    rp = R_EARTH + 300.0
    a = np.array([rp, (rp + R_EARTH + 1000.0) / 2])
    e = np.array([0.0, (a[1] - rp) / a[1]])
    days = propagate(a, e, B, exponential(ATM), max_days=1e6)
    assert days[1] > days[0]


def test_points_needed_grows_with_eccentricity():
    n = points_needed(np.array([R_EARTH + 400, R_EARTH + 400, 24000.0]), np.array([0.0, 0.01, 0.73]))
    assert n[0] == 8 and n[0] <= n[1] < n[2] and n[2] >= 256


def test_drag_rates_converge_for_a_transfer_orbit():
    # Rocket body in a geostationary transfer orbit: perigee 200 km, apogee 35,786 km.
    rp, ra = R_EARTH + 200.0, R_EARTH + 35786.0
    a, e = (rp + ra) / 2, (ra - rp) / (ra + rp)
    density = lambda h, rows: ATM.density(h)
    auto = drag_rates(a, e, B, density)
    fine = orbit_averaged_rates(a, e, B, ATM.density, n_points=8192)
    assert auto[0][0] == pytest.approx(fine[0], rel=1e-4)
    assert auto[1][0] == pytest.approx(fine[1], rel=1e-4)
