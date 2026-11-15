import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from atmosphere import ExponentialAtmosphere
from orbit_decay import MU, R_EARTH, circular_decay_rate
from physics_model import fit_drag_parameter, prepare, semi_major_axis, unit_decay_rate

ATM = ExponentialAtmosphere(rho0=3.7e-12, h0=400.0, scale_height=60.0)


def mean_motion(a_km):
    return np.sqrt(MU / (a_km * 1e3) ** 3) * 86400 / (2 * np.pi)


def synthetic_history(norad, B, h_start=350.0, days=60, raise_on_day=None):
    """Daily orbits of a circular object decaying with drag parameter B."""
    t = np.arange(days, dtype=float)
    sol = solve_ivp(lambda _, a: circular_decay_rate(a, B, ATM.density), (0, days), [R_EARTH + h_start],
                    t_eval=t, rtol=1e-10, atol=1e-9)
    a = sol.y[0]
    if raise_on_day is not None:
        a = a + np.where(t >= raise_on_day, 5.0 * (t - raise_on_day + 1), 0.0)  # thruster firing
    return pd.DataFrame({"NORAD_CAT_ID": norad, "EPOCH": pd.Timestamp("2020-01-01") + pd.to_timedelta(t, unit="D"),
                         "MEAN_MOTION": mean_motion(a), "eccentricity": 0.0, "f107_81d": 150.0, "ap": 10.0})


def test_semi_major_axis_inverts_mean_motion():
    assert semi_major_axis(mean_motion(6778.0)) == pytest.approx(6778.0)


def test_fit_recovers_drag_parameter_for_several_objects():
    df = prepare(pd.concat([synthetic_history(1, 0.01), synthetic_history(2, 0.05, h_start=420.0)]))
    B, n = fit_drag_parameter(df, unit_decay_rate(df, ATM), window=30)
    last = df.groupby("NORAD_CAT_ID").cumcount(ascending=False) == 0
    np.testing.assert_allclose(B[last], [0.01, 0.05], rtol=1e-3)
    assert n[last].tolist() == [31, 31]   # the window includes both ends


def test_fit_needs_enough_points():
    df = prepare(synthetic_history(1, 0.01, days=3))
    B, _ = fit_drag_parameter(df, unit_decay_rate(df, ATM))
    assert np.isnan(B).all()


def test_orbit_raise_gives_negative_drag_parameter():
    df = prepare(synthetic_history(1, 0.01, raise_on_day=40))
    B, _ = fit_drag_parameter(df, unit_decay_rate(df, ATM), window=15)
    assert B[-1] < 0
