import numpy as np
import pytest

from atmosphere import ExponentialAtmosphere
from lunisolar import (J2, JD_J2000, MU_EARTH, MU_MOON, R_EARTH, moon_position, orbit_vectors, propagate_lunisolar,
                       secular_rates, sun_position, tide_gradients, milankovitch)
from orbit_decay import propagate

ATM = ExponentialAtmosphere(rho0=3.7e-12, h0=400.0, scale_height=60.0)


def angles(e_vec, j_vec):
    """Right ascension of the node and argument of perigee (degrees) from the vectors."""
    w = j_vec / np.linalg.norm(j_vec)
    node = np.array([-w[1], w[0], 0.0])  # z x w
    raan = np.degrees(np.arctan2(node[1], node[0]))
    p = e_vec / np.linalg.norm(e_vec)
    arg = np.degrees(np.arctan2(np.dot(np.cross(node / np.linalg.norm(node), p), w), np.dot(node, p) / np.linalg.norm(node)))
    return raan, arg


def test_j2_precession_matches_the_closed_form():
    a, e, i = 7500.0, 0.1, 50.0
    e_vec, j_vec = orbit_vectors(e, i, 30.0, 40.0)
    de, dj = secular_rates(np.array([a]), e_vec[None], j_vec[None], np.array([JD_J2000]), sun=False, moon=False)
    dt = 1e-3  # days
    raan0, arg0 = angles(e_vec, j_vec)
    raan1, arg1 = angles(e_vec + dt * de[0], j_vec + dt * dj[0])
    n = np.sqrt(MU_EARTH / a ** 3) * 86400  # rad/day
    p = a * (1 - e ** 2)
    raan_rate = -1.5 * n * J2 * (R_EARTH / p) ** 2 * np.cos(np.radians(i))
    arg_rate = 0.75 * n * J2 * (R_EARTH / p) ** 2 * (5 * np.cos(np.radians(i)) ** 2 - 1)
    assert np.radians(raan1 - raan0) / dt == pytest.approx(raan_rate, rel=1e-3)
    assert np.radians(arg1 - arg0) / dt == pytest.approx(arg_rate, rel=1e-3)


def test_rates_keep_e_and_j_perpendicular_and_their_lengths_consistent():
    e_vec, j_vec = orbit_vectors(0.7, 27.0, 100.0, 200.0)
    for jd in (JD_J2000, JD_J2000 + 100.3):
        de, dj = secular_rates(np.array([24000.0]), e_vec[None], j_vec[None], np.array([jd]))
        assert np.dot(de[0], j_vec) + np.dot(e_vec, dj[0]) == pytest.approx(0, abs=1e-12)   # e.j stays 0
        assert np.dot(e_vec, de[0]) + np.dot(j_vec, dj[0]) == pytest.approx(0, abs=1e-12)   # e^2 + j^2 stays 1


def test_kozai_a_distant_ring_grows_the_eccentricity_of_an_inclined_orbit():
    """A perturber spread around a ring in the xy plane: e grows for i = 60 deg, w = 45 deg, and shrinks for w = 135 deg."""
    ring = np.array([[np.cos(t), np.sin(t), 0.0] for t in np.linspace(0, 2 * np.pi, 360, endpoint=False)]) * 384400.0
    a = np.array([24000.0])
    for arg, sign in ((45.0, 1), (135.0, -1)):
        e_vec, j_vec = orbit_vectors(0.3, 60.0, 0.0, arg)
        grads = [tide_gradients(a, e_vec[None], j_vec[None], MU_MOON, r[None]) for r in ring]
        grad_e, grad_j = np.mean([g[0] for g in grads], axis=0), np.mean([g[1] for g in grads], axis=0)
        de, _ = milankovitch(a, e_vec[None], j_vec[None], grad_e, grad_j)
        assert np.sign(np.dot(de[0], e_vec)) == sign


def test_sun_and_moon_are_where_they_should_be():
    solstice = 2461213.0  # 2026-06-21, about the June solstice
    sun = sun_position(np.array([solstice]))[0]
    assert np.degrees(np.arcsin(sun[2] / np.linalg.norm(sun))) == pytest.approx(23.44, abs=0.05)
    assert np.linalg.norm(sun) / 149597870.7 == pytest.approx(1.0163, abs=0.001)
    days = JD_J2000 + np.arange(0, 3000, 0.7)
    distance = np.linalg.norm(moon_position(days), axis=1)
    assert 356000 < distance.min() < 360000 and 404000 < distance.max() < 407500
    declination = np.degrees(np.arcsin(moon_position(days)[:, 2] / distance))
    assert 28.0 < np.abs(declination).max() < 29.0  # about 28.7 deg at a major lunar standstill


def test_without_the_new_forces_it_matches_the_drag_propagator():
    a = np.array([R_EARTH + 400.0, R_EARTH + 1200.0])
    e = np.array([0.0005, 0.06])
    B = np.array([0.02, 0.05])
    density = lambda h, idx, t: ATM.density(h)
    expected = propagate(a, e, B, density, max_days=5000)
    e_vec, j_vec = orbit_vectors(e, [51.6, 28.0], [10.0, 20.0], [30.0, 40.0])
    got = propagate_lunisolar(a, e_vec, j_vec, B, density, JD_J2000, max_days=5000, max_step=30.0,
                              sun=False, moon=False, j2=False)
    np.testing.assert_allclose(got, expected, rtol=0.01)


def test_days_from_rows_reads_the_elements_and_a_utc_epoch():
    import pandas as pd
    from lunisolar import days_from_rows

    rows = pd.DataFrame({"a_km": [R_EARTH + 500.0], "eccentricity": [0.02], "INCLINATION": [28.0],
                         "RA_OF_ASC_NODE": [20.0], "ARG_OF_PERICENTER": [40.0],
                         "EPOCH": [pd.Timestamp("2000-01-01 12:00")]})
    density = lambda h, idx, t: ATM.density(h)
    e_vec, j_vec = orbit_vectors([0.02], [28.0], [20.0], [40.0])
    expected = propagate_lunisolar(rows["a_km"].to_numpy(), e_vec, j_vec, [0.05], density, JD_J2000, max_days=1000)
    for epochs in (rows["EPOCH"], rows["EPOCH"].dt.tz_localize("UTC")):
        got = days_from_rows(rows.assign(EPOCH=epochs), [0.05], density, max_days=1000)
        np.testing.assert_allclose(got, expected)
