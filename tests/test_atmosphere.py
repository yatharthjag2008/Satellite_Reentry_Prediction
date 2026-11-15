import numpy as np
import pytest

from atmosphere import APS, DAYS_OF_YEAR, F107S, TABLE_FILE, MsisAtmosphere

pytestmark = pytest.mark.skipif(not TABLE_FILE.exists(), reason="run src/atmosphere.py to build the MSIS table")


@pytest.fixture(scope="module")
def atm():
    return MsisAtmosphere()


def test_matches_table_at_grid_points(atm):
    i_h, i_f, i_a, m = 64, 4, 5, 3   # 400 km, F10.7 140, Ap 15, April
    h, f, ap, doy = atm.altitudes[i_h], F107S[i_f], APS[i_a], DAYS_OF_YEAR[m]
    assert np.log10(atm.density(h, f, ap, doy)) == pytest.approx(atm.log_rho[i_h, i_f, i_a, m])


def test_density_increases_with_solar_flux_and_ap_and_falls_with_height(atm):
    assert atm.density(400, 200, 15) > atm.density(400, 100, 15)
    assert atm.density(400, 150, 100) > atm.density(400, 150, 5)
    assert atm.density(300, 150, 15) > atm.density(500, 150, 15)


def test_known_value_at_400_km(atm):
    # Moderate solar activity: published NRLMSIS values at 400 km are a few times 1e-12 kg/m^3.
    assert 1e-12 < atm.density(400, 150, 15) < 1e-11


def test_time_of_year_wraps_around(atm):
    assert atm.density(400, 150, 15, doy=0.0) == pytest.approx(atm.density(400, 150, 15, doy=365.25), rel=1e-9)


def test_profile_matches_full_interpolation(atm):
    f107a, ap = np.array([75.0, 160.0, 245.0]), np.array([4.0, 18.0, 60.0])
    h = np.array([[150.0, 420.0], [333.0, 700.0], [210.0, 999.0]])
    expected = atm.density(h, f107a[:, None], ap[:, None], 100.0)
    assert atm.profile(f107a, ap, doy=100.0)(h) == pytest.approx(expected, rel=1e-9)
