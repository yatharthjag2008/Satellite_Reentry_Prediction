import numpy as np
import pandas as pd
import pytest

from ground_track import MAX_WINDOW_DAYS, gmst, ground_track, latitude_band, satellite, subpoints


def orbit(**changes):
    base = {"NORAD_CAT_ID": 1, "EPOCH": "2026-01-01T00:00:00", "ECCENTRICITY": 0.001, "ARG_OF_PERICENTER": 0.0,
            "INCLINATION": 51.6, "MEAN_ANOMALY": 0.0, "MEAN_MOTION": 15.9, "RA_OF_ASC_NODE": 0.0}
    return {**base, **changes}


def test_gmst_at_j2000():
    assert np.degrees(gmst(2451545.0)) == pytest.approx(280.46061837, abs=1e-6)


def test_a_geostationary_orbit_stays_over_one_longitude():
    geo = orbit(ECCENTRICITY=0.0, INCLINATION=0.0, MEAN_MOTION=1.00273791)
    lat, lon = subpoints(satellite(geo), pd.date_range("2026-01-01", periods=25, freq="1h"))
    assert np.abs(lat).max() < 0.05
    assert lon.max() - lon.min() < 0.05


def test_track_reaches_but_never_passes_the_inclination():
    start = pd.Timestamp("2026-01-01")
    lat, lon = ground_track(orbit(), start, start + pd.Timedelta(hours=6))
    assert lat.max() == pytest.approx(51.6, abs=0.5) and lat.min() == pytest.approx(-51.6, abs=0.5)
    assert np.abs(lon).max() <= 180


def test_no_track_for_a_wide_window():
    start = pd.Timestamp("2026-01-01")
    assert ground_track(orbit(), start, start + pd.Timedelta(days=MAX_WINDOW_DAYS + 0.1)) is None


def test_latitude_band_of_a_retrograde_orbit():
    assert latitude_band(51.6) == 51.6
    assert latitude_band(97.4) == pytest.approx(82.6)
