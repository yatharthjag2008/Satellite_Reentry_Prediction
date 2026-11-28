import numpy as np
import pytest
import pandas as pd

from predict_reentry import _duration, describe

EPOCH = pd.Timestamp("2026-11-01 06:00")


def result(**changes):
    base = {"epoch": EPOCH, "perigee": 300.0, "apogee": 320.0, "B": 0.02, "perigee_rate": -0.5,
            "physics_days": 40.0, "hybrid_days": 36.0, "range": (25.0, 52.0)}
    return {**base, **changes}


def test_duration_picks_sensible_units():
    assert _duration(0.5) == "12 hours"
    assert _duration(40) == "40 days"
    assert _duration(200) == "7 months"
    assert _duration(3650) == "10.0 years"


def test_describe_shows_both_models_and_the_range():
    text = "\n".join(describe(result()))
    assert "Physics model: about 40 days (around 2026-12-11)" in text
    assert "Hybrid model:  about 36 days (around 2026-12-07)" in text
    assert "90% range:     2026-11-26 to 2026-12-23" in text


def test_describe_explains_an_orbit_that_is_not_sinking():
    text = "\n".join(describe(result(B=-0.001, physics_days=np.nan, hybrid_days=None, range=None)))
    assert "Not sinking" in text


def test_describe_far_future_skips_the_hybrid():
    text = "\n".join(describe(result(physics_days=3000.0, hybrid_days=None, range=None)))
    assert "8.2 years" in text and "beyond what the hybrid model was trained on" in text


def test_no_recent_orbits():
    assert "No orbits" in describe(None)[0]


def test_describe_shows_the_time_of_day_when_close():
    text = "\n".join(describe(result(physics_days=0.75, hybrid_days=0.8, range=(0.5, 1.25))))
    assert "Physics model: about 18 hours (around 2026-11-02 00:00 UTC)" in text
    assert "90% range:     2026-11-01 18:00 UTC to 2026-11-02 12:00 UTC" in text


def test_describe_explains_the_short_window_close_to_reentry():
    text = "\n".join(describe(result(physics_days=2.0, hybrid_days=None, range=None, short_window=True)))
    assert "last 8 days of tracking" in text
    assert "Hybrid" not in text and "two years" not in text


def test_describe_warns_that_satellites_can_reboost():
    assert "thrusters" in "\n".join(describe(result(is_payload=True)))
    assert "thrusters" not in "\n".join(describe(result(is_payload=False)))


def test_describe_mentions_the_solar_cycle_for_long_predictions():
    text = "\n".join(describe(result(physics_days=2600.0, hybrid_days=None, range=None, solar_cycle=True)))
    assert "11-year cycle" in text and "7.1 years" in text


def test_describe_mentions_the_sun_and_moon_for_transfer_orbits():
    text = "\n".join(describe(result(apogee=30000.0, physics_days=900.0, hybrid_days=None, range=None,
                                      solar_cycle=True, lunisolar=True)))
    assert "Moon's and Sun's gravity" in text and "No hybrid model" in text and "11-year" not in text


def test_transfer_orbits_are_recognised_only_with_their_angles():
    from predict_reentry import is_transfer_orbit
    row = pd.DataFrame({"APOAPSIS": [30000.0], "RA_OF_ASC_NODE": [10.0], "ARG_OF_PERICENTER": [20.0]})
    assert is_transfer_orbit(row)
    assert not is_transfer_orbit(row.assign(APOAPSIS=600.0))
    assert not is_transfer_orbit(row.drop(columns="ARG_OF_PERICENTER"))


def test_describe_explains_a_fit_too_large_to_be_drag():
    text = "\n".join(describe(result(B=5.0, physics_days=np.nan, hybrid_days=120.0, range=(80.0, 180.0))))
    assert "faster than drag can" in text and "30 years" not in text
    assert "Hybrid model:  about 4 months" in text and "90% range" in text


def test_describe_says_which_latitudes_it_can_fall_on():
    assert "between 52°S and 52°N" in "\n".join(describe(result(inclination=51.6)))
    assert "Where:" not in "\n".join(describe(result(B=-0.001, physics_days=np.nan, inclination=51.6)))


def test_describe_says_which_assumptions_put_it_beyond_thirty_years():
    def text(**changes):
        return "\n".join(describe(result(physics_days=np.nan, hybrid_days=None, range=None, **changes)))
    assert "at today's solar activity" in text()
    assert "11-year cycle" in text(solar_cycle=True) and "today's" not in text(solar_cycle=True)
    assert "Moon's and Sun's gravity" in text(solar_cycle=True, lunisolar=True)


def test_transfer_range_takes_the_wider_widths(tmp_path, monkeypatch):
    import json
    import predict_reentry
    path = tmp_path / "widths.json"
    path.write_text(json.dumps({"bands": [0, 3, 10, 30, 100, 1e9], "widths": {
        "30": [[0.1, 0.1, 0.1, 0.1, 0.3]] * 3, "transfer": [[0.2, 0.2, 0.2, 0.2, 0.2]] * 3}}))
    monkeypatch.setattr(predict_reentry, "RANGES_FILE", path)
    row = pd.DataFrame({"OBJECT_TYPE": ["R/B"], "is_starlink": [0]})
    low, high = predict_reentry.transfer_range(50.0, row)
    assert high / 50.0 == pytest.approx(10 ** 0.2) and low * high == pytest.approx(2500.0)
    assert predict_reentry.transfer_range(200.0, row)[1] / 200.0 == pytest.approx(10 ** 0.3)
    assert predict_reentry.transfer_range(400.0, row) is None  # beyond a year
