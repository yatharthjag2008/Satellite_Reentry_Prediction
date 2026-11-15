import numpy as np
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
