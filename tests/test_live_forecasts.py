import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from atmosphere import MsisAtmosphere
from live_forecasts import (ORBIT_COLUMNS, add_to_log, fit_label, forecast_all, merge_orbits, ranges, scored_forecasts,
                            scores, to_orbits)
from orbit_decay import MU, R_EARTH, circular_decay_rate

START = pd.Timestamp("2024-03-01")


def mean_motion(a_km):
    return np.sqrt(MU / (a_km * 1e3) ** 3) * 86400 / (2 * np.pi)


def orbits(norad, a_km, days):
    """Orbits in the collected format for a circular orbit with semi-major axis a_km at each time (days)."""
    h = np.asarray(a_km) - R_EARTH
    epochs = START + pd.to_timedelta(days, unit="D")
    # Parsed from text, as CelesTrak's are, so the times have the same resolution as the space weather's.
    epochs = pd.to_datetime(pd.Series(epochs).dt.strftime("%Y-%m-%dT%H:%M:%S"))
    return pd.DataFrame({"NORAD_CAT_ID": norad, "EPOCH": epochs,
                         "MEAN_MOTION": mean_motion(np.asarray(a_km)), "ECCENTRICITY": 0.0, "INCLINATION": 51.6,
                         "BSTAR": 1e-4, "MEAN_MOTION_DOT": 1e-4, "APOAPSIS": h, "PERIAPSIS": h,
                         "RA_OF_ASC_NODE": 10.0, "ARG_OF_PERICENTER": 0.0, "MEAN_ANOMALY": 0.0})[ORBIT_COLUMNS]


def decaying(norad, B, h_start, days, atm, end=25.0, every=0.25):
    """Orbits every `every` days for `days` days up to day `end`, decaying by drag with parameter B."""
    t = np.arange(0, days, every)
    density = lambda h: atm.density(h, 150.0, 10.0, 60.0)
    sol = solve_ivp(lambda _, a: circular_decay_rate(a, B, density), (0, t[-1]), [R_EARTH + h_start], t_eval=t,
                    rtol=1e-9, atol=1e-9)
    return orbits(norad, sol.y[0], end - days + t)


def test_to_orbits_converts_celestrak_rows():
    gp = pd.DataFrame({"OBJECT_NAME": ["X"], "NORAD_CAT_ID": [1], "EPOCH": ["2026-11-27T13:34:19.706304"],
                       "MEAN_MOTION": [mean_motion(R_EARTH + 300.0)], "ECCENTRICITY": [0.001], "INCLINATION": [51.6],
                       "BSTAR": [1e-4], "MEAN_MOTION_DOT": [1e-4], "RA_OF_ASC_NODE": [10.0],
                       "ARG_OF_PERICENTER": [20.0], "MEAN_ANOMALY": [30.0]})
    out = to_orbits(gp)
    assert list(out.columns) == ORBIT_COLUMNS
    a = R_EARTH + 300.0
    assert out["PERIAPSIS"].iloc[0] == pytest.approx(a * 0.999 - R_EARTH, abs=1e-6)
    assert out["APOAPSIS"].iloc[0] == pytest.approx(a * 1.001 - R_EARTH, abs=1e-6)
    assert out["EPOCH"].iloc[0] == pd.Timestamp("2026-11-27 13:34:19.706304")


def test_merge_orbits_drops_repeats_and_old_orbits():
    old = orbits(1, [R_EARTH + 300] * 3, [0, 10, 50])
    new = orbits(1, [R_EARTH + 290] * 2, [50, 51])  # day 50 again, with a corrected value
    now = START + pd.Timedelta(days=52)
    out = merge_orbits([old, new], now, days=45)
    assert out["EPOCH"].tolist() == [START + pd.Timedelta(days=d) for d in (10, 50, 51)]
    assert out["PERIAPSIS"].iloc[1] == pytest.approx(290)  # the newer copy wins


def test_fit_label_matches_the_history_each_object_has():
    assert fit_label(np.array([3, 13.9, 14, 29.9, 30, 40])).tolist() == ["7", "7", "14", "14", "30", "30"]


def test_ranges_use_the_window_group_and_band():
    widths = {"widths": {"30": [[0.1] * 5, [0.2] * 5, [0.3, 0.3, 0.3, 0.3, None]]}}
    low, high = ranges(np.array([5.0, 5.0, 200.0, np.nan]), np.array(["30"] * 4), np.array([0, 1, 2, 0]), widths)
    np.testing.assert_allclose(low[:2], [5 / 10 ** 0.1, 5 / 10 ** 0.2])
    np.testing.assert_allclose(high[:2], [5 * 10 ** 0.1, 5 * 10 ** 0.2])
    assert np.isnan(low[2]) and np.isnan(high[3])  # no width for that cell; no forecast


def test_log_keeps_the_first_forecast_from_each_orbit():
    now = START + pd.Timedelta(days=1)
    latest = pd.DataFrame({"NORAD_CAT_ID": [1, 2], "EPOCH": [START, START], "days": [5.0, np.nan],
                           "low_days": [4.0, np.nan], "high_days": [6.0, np.nan], "fit": ["8-day", "7"],
                           "status": ["forecast", "not sinking"]})
    first = add_to_log(pd.DataFrame(), latest, now)
    again = add_to_log(first, latest.assign(days=9.0), now + pd.Timedelta(hours=6))
    assert len(first) == 1 and len(again) == 1 and again["days"].iloc[0] == 5.0


def test_scores_compare_with_the_decay_date_at_midday():
    made = START + pd.Timedelta(days=10)
    log = pd.DataFrame({
        "NORAD_CAT_ID": [1, 1, 2, 3],
        "EPOCH": [START, START + pd.Timedelta(days=9.5), START + pd.Timedelta(days=12), START],
        "days": [12.0, 0.5, 20.0, 50.0],
        "low_days": [8.0, 0.3, 21.0, 40.0], "high_days": [15.0, 1.2, 30.0, 60.0],
        "fit": ["30", "8-day", "30", "30"], "made": [made] * 4})
    satcat = pd.DataFrame({"NORAD_CAT_ID": [1, 2, 3], "OBJECT_NAME": ["A", "B", "C"], "OBJECT_TYPE": ["DEB"] * 3,
                           "DECAY_DATE": ["2024-03-11", "2024-03-21", np.nan]})
    m = scored_forecasts(log, satcat)
    assert m["actual"].tolist() == [10.5, 1.0, 8.5]  # object 3 hasn't reentered
    out = scores(log, satcat)
    assert out["overall"]["objects"] == 2 and out["overall"]["forecasts"] == 3
    assert out["overall"]["in_range"] == pytest.approx(2 / 3)  # object 2's range starts too late
    under_week, weeks = out["bands"][0], out["bands"][1]
    assert under_week["forecasts"] == 1 and under_week["median_error"] == pytest.approx(0.5)
    assert weeks["forecasts"] == 2
    recent = {r["id"]: r for r in out["recent"]}
    assert recent[2]["decay_date"] == "2024-03-21"  # most recent reentry first
    assert out["recent"][0]["id"] == 2
    assert recent[1]["leads"]["1"]["error_days"] == pytest.approx(-0.5)
    assert recent[1]["leads"]["7"]["error_days"] == pytest.approx(1.5)
    assert recent[2]["leads"] == {"7": {"reentry": "2024-04-02T00:00Z", "error_days": 11.5}}


def test_scores_with_nothing_logged_yet():
    out = scores(pd.DataFrame(columns=["NORAD_CAT_ID", "EPOCH", "days", "low_days", "high_days", "fit", "made"]),
                 pd.DataFrame(columns=["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "DECAY_DATE"]))
    assert out["since"] is None and out["overall"]["forecasts"] == 0 and out["recent"] == []


def test_forecast_all_statuses():
    atm = MsisAtmosphere()
    widths = {"widths": {label: [[0.1] * 5] * 3 for label in ("7", "14", "30", "8-day")}}
    t = np.arange(5, 25, 0.25)
    a_high = R_EARTH + 700 - 3.0 * t  # sinking 3 km/day at 700 km: far too fast for drag
    history = pd.concat([
        decaying(900001, 0.02, 260.0, 6.0, atm),          # days from reentry: the 8-day fit
        decaying(900002, 0.02, 330.0, 25.0, atm),         # weeks to months away
        orbits(900003, R_EARTH + 400 + 0.1 * t, t),  # raising its orbit
        orbits(900004, a_high, t),
        orbits(900005, [R_EARTH + 300] * 2, [24.0, 24.5]),  # only just appeared
    ])
    now = history["EPOCH"].max() + pd.Timedelta(hours=1)
    latest = forecast_all(history, atm, widths, now).set_index("NORAD_CAT_ID")

    assert latest.loc[900001, "status"] == "forecast" and latest.loc[900001, "fit"] == "8-day"
    assert 0 < latest.loc[900001, "days"] < 10
    assert latest.loc[900002, "status"] == "forecast" and latest.loc[900002, "fit"] == "14"
    assert latest.loc[900002, "days"] > 10
    row = latest.loc[900002]
    assert row["low_days"] < row["days"] < row["high_days"]
    assert latest.loc[900003, "status"] == "not sinking"
    assert latest.loc[900004, "status"] == "not drag"
    assert latest.loc[900005, "status"] == "too few orbits"


def test_forecast_all_skips_objects_that_reentered_or_went_quiet():
    atm = MsisAtmosphere()
    widths = {"widths": {label: [[0.1] * 5] * 3 for label in ("7", "14", "30", "8-day")}}
    history = pd.concat([decaying(900001, 0.02, 330.0, 10.0, atm), decaying(900002, 0.02, 330.0, 10.0, atm)])
    now = history["EPOCH"].max() + pd.Timedelta(days=1)
    assert forecast_all(history, atm, widths, now, decayed_ids=[900001])["NORAD_CAT_ID"].tolist() == [900002]
    assert forecast_all(history, atm, widths, now + pd.Timedelta(days=5)).empty


def test_tracks_only_for_short_windows_from_now_on():
    from live_forecasts import tracks

    epoch = pd.Timestamp("2026-01-01")
    row = {"NORAD_CAT_ID": 1, "EPOCH": epoch, "ECCENTRICITY": 0.001, "ARG_OF_PERICENTER": 0.0, "INCLINATION": 51.6,
           "MEAN_ANOMALY": 0.0, "MEAN_MOTION": 15.9, "RA_OF_ASC_NODE": 0.0, "status": "forecast"}
    latest = pd.DataFrame([{**row, "days": 1.0, "low_days": 0.5, "high_days": 1.5},
                           {**row, "NORAD_CAT_ID": 2, "days": 5.0, "low_days": 3.0, "high_days": 8.0}])
    now = epoch + pd.Timedelta(days=0.75)  # part of the first window has passed
    out = tracks(latest, now)
    assert list(out) == ["1"]
    assert out["1"]["start"] == "2026-01-01T18:00Z" and out["1"]["end"] == "2026-01-02T12:00Z"
    assert len(out["1"]["lat"]) == 18 * 30 + 1  # every 2 minutes for 18 hours
    assert abs(out["1"]["at"][0]) <= 51.7
