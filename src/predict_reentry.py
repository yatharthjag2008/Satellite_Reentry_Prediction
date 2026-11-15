"""Predict when one object still in orbit will reenter.

Downloads the object's last HISTORY_DAYS of orbits from Space-Track and fresh
space weather from CelesTrak, builds the same features as the training table,
and runs the physics model and the physics + ML hybrid. Used by
lookup_satellite.py; can also be run on its own:

    python src/predict_reentry.py 25544
"""

import io
import json
import sys
import time

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

from atmosphere import MsisAtmosphere
from build_reentry_dataset import DATA_DIR, R_EARTH, add_catalog_info, add_space_weather, add_trends
from download_history import FIELDS
from physics_model import fit_drag_parameter, forecast, prepare, unit_decay_rate
from train_reentry_model import HYBRID_FEATURES, MODELS_DIR
from uncertainty import group_of

HISTORY_DAYS = 45
SW_URL = "https://celestrak.org/SpaceData/SW-All.csv"
SW_MAX_AGE = 86400  # re-download space weather once a day
MAX_PHYSICS_DAYS = 365.25 * 30
# Close to reentry, B fitted over the last few days of every orbit is far more accurate (EXPERIMENTS.md 015).
SHORT_WINDOW = 8
SHORT_WINDOW_BELOW = 10  # days: switch when the usual prediction is closer than this
CONFORMAL_FILE = MODELS_DIR / "conformal_widths.json"


def refresh_space_weather():
    path = DATA_DIR / "SW-All.csv"
    if path.exists() and time.time() - path.stat().st_mtime < SW_MAX_AGE:
        return
    response = requests.get(SW_URL, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def fetch_history(st, norad_id):
    """Every orbit of the object from the last HISTORY_DAYS."""
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    text = st.get(f"query/class/gp_history/NORAD_CAT_ID/{norad_id}/EPOCH/>{start}"
                  f"/predicates/{FIELDS}/format/csv")
    history = pd.read_csv(io.StringIO(text), parse_dates=["EPOCH"]) if text.strip() else pd.DataFrame()
    return history.sort_values("EPOCH", ignore_index=True) if len(history) else history


def features(history):
    df = add_trends(history.sort_values("EPOCH", ignore_index=True))
    df = add_space_weather(df)
    df = add_catalog_info(df)
    ra, rp = df["APOAPSIS"] + R_EARTH, df["PERIAPSIS"] + R_EARTH
    df["eccentricity"] = (ra - rp) / (ra + rp)
    df["is_debris"] = (df["OBJECT_TYPE"] == "DEB").astype(int)
    df["is_rocket_body"] = (df["OBJECT_TYPE"] == "R/B").astype(int)
    return prepare(df)


def conformal_range(days, row):
    """90% range (low, high) from uncertainty.py's calibration, if it has been run."""
    if not CONFORMAL_FILE.exists():
        return None
    saved = json.loads(CONFORMAL_FILE.read_text())
    band = int(np.digitize(days, saved["bands"][1:-1]))
    factor = 10 ** saved["widths"][int(group_of(row)[0])][band]
    return days / factor, days * factor


def predict(st, norad_id):
    """Dict with the latest orbit and the predictions, or None if Space-Track has no recent orbits."""
    refresh_space_weather()
    history = fetch_history(st, norad_id)
    if len(history) < 2:
        return None
    every = features(history)
    # As in training: B and the hybrid's inputs come from one orbit a day (the last of each day).
    df = every.groupby(every["EPOCH"].dt.floor("D")).tail(1).reset_index(drop=True)
    atm = MsisAtmosphere()
    df["B_fit"], df["n_fit"] = fit_drag_parameter(df, unit_decay_rate(df, atm))
    latest = df.iloc[[-1]].copy()
    physics = forecast(latest, latest["B_fit"].to_numpy(), atm, max_days=MAX_PHYSICS_DAYS)[0]

    result = {"epoch": latest["EPOCH"].iloc[0], "perigee": latest["PERIAPSIS"].iloc[0],
              "apogee": latest["APOAPSIS"].iloc[0], "B": latest["B_fit"].iloc[0],
              "perigee_rate": latest["perigee_rate_30d"].iloc[0], "physics_days": physics,
              "hybrid_days": None, "range": None, "short_window": False,
              "is_payload": latest["OBJECT_TYPE"].iloc[0] == "PAY"}
    if np.isfinite(physics) and physics < SHORT_WINDOW_BELOW:
        B_short = fit_drag_parameter(every, unit_decay_rate(every, atm), SHORT_WINDOW)[0][-1]
        if np.isfinite(B_short) and B_short > 0:
            result["physics_days"] = forecast(every.iloc[[-1]], np.array([B_short]), atm)[0]
            result["short_window"] = True
        return result
    # The hybrid only learned from objects in their final year, so only trust it inside that range.
    if not np.isfinite(physics) or physics > 2 * 365:
        return result
    latest["log_physics_days"] = np.log10(physics)
    latest["log_B"] = np.log10(latest["B_fit"].where(latest["B_fit"] > 0))
    hybrid = xgb.XGBRegressor()
    hybrid.load_model(MODELS_DIR / "reentry_hybrid.json")
    days = physics * 10 ** hybrid.predict(latest[HYBRID_FEATURES])[0]
    result["hybrid_days"] = days
    result["range"] = conformal_range(days, latest)
    return result


def describe(result):
    """Lines of text for the lookup tool."""
    if result is None:
        return ["No orbits from the last 45 days on Space-Track (it may already have reentered)."]
    lines = [f"Latest orbit {result['epoch']:%Y-%m-%d %H:%M} UTC: "
             f"{result['perigee']:.0f} x {result['apogee']:.0f} km"]
    physics = result["physics_days"]
    if not np.isfinite(result["B"]) or result["B"] <= 0:
        lines.append("Not sinking over the last month (it may be holding its altitude with thrusters),")
        lines.append("so drag alone can't say when it will come down.")
        return lines
    if not np.isfinite(physics):
        lines.append(f"Physics model: more than {MAX_PHYSICS_DAYS / 365.25:.0f} years at today's solar activity.")
        return lines
    # Within a few days, the time of day matters; further out, the date is enough.
    when = "%Y-%m-%d %H:%M UTC" if physics < 3 else "%Y-%m-%d"
    end = result["epoch"] + pd.Timedelta(days=physics)
    lines.append(f"Physics model: about {_duration(physics)} (around {end:{when}})")
    if result.get("short_window"):
        lines.append(f"(drag measured over the last {SHORT_WINDOW} days of tracking: our most accurate")
        lines.append(" method this close to reentry; the official forecasts are better in the final day)")
    if result["hybrid_days"] is not None:
        end = result["epoch"] + pd.Timedelta(days=result["hybrid_days"])
        lines.append(f"Hybrid model:  about {_duration(result['hybrid_days'])} (around {end:{when}})")
        if result["range"] is not None:
            lo, hi = (result["epoch"] + pd.Timedelta(days=d) for d in result["range"])
            lines.append(f"90% range:     {lo:{when}} to {hi:{when}}")
    elif physics > 2 * 365:
        lines.append("More than about two years away, beyond what the hybrid model was trained on.")
    if result.get("is_payload") and not result.get("short_window"):
        lines.append("Note: if this satellite still uses thrusters to hold its orbit, it will stay up longer.")
    return lines


def _duration(days):
    if days < 2:
        return f"{days * 24:.0f} hours"
    if days < 90:
        return f"{days:.0f} days"
    if days < 730:
        return f"{days / 30.44:.0f} months"
    return f"{days / 365.25:.1f} years"


def main():
    from spacetrack import SpaceTrack

    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        raise SystemExit("usage: python src/predict_reentry.py NORAD_ID")
    st = SpaceTrack()
    try:
        for line in describe(predict(st, int(sys.argv[1]))):
            print(line)
    finally:
        st.close()


if __name__ == "__main__":
    main()
