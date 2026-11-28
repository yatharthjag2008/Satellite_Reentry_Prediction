"""Predict when one object still in orbit will reenter.

Downloads the object's last HISTORY_DAYS of orbits from Space-Track and fresh
space weather from CelesTrak, builds the same features as the training table,
and runs the physics model and the physics + ML hybrid. A transfer orbit
(apogee above 5,000 km) is simulated with the Moon's and Sun's gravity as well
(lunisolar.py, EXPERIMENTS.md 025). Used by lookup_satellite.py; can also be run
on its own:

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
from ground_track import latitude_band
from build_reentry_dataset import DATA_DIR, R_EARTH, add_catalog_info, add_space_weather, add_trends
from download_history import FIELDS
from physics_model import MAX_B, fit_drag_parameter, forecast, plausible, prepare, unit_decay_rate
from train_reentry_model import HYBRID_FEATURES, MODELS_DIR, NO_PHYSICS_DAYS, RESULTS_DIR
from uncertainty import group_of

HISTORY_DAYS = 45
SW_URL = "https://celestrak.org/SpaceData/SW-All.csv"
SW_MAX_AGE = 86400  # re-download space weather once a day
MAX_PHYSICS_DAYS = 365.25 * 30
# Close to reentry, B fitted over the last few days of every orbit is far more accurate (EXPERIMENTS.md 015).
SHORT_WINDOW = 8
SHORT_WINDOW_BELOW = 10  # days: switch when the usual prediction is closer than this
ANGLE_FIELDS = ",RA_OF_ASC_NODE,ARG_OF_PERICENTER"  # where a transfer orbit faces the Sun and Moon
CONFORMAL_FILE = MODELS_DIR / "conformal_widths.json"
RANGES_FILE = RESULTS_DIR / "live_forecast_widths.json"  # physics ranges from forecast_ranges.py


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
                  f"/predicates/{FIELDS}{ANGLE_FIELDS}/format/csv")
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


def short_window_range(days, row):
    """90% range (low, high) for the 8-day-B physics prediction from every orbit (EXPERIMENTS.md 023)."""
    saved = json.loads(RANGES_FILE.read_text())
    widths = saved["widths"].get(f"{SHORT_WINDOW}-day, every orbit")
    width = widths[int(group_of(row)[0])][int(np.digitize(days, saved["bands"][1:-1]))] if widths else None
    return None if width is None else (days / 10 ** width, days * 10 ** width)


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
    lunisolar = is_transfer_orbit(latest) and plausible(latest["B_fit"].iloc[0])
    if lunisolar:
        # The Sun and Moon move a stretched orbit's perigee, which drag alone misses (EXPERIMENTS.md 025).
        physics = years_ahead(latest, atm)

    result = {"epoch": latest["EPOCH"].iloc[0], "perigee": latest["PERIAPSIS"].iloc[0],
              "apogee": latest["APOAPSIS"].iloc[0], "B": latest["B_fit"].iloc[0],
              "inclination": latest["INCLINATION"].iloc[0],
              "perigee_rate": latest["perigee_rate_30d"].iloc[0], "physics_days": physics,
              "hybrid_days": None, "range": None, "short_window": False, "lunisolar": lunisolar,
              "is_payload": latest["OBJECT_TYPE"].iloc[0] == "PAY"}
    if np.isfinite(physics) and physics < SHORT_WINDOW_BELOW:
        B_short = fit_drag_parameter(every, unit_decay_rate(every, atm), SHORT_WINDOW)[0][-1]
        if plausible(B_short):
            result["physics_days"] = forecast(every.iloc[[-1]], np.array([B_short]), atm)[0]
            result["short_window"] = True
            if np.isfinite(result["physics_days"]):
                result["range"] = short_window_range(result["physics_days"], every.iloc[[-1]])
        return result
    if lunisolar:
        # The hybrid learned to correct drag-only physics, so it isn't used on top of this.
        result["solar_cycle"] = True
        return result
    B = latest["B_fit"].iloc[0]
    if (not np.isfinite(physics) or physics > 365) and plausible(B):
        # Over years the Sun's cycle matters: use the solar-cycle forecast instead (EXPERIMENTS.md 017).
        result["physics_days"] = physics = years_ahead(latest, atm)
        result["solar_cycle"] = True
    # The hybrid only learned from objects in their final year, so only trust it inside that range.
    # Where B is too large to be drag the physics has no answer, and the hybrid works from
    # NO_PHYSICS_DAYS, as it did in training (EXPERIMENTS.md 022).
    if B > MAX_B:
        base = NO_PHYSICS_DAYS
        latest["log_physics_days"] = np.nan
    elif not np.isfinite(physics) or physics > 2 * 365:
        return result
    else:
        base = physics
        latest["log_physics_days"] = np.log10(physics)
    latest["log_B"] = np.log10(latest["B_fit"].where(latest["B_fit"] > 0))
    hybrid = xgb.XGBRegressor()
    hybrid.load_model(MODELS_DIR / "reentry_hybrid.json")
    days = base * 10 ** hybrid.predict(latest[HYBRID_FEATURES])[0]
    result["hybrid_days"] = days
    result["range"] = conformal_range(days, latest)
    return result


def is_transfer_orbit(latest):
    """True if the orbit reaches high enough for the Sun and Moon to matter, and its angles are known."""
    from lifetime_model import uses_lunisolar
    return bool(uses_lunisolar(latest)[0])


def years_ahead(latest, atm):
    """Days until reentry under the solar-cycle forecast and typical geomagnetic activity.

    lifetime_model.simulate adds the Sun and Moon for transfer orbits.
    """
    from lifetime_model import MAX_MONTHS, monthly_weather, simulate, typical_ap
    from solar_cycle import forecast_monthly

    month = latest["EPOCH"].iloc[0].to_period("M").start_time
    flux = forecast_monthly(month, MAX_MONTHS)[None, :]
    ap = np.full_like(flux, typical_ap(monthly_weather()[1], month))
    return simulate(latest.reset_index(drop=True), flux, ap, atm)[0]


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
    if result["B"] > MAX_B:
        lines.append("Physics model: no answer. Over the last month the orbit changed faster than drag can")
        lines.append("explain (thrusters, or the Sun's and Moon's gravity reshaping a stretched orbit).")
    elif not np.isfinite(physics):
        lines.append(f"Physics model: more than {MAX_PHYSICS_DAYS / 365.25:.0f} years at today's solar activity.")
        return lines
    # Within a few days, the time of day matters; further out, the date is enough.
    soonest = physics if np.isfinite(physics) else result["hybrid_days"] or np.inf
    when = "%Y-%m-%d %H:%M UTC" if soonest < 3 else "%Y-%m-%d"
    if np.isfinite(physics):
        end = result["epoch"] + pd.Timedelta(days=physics)
        lines.append(f"Physics model: about {_duration(physics)} (around {end:{when}})")
    if result.get("lunisolar") and not result.get("short_window"):
        lines.append("(including the Moon's and Sun's gravity, which slowly move the low point of a")
        lines.append(" stretched orbit like this one; tested to a median error of 28% on lifetimes)")
    elif result.get("solar_cycle"):
        lines.append("(with a forecast of the Sun's 11-year cycle; a rough guide this far ahead)")
    if result.get("short_window"):
        lines.append(f"(drag measured over the last {SHORT_WINDOW} days of tracking: our most accurate")
        lines.append(" method this close to reentry; the official forecasts are better in the final day)")
    if result["hybrid_days"] is not None or result["range"] is not None:
        if result["hybrid_days"] is not None:
            end = result["epoch"] + pd.Timedelta(days=result["hybrid_days"])
            lines.append(f"Hybrid model:  about {_duration(result['hybrid_days'])} (around {end:{when}})")
        if result["range"] is not None:
            lo, hi = (result["epoch"] + pd.Timedelta(days=d) for d in result["range"])
            lines.append(f"90% range:     {lo:{when}} to {hi:{when}}")
    elif result.get("lunisolar") and not result.get("short_window"):
        lines.append("No hybrid model here: it learned to correct drag alone, which misses the Moon and Sun.")
    elif np.isfinite(physics) and physics > 2 * 365:
        lines.append("More than about two years away, beyond what the hybrid model was trained on.")
    if result.get("inclination") is not None and (np.isfinite(physics) or result["hybrid_days"] is not None):
        band = latitude_band(result["inclination"])
        lines.append(f"Where: anywhere between {band:.0f}°S and {band:.0f}°N, the latitudes its orbit passes over")
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
