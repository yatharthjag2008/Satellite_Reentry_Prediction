"""Part 1: predict an object's whole lifetime from its first weeks in orbit.

For every satellite and rocket body launched since 2000 that has reentered:
1. fit the drag parameter B from its orbits 10-40 days after launch
2. simulate forward from its orbit on about day 40 until it reenters, under three
   assumptions about the Sun: today's flux held fixed, the solar-cycle forecast
   (solar_cycle.py), and the flux that actually happened (an upper bound on what
   a perfect forecast could do)

Saves one row per object to data/spacetrack/lifetime_predictions.parquet and
prints the median error by lifetime and object type.

Run download_early_orbits.py first.

    python src/lifetime_model.py
"""

import time

import numpy as np
import pandas as pd

from atmosphere import MsisAtmosphere
from build_reentry_dataset import DATA_DIR, R_EARTH, add_space_weather
from download_early_orbits import OUT_FILE as EARLY_FILE
from orbit_decay import propagate
from physics_model import fit_drag_parameter, prepare, unit_decay_rate
from solar_cycle import forecast_monthly

OUT_FILE = DATA_DIR / "spacetrack" / "lifetime_predictions.parquet"
FIT_FROM, FIT_TO = 10, 40  # days after launch: deployment and early maneuvers are mostly over by day 10
MAX_YEARS = 30
MAX_MONTHS = MAX_YEARS * 12 + 2
LIFETIME_BANDS = [(0, 365, "under 1 year"), (365, 3 * 365, "1-3 years"), (3 * 365, 10 * 365, "3-10 years"),
                  (10 * 365, np.inf, "10+ years")]


def early_table():
    """One row per object: its orbit on about day 40, the fitted B and the true days left."""
    satcat = pd.read_csv(DATA_DIR / "satcat.csv", parse_dates=["LAUNCH_DATE", "DECAY_DATE"],
                         usecols=["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "LAUNCH_DATE", "DECAY_DATE", "RCS"])
    df = pd.read_parquet(EARLY_FILE).merge(satcat, on="NORAD_CAT_ID")
    since_launch = (df["EPOCH"] - df["LAUNCH_DATE"]).dt.total_seconds() / 86400
    bad = (df["PERIAPSIS"] < 80) | (df["APOAPSIS"] > 100_000) | (df["APOAPSIS"] < df["PERIAPSIS"])
    df = df[(since_launch >= FIT_FROM) & (since_launch < FIT_TO) & ~bad]
    df = df.sort_values("EPOCH").groupby(["NORAD_CAT_ID", df["EPOCH"].dt.floor("D")]).tail(1)  # one a day
    df = add_space_weather(df.sort_values("EPOCH", ignore_index=True))
    ra, rp = df["APOAPSIS"] + R_EARTH, df["PERIAPSIS"] + R_EARTH
    df["eccentricity"] = (ra - rp) / (ra + rp)
    df = prepare(df)

    atm = MsisAtmosphere()
    df["B_fit"], df["n_fit"] = fit_drag_parameter(df, unit_decay_rate(df, atm), window=FIT_TO - FIT_FROM)
    last = df.groupby("NORAD_CAT_ID").tail(1).reset_index(drop=True)
    last["days_left"] = (last["DECAY_DATE"] + pd.Timedelta(hours=12) - last["EPOCH"]).dt.total_seconds() / 86400
    last["is_starlink"] = last["OBJECT_NAME"].str.startswith("STARLINK", na=False).astype(int)
    return last[last["days_left"] > 0].reset_index(drop=True)


def monthly_weather():
    """Monthly trailing-81-day flux and mean Ap, as recorded (the 'perfect forecast')."""
    sw = pd.read_csv(DATA_DIR / "SW-All.csv", parse_dates=["DATE"]).set_index("DATE")
    return sw["F10.7_OBS_LAST81"].resample("MS").mean(), sw["AP_AVG"].resample("MS").mean()


def weather_paths(table):
    """(flux, ap) arrays of shape (objects, months) for each assumption about the future Sun."""
    flux_rec, ap_rec = monthly_weather()
    n = len(table)
    start = table["EPOCH"].dt.to_period("M").dt.start_time
    # Typical Ap: the average over the 11 years before each prediction, known at the time.
    ap_typical = np.array([ap_rec[d - pd.DateOffset(years=11):d - pd.DateOffset(days=1)].mean() for d in start])

    fixed = np.repeat(table["f107_81d"].to_numpy()[:, None], MAX_MONTHS, axis=1)
    forecasts = {d: forecast_monthly(d, MAX_MONTHS) for d in start.unique()}
    cycle = np.stack([forecasts[d] for d in start])
    actual = np.stack([flux_rec[d:].to_numpy()[:MAX_MONTHS] for d in start]) if n else np.empty((0, MAX_MONTHS))
    actual_ap = np.stack([ap_rec[d:].to_numpy()[:MAX_MONTHS] for d in start]) if n else np.empty((0, MAX_MONTHS))
    typical = np.repeat(ap_typical[:, None], MAX_MONTHS, axis=1)
    return {"today's flux held fixed": (fixed, typical),
            "solar-cycle forecast": (cycle, typical),
            "actual flux (perfect forecast)": (actual, actual_ap)}


def simulate(table, flux, ap, atm):
    """Days until reentry under a month-by-month flux and Ap path for each object."""
    def density(h, idx, t):
        month = np.clip((t / 30.44).astype(int), 0, flux.shape[1] - 1)
        return atm.density(h, flux[idx[:, None], month], ap[idx[:, None], month])

    ok = (table["B_fit"] > 0).to_numpy()
    days = np.full(len(table), np.nan)
    rows = np.flatnonzero(ok)
    days[rows] = propagate(table["a_km"].to_numpy()[rows], table["eccentricity"].to_numpy()[rows],
                           table["B_fit"].to_numpy()[rows],
                           lambda h, idx, t: density(h, rows[idx], t), max_days=MAX_YEARS * 365.25)
    return days


def error_by_band(table, predictions):
    truth = table["days_left"].to_numpy()
    rows = {}
    for name, pred in predictions.items():
        err = np.abs(np.where(np.isfinite(pred), pred, MAX_YEARS * 365.25) / truth - 1)
        row = {label: np.median(err[m]) if (m := (truth >= lo) & (truth < hi)).any() else np.nan
               for lo, hi, label in LIFETIME_BANDS}
        row["overall"], row["within 25%"] = np.median(err), (err <= 0.25).mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def main():
    start = time.time()
    table = early_table()
    print(f"{len(table):,} objects; B fitted for {(table['B_fit'] > 0).mean():.0%}  [{time.time() - start:.0f} s]")
    atm = MsisAtmosphere()
    predictions = {}
    for name, (flux, ap) in weather_paths(table).items():
        predictions[name] = simulate(table, flux, ap, atm)
        table[name] = predictions[name]
        print(f"simulated: {name}  [{time.time() - start:.0f} s]")
    table.to_parquet(OUT_FILE, index=False)

    usable = (table["B_fit"] > 0).to_numpy()
    groups = {"all with a fitted B": usable,
              "rocket bodies": usable & (table["OBJECT_TYPE"] == "R/B").to_numpy(),
              "satellites (not Starlink)": usable & (table["OBJECT_TYPE"] == "PAY").to_numpy()
              & (table["is_starlink"] == 0).to_numpy()}
    for group, mask in groups.items():
        print(f"\n{group} ({mask.sum():,} objects): median error by true remaining lifetime")
        print(error_by_band(table[mask], {k: v[mask] for k, v in predictions.items()})
              .to_string(float_format="{:.0%}".format))


if __name__ == "__main__":
    main()
