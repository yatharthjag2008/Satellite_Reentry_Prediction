"""Build the Part 2 training table: one row per object per day in its final year.

Reads the orbit histories saved by download_history.py and adds:
- how fast the orbit is shrinking (perigee and apogee change over the past 7 and 30 days)
- space weather known on that day (solar flux and geomagnetic activity, from the day before)
- object type and size from the catalog
- the target: days until reentry

The result is derived from Space-Track data, so it's saved in the git-ignored data/spacetrack/.

    python src/build_reentry_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_DIR = DATA_DIR / "spacetrack" / "reentry_history"
OUT_FILE = DATA_DIR / "spacetrack" / "reentry_dataset.csv.gz"
R_EARTH = 6378.137  # km


def load_history():
    files = sorted(HISTORY_DIR.glob("decayed_*.csv.gz"))
    if not files:
        raise SystemExit(f"No orbit histories in {HISTORY_DIR}. Run src/download_history.py first.")
    history = pd.concat([pd.read_csv(f, parse_dates=["EPOCH", "DECAY_DATE"]) for f in files], ignore_index=True)
    print(f"{len(files)} months, {history['NORAD_CAT_ID'].nunique():,} objects, {len(history):,} daily orbits")

    # A few archived orbits are physically impossible (perigee below 80 km, apogee past the Moon).
    bad = (history["PERIAPSIS"] < 80) | (history["APOAPSIS"] > 100_000) | (history["APOAPSIS"] < history["PERIAPSIS"])
    print(f"dropping {bad.sum():,} impossible orbits")
    history = history[~bad]
    return history.drop_duplicates(["NORAD_CAT_ID", "EPOCH"]).sort_values("EPOCH", ignore_index=True)


def add_trends(df):
    """Perigee and apogee change (km/day) since the orbit about 7 and 30 days earlier."""
    for days in (7, 30):
        past = df[["NORAD_CAT_ID", "EPOCH", "PERIAPSIS", "APOAPSIS"]].rename(
            columns={"PERIAPSIS": "perigee_then", "APOAPSIS": "apogee_then"})
        past["then"] = past["EPOCH"]
        past["EPOCH"] = past["EPOCH"] + pd.Timedelta(days=days)
        # The latest orbit at least `days` ago, but no more than 1.5x that.
        match = pd.merge_asof(df[["NORAD_CAT_ID", "EPOCH"]], past, on="EPOCH", by="NORAD_CAT_ID",
                              direction="backward", tolerance=pd.Timedelta(days=days / 2))
        elapsed = (df["EPOCH"] - match["then"]).dt.total_seconds() / 86400
        df[f"perigee_rate_{days}d"] = (df["PERIAPSIS"] - match["perigee_then"]) / elapsed
        df[f"apogee_rate_{days}d"] = (df["APOAPSIS"] - match["apogee_then"]) / elapsed
    return df


def add_space_weather(df):
    """Solar and geomagnetic activity from the day before each orbit, so nothing comes from the future."""
    sw = pd.read_csv(DATA_DIR / "SW-All.csv", parse_dates=["DATE"])
    sw = sw[["DATE", "F10.7_OBS", "F10.7_OBS_LAST81", "AP_AVG"]].rename(
        columns={"F10.7_OBS": "f107", "F10.7_OBS_LAST81": "f107_81d", "AP_AVG": "ap"})
    sw["ap_7d"] = sw["ap"].rolling(7, min_periods=1).mean()
    sw["EPOCH"] = sw.pop("DATE") + pd.Timedelta(days=1)  # a day's values are complete by the next day
    return pd.merge_asof(df, sw.dropna(subset=["f107"]), on="EPOCH", direction="backward")


def add_catalog_info(df):
    satcat = pd.read_csv(DATA_DIR / "satcat.csv", usecols=["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "RCS"])
    df = df.merge(satcat, on="NORAD_CAT_ID", how="left")
    df["is_starlink"] = df["OBJECT_NAME"].str.startswith("STARLINK", na=False).astype(int)
    return df


def main():
    df = load_history()
    df = add_trends(df)
    df = add_space_weather(df)
    df = add_catalog_info(df)

    ra, rp = df["APOAPSIS"] + R_EARTH, df["PERIAPSIS"] + R_EARTH
    df["eccentricity"] = (ra - rp) / (ra + rp)
    # DECAY_DATE is only a date, so assume reentry at midday.
    df["days_left"] = (df["DECAY_DATE"] + pd.Timedelta(hours=12) - df["EPOCH"]).dt.total_seconds() / 86400
    df = df[df["days_left"] > 0]

    cols = ["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "DECAY_DATE", "EPOCH",
            "PERIAPSIS", "APOAPSIS", "eccentricity", "INCLINATION", "MEAN_MOTION", "MEAN_MOTION_DOT", "BSTAR",
            "perigee_rate_7d", "apogee_rate_7d", "perigee_rate_30d", "apogee_rate_30d",
            "f107", "f107_81d", "ap", "ap_7d", "RCS", "is_starlink", "days_left"]
    df[cols].to_csv(OUT_FILE, index=False, float_format="%.6g", compression="gzip")
    print(f"saved {len(df):,} rows to {OUT_FILE}")
    print(df[cols].describe().T[["count", "mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
