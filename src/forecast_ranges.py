"""90% ranges for the website's live physics forecasts (live_forecasts.py).

The live forecasts use the physics model only (the trained hybrid stays private,
see the README), on orbits collected from CelesTrak. An object is often first
seen only weeks before it reenters, so B is fitted over whatever history exists.
Each kind of forecast gets its own Mondrian conformal widths (uncertainty.py),
calibrated on the latest RECALIBRATION_DAYS of reentries:

- "7", "14", "30": B fitted over that many days of one-orbit-a-day history, from
  the physics backtest (physics_model.py). With the weather held fixed, lifetime
  scales as 1/B, so a 7-day prediction is the 30-day one times B30 / B7
  (EXPERIMENTS.md 014).
- "8-day": within 10 days of reentry, B fitted over the last 8 days of every
  orbit (EXPERIMENTS.md 015), from download_final_orbits.py's orbits thinned to
  one every 6 hours, as often as the live job collects them.

A fitted B above MAX_B means something other than drag is moving the orbit
(EXPERIMENTS.md 021), so those forecasts are left out here as they are live.

The truth is the catalog decay date at midday, as everywhere else in this
project and as the live forecasts are scored. Only the widths, a few aggregate
numbers, are saved, to results/live_forecast_widths.json.

    python src/forecast_ranges.py
"""

import json

import numpy as np
import pandas as pd

from atmosphere import MsisAtmosphere
from build_reentry_dataset import DATA_DIR
from physics_model import fit_drag_parameter, forecast, unit_decay_rate
from predict_reentry import SHORT_WINDOW, SHORT_WINDOW_BELOW, features
from train_reentry_model import DATA_FILE, PHYSICS_FILE, RESULTS_DIR
from uncertainty import BANDS, MIN_CELL, RECALIBRATION_DAYS, band_of, conformal_quantile, group_of

OUT_FILE = RESULTS_DIR / "live_forecast_widths.json"
GROUPS = ["debris and rocket bodies", "other payloads", "Starlink"]
DAILY_WINDOWS = (7, 14, 30)
MAX_B = 1.0  # m^2/kg: higher fits aren't drag (EXPERIMENTS.md 021)
COLLECT_EVERY = "6h"


def widths_table(pred, truth, groups):
    """Conformal quantile of |log10 error| for each (group, band), falling back to the band across groups."""
    ok = np.isfinite(pred) & (pred > 0) & (truth > 0)
    pred, truth, groups = pred[ok], truth[ok], groups[ok]
    scores = np.abs(np.log10(truth / pred))
    bands = band_of(pred)
    table, counts = [], []
    for group in range(len(GROUPS)):
        row, row_counts = [], []
        for band in range(len(BANDS) - 1):
            cell = scores[(bands == band) & (groups == group)]
            if len(cell) < MIN_CELL:
                cell = scores[bands == band]
            row.append(round(float(conformal_quantile(cell)), 4) if len(cell) else None)
            row_counts.append(int(len(cell)))
        table.append(row)
        counts.append(row_counts)
    return table, counts


def recent(df):
    return df[df["DECAY_DATE"] >= df["DECAY_DATE"].max() - pd.Timedelta(days=RECALIBRATION_DAYS)]


def daily_widths():
    cols = ["NORAD_CAT_ID", "EPOCH", "DECAY_DATE", "days_left", "OBJECT_TYPE", "is_starlink"]
    df = pd.read_parquet(DATA_FILE, columns=cols).merge(pd.read_parquet(PHYSICS_FILE), on=["NORAD_CAT_ID", "EPOCH"])
    df = recent(df)
    groups, truth = group_of(df), df["days_left"].to_numpy()
    out = {}
    for window in DAILY_WINDOWS:
        B_w = df["B_fit"] if window == 30 else df[f"B_fit_{window}d"]
        pred = (df["physics_days"] * df["B_fit"] / B_w.where((B_w > 0) & (B_w <= MAX_B))).to_numpy()
        out[str(window)], counts = widths_table(pred, truth, groups)
        print(f"{window}-day B: {df['NORAD_CAT_ID'].nunique():,} objects, rows per band {counts[0]}")
    return out, df["DECAY_DATE"].min(), df["DECAY_DATE"].max()


def short_widths(atm):
    from download_final_orbits import OUT_FILE as FINAL_ORBITS_FILE

    orbits = pd.read_parquet(FINAL_ORBITS_FILE)
    orbits = orbits.groupby([orbits["NORAD_CAT_ID"], orbits["EPOCH"].dt.floor(COLLECT_EVERY)]).tail(1)
    every = features(orbits)
    decay = pd.read_csv(DATA_DIR / "satcat.csv", usecols=["NORAD_CAT_ID", "DECAY_DATE"], parse_dates=["DECAY_DATE"])
    every = every.merge(decay, on="NORAD_CAT_ID", how="inner").dropna(subset=["DECAY_DATE"])
    every = recent(every).reset_index(drop=True)
    B = fit_drag_parameter(every, unit_decay_rate(every, atm), SHORT_WINDOW)[0]
    B[B > MAX_B] = np.nan
    pred = forecast(every, B, atm, max_days=4 * SHORT_WINDOW_BELOW)
    truth = (every["DECAY_DATE"] + pd.Timedelta(hours=12) - every["EPOCH"]).dt.total_seconds().to_numpy() / 86400
    near = np.isfinite(pred) & (pred < SHORT_WINDOW_BELOW)
    table, counts = widths_table(pred[near], truth[near], group_of(every)[near])
    print(f"{SHORT_WINDOW}-day B, every {COLLECT_EVERY}: {every['NORAD_CAT_ID'].nunique():,} objects, "
          f"rows per band {counts[0][:2]}")
    return table


def main():
    atm = MsisAtmosphere()
    widths, first, last = daily_widths()
    widths[f"{SHORT_WINDOW}-day"] = short_widths(atm)
    OUT_FILE.write_text(json.dumps({
        "coverage": 0.9, "bands": BANDS[:-1] + [1e9], "groups": GROUPS,
        "calibrated_on": f"reentries from {first:%Y-%m-%d} to {last:%Y-%m-%d}",
        "note": "log10 widths: the range is the prediction divided and multiplied by 10^width",
        "widths": widths,
    }, indent=1) + "\n")
    for name, table in widths.items():
        for group, row in zip(GROUPS, table):
            factors = ", ".join("—" if w is None else f"{10 ** w:.2f}x" for w in row)
            print(f"{name:>6} {group:>24}: {factors}")
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
