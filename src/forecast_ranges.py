"""90% ranges for the physics forecasts: the website's (live_forecasts.py) and the
lookup tool's in the final days (predict_reentry.py).

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
- "8-day, every orbit": the same with every orbit, as the lookup tool gets them
  from Space-Track, scored against the reentry time of the final official (TIP)
  message rather than the catalog date (EXPERIMENTS.md 023).
- "transfer": transfer orbits simulated with the Moon's and Sun's gravity too
  (EXPERIMENTS.md 026), from the weekly physics rows with their angles
  (download_final_year_angles.py). Only about 30 transfer orbits reenter a year,
  so these use the latest TRANSFER_YEARS years of reentries.

A fitted B above MAX_B means something other than drag is moving the orbit
(EXPERIMENTS.md 021), so those forecasts are left out here as they are live.

Except for "8-day, every orbit", the truth is the catalog decay date at midday,
as everywhere else in this project and as the live forecasts are scored. Only the widths, a few aggregate
numbers, are saved, to results/live_forecast_widths.json.

    python src/forecast_ranges.py
"""

import json

import numpy as np
import pandas as pd

from atmosphere import MsisAtmosphere
from build_reentry_dataset import DATA_DIR
from physics_model import MAX_B, fit_drag_parameter, forecast, plausible, semi_major_axis, unit_decay_rate
from predict_reentry import RANGES_FILE as OUT_FILE, SHORT_WINDOW, SHORT_WINDOW_BELOW, features
from train_reentry_model import DATA_FILE, PHYSICS_FILE
from uncertainty import BANDS, MIN_CELL, RECALIBRATION_DAYS, band_of, conformal_quantile, group_of

GROUPS = ["debris and rocket bodies", "other payloads", "Starlink"]
DAILY_WINDOWS = (7, 14, 30)
TRANSFER_YEARS = 3
TRANSFER_MAX_DAYS = 365.0  # as far ahead as the live forecasts go
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
    pred = forecast(every, B, atm, max_days=4 * SHORT_WINDOW_BELOW)
    truth = (every["DECAY_DATE"] + pd.Timedelta(hours=12) - every["EPOCH"]).dt.total_seconds().to_numpy() / 86400
    near = np.isfinite(pred) & (pred < SHORT_WINDOW_BELOW)
    table, counts = widths_table(pred[near], truth[near], group_of(every)[near])
    print(f"{SHORT_WINDOW}-day B, every {COLLECT_EVERY}: {every['NORAD_CAT_ID'].nunique():,} objects, "
          f"rows per band {counts[0][:2]}")
    return table


def transfer_widths(atm):
    """Widths for the Sun-and-Moon physics on transfer orbits, or None without their angles."""
    from download_final_year_angles import OUT_FILE as ANGLES_FILE
    from lunisolar import days_from_rows

    if not ANGLES_FILE.exists():
        return None
    cols = ["NORAD_CAT_ID", "EPOCH", "DECAY_DATE", "days_left", "OBJECT_TYPE", "is_starlink", "MEAN_MOTION",
            "eccentricity", "INCLINATION", "f107_81d", "ap_7d"]
    df = (pd.read_parquet(DATA_FILE, columns=cols).merge(pd.read_parquet(PHYSICS_FILE), on=["NORAD_CAT_ID", "EPOCH"])
          .merge(pd.read_parquet(ANGLES_FILE), on=["NORAD_CAT_ID", "EPOCH"]))
    df = df[plausible(df["B_fit"].to_numpy())]
    df = df[df["DECAY_DATE"] >= df["DECAY_DATE"].max() - pd.Timedelta(days=365 * TRANSFER_YEARS)].reset_index(drop=True)
    df["a_km"] = semi_major_axis(df["MEAN_MOTION"])
    profile = atm.profile(df["f107_81d"].to_numpy(), df["ap_7d"].to_numpy())
    pred = days_from_rows(df, df["B_fit"].to_numpy(), lambda h, idx, t: profile(h, idx), max_days=TRANSFER_MAX_DAYS)
    table, counts = widths_table(pred, df["days_left"].to_numpy(), group_of(df))
    print(f"transfer, Sun and Moon: {df['NORAD_CAT_ID'].nunique():,} objects, rows per band {counts[0]}")
    return table


def final_reentry_times():
    """Each object's reentry time from its final TIP message, issued after reentry (compare_official.py)."""
    from download_tip import OUT_FILE as TIP_FILE

    tip = pd.read_csv(TIP_FILE, parse_dates=["MSG_EPOCH", "DECAY_EPOCH"]).sort_values(["NORAD_CAT_ID", "MSG_EPOCH"])
    return tip.groupby("NORAD_CAT_ID")["DECAY_EPOCH"].last().rename("reentry")


def fresh_predictions(atm):
    """From every orbit in the final 40 days: the 8-day-B prediction and the true days left, where under 10 days.

    Orbits within an hour of reentry are left out, as in compare_official.py.
    """
    from download_final_orbits import OUT_FILE as FINAL_ORBITS_FILE

    every = features(pd.read_parquet(FINAL_ORBITS_FILE))
    every["B_short"] = fit_drag_parameter(every, unit_decay_rate(every, atm), SHORT_WINDOW)[0]
    every = every.join(final_reentry_times(), on="NORAD_CAT_ID").dropna(subset=["reentry"]).reset_index(drop=True)
    every["pred"] = forecast(every, every["B_short"].to_numpy(), atm, max_days=4 * SHORT_WINDOW_BELOW)
    every["truth"] = (every["reentry"] - every["EPOCH"]).dt.total_seconds() / 86400
    every["group"] = group_of(every)
    near = np.isfinite(every["pred"]) & (every["pred"] < SHORT_WINDOW_BELOW) & (every["truth"] > 1 / 24)
    return every[near].reset_index(drop=True)


def fresh_widths(atm):
    rows = fresh_predictions(atm)
    rows = rows[rows["reentry"] >= rows["reentry"].max() - pd.Timedelta(days=RECALIBRATION_DAYS)]
    table, counts = widths_table(rows["pred"].to_numpy(), rows["truth"].to_numpy(), rows["group"].to_numpy())
    print(f"{SHORT_WINDOW}-day B, every orbit: {rows['NORAD_CAT_ID'].nunique():,} objects, rows per band {counts[0][:2]}")
    return table


def main():
    atm = MsisAtmosphere()
    widths, first, last = daily_widths()
    widths[f"{SHORT_WINDOW}-day"] = short_widths(atm)
    widths[f"{SHORT_WINDOW}-day, every orbit"] = fresh_widths(atm)
    transfer = transfer_widths(atm)
    if transfer is not None:
        widths["transfer"] = transfer
    OUT_FILE.write_text(json.dumps({
        "coverage": 0.9, "bands": BANDS[:-1] + [1e9], "groups": GROUPS,
        "calibrated_on": f"reentries from {first:%Y-%m-%d} to {last:%Y-%m-%d}",
        "note": "log10 widths: the range is the prediction divided and multiplied by 10^width",
        "widths": widths,
    }, indent=1) + "\n")
    for name, table in widths.items():
        for group, row in zip(GROUPS, table):
            factors = ", ".join("—" if w is None else f"{10 ** w:.2f}x" for w in row)
            print(f"{name:>18} {group:>24}: {factors}")
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
