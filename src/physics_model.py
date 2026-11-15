"""Physics-based predictions of days until reentry (docs/derivation.md).

For each object and day in the training table:
1. fit the drag parameter B from the orbit's decay over the past FIT_WINDOW days
2. simulate the orbit forward with NRLMSIS density, holding the current space
   weather fixed, until it reenters

Predictions are made for one day a week per object (simulating all 3.7 million
rows would take hours) and saved to data/spacetrack/physics_predictions.parquet.

    python src/physics_model.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from atmosphere import MsisAtmosphere
from orbit_decay import MU, drag_rates, propagate

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "spacetrack" / "reentry_dataset.parquet"
OUT_FILE = ROOT / "data" / "spacetrack" / "physics_predictions.parquet"

FIT_WINDOW = 30   # days of orbit history used to fit B
OTHER_WINDOWS = (7, 14, 60)  # also saved: how B differs over shorter and longer windows shows if drag is changing
MIN_FIT_POINTS = 5
EVERY = 7         # predict from one orbit a week per object
MAX_DAYS = 3650.0


def semi_major_axis(mean_motion):
    """Semi-major axis (km) from mean motion (revolutions per day), by Kepler's third law."""
    n = np.asarray(mean_motion) * 2 * np.pi / 86400
    return (MU / n**2) ** (1 / 3) / 1e3


def prepare(df):
    df = df.sort_values(["NORAD_CAT_ID", "EPOCH"], ignore_index=True)
    df["a_km"] = semi_major_axis(df["MEAN_MOTION"])
    df["doy"] = df["EPOCH"].dt.day_of_year + df["EPOCH"].dt.hour / 24
    return df


def unit_decay_rate(df, atm, batch=200_000):
    """g = da/dt (km/day) at each observed orbit if B were 1 m^2/kg, using that day's space weather."""
    g = np.empty(len(df))
    for start in range(0, len(df), batch):
        part = df.iloc[start:start + batch]
        f107a, ap, doy = (part[c].to_numpy() for c in ("f107_81d", "ap", "doy"))
        density = lambda h, rows: atm.density(h, f107a[rows, None], ap[rows, None], doy[rows, None])
        g[start:start + batch] = drag_rates(part["a_km"].to_numpy(), part["eccentricity"].to_numpy(), 1.0, density)[0]
    return g


def fit_drag_parameter(df, g, window=FIT_WINDOW):
    """Least-squares slope of a against G = integral of g dt over the past `window` days, for every row.

    Returns B (m^2/kg) and the number of orbits in each fit (docs/derivation.md, section 6).
    """
    obj = df["NORAD_CAT_ID"].to_numpy()
    days = (df["EPOCH"] - df["EPOCH"].min()).dt.total_seconds().to_numpy() / 86400
    new_obj = np.r_[True, obj[1:] != obj[:-1]]

    # G: trapezoid-rule integral of g along each object's history, restarting at each object.
    step = np.where(new_obj, 0.0, 0.5 * (g + np.r_[0.0, g[:-1]]) * np.diff(days, prepend=days[0]))
    G = pd.Series(step).groupby(obj).cumsum().to_numpy()
    # Offset a by each object's first value, so the running sums stay small and precise.
    a = df["a_km"].to_numpy() - df.groupby("NORAD_CAT_ID")["a_km"].transform("first").to_numpy()

    # Window sums from per-object cumulative sums: rows from `first` up to and including this one.
    key = obj * 1e5 + days  # sorts the same as (object, time)
    first = np.searchsorted(key, key - window, side="left")
    sums = {}
    for name, values in {"n": np.ones_like(a), "a": a, "G": G, "aG": a * G, "GG": G * G}.items():
        c = np.r_[0.0, np.cumsum(values)]
        sums[name] = c[np.arange(1, len(a) + 1)] - c[first]
    n = sums["n"]
    cov = sums["aG"] - sums["a"] * sums["G"] / n
    var = sums["GG"] - sums["G"] ** 2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        B = np.where((n >= MIN_FIT_POINTS) & (var > 0), cov / var, np.nan)
    return B, n.astype(int)


def forecast(df, B, atm, max_days=MAX_DAYS):
    """Days until reentry from each row's orbit, holding its 81-day flux and 7-day Ap fixed."""
    days = np.full(len(df), np.nan)
    ok = np.isfinite(B) & (B > 0)
    rows = df[ok]
    profile = atm.profile(rows["f107_81d"].to_numpy(), rows["ap_7d"].to_numpy())
    days[ok] = propagate(rows["a_km"].to_numpy(), rows["eccentricity"].to_numpy(), B[ok],
                         lambda h, idx, t: profile(h, idx), max_days=max_days)
    return days


def main():
    start = time.time()
    atm = MsisAtmosphere()
    df = prepare(pd.read_parquet(DATA_FILE))
    print(f"{len(df):,} orbits, {df['NORAD_CAT_ID'].nunique():,} objects")

    g = unit_decay_rate(df, atm)
    B, n_fit = fit_drag_parameter(df, g)
    df["B_fit"], df["n_fit"] = B, n_fit
    for window in OTHER_WINDOWS:
        df[f"B_fit_{window}d"] = fit_drag_parameter(df, g, window)[0]
    print(f"fitted B for {np.isfinite(B).mean():.0%} of rows; {(B <= 0).mean():.1%} negative (orbit raised)"
          f"  [{time.time() - start:.0f} s]")

    weekly = df[df.groupby("NORAD_CAT_ID").cumcount() % EVERY == 0].copy()
    weekly["physics_days"] = forecast(weekly, weekly["B_fit"].to_numpy(), atm)
    print(f"simulated {len(weekly):,} orbits  [{time.time() - start:.0f} s]")

    cols = ["NORAD_CAT_ID", "EPOCH", "B_fit", "n_fit", "physics_days"] + [f"B_fit_{w}d" for w in OTHER_WINDOWS]
    weekly[cols].to_parquet(OUT_FILE, index=False)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
