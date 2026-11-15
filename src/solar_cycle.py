"""Forecast solar flux years ahead from the shape of past solar cycles (for Part 1).

A satellite with years to live will see the Sun's activity rise and fall with the
~11-year cycle, so holding today's flux fixed (fine for Part 2) gives poor
lifetimes. This forecast, made on a given date using only data before it:

1. finds the past solar minima in the 13-month smoothed F10.7 flux
2. averages the completed cycles into a mean cycle shape (flux vs. months since minimum)
3. scales that shape to fit the current cycle so far (a strong cycle stays strong)
4. starts from today's 81-day flux and blends into the scaled shape over about a year

Beyond the current cycle, it assumes average cycles.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from build_reentry_dataset import DATA_DIR

QUIET_FLUX = 65.0      # F10.7 never drops far below this, even at solar minimum
CYCLE_MONTHS = 132     # average cycle length, used after the current cycle ends
SHAPE_MONTHS = 156     # how far past each minimum the mean shape is averaged
BLEND_MONTHS = 12.0    # how quickly the forecast moves from today's flux to the cycle shape
CONFIRM_MONTHS = 12    # a minimum only counts once the flux has been rising this long


@lru_cache(maxsize=1)
def monthly_flux():
    """Observed monthly mean F10.7 (predicted values at the end of the file left out)."""
    sw = pd.read_csv(DATA_DIR / "SW-All.csv", parse_dates=["DATE"])
    sw = sw[sw["F10.7_DATA_TYPE"].isin(["OBS", "INT"])]
    return sw.set_index("DATE")["F10.7_OBS"].resample("MS").mean().dropna()


def solar_minima(smoothed):
    """Months where the smoothed flux is the lowest within four years either side."""
    is_min = smoothed == smoothed.rolling(97, center=True, min_periods=40).min()
    return list(smoothed.index[is_min])


def mean_cycle_shape(smoothed, minima):
    """Average smoothed flux by months since minimum, over every cycle that has finished."""
    rows = []
    for start, end in zip(minima[:-1], minima[1:]):
        cycle = smoothed[start:].iloc[:SHAPE_MONTHS].to_numpy()
        rows.append(np.pad(cycle, (0, SHAPE_MONTHS - len(cycle)), constant_values=np.nan))
    return np.nanmean(rows, axis=0)


def forecast_monthly(date, months, flux=None):
    """Forecast of monthly F10.7 for the `months` months starting at `date`, using only earlier data."""
    flux = monthly_flux() if flux is None else flux
    date = pd.Timestamp(date)
    known = flux[flux.index < date.to_period("M").start_time]
    smoothed = known.rolling(13, center=True, min_periods=7).mean()
    minima = [m for m in solar_minima(smoothed) if m <= known.index[-1] - pd.DateOffset(months=CONFIRM_MONTHS)]
    shape = mean_cycle_shape(smoothed, minima)
    last_min = minima[-1]
    elapsed = (date.year - last_min.year) * 12 + date.month - last_min.month

    # How strong is this cycle compared with the average shape so far?
    so_far = known[last_min:].to_numpy()[:min(elapsed, SHAPE_MONTHS)]
    base = shape[:len(so_far)] - QUIET_FLUX
    scale = np.dot(so_far - QUIET_FLUX, base) / np.dot(base, base) if len(so_far) >= 12 else 1.0

    k = elapsed + np.arange(months)
    in_cycle = k < CYCLE_MONTHS
    phase = np.where(in_cycle, k, (k - CYCLE_MONTHS) % CYCLE_MONTHS)  # later cycles start again from a minimum
    model = QUIET_FLUX + np.where(in_cycle, scale, 1.0) * (shape[phase] - QUIET_FLUX)
    now = known.iloc[-3:].mean()  # about the 81-day average
    weight = np.exp(-np.arange(months) / BLEND_MONTHS)
    return np.maximum(weight * now + (1 - weight) * model, QUIET_FLUX)


def forecast_daily(date, days):
    """Daily 81-day-average flux forecast for `days` days from `date` (linear between months)."""
    months = int(np.ceil(days / 30.44)) + 2
    monthly = forecast_monthly(date, months)
    mid_month = np.arange(months) * 30.44 + 15.2 - pd.Timestamp(date).day
    return np.interp(np.arange(days), mid_month, monthly)
