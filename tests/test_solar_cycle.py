import numpy as np
import pandas as pd
import pytest

from solar_cycle import forecast_daily, forecast_monthly, monthly_flux


def test_forecast_uses_only_data_before_the_date():
    flux = monthly_flux()
    date = pd.Timestamp("2012-07-01")
    changed = flux.copy()
    changed[changed.index >= date] = 999.0  # rewrite the future
    np.testing.assert_allclose(forecast_monthly(date, 60, changed), forecast_monthly(date, 60, flux))


def test_forecast_from_a_minimum_peaks_a_few_years_later():
    f = forecast_monthly("2020-06-01", 144)
    peak_year = np.argmax(f) / 12
    assert 2.5 < peak_year < 6.5
    assert f.max() > 1.5 * f[0]


def test_forecast_beats_persistence_two_to_six_years_ahead():
    flux = monthly_flux()
    forecast_err, persist_err = [], []
    for date in pd.date_range("1990-01-01", "2019-01-01", freq="12MS"):
        actual = flux[date:].to_numpy()[24:72]
        forecast_err.append(np.mean(np.abs(forecast_monthly(date, 72)[24:24 + len(actual)] - actual)))
        persist_err.append(np.mean(np.abs(flux[:date].iloc[-4:-1].mean() - actual)))
    assert np.mean(forecast_err) < 0.6 * np.mean(persist_err)


def test_daily_forecast_starts_near_today_and_stays_physical():
    daily = forecast_daily("2024-03-15", 3650)
    recent = monthly_flux()[:"2024-02-29"].iloc[-3:].mean()
    assert daily[0] == pytest.approx(recent, rel=0.1)
    assert daily.min() >= 65 and len(daily) == 3650
