import numpy as np
import pandas as pd

from build_site import CATALOG_COLUMNS, compact_catalog, daily_solar_flux


def test_compact_catalog_keeps_every_column_with_nulls_and_short_numbers():
    row = {name: "x" for name in CATALOG_COLUMNS}
    satcat = pd.DataFrame([row, row])
    satcat["NORAD_CAT_ID"] = [25544, 20580]
    satcat["LAUNCH_DATE"] = ["1998-11-20", "1990-04-24"]
    satcat["DECAY_DATE"] = [np.nan, "2030-01-02"]
    satcat["PERIGEE"] = [415.0, np.nan]
    satcat["PERIOD"] = [92.8, 95.2]

    out = compact_catalog(satcat)
    columns = out["columns"]

    assert list(columns) == CATALOG_COLUMNS
    assert out["count"] == 2
    assert columns["NORAD_CAT_ID"] == [25544, 20580]
    assert columns["DECAY_DATE"] == [None, "2030-01-02"]
    assert columns["PERIGEE"] == [415, None] and isinstance(columns["PERIGEE"][0], int)
    assert columns["PERIOD"] == [92.8, 95.2]
    assert out["updated"] == "2030-01-02"


def test_daily_solar_flux_fills_forward_and_stops_at_the_given_day():
    sw = pd.DataFrame({
        "DATE": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-02-01"]),
        "F10.7_OBS_LAST81": [150.4, np.nan, 120.0],
    })
    out = daily_solar_flux(sw, pd.Timestamp("2000-02-02"))

    assert out["start"] == "2000-01-01"
    assert len(out["values"]) == 33  # 1 Jan to 2 Feb
    assert out["values"][:3] == [150, 150, 150]  # the missing day keeps the last known value
    assert out["values"][-2:] == [120, 120]


def test_source_prefers_the_forecast_runs_fresher_copy(tmp_path, monkeypatch):
    import build_site

    monkeypatch.setattr(build_site, "FORECAST_DIR", tmp_path)
    assert build_site.source("satcat.csv") == build_site.DATA_DIR / "satcat.csv"
    (tmp_path / "satcat.csv").write_text("x")
    assert build_site.source("satcat.csv") == tmp_path / "satcat.csv"
