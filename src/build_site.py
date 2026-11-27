"""Build the public website into _site/, which GitHub Pages serves.

The site only uses public data: the CelesTrak catalog and space weather in
data/, the plots in results/ and, if the forecast-data branch is checked out in
forecast-data/, the live physics forecasts from live_forecasts.py with that
run's fresher catalog and weather. Space-Track data and the trained models stay
on your own computer (see the README).

    git worktree add forecast-data forecast-data    # optional: the live forecasts
    python src/build_site.py
    python -m http.server -d _site      # then open http://localhost:8000
"""

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"
OUT_DIR = ROOT / "_site"
FORECAST_DIR = ROOT / "forecast-data"
FORECAST_FILES = ["forecasts.json", "scores.json", "tracks.json"]

CATALOG_COLUMNS = [
    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "OBJECT_TYPE", "OPS_STATUS_CODE", "OWNER",
    "LAUNCH_DATE", "LAUNCH_SITE", "DECAY_DATE", "PERIOD", "INCLINATION", "APOGEE", "PERIGEE",
    "RCS", "ORBIT_CENTER", "ORBIT_TYPE",
]
DATE_COLUMNS = ["LAUNCH_DATE", "DECAY_DATE"]
PLOTS = [
    "error_by_horizon.png", "predicted_vs_actual.png",
    "backtest_by_year.png", "lifetime_predicted_vs_actual.png",
]


def _json_value(value):
    """None for missing values, and whole numbers without a trailing '.0' to keep the file small."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def compact_catalog(satcat):
    """The catalog as one list per column, which is much smaller as JSON than one object per row."""
    columns = {}
    for name in CATALOG_COLUMNS:
        values = satcat[name]
        if name in DATE_COLUMNS:
            values = pd.to_datetime(values).dt.strftime("%Y-%m-%d")
        columns[name] = [_json_value(v) for v in values.tolist()]
    dates = pd.concat([pd.to_datetime(satcat[c]) for c in DATE_COLUMNS])
    return {"updated": f"{dates.max():%Y-%m-%d}", "count": len(satcat), "columns": columns}


def daily_solar_flux(sw, until):
    """81-day average solar flux for every day, as the lookup uses it: the last value known on or before that day.

    SW-All.csv is daily up to a few weeks ahead and monthly after that, so the
    days are filled forward; nothing past `until` is kept.
    """
    flux = sw.set_index("DATE")["F10.7_OBS_LAST81"].dropna()
    days = pd.date_range(flux.index[0], until, freq="D")
    values = flux.reindex(days, method="ffill")
    return {"start": f"{days[0]:%Y-%m-%d}", "values": values.round().astype(int).tolist()}


def source(name):
    """The forecast run's copy of a CelesTrak file if there is one, since it's refreshed daily."""
    fresh = FORECAST_DIR / name
    return fresh if fresh.exists() else DATA_DIR / name


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    shutil.copytree(SITE_DIR, OUT_DIR)

    satcat = pd.read_csv(source("satcat.csv"))
    write_json(OUT_DIR / "data" / "catalog.json", compact_catalog(satcat))

    sw = pd.read_csv(source("SW-All.csv"), parse_dates=["DATE"], usecols=["DATE", "F10.7_OBS_LAST81"])
    write_json(OUT_DIR / "data" / "solar_flux.json", daily_solar_flux(sw, pd.Timestamp.today().normalize()))

    for name in FORECAST_FILES:
        if (FORECAST_DIR / name).exists():
            shutil.copy2(FORECAST_DIR / name, OUT_DIR / "data" / name)

    (OUT_DIR / "plots").mkdir()
    for plot in PLOTS:
        shutil.copy2(RESULTS_DIR / plot, OUT_DIR / "plots" / plot)

    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(OUT_DIR).as_posix():<45} {path.stat().st_size / 1024:>8,.0f} KB")


if __name__ == "__main__":
    main()
