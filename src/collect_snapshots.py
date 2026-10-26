"""Daily CelesTrak collector for Part 2 (days-until-reentry).

Each run saves one snapshot of current orbit elements for low-Earth-orbit objects,
plus fresh copies of the satellite catalog (for decay dates) and space weather.
Run it once a day; it skips itself if today's snapshot already exists, which also
keeps us within CelesTrak's usage guidelines.

    python src/collect_snapshots.py
"""

import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
LOG_FILE = DATA_DIR / "collect_log.txt"

GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
# CelesTrak has no "everything" group, so combine the groups that cover LEO objects.
GP_QUERIES = [
    {"SPECIAL": "DECAYING"},          # objects expected to reenter soon
    {"GROUP": "active"},              # active payloads (includes Starlink)
    {"GROUP": "last-30-days"},        # recent launches, incl. rocket bodies
    {"GROUP": "analyst"},
    {"GROUP": "cosmos-1408-debris"},
    {"GROUP": "fengyun-1c-debris"},
    {"GROUP": "iridium-33-debris"},
    {"GROUP": "cosmos-2251-debris"},
]
SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
SPACE_WEATHER_URL = "https://celestrak.org/SpaceData/SW-All.csv"

MAX_PERIGEE_KM = 700  # only keep objects low enough to plausibly decay
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km
HEADERS = {"User-Agent": "satellite-reentry-student-project"}


def log(message):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def download(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response.text


def add_altitudes(df):
    """Perigee/apogee altitude (km) from mean motion (rev/day) and eccentricity."""
    n_rad_per_s = df["MEAN_MOTION"] * 2 * np.pi / 86400
    semi_major_axis = (MU_EARTH / n_rad_per_s**2) ** (1 / 3)
    df["PERIGEE_KM"] = semi_major_axis * (1 - df["ECCENTRICITY"]) - R_EARTH
    df["APOGEE_KM"] = semi_major_axis * (1 + df["ECCENTRICITY"]) - R_EARTH
    return df


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_file = SNAPSHOT_DIR / f"gp_{today}.csv"
    if out_file.exists():
        log(f"{out_file.name} already exists, skipping")
        return

    frames = []
    failed = 0
    for query in GP_QUERIES:
        try:
            text = download(GP_URL, {**query, "FORMAT": "csv"})
            frames.append(pd.read_csv(io.StringIO(text)))
        except Exception as e:  # one failed group shouldn't lose the whole day
            log(f"failed {query}: {e}")
            failed += 1
        time.sleep(2)

    # An incomplete day is saved separately so a later run retries it
    # (CelesTrak returns 403 if the same group is fetched twice within 2 hours).
    if failed:
        out_file = SNAPSHOT_DIR / f"gp_{today}_partial.csv"

    if not frames:
        log("no data downloaded")
        sys.exit(1)

    gp = pd.concat(frames, ignore_index=True)
    gp = gp.drop_duplicates(subset=["NORAD_CAT_ID", "EPOCH"])
    gp = add_altitudes(gp)
    gp = gp[gp["PERIGEE_KM"] < MAX_PERIGEE_KM]
    gp["DOWNLOADED_UTC"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gp.to_csv(out_file, index=False)
    log(f"saved {len(gp)} objects to {out_file.name}")

    for url, name in [(SATCAT_URL, "satcat.csv"), (SPACE_WEATHER_URL, "SW-All.csv")]:
        try:
            (DATA_DIR / name).write_text(download(url), encoding="utf-8")
            log(f"refreshed {name}")
        except Exception as e:
            log(f"failed to refresh {name}: {e}")


if __name__ == "__main__":
    main()
