"""Download the first weeks of orbits for satellites and rocket bodies that later reentered (Part 1).

Part 1 asks: from an object's first month in orbit, how long will it stay up?
That needs each object's early orbits, which the Part 2 download (final year
only) doesn't have unless the object lived less than a year. Debris is left out
because its catalog launch date is its parent's, not when the piece appeared.

Saved to the git-ignored data/spacetrack/early_orbits.parquet. About 2 hours,
because of Space-Track's rate limits; rerunning resumes where it stopped.

    python src/download_early_orbits.py
"""

from pathlib import Path

import pandas as pd

from build_reentry_dataset import DATA_DIR
from download_history import BATCH_SIZE, fetch
from spacetrack import SpaceTrack

OUT_DIR = DATA_DIR / "spacetrack" / "early_orbits"
OUT_FILE = DATA_DIR / "spacetrack" / "early_orbits.parquet"
EARLY_DAYS = 40
LAUNCHED_FROM = "2000-01-01"


def objects():
    satcat = pd.read_csv(DATA_DIR / "satcat.csv", parse_dates=["LAUNCH_DATE", "DECAY_DATE"])
    keep = (satcat["OBJECT_TYPE"].isin(["PAY", "R/B"]) & (satcat["LAUNCH_DATE"] >= LAUNCHED_FROM)
            & satcat["DECAY_DATE"].notna())
    return satcat.loc[keep, ["NORAD_CAT_ID", "LAUNCH_DATE"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = objects()
    by_month = todo.groupby(todo["LAUNCH_DATE"].dt.to_period("M"))
    st = SpaceTrack()
    try:
        for i, (month, group) in enumerate(by_month, start=1):
            path = OUT_DIR / f"launched_{month}.parquet"
            if path.exists():
                continue
            ids = sorted(group["NORAD_CAT_ID"])
            start, end = month.start_time, month.end_time + pd.Timedelta(days=EARLY_DAYS + 1)
            parts = [fetch(st, ids[k:k + BATCH_SIZE], start, end) for k in range(0, len(ids), BATCH_SIZE)]
            orbits = pd.concat(parts, ignore_index=True)
            if len(orbits):
                launch = orbits["NORAD_CAT_ID"].map(group.set_index("NORAD_CAT_ID")["LAUNCH_DATE"])
                orbits = orbits[orbits["EPOCH"] < launch + pd.Timedelta(days=EARLY_DAYS)]
            orbits.to_parquet(path, index=False)
            print(f"[{i}/{len(by_month)}] {month}: {orbits['NORAD_CAT_ID'].nunique() if len(orbits) else 0}"
                  f"/{len(ids)} objects, {len(orbits):,} orbits", flush=True)
    finally:
        st.close()
    files = sorted(OUT_DIR.glob("launched_*.parquet"))
    early = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    early.to_parquet(OUT_FILE, index=False)
    print(f"saved {len(early):,} orbits of {early['NORAD_CAT_ID'].nunique():,} objects to {OUT_FILE}")


if __name__ == "__main__":
    main()
