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



def batches(todo, max_objects=BATCH_SIZE, max_months=12):
    """Consecutive launch months packed into one query while they total at most max_objects.

    Before about 2016 most months have only a few launches, so packing them saves
    hundreds of rate-limited queries. A busy month gets a batch of its own.
    """
    counts = todo.groupby("month").size()
    batch, total = [], 0
    for month, n in counts.items():
        if batch and (total + n > max_objects or len(batch) == max_months):
            yield batch
            batch, total = [], 0
        batch.append(month)
        total += n
    if batch:
        yield batch

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = objects()
    todo["month"] = todo["LAUNCH_DATE"].dt.to_period("M")
    todo = todo[[not (OUT_DIR / f"launched_{m}.parquet").exists() for m in todo["month"]]]
    st = SpaceTrack()
    try:
        for months in batches(todo):
            group = todo[todo["month"].isin(months)]
            ids = sorted(group["NORAD_CAT_ID"])
            start, end = months[0].start_time, months[-1].end_time + pd.Timedelta(days=EARLY_DAYS + 1)
            parts = [fetch(st, ids[k:k + BATCH_SIZE], start, end) for k in range(0, len(ids), BATCH_SIZE)]
            orbits = pd.concat(parts, ignore_index=True)
            if len(orbits):
                launch = orbits["NORAD_CAT_ID"].map(group.set_index("NORAD_CAT_ID")["LAUNCH_DATE"])
                orbits = orbits[orbits["EPOCH"] < launch + pd.Timedelta(days=EARLY_DAYS)]
                month_of = orbits["NORAD_CAT_ID"].map(group.set_index("NORAD_CAT_ID")["month"])
            for month in months:  # one file per launch month, so a rerun resumes cleanly
                part = orbits[month_of == month] if len(orbits) else orbits
                path = OUT_DIR / f"launched_{month}.parquet"
                part.to_parquet(path.with_suffix(".tmp"), index=False)
                path.with_suffix(".tmp").replace(path)  # never leave a half-written file to be skipped
            print(f"{months[0]} to {months[-1]}: {len(ids)} objects, {len(orbits):,} orbits", flush=True)
    finally:
        st.close()
    files = sorted(OUT_DIR.glob("launched_*.parquet"))
    early = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    early.to_parquet(OUT_FILE, index=False)
    print(f"saved {len(early):,} orbits of {early['NORAD_CAT_ID'].nunique():,} objects to {OUT_FILE}")


if __name__ == "__main__":
    main()
