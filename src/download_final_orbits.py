"""Download every orbit (not one a day) from the final weeks of objects with official predictions.

The training table keeps one orbit per day, so near reentry our latest orbit is
often half a day old. For a fair comparison with the official predictions, this
fetches all orbits from FINAL_DAYS before each object's reentry. Saved to the
git-ignored data/spacetrack/final_orbits.parquet.

Run download_tip.py first.

    python src/download_final_orbits.py
"""

from pathlib import Path

import pandas as pd

from download_history import BATCH_SIZE, fetch
from download_tip import OUT_FILE as TIP_FILE
from spacetrack import SpaceTrack

OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "spacetrack" / "final_orbits.parquet"
FINAL_DAYS = 40  # 30 days to fit the drag parameter, plus up to 10 days of official messages


def main():
    tip = pd.read_csv(TIP_FILE, parse_dates=["DECAY_EPOCH"])
    reentry = tip.sort_values("DECAY_EPOCH").groupby("NORAD_CAT_ID")["DECAY_EPOCH"].last()
    months = reentry.dt.to_period("M")
    st = SpaceTrack()
    parts = []
    try:
        for month, ids in reentry.groupby(months).groups.items():
            ids = sorted(ids)
            start = month.start_time - pd.Timedelta(days=FINAL_DAYS)
            end = month.end_time + pd.Timedelta(days=1)
            for k in range(0, len(ids), BATCH_SIZE):
                parts.append(fetch(st, ids[k:k + BATCH_SIZE], start, end))
            print(f"{month}: {len(ids)} objects")
    finally:
        st.close()
    orbits = pd.concat(parts, ignore_index=True).drop_duplicates(["NORAD_CAT_ID", "EPOCH"])
    # Keep each object's final FINAL_DAYS only.
    orbits = orbits[orbits["EPOCH"] >= orbits["NORAD_CAT_ID"].map(reentry) - pd.Timedelta(days=FINAL_DAYS)]
    orbits.to_parquet(OUT_FILE, index=False)
    print(f"saved {len(orbits):,} orbits of {orbits['NORAD_CAT_ID'].nunique():,} objects to {OUT_FILE}")


if __name__ == "__main__":
    main()
