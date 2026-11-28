"""Download the orientation of each transfer orbit in the final-year physics predictions.

The Moon's and Sun's pull on a transfer orbit depends on where its perigee points
and how its plane is tilted (lunisolar.py), which download_history.py leaves out.
For every weekly row of physics_model.py's predictions with an apogee above
TRANSFER_APOGEE, this fetches the object's orbits over the span of those rows,
with the angles, and keeps the ones at those exact moments (EXPERIMENTS.md 026).

Saved to the git-ignored data/spacetrack/final_year_angles.parquet. About 20 minutes.

    python src/download_final_year_angles.py
"""

import pandas as pd

from download_history import fetch
from lifetime_model import ANGLES, TRANSFER_APOGEE
from physics_model import DATA_FILE, OUT_FILE as PHYSICS_FILE
from spacetrack import SpaceTrack

OUT_FILE = DATA_FILE.parent / "final_year_angles.parquet"
BATCH = 25  # objects per query: a year of orbits each


def transfer_rows():
    """(NORAD_CAT_ID, EPOCH) of the weekly physics rows on transfer orbits."""
    weekly = pd.read_parquet(PHYSICS_FILE, columns=["NORAD_CAT_ID", "EPOCH"])
    orbits = pd.read_parquet(DATA_FILE, columns=["NORAD_CAT_ID", "EPOCH", "APOAPSIS"])
    rows = weekly.merge(orbits, on=["NORAD_CAT_ID", "EPOCH"])
    return rows[rows["APOAPSIS"] > TRANSFER_APOGEE][["NORAD_CAT_ID", "EPOCH"]]


def main():
    todo = transfer_rows()
    spans = todo.groupby("NORAD_CAT_ID")["EPOCH"].agg(["min", "max"]).sort_values("min")
    print(f"{len(todo):,} rows on {len(spans):,} transfer orbits", flush=True)
    st = SpaceTrack()
    parts = []
    try:
        for i in range(0, len(spans), BATCH):
            group = spans.iloc[i:i + BATCH]
            start, end = group["min"].min() - pd.Timedelta(days=1), group["max"].max() + pd.Timedelta(days=1)
            orbits = fetch(st, list(group.index), start, end, "NORAD_CAT_ID,EPOCH," + ",".join(ANGLES))
            if len(orbits):
                parts.append(orbits.merge(todo, on=["NORAD_CAT_ID", "EPOCH"]))  # only the rows' own orbits
            print(f"{i + len(group):,} of {len(spans):,} objects: {len(orbits):,} orbits", flush=True)
    finally:
        st.close()
    found = pd.concat(parts, ignore_index=True).drop_duplicates(["NORAD_CAT_ID", "EPOCH"])
    found.to_parquet(OUT_FILE, index=False)
    print(f"saved the angles of {len(found):,} of {len(todo):,} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
