"""Download the orientation of each transfer orbit at the start of its Part 1 simulation.

The Moon's and Sun's pull on a transfer orbit depends on where its perigee points
and how its plane is tilted (lunisolar.py), which the Part 1 download leaves out.
For every object in lifetime_model.py's table with an apogee above
TRANSFER_APOGEE, this fetches its orbits around the day-40 orbit the simulation
starts from, with the angles, and keeps the one at that exact moment.

Saved to the git-ignored data/spacetrack/transfer_orbits.parquet. About 15 minutes.

    python src/download_transfer_orbits.py
"""

import pandas as pd

from build_reentry_dataset import DATA_DIR
from download_early_orbits import batches
from download_history import FIELDS, fetch
from lifetime_model import OUT_FILE as LIFETIME_FILE, TRANSFER_APOGEE
from spacetrack import SpaceTrack

OUT_FILE = DATA_DIR / "spacetrack" / "transfer_orbits.parquet"
ANGLE_FIELDS = FIELDS + ",RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY"


def main():
    table = pd.read_parquet(LIFETIME_FILE, columns=["NORAD_CAT_ID", "EPOCH", "APOAPSIS"])
    todo = table[table["APOAPSIS"] > TRANSFER_APOGEE].copy()
    todo["month"] = todo["EPOCH"].dt.to_period("M")
    st = SpaceTrack()
    parts = []
    try:
        for months in batches(todo, max_months=6):
            group = todo[todo["month"].isin(months)]
            start, end = group["EPOCH"].min() - pd.Timedelta(days=1), group["EPOCH"].max() + pd.Timedelta(days=1)
            orbits = fetch(st, sorted(group["NORAD_CAT_ID"]), start, end, ANGLE_FIELDS)
            parts.append(orbits)
            print(f"{months[0]} to {months[-1]}: {len(group)} objects, {len(orbits):,} orbits", flush=True)
    finally:
        st.close()
    orbits = pd.concat(parts, ignore_index=True)
    start = orbits.merge(todo[["NORAD_CAT_ID", "EPOCH"]], on=["NORAD_CAT_ID", "EPOCH"])  # the day-40 orbit itself
    start.drop_duplicates(["NORAD_CAT_ID", "EPOCH"]).to_parquet(OUT_FILE, index=False)
    print(f"saved the starting orbit of {start['NORAD_CAT_ID'].nunique()} of {len(todo)} transfer orbits to {OUT_FILE}")


if __name__ == "__main__":
    main()
