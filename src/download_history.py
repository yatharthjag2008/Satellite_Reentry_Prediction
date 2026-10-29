"""Download the final year of orbits for every reentered object, from Space-Track (Part 2).

For each object in data/satcat.csv that reentered since START_YEAR, this saves its orbit
history over the DAYS_BEFORE_DECAY days before reentry, thinned to one orbit per day
(the last one each day). Each orbit later becomes a training row labeled with how many
days the object had left.

Output goes to data/spacetrack/reentry_history/, one file per month of reentries. That
folder is git-ignored because Space-Track's terms don't allow republishing their data.

Space-Track limits how fast you can query, so the full download takes a few hours. It
saves each month as it goes; if it stops, run it again and it continues where it left off.

    python src/download_history.py
"""

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from spacetrack import SpaceTrack

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "spacetrack" / "reentry_history"
LOG_FILE = OUT_DIR / "download_log.txt"

START_YEAR = 2000
DAYS_BEFORE_DECAY = 365
BATCH_SIZE = 50  # objects per query; Space-Track refuses queries that return too much
FIELDS = "NORAD_CAT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,BSTAR,MEAN_MOTION_DOT,APOAPSIS,PERIAPSIS"


def log(message):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch(st, ids, start, end):
    """Orbit history for these objects between two dates, splitting the request if it's too big."""
    text = st.get(f"query/class/gp_history/NORAD_CAT_ID/{','.join(map(str, ids))}"
                  f"/EPOCH/{start:%Y-%m-%d}--{end:%Y-%m-%d}/predicates/{FIELDS}/format/csv")
    if not text.strip():
        return pd.DataFrame()
    if not text.startswith("NORAD_CAT_ID"):
        if "out of bounds" in text and len(ids) > 1:
            half = len(ids) // 2
            return pd.concat([fetch(st, ids[:half], start, end), fetch(st, ids[half:], start, end)])
        raise RuntimeError(f"unexpected reply from Space-Track: {text[:200]}")
    return pd.read_csv(io.StringIO(text), parse_dates=["EPOCH"])


def thin(history, decays):
    """Keep orbits from the last DAYS_BEFORE_DECAY days of each object, one per day."""
    history = history.merge(decays, on="NORAD_CAT_ID")
    days_left = (history["DECAY_DATE"] - history["EPOCH"]).dt.total_seconds() / 86400
    history = history[(days_left > -1) & (days_left <= DAYS_BEFORE_DECAY)].sort_values("EPOCH")
    return history.groupby(["NORAD_CAT_ID", history["EPOCH"].dt.date]).tail(1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    satcat = pd.read_csv(DATA_DIR / "satcat.csv", parse_dates=["DECAY_DATE"])
    decayed = satcat.loc[(satcat["DECAY_DATE"].dt.year >= START_YEAR) & (satcat["ORBIT_CENTER"] == "EA"),
                         ["NORAD_CAT_ID", "DECAY_DATE"]]

    # Skip the current month: more reentries will still be added to it.
    this_month = pd.Timestamp.today().to_period("M")
    todo = [(month, group) for month, group in decayed.groupby(decayed["DECAY_DATE"].dt.to_period("M"))
            if month < this_month and not (OUT_DIR / f"decayed_{month}.csv.gz").exists()]
    queries = sum(-(-len(group) // BATCH_SIZE) for _, group in todo)
    log(f"{len(todo)} months to download ({queries} queries, about {queries * 15 / 3600:.1f} hours)")

    st = SpaceTrack()
    try:
        for i, (month, group) in enumerate(todo, start=1):
            start = month.start_time - pd.Timedelta(days=DAYS_BEFORE_DECAY)
            end = month.end_time + pd.Timedelta(days=1)
            ids = group["NORAD_CAT_ID"].tolist()
            frames = [fetch(st, ids[k:k + BATCH_SIZE], start, end) for k in range(0, len(ids), BATCH_SIZE)]
            frames = [f for f in frames if len(f)]
            history = thin(pd.concat(frames), group) if frames else pd.DataFrame(columns=FIELDS.split(",") + ["DECAY_DATE"])

            # Write to a temporary name first so an interrupted run never leaves a half-written month.
            out = OUT_DIR / f"decayed_{month}.csv.gz"
            tmp = out.with_name(out.name + ".tmp")
            history.to_csv(tmp, index=False, compression="gzip")
            tmp.replace(out)
            log(f"[{i}/{len(todo)}] {month}: {history['NORAD_CAT_ID'].nunique()}/{len(ids)} objects, "
                f"{len(history):,} daily orbits")
    finally:
        st.close()
    log("done")


if __name__ == "__main__":
    main()
