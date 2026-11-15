"""Download the official reentry predictions (TIP messages) from Space-Track.

The US Space Force issues Tracking and Impact Prediction (TIP) messages for objects
in their last few days, each with a predicted reentry time and a +/- window. They
are the benchmark the models here are compared against. Saved to the git-ignored
data/spacetrack/tip_messages.csv.gz.

    python src/download_tip.py
"""

import io
from pathlib import Path

import pandas as pd

from spacetrack import SpaceTrack

OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "spacetrack" / "tip_messages.csv.gz"
START = "2023-12-01"  # a month before the test period starts
FIELDS = "NORAD_CAT_ID,MSG_EPOCH,DECAY_EPOCH,WINDOW,HIGH_INTEREST"


def main():
    months = pd.date_range(START, pd.Timestamp.now(), freq="MS")
    st = SpaceTrack()
    parts = []
    try:
        for start, end in zip(months, list(months[1:]) + [pd.Timestamp.now().normalize() + pd.Timedelta(days=1)]):
            text = st.get(f"query/class/tip/MSG_EPOCH/{start:%Y-%m-%d}--{end:%Y-%m-%d}"
                          f"/predicates/{FIELDS}/format/csv")
            part = pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()
            parts.append(part)
            print(f"{start:%Y-%m}: {len(part):,} messages")
    finally:
        st.close()
    tip = pd.concat(parts, ignore_index=True).drop_duplicates()
    tip.to_csv(OUT_FILE, index=False, compression="gzip")
    print(f"saved {len(tip):,} messages for {tip['NORAD_CAT_ID'].nunique():,} objects to {OUT_FILE}")


if __name__ == "__main__":
    main()
