"""Look up any object in the satellite catalog and print its stats.

Type a name (e.g. ISS, HUBBLE, STARLINK-1007) or a NORAD catalog number.
Partial names work; if several objects match you pick one from a list.
Press Enter on an empty line (or type q) to quit.

    python src/lookup_satellite.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
R_EARTH = 6378.137  # km
MAX_LISTED = 20

OBJECT_TYPES = {"PAY": "payload (satellite)", "R/B": "rocket body", "DEB": "debris", "UNK": "unknown"}
OPS_STATUS = {
    "+": "operational", "-": "not operational", "P": "partially operational",
    "B": "backup", "S": "spare", "X": "extended mission", "D": "decayed", "?": "unknown",
}
ORBIT_TYPES = {"ORB": "in orbit", "LAN": "landed", "IMP": "impacted", "DOC": "docked", "R/T": "round trip"}


def load_catalog():
    satcat = pd.read_csv(DATA_DIR / "satcat.csv", parse_dates=["LAUNCH_DATE", "DECAY_DATE"])
    satcat["NAME_UPPER"] = satcat["OBJECT_NAME"].str.upper()
    return satcat


def load_space_weather():
    sw = pd.read_csv(DATA_DIR / "SW-All.csv", parse_dates=["DATE"], usecols=["DATE", "F10.7_OBS_LAST81"])
    return sw.set_index("DATE")["F10.7_OBS_LAST81"]


def find(satcat, query):
    """Matching rows, best matches first: exact name, then starts with, then contains."""
    query = query.strip().upper()
    if query.isdigit():
        return satcat[satcat["NORAD_CAT_ID"] == int(query)]

    names = satcat["NAME_UPPER"]
    matches = satcat[names.str.contains(query, regex=False, na=False)].copy()
    matches["rank"] = 2
    matches.loc[matches["NAME_UPPER"].str.startswith(query), "rank"] = 1
    matches.loc[matches["NAME_UPPER"] == query, "rank"] = 0
    return matches.sort_values(["rank", "LAUNCH_DATE"])


def choose(matches):
    """Let the user pick one row when a search returns several."""
    print(f"\n{len(matches):,} matches" + (f", showing the first {MAX_LISTED}" if len(matches) > MAX_LISTED else "") + ":")
    shown = matches.head(MAX_LISTED)
    for i, (_, row) in enumerate(shown.iterrows(), start=1):
        launched = row["LAUNCH_DATE"].date() if pd.notna(row["LAUNCH_DATE"]) else "?"
        print(f"  {i:>2}. {row['OBJECT_NAME']:<28} NORAD {row['NORAD_CAT_ID']:<6}  {row['OBJECT_TYPE']:<4} launched {launched}")

    pick = input("Pick a number (or Enter to search again): ").strip()
    if pick.isdigit() and 1 <= int(pick) <= len(shown):
        return shown.iloc[int(pick) - 1]
    return None


def fmt(value, unit="", decimals=0):
    if pd.isna(value):
        return "unknown"
    return f"{value:,.{decimals}f}{unit}"


def describe_duration(days):
    if days < 1:
        return "less than a day"
    if days < 60:
        return f"{days:,} days"
    if days < 730:
        return f"{days:,} days (~{days / 30.44:.0f} months)"
    return f"{days:,} days (~{days / 365.25:.1f} years)"


def show(row, f107_81d):
    reentered = pd.notna(row["DECAY_DATE"])
    launch = row["LAUNCH_DATE"]

    print("\n" + "=" * 60)
    print(f"  {row['OBJECT_NAME']}")
    print("=" * 60)
    print(f"  NORAD ID           {row['NORAD_CAT_ID']}")
    print(f"  International ID   {row['OBJECT_ID']}")
    print(f"  Type               {OBJECT_TYPES.get(row['OBJECT_TYPE'], row['OBJECT_TYPE'])}")
    print(f"  Owner              {row['OWNER']}")
    print(f"  Status             {OPS_STATUS.get(row['OPS_STATUS_CODE'], 'not listed')}")

    print("\n  Launch")
    print(f"    Date             {launch.date() if pd.notna(launch) else 'unknown'}")
    print(f"    Site             {row['LAUNCH_SITE']}")
    if pd.notna(launch):
        known = f107_81d[:launch]
        if len(known):
            print(f"    Solar flux       {known.iloc[-1]:.0f} sfu (81-day average at launch)")

    print("\n  Orbit (as stored in the catalog)")
    if row["ORBIT_CENTER"] != "EA":
        print(f"    Centre           {row['ORBIT_CENTER']} (not orbiting Earth)")
    print(f"    Perigee          {fmt(row['PERIGEE'], ' km')}")
    print(f"    Apogee           {fmt(row['APOGEE'], ' km')}")
    print(f"    Inclination      {fmt(row['INCLINATION'], '°', 2)}")
    print(f"    Period           {fmt(row['PERIOD'], ' min', 1)}")
    if pd.notna(row["PERIGEE"]) and pd.notna(row["APOGEE"]):
        ra, rp = row["APOGEE"] + R_EARTH, row["PERIGEE"] + R_EARTH
        print(f"    Eccentricity     {(ra - rp) / (ra + rp):.4f}")
    print(f"    Radar size (RCS) {fmt(row['RCS'], ' m²', 3)}")

    print("\n  Lifetime")
    if reentered:
        print(f"    Reentered        {row['DECAY_DATE'].date()} ({ORBIT_TYPES.get(row['ORBIT_TYPE'], row['ORBIT_TYPE'])})")
        if pd.notna(launch):
            print(f"    Time in orbit    {describe_duration((row['DECAY_DATE'] - launch).days)}")
        if pd.notna(row["PERIGEE"]) and row["PERIGEE"] < 250:
            print("    Note             perigee under 250 km: probably its final orbit before reentry")
    else:
        print(f"    Still up         yes ({ORBIT_TYPES.get(row['ORBIT_TYPE'], row['ORBIT_TYPE'])})")
        if pd.notna(launch):
            print(f"    Time so far      {describe_duration((pd.Timestamp.today() - launch).days)}")
    print()


def main():
    print("Loading catalog...")
    satcat = load_catalog()
    f107_81d = load_space_weather()
    print(f"{len(satcat):,} objects loaded. Type a name or NORAD ID (Enter or q to quit).")

    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ("", "q", "quit", "exit"):
            break

        matches = find(satcat, query)
        if matches.empty:
            print(f"No objects match '{query}'.")
            continue

        row = matches.iloc[0] if len(matches) == 1 else choose(matches)
        if row is not None:
            show(row, f107_81d)


if __name__ == "__main__":
    main()
