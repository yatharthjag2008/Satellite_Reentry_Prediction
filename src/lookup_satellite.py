"""Look up any object in the satellite catalog and print its stats.

Type a name (e.g. ISS, HUBBLE, STARLINK-1007) or a NORAD catalog number.
Partial names work; if several objects match you pick one from a list.
Press Enter on an empty line (or type q) to quit.

    python src/lookup_satellite.py
"""

import shutil
from itertools import zip_longest
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


def section(title, rows, width=None):
    """A titled block of aligned 'label  value' lines."""
    width = width or max(len(label) for label, _ in rows)
    return [title] + [f"{label:<{width}}  {value}" for label, value in rows]


def show(row, f107_81d):
    reentered = pd.notna(row["DECAY_DATE"])
    launch = row["LAUNCH_DATE"]
    where = ORBIT_TYPES.get(row["ORBIT_TYPE"], row["ORBIT_TYPE"])

    launch_rows = [("Date", launch.date() if pd.notna(launch) else "unknown"), ("Site", row["LAUNCH_SITE"])]
    if pd.notna(launch):
        known = f107_81d[:launch]
        if len(known):
            launch_rows.append(("Solar flux", f"{known.iloc[-1]:.0f} sfu (81-day avg)"))

    if reentered:
        life_rows = [("Reentered", f"{row['DECAY_DATE'].date()} ({where})")]
        if pd.notna(launch):
            life_rows.append(("In orbit", describe_duration((row["DECAY_DATE"] - launch).days)))
    else:
        life_rows = [("Status", f"still up ({where})")]
        if pd.notna(launch):
            life_rows.append(("So far", describe_duration((pd.Timestamp.today() - launch).days)))

    orbit_rows = [
        ("Perigee", fmt(row["PERIGEE"], " km")),
        ("Apogee", fmt(row["APOGEE"], " km")),
        ("Inclination", fmt(row["INCLINATION"], "°", 2)),
        ("Period", fmt(row["PERIOD"], " min", 1)),
    ]
    if pd.notna(row["PERIGEE"]) and pd.notna(row["APOGEE"]):
        ra, rp = row["APOGEE"] + R_EARTH, row["PERIGEE"] + R_EARTH
        orbit_rows.append(("Eccentricity", f"{(ra - rp) / (ra + rp):.4f}"))
    orbit_rows.append(("Radar size", fmt(row["RCS"], " m²", 3)))
    if row["ORBIT_CENTER"] != "EA":
        orbit_rows.insert(0, ("Centre", f"{row['ORBIT_CENTER']} (not Earth)"))

    label_width = max(len(label) for label, _ in launch_rows + life_rows)
    left = section("LAUNCH", launch_rows, label_width) + [""] + section("LIFETIME", life_rows, label_width)
    right = section("ORBIT (catalog)", orbit_rows)

    # Side by side if the terminal is wide enough, stacked otherwise.
    term_width = shutil.get_terminal_size().columns
    left_width = max(len(line) for line in left) + 6
    if left_width + max(len(line) for line in right) < term_width - 2:
        body = [f"{l:<{left_width}}{r}".rstrip() for l, r in zip_longest(left, right, fillvalue="")]
    else:
        body = left + [""] + right

    summary = [OBJECT_TYPES.get(row["OBJECT_TYPE"], row["OBJECT_TYPE"]), f"owner {row['OWNER']}"]
    if row["OPS_STATUS_CODE"] in OPS_STATUS:
        summary.append(OPS_STATUS[row["OPS_STATUS_CODE"]])
    summary += [f"NORAD {row['NORAD_CAT_ID']}", row["OBJECT_ID"]]

    rule_width = min(term_width - 1, 80)
    print(f"\n== {row['OBJECT_NAME']} " + "=" * max(0, rule_width - len(row["OBJECT_NAME"]) - 4))
    print(" " + " · ".join(summary) + "\n")
    for line in body:
        print(" " + line)
    if reentered and pd.notna(row["PERIGEE"]) and row["PERIGEE"] < 250:
        print("\n Note: perigee under 250 km, so this is probably its final orbit before reentry.")
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

        exact = matches[matches["rank"] == 0] if "rank" in matches else matches
        if len(exact) == 1:
            row = exact.iloc[0]
        else:
            row = matches.iloc[0] if len(matches) == 1 else choose(matches)
        if row is not None:
            show(row, f107_81d)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):  # Ctrl+Z / Ctrl+C
        print()
