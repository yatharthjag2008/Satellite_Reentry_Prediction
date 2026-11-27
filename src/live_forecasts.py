"""Live reentry forecasts for the website, from orbits collected from CelesTrak.

GitHub Actions runs this every 6 hours (.github/workflows/forecasts.yml):

1. collect: CelesTrak's decaying objects every run and, once a day, the larger
   groups below MAX_PERIGEE_KM, the satellite catalog and the space weather. New
   orbits are added to data/live/orbits.parquet, which keeps HISTORY_DAYS.
2. forecast: the physics model for every object still tracked, as the lookup
   tool runs it: B fitted over up to 30 days of one orbit a day and, within 10
   days of reentry, over the last 8 days of every orbit (EXPERIMENTS.md 015).
   An object first seen recently gets a fit over the days there are, with a
   90% range calibrated for that (forecast_ranges.py).
3. score: once the catalog lists an object as reentered, every forecast made
   for it is compared with the decay date (taken as midday, as in training).

The trained hybrid isn't used: like the Space-Track data it was trained on, it
stays private. The collected orbits stay private too, in the Actions cache,
because CelesTrak's orbits come from Space-Track. Only our own forecasts are
published, to the forecast-data branch, which the website reads.

    python src/live_forecasts.py                  # collect, forecast, score
    python src/live_forecasts.py --no-download    # forecast and score from what's collected
"""

import argparse
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from atmosphere import MsisAtmosphere
from build_reentry_dataset import DATA_DIR
from collect_snapshots import GP_QUERIES, GP_URL, HEADERS, SATCAT_URL, SPACE_WEATHER_URL, add_altitudes
from download_history import FIELDS
from forecast_ranges import MAX_B, OUT_FILE as WIDTHS_FILE
from physics_model import fit_drag_parameter, forecast, unit_decay_rate
from predict_reentry import SHORT_WINDOW, SHORT_WINDOW_BELOW, features
from train_reentry_model import HORIZONS
from uncertainty import band_of, group_of

ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = DATA_DIR / "live"
ORBITS_FILE = LIVE_DIR / "orbits.parquet"
STATE_FILE = LIVE_DIR / "state.json"
OUT_DIR = ROOT / "forecast-data"

ORBIT_COLUMNS = FIELDS.split(",")
HISTORY_DAYS = 45
FULL_EVERY = pd.Timedelta(hours=20)  # the large groups, catalog and weather: about once a day
MAX_PERIGEE_KM = 500   # from the large groups, keep objects low enough to come down within a few years
MAX_ORBIT_AGE = 3.0    # days: no newer orbit than this means it has reentered or isn't tracked
MAX_FORECAST_DAYS = 365.0
LEADS = (1, 3, 7, 30)  # days before reentry: the forecasts shown for each recent reentry
RECENT_REENTRIES = 25


def log(message):
    print(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S}Z  {message}", flush=True)


def utc_now():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def iso(t):
    return f"{t:%Y-%m-%dT%H:%MZ}"


# ---------------------------------------------------------------- collect

def to_orbits(gp):
    """CelesTrak GP rows with the columns of Space-Track's gp_history that the models use."""
    gp = add_altitudes(gp.copy()).rename(columns={"PERIGEE_KM": "PERIAPSIS", "APOGEE_KM": "APOAPSIS"})
    gp["EPOCH"] = pd.to_datetime(gp["EPOCH"])
    return gp[ORBIT_COLUMNS]


def merge_orbits(frames, now, days=HISTORY_DAYS):
    """The orbits in all frames without repeats, keeping the last `days` days."""
    both = pd.concat(frames, ignore_index=True).drop_duplicates(["NORAD_CAT_ID", "EPOCH"], keep="last")
    both = both[both["EPOCH"] >= now - pd.Timedelta(days=days)]
    return both.sort_values(["NORAD_CAT_ID", "EPOCH"], ignore_index=True)


def download(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response.text


def fetch_orbits(full):
    """New orbits: the decaying group, plus the large groups (below MAX_PERIGEE_KM) if `full`."""
    assert GP_QUERIES[0] == {"SPECIAL": "DECAYING"}
    frames, failed = [], 0
    for i, query in enumerate(GP_QUERIES if full else GP_QUERIES[:1]):
        try:
            orbits = to_orbits(pd.read_csv(io.StringIO(download(GP_URL, {**query, "FORMAT": "csv"}))))
        except Exception as e:  # one failed group shouldn't lose the rest
            log(f"failed {query}: {e}")
            failed += 1
            continue
        if i > 0:
            orbits = orbits[orbits["PERIAPSIS"] < MAX_PERIGEE_KM]
        log(f"{query}: {len(orbits):,} orbits")
        frames.append(orbits)
        time.sleep(2)
    return frames, failed


def collect(now):
    """Add new CelesTrak orbits to the history; refresh the catalog and weather about once a day."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    last_full = pd.Timestamp(state["last_full"]) if "last_full" in state else None
    full = last_full is None or now - last_full >= FULL_EVERY

    frames, failed = fetch_orbits(full)
    if ORBITS_FILE.exists():
        frames.insert(0, pd.read_parquet(ORBITS_FILE))
    if not frames:
        raise SystemExit("no orbits downloaded and none collected before")
    before = len(frames[0]) if ORBITS_FILE.exists() else 0
    orbits = merge_orbits(frames, now)
    orbits.to_parquet(ORBITS_FILE, index=False)
    log(f"history: {len(orbits):,} orbits of {orbits['NORAD_CAT_ID'].nunique():,} objects ({len(orbits) - before:+,})")

    if full:
        for url, name in [(SATCAT_URL, "satcat.csv"), (SPACE_WEATHER_URL, "SW-All.csv")]:
            try:
                (LIVE_DIR / name).write_text(download(url), encoding="utf-8")
                log(f"refreshed {name}")
            except Exception as e:
                log(f"failed to refresh {name}: {e}")
                failed += 1
        if not failed:  # otherwise try the large groups again next run
            state["last_full"] = now.isoformat()
    STATE_FILE.write_text(json.dumps(state))


def use_live_catalog():
    """The models read data/satcat.csv and data/SW-All.csv, so copy the collected ones over them."""
    for name in ("satcat.csv", "SW-All.csv"):
        if (LIVE_DIR / name).exists():
            shutil.copyfile(LIVE_DIR / name, DATA_DIR / name)


# ---------------------------------------------------------------- forecast

def fit_label(tracked_days):
    """Which calibration fits the history an object has (forecast_ranges.py)."""
    return np.where(tracked_days < 14, "7", np.where(tracked_days < 30, "14", "30"))


def ranges(days, labels, groups, widths):
    """(low, high) of each forecast's 90% range."""
    w = widths["widths"]
    bands = band_of(days)
    width = np.array([w[label][g][b] if np.isfinite(d) and w[label][g][b] is not None else np.nan
                      for d, label, g, b in zip(days, labels, groups, bands)], dtype=float)
    factor = 10 ** width
    return days / factor, days * factor


def plausible(B):
    return np.isfinite(B) & (B > 0) & (B <= MAX_B)


def forecast_all(orbits, atm, widths, now, decayed_ids=()):
    """One row per object still tracked: its latest orbit, status and, if it has one, the forecast."""
    every = features(orbits)
    every["B_short"] = fit_drag_parameter(every, unit_decay_rate(every, atm), SHORT_WINDOW)[0]
    every["first_seen"] = every.groupby("NORAD_CAT_ID")["EPOCH"].transform("min")
    # As in training: B from one orbit a day (the last of each day).
    daily = every.groupby([every["NORAD_CAT_ID"], every["EPOCH"].dt.floor("D")]).tail(1).reset_index(drop=True)
    daily["B_fit"] = fit_drag_parameter(daily, unit_decay_rate(daily, atm))[0]
    latest = daily.groupby("NORAD_CAT_ID").tail(1).reset_index(drop=True)
    latest = latest[(now - latest["EPOCH"]).dt.total_seconds() / 86400 <= MAX_ORBIT_AGE]
    latest = latest[~latest["NORAD_CAT_ID"].isin(decayed_ids)].reset_index(drop=True)

    B, B_short = latest["B_fit"].to_numpy(), latest["B_short"].to_numpy()
    days = np.full(len(latest), np.nan)
    if plausible(B).any():
        days = forecast(latest, np.where(plausible(B), B, np.nan), atm, max_days=MAX_FORECAST_DAYS)
    # Within 10 days, the 8-day fit; also for an object seen too briefly for the daily fit.
    short = plausible(B_short) & ((plausible(B) & (days < SHORT_WINDOW_BELOW)) | ~np.isfinite(B))
    days_short = np.full(len(latest), np.nan)
    if short.any():
        days_short[short] = forecast(latest[short], B_short[short], atm, max_days=4 * SHORT_WINDOW_BELOW)
    short &= np.isfinite(days_short) & (plausible(B) | (days_short < SHORT_WINDOW_BELOW))
    days = np.where(short, days_short, days)

    tracked = (latest["EPOCH"] - latest["first_seen"]).dt.total_seconds().to_numpy() / 86400
    latest["tracked_days"] = tracked
    latest["fit"] = np.where(short, f"{SHORT_WINDOW}-day", fit_label(tracked))
    latest["days"] = days
    latest["group"] = group_of(latest)
    latest["low_days"], latest["high_days"] = ranges(days, latest["fit"], latest["group"], widths)
    B_used = np.where(short, B_short, B)
    latest["status"] = np.select(
        [np.isfinite(days), ~np.isfinite(B_used), B_used <= 0, B_used > MAX_B],
        ["forecast", "too few orbits", "not sinking", "not drag"], default="over a year")
    return latest


def text(value):
    return None if pd.isna(value) else str(value)


def forecast_records(latest):
    """The forecasts, soonest first, for the website."""
    rows = latest[latest["status"] == "forecast"].sort_values("days")
    records = []
    for r in rows.itertuples():
        epoch = r.EPOCH
        records.append({
            "id": int(r.NORAD_CAT_ID), "name": text(r.OBJECT_NAME) or str(r.NORAD_CAT_ID),
            "type": text(r.OBJECT_TYPE), "group": int(r.group),
            "epoch": iso(epoch), "perigee": round(float(r.PERIAPSIS)), "apogee": round(float(r.APOAPSIS)),
            "inclination": round(float(r.INCLINATION), 1), "fit": r.fit, "tracked": round(float(r.tracked_days), 1),
            "days": round(float(r.days), 2),
            "reentry": iso(epoch + pd.Timedelta(days=r.days)),
            "low": iso(epoch + pd.Timedelta(days=r.low_days)) if np.isfinite(r.low_days) else None,
            "high": iso(epoch + pd.Timedelta(days=r.high_days)) if np.isfinite(r.high_days) else None,
        })
    return records


def add_to_log(log_df, latest, now):
    """New forecasts (one per object and orbit) added to the log of everything forecast so far."""
    new = latest[latest["status"] == "forecast"][["NORAD_CAT_ID", "EPOCH", "days", "low_days", "high_days", "fit"]]
    new = new.assign(made=now.floor("min"))
    both = pd.concat([log_df, new], ignore_index=True) if len(log_df) else new
    return both.drop_duplicates(["NORAD_CAT_ID", "EPOCH"], keep="first").sort_values(["made", "NORAD_CAT_ID"],
                                                                                    ignore_index=True)


# ---------------------------------------------------------------- score

def scored_forecasts(log_df, satcat):
    """Logged forecasts of objects that have reentered, with the true days left from each orbit."""
    decays = satcat.dropna(subset=["DECAY_DATE"])[["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "DECAY_DATE"]]
    m = log_df.merge(decays, on="NORAD_CAT_ID")
    m["EPOCH"] = pd.to_datetime(m["EPOCH"])
    reentry = pd.to_datetime(m["DECAY_DATE"]) + pd.Timedelta(hours=12)
    m["actual"] = (reentry - m["EPOCH"]).dt.total_seconds() / 86400
    m["reentry"] = reentry
    return m[m["actual"] > 0].reset_index(drop=True)


def summarize(m):
    err = np.abs(m["days"] / m["actual"] - 1)
    inside = (m["actual"] >= m["low_days"]) & (m["actual"] <= m["high_days"])
    has_range = m["low_days"].notna()
    return {"forecasts": int(len(m)), "objects": int(m["NORAD_CAT_ID"].nunique()),
            "median_error": float(err.median()) if len(m) else None,
            "within_25": float((err <= 0.25).mean()) if len(m) else None,
            "in_range": float(inside[has_range].mean()) if has_range.any() else None}


def scores(log_df, satcat):
    m = scored_forecasts(log_df, satcat)
    bands = [{"label": label, **summarize(m[(m["actual"] >= lo) & (m["actual"] < hi)])} for lo, hi, label in HORIZONS]
    recent = []
    for norad, rows in sorted(m.groupby("NORAD_CAT_ID"), key=lambda kv: kv[1]["reentry"].iloc[0], reverse=True):
        if len(recent) == RECENT_REENTRIES:
            break
        leads = {}
        for lead in LEADS:
            # The last forecast made at least `lead` days before reentry, if made by 1.5x that.
            before = rows[(rows["actual"] >= lead) & (rows["actual"] <= 1.5 * lead)]
            if len(before):
                r = before.loc[before["actual"].idxmin()]
                leads[str(lead)] = {"reentry": iso(r["EPOCH"] + pd.Timedelta(days=r["days"])),
                                    "error_days": round(float(r["days"] - r["actual"]), 2)}
        if leads:
            first = rows.iloc[0]
            recent.append({"id": int(norad), "name": text(first["OBJECT_NAME"]), "type": text(first["OBJECT_TYPE"]),
                           "decay_date": f"{pd.Timestamp(first['DECAY_DATE']):%Y-%m-%d}", "leads": leads})
    since = log_df["made"].min()
    return {"since": iso(since) if len(log_df) else None, "overall": summarize(m), "bands": bands,
            "recent": recent}


# ---------------------------------------------------------------- main

DATA_README = """# Live reentry forecasts

Written every 6 hours by `src/live_forecasts.py` (workflow `forecasts.yml` on the
main branch) and read by the website. This branch is replaced on every run.

- `forecasts.json`: the latest physics forecast for every tracked object expected
  to reenter within a year, with its 90% range
- `scores.json`: how the forecasts did, once the objects reentered
- `log.csv.gz`: every forecast made, one row per object and orbit: the orbit's
  epoch, the days predicted from it, the 90% range in days and the fit used
- `satcat.csv`, `SW-All.csv`: the CelesTrak catalog and space weather used

The orbits themselves aren't published: they come from CelesTrak, which gets
them from Space-Track.
"""


def write_json(path, data):
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-download", action="store_true", help="use only the orbits collected so far")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="where to write the published files")
    args = parser.parse_args()
    now = utc_now()

    if not args.no_download:
        collect(now)
    use_live_catalog()
    if not ORBITS_FILE.exists():
        raise SystemExit("no orbits collected yet")

    widths = json.loads(WIDTHS_FILE.read_text())
    satcat = pd.read_csv(DATA_DIR / "satcat.csv", usecols=["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "DECAY_DATE"])
    orbits = pd.read_parquet(ORBITS_FILE)
    latest = forecast_all(orbits, MsisAtmosphere(), widths, now,
                          decayed_ids=satcat.loc[satcat["DECAY_DATE"].notna(), "NORAD_CAT_ID"])
    counts = latest["status"].value_counts().to_dict()
    log("objects: " + ", ".join(f"{n:,} {status}" for status, n in counts.items()))

    args.out.mkdir(parents=True, exist_ok=True)
    # The website's catalog lookup uses the same fresh catalog and weather (CelesTrak's, as in data/).
    for name in ("satcat.csv", "SW-All.csv"):
        if (LIVE_DIR / name).exists():
            shutil.copyfile(LIVE_DIR / name, args.out / name)
    (args.out / "README.md").write_text(DATA_README, encoding="utf-8")
    log_file = args.out / "log.csv.gz"
    old_log = (pd.read_csv(log_file, parse_dates=["EPOCH", "made"]) if log_file.exists()
               else pd.DataFrame(columns=["NORAD_CAT_ID", "EPOCH", "days", "low_days", "high_days", "fit", "made"]))
    forecast_log = add_to_log(old_log, latest, now)
    forecast_log.to_csv(log_file, index=False, float_format="%.4f", date_format="%Y-%m-%dT%H:%M:%S")

    write_json(args.out / "forecasts.json", {
        "made": iso(now), "orbits_through": iso(orbits["EPOCH"].max()), "counts": counts,
        "calibrated_on": widths["calibrated_on"], "objects": forecast_records(latest)})
    result = scores(forecast_log, satcat)
    write_json(args.out / "scores.json", result)
    log(f"logged {len(forecast_log) - len(old_log):,} new forecasts ({len(forecast_log):,} in all); "
        f"scored {result['overall']['forecasts']:,} on {result['overall']['objects']:,} reentered objects")


if __name__ == "__main__":
    main()
