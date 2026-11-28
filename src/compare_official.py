"""Compare the models with the official US Space Force reentry predictions.

For objects that reentered from 2024 on, each TIP message gives an official
predicted reentry time issued at a known moment (MSG_EPOCH). For every message,
the models predict from the latest orbit available at that moment, and both are
scored against the final reentry time.

The final TIP message for each object is issued after reentry and is used as the
true reentry time (catalog decay dates are only to the day). Messages issued
within an hour of reentry are left out.

The physics here is drag only: these orbits don't include the angles that the
Sun-and-Moon simulation needs (EXPERIMENTS.md 026). Only 0.7% of the messages
(73, on 21 objects) are for transfer orbits, so this barely changes the table.

Run download_tip.py and train_reentry_model.py first.

    python src/compare_official.py           # one orbit a day, as in the training table
    python src/compare_official.py --fresh   # every orbit (run download_final_orbits.py first)
"""

import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from atmosphere import MsisAtmosphere
from download_final_orbits import OUT_FILE as FINAL_ORBITS_FILE
from download_tip import OUT_FILE as TIP_FILE
from physics_model import fit_drag_parameter, forecast, prepare, unit_decay_rate
from train_reentry_model import (DATA_FILE, FEATURES, HYBRID_FEATURES, MODELS_DIR, NO_PHYSICS_DAYS, RESULTS_DIR,
                                 TEST_FROM)

LEAD_BANDS = [(0, 1, "under 1 day"), (1, 2, "1-2 days"), (2, 4, "2-4 days"), (4, 30, "4+ days")]
MAX_ORBIT_AGE = 2.0  # days: skip a message if our latest orbit is older than this


def official_messages():
    tip = pd.read_csv(TIP_FILE, parse_dates=["MSG_EPOCH", "DECAY_EPOCH"]).sort_values(["NORAD_CAT_ID", "MSG_EPOCH"])
    final = tip.groupby("NORAD_CAT_ID").tail(1).set_index("NORAD_CAT_ID")["DECAY_EPOCH"].rename("reentry")
    tip = tip.join(final, on="NORAD_CAT_ID")
    tip = tip[tip.groupby("NORAD_CAT_ID").cumcount(ascending=False) > 0]  # drop each final message
    tip["lead_days"] = (tip["reentry"] - tip["MSG_EPOCH"]).dt.total_seconds() / 86400
    return tip[tip["lead_days"] > 1 / 24]


def daily_orbits(tip, atm):
    """The training table's orbits (one a day) for the objects with official messages."""
    df = pd.read_parquet(DATA_FILE)
    df = prepare(df[(df["DECAY_DATE"] >= TEST_FROM) & df["NORAD_CAT_ID"].isin(tip["NORAD_CAT_ID"])])
    df["B_fit"], df["n_fit"] = fit_drag_parameter(df, unit_decay_rate(df, atm))
    return df


def fresh_orbits(atm):
    """Every orbit from the final weeks, with the same features as the training table.

    B is still fitted on one orbit a day, as in training (the hybrid uses the number
    of orbits in the fit as an input), and each orbit takes the latest daily fit.
    """
    from predict_reentry import features

    orbits = features(pd.read_parquet(FINAL_ORBITS_FILE))
    orbits["day"] = orbits["EPOCH"].dt.floor("D")
    daily = prepare(orbits.groupby(["NORAD_CAT_ID", "day"]).tail(1))
    daily["B_fit"], daily["n_fit"] = fit_drag_parameter(daily, unit_decay_rate(daily, atm))
    return pd.merge_asof(orbits.sort_values("EPOCH"), daily[["NORAD_CAT_ID", "EPOCH", "B_fit", "n_fit"]]
                         .sort_values("EPOCH"), on="EPOCH", by="NORAD_CAT_ID", direction="backward")


def our_predictions(tip, df, atm):
    """Physics, XGBoost and hybrid predictions from the latest orbit before each message."""
    matched = pd.merge_asof(tip.sort_values("MSG_EPOCH"), df.sort_values("EPOCH"), left_on="MSG_EPOCH",
                            right_on="EPOCH", by="NORAD_CAT_ID", direction="backward")
    matched = matched[(matched["MSG_EPOCH"] - matched["EPOCH"]).dt.total_seconds() / 86400 <= MAX_ORBIT_AGE]
    matched = matched.reset_index(drop=True)

    matched["physics_days"] = forecast(matched, matched["B_fit"].to_numpy(), atm)
    matched["is_debris"] = (matched["OBJECT_TYPE"] == "DEB").astype(int)
    matched["is_rocket_body"] = (matched["OBJECT_TYPE"] == "R/B").astype(int)
    matched["log_physics_days"] = np.log10(matched["physics_days"].clip(lower=0.01))
    matched["log_B"] = np.log10(matched["B_fit"].where(matched["B_fit"] > 0))
    physics = matched["physics_days"].fillna(NO_PHYSICS_DAYS).clip(lower=0.01).to_numpy()

    booster, hybrid = xgb.XGBRegressor(), xgb.XGBRegressor()
    booster.load_model(MODELS_DIR / "reentry_xgb.json")
    hybrid.load_model(MODELS_DIR / "reentry_hybrid.json")
    days = {"physics": physics,
            "XGBoost": 10 ** booster.predict(matched[FEATURES]),
            "hybrid": physics * 10 ** hybrid.predict(matched[HYBRID_FEATURES])}
    for name, d in days.items():
        matched[name] = matched["EPOCH"] + pd.to_timedelta(d, unit="D")
    return matched


def main():
    fresh = "--fresh" in sys.argv
    tip = official_messages()
    print(f"{len(tip):,} official predictions for {tip['NORAD_CAT_ID'].nunique():,} objects")
    atm = MsisAtmosphere()
    m = our_predictions(tip, fresh_orbits(atm) if fresh else daily_orbits(tip, atm), atm)
    m = m.rename(columns={"DECAY_EPOCH": "official"})
    print(f"{len(m):,} have one of our orbits from within {MAX_ORBIT_AGE:.0f} days before\n")

    models = ["official", "physics", "XGBoost", "hybrid"]
    rows = {}
    for lo, hi, label in LEAD_BANDS:
        band = m[(m["lead_days"] >= lo) & (m["lead_days"] < hi)]
        row = {"messages": len(band), "objects": band["NORAD_CAT_ID"].nunique(),
               "our orbit's age (hours)": ((band["MSG_EPOCH"] - band["EPOCH"]).dt.total_seconds() / 3600).median()}
        for name in models:
            hours = (band[name] - band["reentry"]).abs().dt.total_seconds() / 3600
            row[f"{name} (hours)"] = hours.median()
            row[f"{name} (%)"] = (hours / 24 / band["lead_days"]).median()
        rows[label] = row
    table = pd.DataFrame(rows).T
    print("Median error of the predicted reentry time, by how long before reentry the prediction was made")
    fmt = {c: ("{:.0f}".format if c in ("messages", "objects") else "{:.1f}".format if "hours" in c
               else "{:.0%}".format) for c in table.columns}
    print(table.to_string(formatters=fmt))
    table.to_csv(RESULTS_DIR / f"official_comparison{'_fresh' if fresh else ''}.csv", float_format="%.4f")

    inside = (m["reentry"] - m["official"]).abs().dt.total_seconds() / 60 <= m["WINDOW"]
    print(f"\nthe true reentry falls inside the official +/- window for {inside.mean():.0%} of messages")


if __name__ == "__main__":
    main()
