"""Part 1: compare physics, ML and a hybrid for whole lifetimes from the first weeks in orbit.

Every object here has already reentered, so a time split would be biased: of the
objects launched recently, only the short-lived ones have come down yet. Instead
the models are cross-validated by launch (all objects from one launch date are in
the same fold), so no model is tested on a sibling of an object it trained on.

Run lifetime_model.py first.

    python src/train_lifetime_model.py
    python src/train_lifetime_model.py --time-split   # the time-split check (EXPERIMENTS.md 019)
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from lifetime_model import MAX_YEARS, OUT_FILE, error_by_band, object_groups
from train_reentry_model import HYBRID_PARAMS, RESULTS_DIR, XGB_PARAMS

PHYSICS = "solar-cycle forecast"
FEATURES = ["PERIAPSIS", "APOAPSIS", "eccentricity", "INCLINATION", "MEAN_MOTION", "MEAN_MOTION_DOT", "BSTAR",
            "f107_81d", "RCS", "is_rocket_body", "is_starlink", "log_B", "n_fit"]
HYBRID_FEATURES = FEATURES + ["log_physics_days"]
FOLDS = 5


def load():
    df = pd.read_parquet(OUT_FILE)
    df["is_rocket_body"] = (df["OBJECT_TYPE"] == "R/B").astype(int)
    df["log_B"] = np.log10(df["B_fit"].where(df["B_fit"] > 0))
    df["physics_filled"] = df[PHYSICS].fillna(MAX_YEARS * 365.25).clip(lower=0.5)
    df["log_physics_days"] = np.log10(df["physics_filled"])
    return df


def cross_validated(df):
    """Out-of-fold predictions of XGBoost and the hybrid, folds grouped by launch date."""
    xgb_pred, hybrid_pred = np.empty(len(df)), np.empty(len(df))
    y = np.log10(df["days_left"])
    residual = y - df["log_physics_days"]
    for train, test in GroupKFold(FOLDS).split(df, groups=df["LAUNCH_DATE"]):
        model = xgb.XGBRegressor(**XGB_PARAMS).fit(df.iloc[train][FEATURES], y.iloc[train])
        xgb_pred[test] = 10 ** model.predict(df.iloc[test][FEATURES])
        hybrid = xgb.XGBRegressor(**HYBRID_PARAMS).fit(df.iloc[train][HYBRID_FEATURES], residual.iloc[train])
        hybrid_pred[test] = df.iloc[test]["physics_filled"].to_numpy() * 10 ** hybrid.predict(df.iloc[test][HYBRID_FEATURES])
    return xgb_pred, hybrid_pred


def plot(df, predictions, path):
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.5 * len(predictions), 5.5), sharey=True)
    truth = df["days_left"].to_numpy() / 365.25
    for ax, (name, pred) in zip(axes, predictions.items()):
        ax.scatter(truth, pred / 365.25, s=4, alpha=0.3)
        ax.plot([0.01, 40], [0.01, 40], color="black", lw=1, label="perfect prediction")
        ax.fill_between([0.01, 40], [0.0075, 30], [0.0125, 50], color="grey", alpha=0.2, label="within 25%")
        ax.set(xscale="log", yscale="log", xlim=(0.05, 30), ylim=(0.05, 30), title=name,
               xlabel="actual years until reentry")
    axes[0].set_ylabel("predicted years until reentry")
    axes[0].legend(loc="upper left")
    fig.suptitle("Lifetime from the first 40 days in orbit (cross-validated by launch)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def time_split():
    """A time split that is fair despite censoring.

    Keep launches up to CUTOFF_LAUNCH whose objects lived at most MAX_LIFE years.
    Every such object launched by then has reentered by 2026, so train (launched
    before SPLIT) and test (launched SPLIT to CUTOFF_LAUNCH) are censored the same way.
    """
    SPLIT, CUTOFF_LAUNCH, MAX_LIFE = "2012-01-01", "2016-01-01", 10
    df = load()
    life_years = (df["DECAY_DATE"] - df["LAUNCH_DATE"]).dt.days / 365.25
    df = df[(df["LAUNCH_DATE"] < CUTOFF_LAUNCH) & (life_years <= MAX_LIFE)].reset_index(drop=True)
    train, test = df[df["LAUNCH_DATE"] < SPLIT], df[df["LAUNCH_DATE"] >= SPLIT]
    print(f"train: {len(train):,} objects launched before {SPLIT[:4]}; test: {len(test):,} launched "
          f"{SPLIT[:4]}-{int(CUTOFF_LAUNCH[:4]) - 1}; all lived at most {MAX_LIFE} years\n")
    y = np.log10(train["days_left"])
    booster = xgb.XGBRegressor(**XGB_PARAMS).fit(train[FEATURES], y)
    hybrid = xgb.XGBRegressor(**HYBRID_PARAMS).fit(train[HYBRID_FEATURES], y - train["log_physics_days"])
    predictions = {"physics (solar-cycle forecast)": test[PHYSICS].to_numpy(),
                   "XGBoost": 10 ** booster.predict(test[FEATURES]),
                   "physics + ML hybrid": test["physics_filled"].to_numpy() * 10 ** hybrid.predict(test[HYBRID_FEATURES])}
    for group, mask in object_groups(test).items():
        if mask.any():
            print(f"{group} ({mask.sum():,} objects)")
            print(error_by_band(test[mask], {k: v[mask] for k, v in predictions.items()})
                  .to_string(float_format="{:.0%}".format) + "\n")


def main():
    if "--time-split" in sys.argv:
        return time_split()
    df = load()
    print(f"{len(df):,} objects from {df['LAUNCH_DATE'].nunique():,} launches\n")
    xgb_pred, hybrid_pred = cross_validated(df)
    predictions = {"physics (today's flux held fixed)": df["today's flux held fixed"].to_numpy(),
                   "physics (solar-cycle forecast)": df[PHYSICS].to_numpy(),
                   "physics (perfect solar forecast)": df["actual flux (perfect forecast)"].to_numpy(),
                   "XGBoost": xgb_pred, "physics + ML hybrid": hybrid_pred}

    tables = []
    for group, mask in object_groups(df).items():
        if not mask.any():
            continue
        table = error_by_band(df[mask], {k: v[mask] for k, v in predictions.items()})
        print(f"{group} ({mask.sum():,} objects): median error by true remaining lifetime")
        print(table.to_string(float_format="{:.0%}".format) + "\n")
        tables.append(table.assign(group=group))
    pd.concat(tables).to_csv(RESULTS_DIR / "lifetime_error_table.csv", float_format="%.4f")
    plot(df, {k: predictions[k] for k in ("physics (solar-cycle forecast)", "XGBoost", "physics + ML hybrid")},
         RESULTS_DIR / "lifetime_predicted_vs_actual.png")
    print(f"saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
