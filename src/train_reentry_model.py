"""Train and compare models that predict days until reentry (Part 2).

Trains on objects that reentered before TEST_FROM and tests on objects that reentered
after it, so every test object is one the models never saw (and from a later time).

Models compared:
- rule of thumb: extrapolate the current perigee decay rate down to 120 km
- linear regression on log(days left)
- XGBoost on log(days left)
- Gaussian process on a subsample, which also gives an uncertainty range

Prints error tables, saves plots to results/, and saves the XGBoost model to models/.

    python src/train_reentry_model.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "spacetrack" / "reentry_dataset.csv.gz"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

TEST_FROM = "2024-01-01"
FEATURES = ["PERIAPSIS", "APOAPSIS", "eccentricity", "INCLINATION", "MEAN_MOTION", "MEAN_MOTION_DOT", "BSTAR",
            "perigee_rate_7d", "apogee_rate_7d", "perigee_rate_30d", "apogee_rate_30d",
            "f107", "f107_81d", "ap", "ap_7d", "RCS", "is_starlink", "is_debris", "is_rocket_body"]
GP_FEATURES = ["PERIAPSIS", "APOAPSIS", "perigee_rate_7d", "perigee_rate_30d", "BSTAR", "f107_81d"]
GP_TRAIN_SIZE = 3000
HORIZONS = [(0, 7, "under 1 week"), (7, 30, "1-4 weeks"), (30, 90, "1-3 months"), (90, 366, "3-12 months")]


def load():
    df = pd.read_csv(DATA_FILE, parse_dates=["EPOCH", "DECAY_DATE"])
    df["is_debris"] = (df["OBJECT_TYPE"] == "DEB").astype(int)
    df["is_rocket_body"] = (df["OBJECT_TYPE"] == "R/B").astype(int)
    train, test = df[df["DECAY_DATE"] < TEST_FROM], df[df["DECAY_DATE"] >= TEST_FROM]
    print(f"train: {train['NORAD_CAT_ID'].nunique():,} objects, {len(train):,} rows (reentered before {TEST_FROM})")
    print(f"test:  {test['NORAD_CAT_ID'].nunique():,} objects, {len(test):,} rows (reentered {TEST_FROM} or later)\n")
    return train, test


def robust_scaler():
    """Rescale each feature by rank, so a few extreme values (maneuvers, mis-tagged orbits) can't dominate."""
    return QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=0)


def rule_of_thumb(df):
    """Days for the average altitude to reach 120 km at its recent sinking rate.

    Uses the average of perigee and apogee because in stretched orbits the apogee
    sinks first while the perigee barely moves.
    """
    altitude = (df["PERIAPSIS"] + df["APOAPSIS"]) / 2
    rate_7d = (df["perigee_rate_7d"] + df["apogee_rate_7d"]) / 2
    rate = rate_7d.fillna((df["perigee_rate_30d"] + df["apogee_rate_30d"]) / 2)
    days = (altitude - 120) / -rate
    return days.where(rate < 0).clip(0.1, 3650).fillna(365)  # not sinking: guess a year


def error_table(predictions, truth, groups):
    """Median error (%) of each model in each horizon, plus the share of predictions within 25%."""
    rows = {}
    for name, pred in predictions.items():
        err = np.abs(pred / truth - 1)
        row = {}
        for lo, hi, label in HORIZONS:
            mask = (truth >= lo) & (truth < hi)
            row[label] = f"{np.median(err[mask]):.0%}"
        row["overall"] = f"{np.median(err):.0%}"
        row["within 25%"] = f"{(err <= 0.25).mean():.0%}"
        rows[name] = row
    table = pd.DataFrame(rows).T
    counts = {label: f"{groups[(truth >= lo) & (truth < hi)].nunique():,} obj" for lo, hi, label in HORIZONS}
    return pd.concat([table, pd.DataFrame(counts, index=["(test objects)"])])


def plot_predictions(truth, pred, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    sample = np.random.default_rng(0).choice(len(truth), size=min(20000, len(truth)), replace=False)
    ax.scatter(truth[sample], pred[sample], s=1, alpha=0.15)
    ax.plot([0.1, 400], [0.1, 400], color="black", lw=1, label="perfect prediction")
    ax.fill_between([0.1, 400], [0.075, 300], [0.125, 500], color="grey", alpha=0.2, label="within 25%")
    ax.set(xscale="log", yscale="log", xlim=(0.1, 400), ylim=(0.1, 400),
           xlabel="actual days until reentry", ylabel="predicted days until reentry",
           title="XGBoost on objects that reentered 2024 or later")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    train, test = load()
    y_train = np.log10(train["days_left"])
    truth = test["days_left"].to_numpy()

    print("training linear regression...")
    linear = make_pipeline(SimpleImputer(strategy="median"), robust_scaler(), LinearRegression())
    linear.fit(train[FEATURES], y_train)

    print("training XGBoost...")
    booster = xgb.XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, tree_method="hist", n_jobs=-1)
    booster.fit(train[FEATURES], y_train)

    print(f"training Gaussian process on {GP_TRAIN_SIZE:,} rows...")
    gp_rows = train.dropna(subset=GP_FEATURES).sample(GP_TRAIN_SIZE, random_state=0)
    gp_scaler = make_pipeline(SimpleImputer(strategy="median"), robust_scaler()).fit(gp_rows[GP_FEATURES])
    kernel = ConstantKernel() * RBF(length_scale=np.ones(len(GP_FEATURES))) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(gp_scaler.transform(gp_rows[GP_FEATURES]), np.log10(gp_rows["days_left"]))
    # Predict in batches: all test rows at once would need tens of GB of memory.
    X_gp = gp_scaler.transform(test[GP_FEATURES])
    batches = [gp.predict(X_gp[k:k + 10_000], return_std=True) for k in range(0, len(X_gp), 10_000)]
    gp_mean = np.concatenate([mean for mean, _ in batches])
    gp_std = np.concatenate([std for _, std in batches])

    predictions = {
        "rule of thumb": rule_of_thumb(test).to_numpy(),
        "linear regression": 10 ** linear.predict(test[FEATURES]),
        "XGBoost": 10 ** booster.predict(test[FEATURES]),
        "Gaussian process": 10 ** gp_mean,
    }

    print("\nMedian error by how far away reentry was (lower is better)")
    print(error_table(predictions, truth, test["NORAD_CAT_ID"]).to_string())

    passive = test["OBJECT_TYPE"].isin(["DEB", "R/B"]).to_numpy()
    print("\nXGBoost by object type (payloads can maneuver; debris and rocket bodies can't)")
    by_type = {"debris + rocket bodies": passive, "payloads": ~passive & (test["is_starlink"] == 0).to_numpy(),
               "Starlink": (test["is_starlink"] == 1).to_numpy()}
    for name, mask in by_type.items():
        if not mask.any():
            continue
        err = np.abs(predictions["XGBoost"][mask] / truth[mask] - 1)
        print(f"  {name:<24} median error {np.median(err):.0%}, within 25%: {(err <= 0.25).mean():.0%}")

    inside = np.abs(np.log10(truth) - gp_mean) <= 1.645 * gp_std
    print(f"\nGaussian process 90% range contains the true answer {inside.mean():.0%} of the time (ideal: 90%)")

    importance = pd.Series(booster.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nXGBoost's most useful features:")
    print(importance.head(8).round(3).to_string())

    plot_predictions(truth, predictions["XGBoost"], RESULTS_DIR / "xgboost_predicted_vs_actual.png")
    booster.save_model(MODELS_DIR / "reentry_xgb.json")
    print(f"\nsaved plot to {RESULTS_DIR / 'xgboost_predicted_vs_actual.png'} and model to {MODELS_DIR / 'reentry_xgb.json'}")


if __name__ == "__main__":
    main()
