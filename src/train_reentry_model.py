"""Train and compare models that predict days until reentry (Part 2).

Trains on objects that reentered before TEST_FROM and tests on objects that reentered
after it, so every test object is one the models never saw (and from a later time).
Uses the one-orbit-a-week rows that physics_model.py simulated, so every model is
scored on exactly the same rows.

Models compared:
- rule of thumb: extrapolate the current sinking rate down to 120 km
- physics: drag simulation with a fitted drag parameter (physics_model.py)
- linear regression on log(days left)
- XGBoost on log(days left)
- Gaussian process on a subsample, which also gives an uncertainty range
- hybrid: XGBoost that learns how far off the physics prediction is

Prints error tables, saves plots and the error table to results/, and saves the
models to models/.

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
DATA_FILE = ROOT / "data" / "spacetrack" / "reentry_dataset.parquet"
PHYSICS_FILE = ROOT / "data" / "spacetrack" / "physics_predictions.parquet"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

TEST_FROM = "2024-01-01"
FEATURES = ["PERIAPSIS", "APOAPSIS", "eccentricity", "INCLINATION", "MEAN_MOTION", "MEAN_MOTION_DOT", "BSTAR",
            "perigee_rate_7d", "apogee_rate_7d", "perigee_rate_30d", "apogee_rate_30d",
            "f107", "f107_81d", "ap", "ap_7d", "RCS", "is_starlink", "is_debris", "is_rocket_body"]
HYBRID_FEATURES = FEATURES + ["log_physics_days", "log_B", "n_fit"]
GP_FEATURES = ["PERIAPSIS", "APOAPSIS", "perigee_rate_7d", "perigee_rate_30d", "BSTAR", "f107_81d"]
GP_TRAIN_SIZE = 3000
NO_PHYSICS_DAYS = 365  # used when the physics model has no answer (orbit raised, too few orbits, or 10+ years)
HORIZONS = [(0, 7, "under 1 week"), (7, 30, "1-4 weeks"), (30, 90, "1-3 months"), (90, 366, "3-12 months")]
XGB_PARAMS = dict(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=-1)


def load(test_from=TEST_FROM, verbose=True):
    df = pd.read_parquet(DATA_FILE).merge(pd.read_parquet(PHYSICS_FILE), on=["NORAD_CAT_ID", "EPOCH"])
    df["is_debris"] = (df["OBJECT_TYPE"] == "DEB").astype(int)
    df["is_rocket_body"] = (df["OBJECT_TYPE"] == "R/B").astype(int)
    df["log_physics_days"] = np.log10(df["physics_days"].clip(lower=0.01))
    df["log_B"] = np.log10(df["B_fit"].where(df["B_fit"] > 0))
    df["physics_filled"] = df["physics_days"].fillna(NO_PHYSICS_DAYS).clip(lower=0.01)
    train, test = df[df["DECAY_DATE"] < test_from], df[df["DECAY_DATE"] >= test_from]
    if verbose:
        print(f"train: {train['NORAD_CAT_ID'].nunique():,} objects, {len(train):,} rows (reentered before {test_from})")
        print(f"test:  {test['NORAD_CAT_ID'].nunique():,} objects, {len(test):,} rows (reentered {test_from} or later)\n")
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


def train_xgboost(train, features, target):
    booster = xgb.XGBRegressor(**XGB_PARAMS)
    booster.fit(train[features], target)
    return booster


def train_hybrid(train):
    """XGBoost on log10(actual / physics): it learns a correction factor for the physics prediction."""
    return train_xgboost(train, HYBRID_FEATURES, np.log10(train["days_left"] / train["physics_filled"]))


def predict_hybrid(booster, df):
    return df["physics_filled"].to_numpy() * 10 ** booster.predict(df[HYBRID_FEATURES])


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


def object_groups(df):
    passive = df["OBJECT_TYPE"].isin(["DEB", "R/B"]).to_numpy()
    starlink = (df["is_starlink"] == 1).to_numpy()
    return {"debris + rocket bodies": passive, "payloads": ~passive & ~starlink, "Starlink": starlink}


def plot_predictions(truth, predictions, path):
    """Predicted vs. actual days, one panel per model."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.5 * len(predictions), 5.5), sharey=True)
    sample = np.random.default_rng(0).choice(len(truth), size=min(20000, len(truth)), replace=False)
    for ax, (name, pred) in zip(np.atleast_1d(axes), predictions.items()):
        ax.scatter(truth[sample], pred[sample], s=1, alpha=0.15)
        ax.plot([0.1, 400], [0.1, 400], color="black", lw=1, label="perfect prediction")
        ax.fill_between([0.1, 400], [0.075, 300], [0.125, 500], color="grey", alpha=0.2, label="within 25%")
        ax.set(xscale="log", yscale="log", xlim=(0.1, 400), ylim=(0.1, 400), title=name,
               xlabel="actual days until reentry")
    np.atleast_1d(axes)[0].set_ylabel("predicted days until reentry")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.suptitle("Objects that reentered 2024 or later")
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
    booster = train_xgboost(train, FEATURES, y_train)

    print("training physics + ML hybrid...")
    hybrid = train_hybrid(train)

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
        "physics": test["physics_filled"].to_numpy(),
        "linear regression": 10 ** linear.predict(test[FEATURES]),
        "XGBoost": 10 ** booster.predict(test[FEATURES]),
        "Gaussian process": 10 ** gp_mean,
        "physics + ML hybrid": predict_hybrid(hybrid, test),
    }

    table = error_table(predictions, truth, test["NORAD_CAT_ID"])
    print("\nMedian error by how far away reentry was (lower is better)")
    print(table.to_string())
    table.to_csv(RESULTS_DIR / "error_table.csv")

    has_physics = test["physics_days"].notna().to_numpy()
    print(f"\nphysics has its own answer for {has_physics.mean():.0%} of test rows; on those rows:")
    print(error_table({k: v[has_physics] for k, v in predictions.items()}, truth[has_physics],
                      test.loc[has_physics, "NORAD_CAT_ID"]).to_string())

    print("\nMedian error by object type (payloads can maneuver; debris and rocket bodies can't)")
    by_type = {}
    for group, mask in object_groups(test).items():
        if mask.any():
            by_type[group] = {name: f"{np.median(np.abs(p[mask] / truth[mask] - 1)):.0%}"
                              for name, p in predictions.items()}
    print(pd.DataFrame(by_type).to_string())

    inside = np.abs(np.log10(truth) - gp_mean) <= 1.645 * gp_std
    print(f"\nGaussian process 90% range contains the true answer {inside.mean():.0%} of the time (ideal: 90%)")

    for name, model, features in (("XGBoost", booster, FEATURES), ("hybrid", hybrid, HYBRID_FEATURES)):
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print(f"\n{name}'s most useful features:")
        print(importance.head(6).round(3).to_string())

    shown = {k: predictions[k] for k in ("physics", "XGBoost", "physics + ML hybrid")}
    plot_predictions(truth, shown, RESULTS_DIR / "predicted_vs_actual.png")
    booster.save_model(MODELS_DIR / "reentry_xgb.json")
    hybrid.save_model(MODELS_DIR / "reentry_hybrid.json")
    print(f"\nsaved results to {RESULTS_DIR} and models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
