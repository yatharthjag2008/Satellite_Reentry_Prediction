"""Find where the saved XGBoost model goes most wrong on the test set.

The predicted-vs-actual plot shows a cloud of objects with 100-365 days left that
the model expects to fall within a few months. This script breaks those badly low
predictions down by object type and by the space weather at the time.

    python src/analyze_errors.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from train_reentry_model import FEATURES, MODELS_DIR, load

FAR = 100      # days left: "far from reentry"
TOO_LOW = 0.5  # a prediction under half the true value counts as badly low


def object_group(df):
    passive = df["OBJECT_TYPE"].isin(["DEB", "R/B"])
    return np.select([passive, df["is_starlink"] == 1], ["debris + rocket bodies", "Starlink"], "other payloads")


def main():
    train, test = load()
    booster = xgb.XGBRegressor()
    booster.load_model(MODELS_DIR / "reentry_xgb.json")
    test = test.assign(pred=10 ** booster.predict(test[FEATURES]))

    far = test[test["days_left"] >= FAR].copy()
    far["badly_low"] = far["pred"] < TOO_LOW * far["days_left"]
    far["group"] = object_group(far)
    print(f"{far['badly_low'].mean():.0%} of predictions with {FAR}+ days left are under half the true value\n")

    summary = far.groupby("group").agg(rows=("badly_low", "size"), badly_low=("badly_low", "mean"))
    summary["share of all badly low"] = far[far["badly_low"]].groupby("group").size() / far["badly_low"].sum()
    print(summary.to_string(formatters={"rows": "{:,}".format, "badly_low": "{:.0%}".format,
                                        "share of all badly low": "{:.0%}".format}))

    print("\nMedian conditions (badly low vs. all far rows):")
    cols = ["PERIAPSIS", "perigee_rate_7d", "MEAN_MOTION_DOT", "f107_81d", "ap_7d"]
    compare = pd.DataFrame({"badly low": far.loc[far["badly_low"], cols].median(), "all": far[cols].median()})
    print(compare.round(4).to_string())

    # Solar flux this high is rare before 2024: how much of the training data looks like it?
    high = far.loc[far["badly_low"], "f107_81d"].median()
    print(f"\nshare of training rows with 81-day solar flux above {high:.0f}: {(train['f107_81d'] > high).mean():.1%}")


if __name__ == "__main__":
    main()
