"""Experiments behind EXPERIMENTS.md.

Choices (model settings, fit window, ...) are made on a validation year, never on
the test set: fit on objects that reentered before 2023, score on those that
reentered during 2023. Only the winning choice is then retrained on everything
before 2024 and scored on the test set by train_reentry_model.py.

    python src/experiments.py hybrid_variants
"""

import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from train_reentry_model import (FEATURES, HORIZONS, HYBRID_FEATURES, XGB_PARAMS, load, predict_hybrid,
                                 train_hybrid, train_xgboost)

VALIDATE_FROM = "2023-01-01"


def validation_split():
    before_test, _ = load(verbose=False)
    fit = before_test[before_test["DECAY_DATE"] < VALIDATE_FROM]
    val = before_test[before_test["DECAY_DATE"] >= VALIDATE_FROM]
    print(f"fit: {fit['NORAD_CAT_ID'].nunique():,} objects; validate: {val['NORAD_CAT_ID'].nunique():,} objects "
          f"(reentered {VALIDATE_FROM[:4]})\n")
    return fit, val


def scores(predictions, truth):
    rows = {}
    for name, pred in predictions.items():
        err = np.abs(pred / truth - 1)
        row = {label: np.median(err[(truth >= lo) & (truth < hi)]) for lo, hi, label in HORIZONS}
        row["overall"] = np.median(err)
        row["within 25%"] = (err <= 0.25).mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def show(table):
    print(table.to_string(float_format="{:.0%}".format))


def hybrid_variants():
    """Ways to combine the physics prediction with ML."""
    fit, val = validation_split()
    truth = val["days_left"].to_numpy()
    y = np.log10(fit["days_left"])
    residual = np.log10(fit["days_left"] / fit["physics_filled"])
    physics = val["physics_filled"].to_numpy()
    preds = {"physics": physics}

    booster = train_xgboost(fit, FEATURES, y)
    preds["XGBoost"] = 10 ** booster.predict(val[FEATURES])
    preds["hybrid (residual, default)"] = predict_hybrid(train_hybrid(fit), val)

    gentle = xgb.XGBRegressor(**{**XGB_PARAMS, "max_depth": 4, "min_child_weight": 50, "reg_lambda": 10.0,
                                 "n_estimators": 400}).fit(fit[HYBRID_FEATURES], residual)
    preds["hybrid (residual, shallow trees)"] = physics * 10 ** gentle.predict(val[HYBRID_FEATURES])

    as_feature = train_xgboost(fit, HYBRID_FEATURES, y)
    preds["XGBoost + physics as a feature"] = 10 ** as_feature.predict(val[HYBRID_FEATURES])

    trusted = (val["physics_days"].notna() & (val["is_starlink"] == 0)).to_numpy()
    preds["physics where it has an answer, else XGBoost"] = np.where(trusted, physics, preds["XGBoost"])

    logs = np.log10(np.column_stack([preds["XGBoost"], preds["XGBoost + physics as a feature"],
                                     preds["hybrid (residual, default)"]]))
    preds["average of the three ML models (log scale)"] = 10 ** logs.mean(axis=1)
    show(scores(preds, truth))


def hybrid_tuning():
    """Small grid over how simple the hybrid's correction trees are."""
    fit, val = validation_split()
    truth = val["days_left"].to_numpy()
    residual = np.log10(fit["days_left"] / fit["physics_filled"])
    physics = val["physics_filled"].to_numpy()
    preds = {}
    for depth in (2, 3, 4, 6):
        for min_child in (50, 500):
            for trees, rate in ((400, 0.05), (1000, 0.02)):
                params = {**XGB_PARAMS, "max_depth": depth, "min_child_weight": min_child, "reg_lambda": 10.0,
                          "n_estimators": trees, "learning_rate": rate}
                model = xgb.XGBRegressor(**params).fit(fit[HYBRID_FEATURES], residual)
                name = f"depth {depth}, min_child {min_child}, {trees} trees x {rate}"
                preds[name] = physics * 10 ** model.predict(val[HYBRID_FEATURES])
                print(f"{name}: {np.median(np.abs(preds[name] / truth - 1)):.1%}")
    print()
    show(scores(preds, truth).sort_values("overall"))


EXPERIMENTS = {"hybrid_variants": hybrid_variants, "hybrid_tuning": hybrid_tuning}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        raise SystemExit(f"usage: python src/experiments.py [{' | '.join(EXPERIMENTS)}]")
    EXPERIMENTS[sys.argv[1]]()
