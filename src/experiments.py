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

from atmosphere import MsisAtmosphere
from build_reentry_dataset import DATA_DIR
from orbit_decay import propagate
from physics_model import semi_major_axis

from train_reentry_model import (BASE_PARAMS, FEATURES, HORIZONS, HYBRID_FEATURES, UNTUNED_PARAMS, load,
                                 predict_hybrid, train_hybrid, train_xgboost)

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

    booster = train_xgboost(fit, FEATURES, y, UNTUNED_PARAMS)
    preds["XGBoost"] = 10 ** booster.predict(val[FEATURES])
    preds["hybrid (residual, default)"] = predict_hybrid(train_hybrid(fit, UNTUNED_PARAMS), val)

    gentle = xgb.XGBRegressor(**{**UNTUNED_PARAMS, "max_depth": 4, "min_child_weight": 50, "reg_lambda": 10.0,
                                 "n_estimators": 400}).fit(fit[HYBRID_FEATURES], residual)
    preds["hybrid (residual, shallow trees)"] = physics * 10 ** gentle.predict(val[HYBRID_FEATURES])

    as_feature = train_xgboost(fit, HYBRID_FEATURES, y, UNTUNED_PARAMS)
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
                params = {**BASE_PARAMS, "max_depth": depth, "min_child_weight": min_child, "reg_lambda": 10.0,
                          "n_estimators": trees, "learning_rate": rate}
                model = xgb.XGBRegressor(**params).fit(fit[HYBRID_FEATURES], residual)
                name = f"depth {depth}, min_child {min_child}, {trees} trees x {rate}"
                preds[name] = physics * 10 ** model.predict(val[HYBRID_FEATURES])
                print(f"{name}: {np.median(np.abs(preds[name] / truth - 1)):.1%}")
    print()
    show(scores(preds, truth).sort_values("overall"))


def xgboost_tuning():
    """The same grid for plain XGBoost, so the comparison with the hybrid is fair."""
    fit, val = validation_split()
    truth = val["days_left"].to_numpy()
    y = np.log10(fit["days_left"])
    untuned = train_xgboost(fit, FEATURES, y, UNTUNED_PARAMS)
    preds = {"default (depth 8, 600 trees x 0.05)": 10 ** untuned.predict(val[FEATURES])}
    for depth in (3, 4, 6):
        for min_child in (50, 500):
            params = {**BASE_PARAMS, "max_depth": depth, "min_child_weight": min_child, "reg_lambda": 10.0,
                      "n_estimators": 400}
            model = xgb.XGBRegressor(**params).fit(fit[FEATURES], y)
            preds[f"depth {depth}, min_child {min_child}, 400 trees"] = 10 ** model.predict(val[FEATURES])
    show(scores(preds, truth).sort_values("overall"))


def solar_forecast(sample=5000):
    """How much would a perfect forecast of solar and geomagnetic activity help the physics?

    No training is involved, so this uses test rows directly: a random sample of
    rows where the physics has a fitted drag parameter.
    """
    _, test = load(verbose=False)
    rows = test[test["B_fit"] > 0].sample(sample, random_state=0).reset_index(drop=True)
    truth = rows["days_left"].to_numpy()
    a, e, B = semi_major_axis(rows["MEAN_MOTION"]), rows["eccentricity"].to_numpy(), rows["B_fit"].to_numpy()
    atm = MsisAtmosphere()

    sw = pd.read_csv(DATA_DIR / "SW-All.csv", parse_dates=["DATE"])
    sw = sw.set_index("DATE")[["F10.7_OBS_CENTER81", "F10.7_OBS_LAST81", "AP_AVG"]].asfreq("D").ffill()
    day0 = sw.index[0]
    ap_daily = sw["AP_AVG"].to_numpy()
    start = ((rows["EPOCH"] - day0).dt.total_seconds() / 86400).to_numpy()
    doy0 = rows["EPOCH"].dt.day_of_year.to_numpy().astype(float)
    f_now, ap_now = rows["f107_81d"].to_numpy(), rows["ap_7d"].to_numpy()

    def run(density, max_step):
        return propagate(a, e, B, density, max_days=3650, max_step=max_step)

    def actual_weather(flux_column):
        flux = sw[flux_column].to_numpy()

        def density(h, idx, t):
            day = np.clip((start[idx, None] + t).astype(int), 0, len(flux) - 1)
            return atm.density(h, flux[day], ap_daily[day], (doy0[idx, None] + t) % 365.25)
        return density

    profile = atm.profile(f_now, ap_now)
    preds = {
        "today's weather held fixed, year-average season (baseline)": run(lambda h, idx, t: profile(h, idx), 30.0),
        "today's weather held fixed, with the seasons":
            run(lambda h, idx, t: atm.density(h, f_now[idx, None], ap_now[idx, None], (doy0[idx, None] + t) % 365.25), 5.0),
        # B was fitted with the trailing 81-day flux, so a consistent forecast uses the same index.
        "perfect forecast, centered 81-day flux": run(actual_weather("F10.7_OBS_CENTER81"), 2.0),
        "perfect forecast, trailing 81-day flux (as in the fit)": run(actual_weather("F10.7_OBS_LAST81"), 2.0),
    }
    preds = {k: np.where(np.isfinite(v), v, 3650.0) for k, v in preds.items()}
    show(scores(preds, truth))


EXPERIMENTS = {"hybrid_variants": hybrid_variants, "hybrid_tuning": hybrid_tuning, "xgboost_tuning": xgboost_tuning,
               "solar_forecast": solar_forecast}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        raise SystemExit(f"usage: python src/experiments.py [{' | '.join(EXPERIMENTS)}]")
    EXPERIMENTS[sys.argv[1]]()
