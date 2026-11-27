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

from train_reentry_model import (BASE_PARAMS, FEATURES, HORIZONS, HYBRID_FEATURES, HYBRID_PARAMS, NO_PHYSICS_DAYS,
                                 UNTUNED_PARAMS, load, predict_hybrid, train_hybrid, train_xgboost)

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

    def cycle_weather():
        from solar_cycle import forecast_monthly

        start = rows["EPOCH"].dt.to_period("M").dt.start_time
        forecasts = {d: forecast_monthly(d, 122) for d in start.unique()}
        flux = np.array([forecasts[d] for d in start])
        return lambda h, idx, t: atm.density(
            h, flux[idx[:, None], np.clip((t / 30.44).astype(int), 0, flux.shape[1] - 1)], ap_now[idx, None])

    profile = atm.profile(f_now, ap_now)
    preds = {
        "today's weather held fixed, year-average season (baseline)": run(lambda h, idx, t: profile(h, idx), 30.0),
        "today's weather held fixed, with the seasons":
            run(lambda h, idx, t: atm.density(h, f_now[idx, None], ap_now[idx, None], (doy0[idx, None] + t) % 365.25), 5.0),
        # B was fitted with the trailing 81-day flux, so a consistent forecast uses the same index.
        "perfect forecast, centered 81-day flux": run(actual_weather("F10.7_OBS_CENTER81"), 2.0),
        "perfect forecast, trailing 81-day flux (as in the fit)": run(actual_weather("F10.7_OBS_LAST81"), 2.0),
        "solar-cycle forecast (solar_cycle.py), Ap held fixed": run(cycle_weather(), 15.0),
    }
    preds = {k: np.where(np.isfinite(v), v, 3650.0) for k, v in preds.items()}
    show(scores(preds, truth))


def add_drag_stability(df):
    """log10 of B over 7, 14 and 60 days relative to B over 30 days: near 0 when drag is steady."""
    for w in (7, 14, 60):
        ratio = df[f"B_fit_{w}d"] / df["B_fit"]
        df[f"B_ratio_{w}d"] = np.log10(ratio.where(ratio > 0))
    return df


STABILITY_FEATURES = ["B_ratio_7d", "B_ratio_14d", "B_ratio_60d"]


def drag_stability():
    """Which window to fit B over, and whether telling the hybrid how steady B is helps."""
    fit, val = (add_drag_stability(d.copy()) for d in validation_split())
    truth = val["days_left"].to_numpy()
    preds = {}
    # With the weather held fixed, lifetime scales exactly as 1/B, so other windows need no new simulation.
    for w in (7, 14, 30, 60):
        B = val["B_fit"] if w == 30 else val[f"B_fit_{w}d"]
        scaled = val["physics_days"] * val["B_fit"] / B
        preds[f"physics, B fitted over {w} days"] = scaled.where(B > 0).fillna(NO_PHYSICS_DAYS).clip(lower=0.01).to_numpy()

    preds["hybrid"] = predict_hybrid(train_hybrid(fit), val)
    features = HYBRID_FEATURES + STABILITY_FEATURES
    residual = np.log10(fit["days_left"] / fit["physics_filled"])
    model = xgb.XGBRegressor(**HYBRID_PARAMS).fit(fit[features], residual)
    preds["hybrid + drag stability"] = val["physics_filled"].to_numpy() * 10 ** model.predict(val[features])
    show(scores(preds, truth))
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nimportance of the stability inputs:", importance[STABILITY_FEATURES].round(3).to_dict())


def final_days_window():
    """In the final days, is B better fitted over a short recent window, using every orbit?

    Uses the official-comparison setup (compare_official.py --fresh). No training is
    involved, so scoring on test objects is fair.
    """
    from compare_official import LEAD_BANDS, official_messages
    from download_final_orbits import OUT_FILE as FINAL_ORBITS_FILE
    from physics_model import fit_drag_parameter, forecast, prepare, unit_decay_rate
    from predict_reentry import features

    atm = MsisAtmosphere()
    tip = official_messages()
    orbits = prepare(features(pd.read_parquet(FINAL_ORBITS_FILE)))
    g = unit_decay_rate(orbits, atm)
    rows = {}
    for window in (1, 2, 4, 8, 30):
        orbits["B_w"] = fit_drag_parameter(orbits, g, window)[0]
        m = pd.merge_asof(tip.sort_values("MSG_EPOCH"), orbits.sort_values("EPOCH"), left_on="MSG_EPOCH",
                          right_on="EPOCH", by="NORAD_CAT_ID", direction="backward")
        m = m[(m["MSG_EPOCH"] - m["EPOCH"]).dt.total_seconds() / 86400 <= 2].reset_index(drop=True)
        days = forecast(m, m["B_w"].to_numpy(), atm)
        hours = (m["EPOCH"] + pd.to_timedelta(np.nan_to_num(days, nan=365.0), unit="D") - m["reentry"]).abs()
        hours = hours.dt.total_seconds() / 3600
        rows[f"B over the last {window} days"] = {label: hours[(m["lead_days"] >= lo) & (m["lead_days"] < hi)].median()
                                                 for lo, hi, label in LEAD_BANDS}
    print("Median error (hours) of the physics prediction, by time before reentry")
    print(pd.DataFrame(rows).T.to_string(float_format="{:.1f}".format))


def hybrid_corrections():
    """What does the hybrid correct? Its correction factor vs. the one actually needed, by condition."""
    import matplotlib.pyplot as plt

    from train_reentry_model import RESULTS_DIR, object_groups

    train, test = load(verbose=False)
    hybrid = train_hybrid(train)
    has_physics = test["physics_days"].notna()
    test = test[has_physics].copy()
    test["learned"] = 10 ** hybrid.predict(test[HYBRID_FEATURES])
    test["needed"] = test["days_left"] / test["physics_filled"]
    groups = object_groups(test)
    test["group"] = np.select(list(groups.values()), list(groups.keys()), default="other")
    bins = {"solar flux (81-day F10.7)": pd.cut(test["f107_81d"], [0, 100, 130, 160, 190, 400]),
            "physics prediction (days)": pd.cut(test["physics_days"], [0, 7, 30, 90, 365, 4000]),
            "object group": test["group"]}
    fig, axes = plt.subplots(1, len(bins), figsize=(16, 4.5), sharey=True)
    for ax, (name, by) in zip(axes, bins.items()):
        table = test.groupby(by, observed=True)[["needed", "learned"]].median()
        print(f"\nMedian correction factor (actual / physics) by {name}")
        print(table.round(2).to_string())
        x = np.arange(len(table))
        ax.bar(x - 0.2, table["needed"], 0.4, label="needed (actual / physics)")
        ax.bar(x + 0.2, table["learned"], 0.4, label="learned by the hybrid")
        ax.axhline(1, color="black", lw=0.8)
        ax.set_xticks(x, [str(i) for i in table.index], rotation=20)
        ax.set_title(name)
    axes[0].set_ylabel("multiply the physics prediction by")
    axes[0].legend()
    fig.suptitle("What the hybrid corrects (test set, rows where the physics has an answer)")
    fig.savefig(RESULTS_DIR / "hybrid_corrections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def drag_cap():
    """Are physically extreme drag fits a sign that something other than drag moves the orbit?

    Rules are chosen on the validation year; the physics has no training, so the
    test set is scored too. Rows a rule flags get no forecast, so the share kept is shown.
    """
    from uncertainty import band_of, conformal_quantile

    _, val = validation_split()
    _, test = load(verbose=False)
    for name, df in [("validation 2023", val), ("test 2024+", test)]:
        df = df[np.isfinite(df["physics_days"]) & (df["B_fit"] > 0)]
        rules = {"no rule": df["B_fit"] > np.inf, "B > 0.5 m^2/kg": df["B_fit"] > 0.5,
                 "B > 1 m^2/kg": df["B_fit"] > 1, "apogee rising over 30 days": df["apogee_rate_30d"] > 0}
        rows = {}
        for rule, flagged in rules.items():
            kept = df[~flagged]
            row = scores({rule: kept["physics_days"].to_numpy()}, kept["days_left"].to_numpy()).iloc[0].to_dict()
            row["rows kept"] = len(kept) / len(df)
            score = np.abs(np.log10(kept["days_left"] / kept["physics_days"]))
            bands = band_of(kept["physics_days"].to_numpy())
            for band, label in [(1, "90% range, predicted 3-10 days"), (2, "predicted 10-30 days")]:
                row[label] = 10 ** conformal_quantile(score[bands == band])
            rows[rule] = row
        print(f"\n{name}: physics median error, and the factor a 90% range spans each way")
        table = pd.DataFrame(rows).T
        print(table.to_string(formatters={c: ("{:.2f}x".format if "predicted" in c else "{:.0%}".format)
                                          for c in table.columns}))


EXPERIMENTS = {"hybrid_corrections": hybrid_corrections, "hybrid_variants": hybrid_variants, "hybrid_tuning": hybrid_tuning, "xgboost_tuning": xgboost_tuning,
               "solar_forecast": solar_forecast, "drag_stability": drag_stability,
               "final_days_window": final_days_window, "drag_cap": drag_cap}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        raise SystemExit(f"usage: python src/experiments.py [{' | '.join(EXPERIMENTS)}]")
    EXPERIMENTS[sys.argv[1]]()
