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


def cap_physics(df, max_b=1.0):
    """The rows' physics answer removed where the fitted B is too large to be drag (experiment 021)."""
    df = df.copy()
    flagged = df["B_fit"] > max_b
    df.loc[flagged, "physics_days"] = np.nan
    df.loc[flagged, "log_physics_days"] = np.nan
    df.loc[flagged, "physics_filled"] = NO_PHYSICS_DAYS
    return df


def hybrid_drag_cap():
    """Does the hybrid do better if the physics gives no answer when B > 1 m^2/kg?"""
    fit, val = validation_split()
    truth = val["days_left"].to_numpy()
    preds = {"physics": val["physics_filled"].to_numpy(),
             "physics, capped": cap_physics(val)["physics_filled"].to_numpy(),
             "hybrid": predict_hybrid(train_hybrid(fit), val),
             "hybrid, capped": predict_hybrid(train_hybrid(cap_physics(fit)), cap_physics(val))}
    show(scores(preds, truth))
    flagged = (val["B_fit"] > 1.0).to_numpy()
    print(f"\n{flagged.mean():.1%} of rows flagged; median error on them:")
    show(scores({name: pred[flagged] for name, pred in preds.items()}, truth[flagged]))


def final_days_ranges(split="2025-07-01"):
    """Do 90% ranges for the 8-day-B physics, calibrated on every orbit, hold on later reentries?

    Calibrated on objects that reentered before `split`, checked on those after it,
    against the reentry time of the final official (TIP) message.
    """
    from forecast_ranges import fresh_predictions, widths_table
    from uncertainty import band_of

    rows = fresh_predictions(MsisAtmosphere())
    before, after = rows[rows["reentry"] < split], rows[rows["reentry"] >= split]
    print(f"calibrate: {before['NORAD_CAT_ID'].nunique():,} objects; check: {after['NORAD_CAT_ID'].nunique():,} "
          f"objects (reentered from {split})")
    table, _ = widths_table(before["pred"].to_numpy(), before["truth"].to_numpy(), before["group"].to_numpy())
    bands = band_of(after["pred"].to_numpy())
    width = np.array([table[g][b] for g, b in zip(after["group"], bands)], dtype=float)
    low, high = after["pred"] / 10 ** width, after["pred"] * 10 ** width
    inside = (after["truth"] >= low) & (after["truth"] <= high)
    hours = (high - low) * 24
    out = {}
    for label, m in [("predicted under 3 days", bands == 0), ("predicted 3-10 days", bands == 1)]:
        for name, g in [("debris and rocket bodies", 0), ("other payloads", 1), ("Starlink", 2), ("all", None)]:
            cell = m & ((after["group"] == g).to_numpy() if g is not None else True)
            out[(label, name)] = {"rows": int(cell.sum()), "contains truth": inside[cell].mean(),
                                  "median width (hours)": hours[cell].median()}
    print(pd.DataFrame(out).T.to_string(formatters={"contains truth": "{:.0%}".format,
                                                     "median width (hours)": "{:.0f}".format}))


def storm_bound(storm_ap=50):
    """In the final week, how much could a forecast of geomagnetic storms help the physics?

    An upper bound: the 8-day-B physics from every orbit (EXPERIMENTS.md 015) is
    rerun with the 3-hourly ap that actually happened until reentry, against the
    usual 7-day average held fixed. No training is involved, so test objects are used.
    """
    from compare_official import LEAD_BANDS
    from forecast_ranges import fresh_predictions

    atm = MsisAtmosphere()
    rows = fresh_predictions(atm)
    rows = rows[rows["truth"] <= 7].reset_index(drop=True)
    sw = pd.read_csv(DATA_DIR / "SW-All.csv", parse_dates=["DATE"]).set_index("DATE").asfreq("D").ffill()
    ap3 = sw[[f"AP{i}" for i in range(1, 9)]].to_numpy().ravel()  # every 3 hours from the first day
    start = ((rows["EPOCH"] - sw.index[0]).dt.total_seconds() / 10800).to_numpy()
    a, e, B = rows["a_km"].to_numpy(), rows["eccentricity"].to_numpy(), rows["B_short"].to_numpy()
    f107a, ap_7d = rows["f107_81d"].to_numpy(), rows["ap_7d"].to_numpy()

    def actual(h, idx, t):
        slot = np.clip((start[idx, None] + t * 8).astype(int), 0, len(ap3) - 1)
        return atm.density(h, f107a[idx, None], ap3[slot])

    held = rows["pred"].to_numpy()
    perfect = propagate(a, e, B, actual, max_days=40, max_step=0.125)
    end = np.clip((start + rows["truth"].to_numpy() * 8).astype(int), 0, len(ap3) - 1)
    peak = np.array([ap3[int(s):max(int(s) + 1, n + 1)].max() for s, n in zip(start, end)])
    stormy = peak >= storm_ap
    print(f"{len(rows):,} orbits of {rows['NORAD_CAT_ID'].nunique():,} objects; "
          f"{stormy.mean():.0%} with ap reaching {storm_ap}+ before reentry")
    out = {}
    for name, m in [("all", np.ones(len(rows), bool)), (f"storm (ap {storm_ap}+) before reentry", stormy),
                    ("no storm", ~stormy)]:
        for label, pred in [("7-day ap held fixed (now)", held), ("actual 3-hourly ap (perfect forecast)", perfect)]:
            hours = np.abs(pred - rows["truth"].to_numpy()) * 24
            out[(name, label)] = {lead: np.nanmedian(hours[m & (rows["truth"] >= lo).to_numpy() & (rows["truth"] < hi).to_numpy()])
                                  for lo, hi, lead in LEAD_BANDS}
    print("Median error (hours) by time before reentry")
    print(pd.DataFrame(out).T.to_string(float_format="{:.1f}".format))


def transfer_orbits(max_step=1.0):
    """Part 1: do the Moon's and Sun's gravity fix the lifetimes of transfer orbits?

    The same objects, starting orbits, fitted B and solar-cycle forecast as Part 1
    (lifetime_model.py), simulated with drag only, drag and J2, and drag, J2, Sun
    and Moon (lunisolar.py). Nothing is fitted, so every transfer orbit is scored.
    Run download_transfer_orbits.py first.
    """
    import time

    from download_transfer_orbits import OUT_FILE as TRANSFER_FILE
    from lifetime_model import MAX_YEARS, OUT_FILE as LIFETIME_FILE, error_by_band, weather_paths
    from lunisolar import orbit_vectors, propagate_lunisolar

    table = pd.read_parquet(LIFETIME_FILE)
    angles = pd.read_parquet(TRANSFER_FILE)[["NORAD_CAT_ID", "EPOCH", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER"]]
    table = table.merge(angles, on=["NORAD_CAT_ID", "EPOCH"])
    table = table[table["B_fit"] > 0].reset_index(drop=True)
    print(f"{len(table)} transfer orbits with a fitted B ({(table['OBJECT_TYPE'] == 'R/B').sum()} rocket bodies)")

    atm = MsisAtmosphere()
    flux, ap = weather_paths(table)["solar-cycle forecast"]

    def density(h, idx, t):
        month = np.clip((t / 30.44).astype(int), 0, flux.shape[1] - 1)
        return atm.density(h, flux[idx[:, None], month], ap[idx[:, None], month])

    e_vec, j_vec = orbit_vectors(table["eccentricity"], table["INCLINATION"], table["RA_OF_ASC_NODE"],
                                 table["ARG_OF_PERICENTER"])
    jd0 = pd.DatetimeIndex(table["EPOCH"]).to_julian_date().to_numpy()
    args = (table["a_km"].to_numpy(), e_vec, j_vec, table["B_fit"].to_numpy(), density, jd0)
    preds = {"drag only (Part 1)": table["solar-cycle forecast"].to_numpy()}
    for name, forces in [("drag + J2", dict(sun=False, moon=False)), ("drag + J2 + Sun + Moon", {})]:
        start = time.time()
        preds[name] = propagate_lunisolar(*args, max_days=MAX_YEARS * 365.25, max_step=max_step, **forces)
        print(f"{name}: {time.time() - start:.0f} s")
    print("\nMedian error by true remaining lifetime (no answer within 30 years counts as 30 years)")
    show(error_by_band(table, preds))
    print("\nshare with an answer within 30 years:",
          {name: f"{np.isfinite(p).mean():.0%}" for name, p in preds.items()})
    truth = table["days_left"].to_numpy()
    for name, p in preds.items():
        ratio = np.where(np.isfinite(p), p, MAX_YEARS * 365.25) / truth
        print(f"{name}: predicted / actual, middle half {np.percentile(ratio, 25):.2f} to {np.percentile(ratio, 75):.2f}")


EXPERIMENTS = {"hybrid_corrections": hybrid_corrections, "hybrid_variants": hybrid_variants, "hybrid_tuning": hybrid_tuning, "xgboost_tuning": xgboost_tuning,
               "solar_forecast": solar_forecast, "drag_stability": drag_stability,
               "final_days_window": final_days_window, "drag_cap": drag_cap,
               "hybrid_drag_cap": hybrid_drag_cap,
               "final_days_ranges": final_days_ranges,
               "storm_bound": storm_bound,
               "transfer_orbits": transfer_orbits}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        raise SystemExit(f"usage: python src/experiments.py [{' | '.join(EXPERIMENTS)}]")
    EXPERIMENTS[sys.argv[1]]()
