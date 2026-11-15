"""Prediction ranges for the hybrid model with split conformal prediction.

A point prediction like "40 days" isn't enough for planning; a range like "30 to
55 days, 90% likely" is. Conformal prediction builds that range from the model's
errors on held-out data, with a guarantee that holds if future errors look like
past ones:

1. fit the hybrid on objects that reentered before CALIBRATE_FROM
2. on objects that reentered between CALIBRATE_FROM and TEST_FROM (never seen in
   training), record the error |log10(actual / predicted)|
3. the range is the prediction times/divided by 10^q, where q is the 90th
   percentile of those errors

Errors grow with the horizon, so step 3 is done separately for each band of
*predicted* days (the actual days aren't known when predicting). This is called
Mondrian conformal prediction.

Because the satellite population keeps changing (far more Starlink reentries
since 2024), a fixed calibration year goes stale. The rolling version
recalibrates every month on the objects that reentered in the previous
RECALIBRATION_DAYS, which are all known by then, separately for each object
group and band.

A second version, conformalized quantile regression (CQR), first trains XGBoost
to predict the 5th and 95th percentiles directly, so the range can be narrow for
easy objects and wide for hard ones, then uses the calibration year to correct it.

    python src/uncertainty.py
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from train_reentry_model import (HORIZONS, HYBRID_FEATURES, HYBRID_PARAMS, MODELS_DIR, RESULTS_DIR, load,
                                 object_groups, predict_hybrid, train_hybrid)

CALIBRATE_FROM = "2023-01-01"
COVERAGE = 0.90
BANDS = [0, 3, 10, 30, 100, np.inf]  # bands of predicted days for Mondrian conformal
RECALIBRATION_DAYS = 365
MIN_CELL = 200  # fewer calibration rows than this in a (group, band) cell: use the band across all groups


def conformal_quantile(scores, coverage=COVERAGE):
    """The ceil((n+1) * coverage)-th smallest score: the finite-sample conformal quantile."""
    scores = np.sort(np.asarray(scores))
    k = int(np.ceil((len(scores) + 1) * coverage)) - 1
    return scores[min(k, len(scores) - 1)]


def band_of(pred_days):
    return np.digitize(pred_days, BANDS[1:-1])


def mondrian_widths(calib_pred, calib_truth):
    """Conformal quantile of |log10 error| for each band of predicted days."""
    scores = np.abs(np.log10(calib_truth / calib_pred))
    bands = band_of(calib_pred)
    return np.array([conformal_quantile(scores[bands == b]) for b in range(len(BANDS) - 1)])


def train_quantiles(train):
    """XGBoost models of the 5th and 95th percentiles of log10(actual / physics)."""
    target = np.log10(train["days_left"] / train["physics_filled"])
    lo_q, hi_q = (1 - COVERAGE) / 2, 1 - (1 - COVERAGE) / 2
    params = {**HYBRID_PARAMS, "objective": "reg:quantileerror", "quantile_alpha": np.array([lo_q, hi_q])}
    return xgb.XGBRegressor(**params).fit(train[HYBRID_FEATURES], target)


def quantile_range(model, df):
    """(low, high) in days from the quantile model."""
    q = model.predict(df[HYBRID_FEATURES])
    base = df["physics_filled"].to_numpy()
    return base * 10 ** q[:, 0], base * 10 ** q[:, 1]


def coverage_table(truth, low, high, groups):
    rows = {}
    for lo, hi, label in HORIZONS:
        m = (truth >= lo) & (truth < hi)
        rows[label] = [((truth >= low) & (truth <= high))[m].mean(), np.median(high[m] / low[m])]
    rows["overall"] = [((truth >= low) & (truth <= high)).mean(), np.median(high / low)]
    for name, m in groups.items():
        rows[name] = [((truth >= low) & (truth <= high))[m].mean(), np.median(high[m] / low[m])]
    return pd.DataFrame(rows, index=["contains truth", "high / low"]).T


def group_of(df):
    """0 = debris and rocket bodies, 1 = other payloads, 2 = Starlink."""
    passive = df["OBJECT_TYPE"].isin(["DEB", "R/B"]).to_numpy()
    return np.where(passive, 0, np.where(df["is_starlink"].to_numpy() == 1, 2, 1))


def rolling_widths(pool, test, score_col, static):
    """Conformal quantile of pool[score_col] for each test row, recalibrated monthly.

    For test rows predicted during month M, uses only pool rows of objects that
    reentered in the RECALIBRATION_DAYS before M starts, so every calibration
    answer was already known. Calibrates each (object group, band) cell separately,
    falling back to the band across all groups, then to `static` (one width per band).
    """
    widths = np.full(len(test), np.nan)
    months = test["EPOCH"].dt.to_period("M").to_numpy()
    for month in np.unique(months):
        start = month.start_time
        recent = pool[(pool["DECAY_DATE"] < start) & (pool["DECAY_DATE"] >= start - pd.Timedelta(days=RECALIBRATION_DAYS))]
        rows = np.flatnonzero(months == month)
        for band in range(len(BANDS) - 1):
            in_band = recent[recent["band"] == band]
            fallback = conformal_quantile(in_band[score_col]) if len(in_band) >= MIN_CELL else static[band]
            for group in range(3):
                cell = in_band[in_band["group"] == group]
                w = conformal_quantile(cell[score_col]) if len(cell) >= MIN_CELL else fallback
                widths[rows[(test["band"].to_numpy()[rows] == band) & (test["group"].to_numpy()[rows] == group)]] = w
    return widths


def main():
    fit_and_calib, test = load()
    fit = fit_and_calib[fit_and_calib["DECAY_DATE"] < CALIBRATE_FROM]
    calib = fit_and_calib[fit_and_calib["DECAY_DATE"] >= CALIBRATE_FROM]
    print(f"fit on {fit['NORAD_CAT_ID'].nunique():,} objects, calibrate on {calib['NORAD_CAT_ID'].nunique():,} "
          f"(reentered {CALIBRATE_FROM[:4]}), test on {test['NORAD_CAT_ID'].nunique():,}\n")
    truth = test["days_left"].to_numpy()
    groups = object_groups(test)

    hybrid = train_hybrid(fit)
    widths = mondrian_widths(predict_hybrid(hybrid, calib), calib["days_left"].to_numpy())
    print("90% range for each band of predicted days (prediction times/divided by):")
    for b, w in enumerate(widths):
        print(f"  {BANDS[b]:>4}-{BANDS[b + 1]:<4} days: x{10 ** w:.2f}")
    pred = predict_hybrid(hybrid, test)
    w = widths[band_of(pred)]
    mondrian = coverage_table(truth, pred / 10 ** w, pred * 10 ** w, groups)

    quant = train_quantiles(fit)
    c_low, c_high = quantile_range(quant, calib)
    c_truth = calib["days_left"].to_numpy()
    # CQR score: how far the truth falls outside the predicted range (negative if inside), in log10 units.
    scores = np.maximum(np.log10(c_low / c_truth), np.log10(c_truth / c_high))
    q = conformal_quantile(scores)
    low, high = quantile_range(quant, test)
    cqr = coverage_table(truth, low / 10 ** q, high * 10 ** q, groups)

    # Rolling recalibration: score every calibration and test row, then recalibrate month by month.
    calib, test = calib.copy(), test.copy()
    for df in (calib, test):
        df["pred"] = predict_hybrid(hybrid, df)
        df["band"], df["group"] = band_of(df["pred"].to_numpy()), group_of(df)
        df["score"] = np.abs(np.log10(df["days_left"] / df["pred"]))
        low_q, high_q = quantile_range(quant, df)
        df["low_q"], df["high_q"] = low_q, high_q
        df["cqr_score"] = np.maximum(np.log10(low_q / df["days_left"]), np.log10(df["days_left"] / high_q))
    pool = pd.concat([calib, test])
    w = rolling_widths(pool, test, "score", widths)
    rolling = coverage_table(truth, test["pred"].to_numpy() / 10 ** w, test["pred"].to_numpy() * 10 ** w, groups)
    q = rolling_widths(pool, test, "cqr_score", np.full(len(BANDS) - 1, q))
    rolling_cqr = coverage_table(truth, test["low_q"].to_numpy() / 10 ** q, test["high_q"].to_numpy() * 10 ** q, groups)

    summary = pd.concat({"Mondrian conformal": mondrian, "conformalized quantile regression": cqr,
                         "Mondrian, rolling": rolling, "CQR, rolling": rolling_cqr}, axis=1)
    print(f"\nTest set (target: {COVERAGE:.0%} of ranges contain the truth)")
    print(summary.to_string(formatters={c: ("{:.0%}".format if c[1] == "contains truth" else "{:.1f}x".format)
                                        for c in summary.columns}))
    summary.to_csv(RESULTS_DIR / "conformal_coverage.csv")
    save_live_widths(test)


def save_live_widths(test):
    """Widths for predict_reentry.py: the saved hybrid's errors on the latest year of reentries.

    The saved hybrid (train_reentry_model.py) never saw these objects, which reentered
    from 2024 on. Saved in the git-ignored models/ because it's derived from Space-Track data.
    """
    hybrid = xgb.XGBRegressor()
    hybrid.load_model(MODELS_DIR / "reentry_hybrid.json")
    recent = test[test["DECAY_DATE"] >= test["DECAY_DATE"].max() - pd.Timedelta(days=RECALIBRATION_DAYS)].copy()
    recent["pred"] = predict_hybrid(hybrid, recent)
    recent["band"], recent["group"] = band_of(recent["pred"].to_numpy()), group_of(recent)
    recent["score"] = np.abs(np.log10(recent["days_left"] / recent["pred"]))
    widths = []
    for group in range(3):
        row = []
        for band in range(len(BANDS) - 1):
            cell = recent[(recent["band"] == band) & (recent["group"] == group)]
            if len(cell) < MIN_CELL:
                cell = recent[recent["band"] == band]
            row.append(conformal_quantile(cell["score"]))
        widths.append(row)
    (MODELS_DIR / "conformal_widths.json").write_text(json.dumps({"bands": BANDS[:-1] + [1e9], "widths": widths}))
    print(f"saved live 90% widths from {recent['NORAD_CAT_ID'].nunique():,} recent reentries")


if __name__ == "__main__":
    main()
