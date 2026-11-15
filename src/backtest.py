"""Year-by-year backtest: does the model hold up across the solar cycle?

The main test set (2024 onward) is near a solar maximum. Here, for each year Y,
the models train on every object that reentered before Y and are tested on the
objects that reentered during Y, the way they would have been used at the time.
That covers quiet-Sun years (2008-2010, 2018-2020) and active ones (2001-2002,
2014, 2024).

    python src/backtest.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_reentry_model import FEATURES, RESULTS_DIR, load, predict_hybrid, train_hybrid, train_xgboost

FIRST_YEAR = 2005  # earlier years leave too few objects to train on


def main():
    df, _ = load(test_from="2100-01-01", verbose=False)
    years = range(FIRST_YEAR, df["DECAY_DATE"].dt.year.max() + 1)
    rows = []
    for year in years:
        train = df[df["DECAY_DATE"] < f"{year}-01-01"]
        test = df[df["DECAY_DATE"].dt.year == year]
        truth = test["days_left"].to_numpy()
        booster = train_xgboost(train, FEATURES, np.log10(train["days_left"]))
        hybrid = train_hybrid(train)
        preds = {"physics": test["physics_filled"].to_numpy(),
                 "XGBoost": 10 ** booster.predict(test[FEATURES]),
                 "hybrid": predict_hybrid(hybrid, test)}
        row = {"year": year, "test objects": test["NORAD_CAT_ID"].nunique(),
               "train objects": train["NORAD_CAT_ID"].nunique(), "mean F10.7": test["f107_81d"].mean()}
        row.update({name: np.median(np.abs(p / truth - 1)) for name, p in preds.items()})
        rows.append(row)
        print(f"{year}: " + ", ".join(f"{k} {v:.0%}" for k, v in row.items() if k in preds)
              + f"  ({row['test objects']:,} objects, F10.7 {row['mean F10.7']:.0f})")

    table = pd.DataFrame(rows).set_index("year")
    table.to_csv(RESULTS_DIR / "backtest_by_year.csv", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ("physics", "XGBoost", "hybrid"):
        ax.plot(table.index, 100 * table[name], marker="o", label=name)
    ax.set(xlabel="year of reentry (trained on all earlier years)", ylabel="median error (%)",
           title="Year-by-year backtest")
    ax.legend(loc="upper left")
    flux = ax.twinx()
    flux.fill_between(table.index, table["mean F10.7"], color="orange", alpha=0.15)
    flux.set_ylabel("average solar flux F10.7 (shaded)")
    fig.savefig(RESULTS_DIR / "backtest_by_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nmedian over years:", ", ".join(f"{n} {table[n].median():.0%}" for n in ("physics", "XGBoost", "hybrid")))


if __name__ == "__main__":
    main()
