"""Explore the CelesTrak data and build the orbital lifetime table for Part 1.

Script version of notebooks/01_explore_data.ipynb. Prints the summary tables,
shows the plots at the end, and saves data/lifetime_dataset.csv.

    python src/explore_data.py

The "# %%" markers also let you run it section by section in VS Code.
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
R_EARTH = 6378.137  # km

plt.rcParams["figure.figsize"] = (10, 5)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

# %% 1. Load the satellite catalog
# OBJECT_TYPE: PAY = payload, R/B = rocket body, DEB = debris, UNK = unknown
# DECAY_DATE blank = still in orbit; APOGEE/PERIGEE in km; RCS in m² (rough size)
satcat = pd.read_csv(DATA_DIR / "satcat.csv", parse_dates=["LAUNCH_DATE", "DECAY_DATE"])
print(f"{len(satcat):,} objects")
print(satcat.head(), "\n")

summary = pd.DataFrame({
    "total": satcat["OBJECT_TYPE"].value_counts(),
    "reentered": satcat.loc[satcat["DECAY_DATE"].notna(), "OBJECT_TYPE"].value_counts(),
})
summary["share_reentered"] = (summary["reentered"] / summary["total"]).round(2)
print(summary, "\n")

# %% 2. Load space weather
sw = pd.read_csv(DATA_DIR / "SW-All.csv", parse_dates=["DATE"])
sw = sw[["DATE", "F10.7_OBS", "F10.7_OBS_LAST81", "KP_SUM", "AP_AVG", "ISN"]].rename(columns={
    "F10.7_OBS": "f107",               # daily solar flux
    "F10.7_OBS_LAST81": "f107_81d",    # trailing 81-day average (only uses past data)
    "KP_SUM": "kp_sum",                # sum of the eight 3-hour Kp values (x10)
    "AP_AVG": "ap",                    # daily geomagnetic activity
    "ISN": "sunspots",
})
sw = sw[sw["DATE"] <= pd.Timestamp.today()]  # file also contains forecasts
print(sw.tail(), "\n")

fig, ax = plt.subplots()
ax.plot(sw["DATE"], sw["f107"], lw=0.3, alpha=0.5, label="daily F10.7")
ax.plot(sw["DATE"], sw["f107_81d"], lw=1.5, label="81-day average")
ax.set(title="Solar activity since 1957: the ~11-year cycle", ylabel="F10.7 solar flux (sfu)")
ax.legend()

# %% 3. Build the orbital lifetime table
# lifetime_days = decay date - launch date (the target for Part 1)
# eccentricity from apogee and perigee (0 = circular orbit)
life = satcat[satcat["DECAY_DATE"].notna() & satcat["PERIGEE"].notna() & satcat["APOGEE"].notna()].copy()
life = life[life["ORBIT_CENTER"] == "EA"]  # orbiting Earth only

life["lifetime_days"] = (life["DECAY_DATE"] - life["LAUNCH_DATE"]).dt.days
life = life[life["lifetime_days"] > 0]

ra, rp = life["APOGEE"] + R_EARTH, life["PERIGEE"] + R_EARTH
life["eccentricity"] = (ra - rp) / (ra + rp)

print(f"{len(life):,} reentered objects with usable orbits")
print(life[["OBJECT_NAME", "OBJECT_TYPE", "LAUNCH_DATE", "DECAY_DATE", "PERIGEE", "APOGEE", "RCS",
            "lifetime_days"]].sample(5, random_state=0), "\n")

fig, ax = plt.subplots()
bins = np.logspace(0, np.log10(life["lifetime_days"].max()), 60)
for obj_type in ["PAY", "R/B", "DEB"]:
    ax.hist(life.loc[life["OBJECT_TYPE"] == obj_type, "lifetime_days"], bins=bins, alpha=0.5, label=obj_type)
ax.set(xscale="log", xlabel="lifetime (days, log scale)", ylabel="objects",
       title="How long objects stayed in orbit")
for days, label in [(30, "1 month"), (365, "1 year"), (3650, "10 years")]:
    ax.axvline(days, color="grey", ls=":", lw=1)
    ax.text(days, ax.get_ylim()[1] * 0.95, label, rotation=90, va="top", ha="right", color="grey")
ax.legend()

# %% 4. How much does altitude matter?
# Lower orbits pass through denser air, so lifetime should drop sharply as perigee gets lower.
low = life[(life["PERIGEE"] < 1000) & (life["APOGEE"] < 2000)]

fig, ax = plt.subplots()
sc = ax.scatter(low["PERIGEE"], low["lifetime_days"], s=2, alpha=0.3, c=low["eccentricity"], cmap="viridis", vmax=0.1)
ax.set(yscale="log", xlabel="perigee (km)", ylabel="lifetime (days, log scale)",
       title="Lifetime vs perigee (low orbits)")
fig.colorbar(sc, label="eccentricity")

# %% 5. Can we trust the catalog's orbit values?
# The catalog stores one orbit per object without saying when it was measured. A perigee
# under ~250 km suggests a near-reentry orbit, which says little about the original orbit.
fig, ax = plt.subplots()
ax.hist(life.loc[life["PERIGEE"] < 1500, "PERIGEE"], bins=150)
ax.axvline(250, color="red", ls="--", label="250 km")
ax.set(xlabel="catalog perigee (km)", ylabel="reentered objects", title="Perigee of objects that have already reentered")
ax.legend()

share_low = (life["PERIGEE"] < 250).mean()
print(f"{share_low:.0%} of reentered objects have a catalog perigee below 250 km (likely a near-reentry orbit)\n")

# Does the pattern change over time? Look at decade of reentry.
life["decay_decade"] = (life["DECAY_DATE"].dt.year // 10) * 10
print(life.groupby("decay_decade").agg(
    objects=("PERIGEE", "size"),
    median_perigee_km=("PERIGEE", "median"),
    share_below_250km=("PERIGEE", lambda p: round((p < 250).mean(), 2)),
), "\n")

# %% 6. Reentries vs solar activity
# A hotter Sun puffs up the atmosphere and increases drag, so solar peaks should line up
# with bursts of reentries. Debris events and megaconstellations add spikes of their own.
per_year = life.groupby(life["DECAY_DATE"].dt.year).size().rename("reentries")
solar_year = sw.groupby(sw["DATE"].dt.year)["f107"].mean().rename("mean_f107")
yearly = pd.concat([per_year, solar_year], axis=1).loc[1958:pd.Timestamp.today().year - 1]

fig, ax1 = plt.subplots()
ax1.bar(yearly.index, yearly["reentries"], color="tab:blue", alpha=0.6)
ax1.set_ylabel("reentries per year", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(yearly.index, yearly["mean_f107"], color="tab:orange", lw=2)
ax2.set_ylabel("mean F10.7", color="tab:orange")
ax1.set_title("Reentries per year vs solar activity")

# %% 7. Add solar activity at launch
# Only use what was known at launch; averaging over the lifetime would leak future information.
life = life.sort_values("LAUNCH_DATE")
life = pd.merge_asof(life, sw[["DATE", "f107", "f107_81d", "ap"]].rename(columns={"DATE": "LAUNCH_DATE"}),
                     on="LAUNCH_DATE", direction="backward")

print("Correlation with log10(lifetime):")
print(life[["PERIGEE", "APOGEE", "eccentricity", "INCLINATION", "RCS", "f107_81d", "ap", "lifetime_days"]].assign(
    log_lifetime=np.log10(life["lifetime_days"])
).corr(numeric_only=True)["log_lifetime"].drop(["lifetime_days", "log_lifetime"]).sort_values(), "\n")

# %% 8. Save the cleaned table
cols = ["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "OWNER", "LAUNCH_DATE", "DECAY_DATE",
        "PERIGEE", "APOGEE", "eccentricity", "INCLINATION", "PERIOD", "RCS",
        "f107", "f107_81d", "ap", "lifetime_days"]
out = DATA_DIR / "lifetime_dataset.csv"
life[cols].to_csv(out, index=False)
print(f"saved {len(life):,} rows to {out}")

# %%
plt.show()
