# Satellite Reentry Prediction

**Given an object's orbit and current space weather, how long until it falls back into Earth's atmosphere?**

Physics-based reentry predictions are hard because the upper atmosphere keeps changing. When the Sun is active, it heats and expands the atmosphere, which increases drag on everything in low orbit. This project uses machine learning on tens of thousands of historical reentries to learn how orbit, object size and solar activity relate to orbital lifetime. All data is public, from [CelesTrak](https://celestrak.org).

> **Status:** data collection, exploratory analysis and a satellite lookup tool are done. The prediction models are in progress (see [Roadmap](#roadmap)).

## What's here

| Script | What it does |
|---|---|
| [`src/lookup_satellite.py`](src/lookup_satellite.py) | Interactive lookup: search any of the ~71,000 tracked objects by name or NORAD ID and see its launch, orbit, size and lifetime |
| [`src/explore_data.py`](src/explore_data.py) | Exploratory analysis of the catalog and space weather data; builds the cleaned `data/lifetime_dataset.csv` used for modeling |
| [`src/collect_snapshots.py`](src/collect_snapshots.py) | Daily collector that saves current orbits of low-Earth-orbit objects, building a dataset to predict days until reentry |

## Quick start

Tested with Python 3.15.

```bash
git clone https://github.com/yatharthjag2008/Satellite_Reentry_Prediction.git
cd Satellite_Reentry_Prediction

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

The catalog and space weather data are included in `data/`, so every script works right away without downloading anything.

## Usage

### Look up a satellite

```bash
python src/lookup_satellite.py
```

Type a full or partial name (`ISS (ZARYA)`, `HST`, `starlink`, `sputnik`) or a NORAD ID (`25544`). If several objects match, pick one from the list. Press Enter on an empty line to quit.

```
== ISS (ZARYA) =================================================================
 payload (satellite) · owner ISS · operational · NORAD 25544 · 1998-067A

 LAUNCH                                     ORBIT (catalog)
 Date        1998-11-20                     Perigee       415 km
 Site        TYMSC                          Apogee        424 km
 Solar flux  130 sfu (81-day avg)           Inclination   51.63°
                                            Period        93.0 min
 LIFETIME                                   Eccentricity  0.0007
 Status      still up (in orbit)            Radar size    399.052 m²
 So far      10,203 days (~27.9 years)
```

Some names to try: `HST` (Hubble), `CSS (TIANHE)` (China's space station), `SPUTNIK 1`, `VANGUARD 1` (the oldest satellite still in orbit), `MIR`, `TIANGONG-1`, `ENVISAT`.

### Run the analysis

```bash
python src/explore_data.py
```

Prints summary tables, opens five plots, and saves `data/lifetime_dataset.csv`. The `# %%` markers also let you run it one section at a time in VS Code.

### Collect daily snapshots

```bash
python src/collect_snapshots.py
```

Run it once a day, for example with Windows Task Scheduler or cron. It skips itself if today's snapshot already exists, which also keeps it within CelesTrak's usage limits. Snapshots go to `data/snapshots/` and are not committed to the repo.

## Early findings

From `explore_data.py`:

- **71,139 tracked objects**, of which **34,155** have reentered with usable orbit data. Debris and rocket bodies have reentered at a much higher rate (65%) than payloads (28%).
- **The catalog's orbit values are mostly near-reentry orbits.** 65% of reentered objects have a catalog perigee below 250 km, which suggests the stored orbit was often measured just before reentry rather than early in the object's life. This holds in every decade since the 1960s.
- **As a result, perigee barely correlates with lifetime** (r = 0.11 with log lifetime), even though altitude should be the strongest driver. The best single correlates are inclination (0.27) and radar cross-section (−0.20).
- **Reentries rise and fall with the ~11-year solar cycle**, as expected: a more active Sun expands the atmosphere and increases drag.

The second finding shapes the modeling plan. The catalog alone is a weak basis for lifetime prediction, which is why Part 2 collects its own dated orbit snapshots.

## Data

| File | Source | Contents |
|---|---|---|
| `data/satcat.csv` | [CelesTrak SATCAT](https://celestrak.org/pub/satcat.csv) | Every tracked object: launch and decay dates, apogee, perigee, inclination, period, radar cross-section (RCS), object type and owner |
| `data/SW-All.csv` | [CelesTrak space weather](https://celestrak.org/SpaceData/SW-All.csv) | Daily space weather since 1957: F10.7 solar flux (with 81-day average), Kp and Ap geomagnetic indices, sunspot number |
| `data/lifetime_dataset.csv` | Built by `explore_data.py` | One row per reentered object: orbit, size, solar activity at launch, and lifetime in days |
| `data/snapshots/` | [CelesTrak GP data](https://celestrak.org/NORAD/elements/gp.php) | Daily orbit elements (including the B* drag term) for objects with perigee under 700 km |

The collector refreshes `satcat.csv` and `SW-All.csv` each time it runs.

## Roadmap

**Part 1: Orbital lifetime from historical data**
- [x] Build the lifetime table (`lifetime_days = DECAY_DATE − LAUNCH_DATE`)
- [x] Add solar activity at launch, using only data known on the launch date to avoid leakage
- [ ] Handle the near-reentry orbit problem, e.g. keep only objects whose catalog orbit looks like an early-life orbit
- [ ] Train and compare linear regression (on log lifetime), XGBoost, and a Gaussian process (which also estimates uncertainty)
- [ ] Show a predicted reentry date in the lookup tool

**Part 2: Days until reentry from daily snapshots**
- [x] Daily collector for low-Earth-orbit objects
- [ ] Label snapshots once each object's `DECAY_DATE` appears in the catalog
- [ ] Features: altitude, rate of altitude loss, eccentricity, B*, F10.7, Kp
- [ ] Train and evaluate models on objects that reentered after collection began

## Project structure

```
Satellite_Reentry_Prediction/
├── data/
│   ├── satcat.csv              satellite catalog
│   ├── SW-All.csv              daily space weather
│   ├── lifetime_dataset.csv    cleaned table for Part 1
│   └── snapshots/              daily orbit snapshots (not in repo)
├── src/
│   ├── collect_snapshots.py
│   ├── explore_data.py
│   └── lookup_satellite.py
├── requirements.txt
└── README.md
```

## Acknowledgments

All orbital, catalog and space weather data comes from [CelesTrak](https://celestrak.org), maintained by Dr. T.S. Kelso.
