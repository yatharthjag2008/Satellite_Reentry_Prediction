# Satellite Reentry Prediction

**Given an object's orbit and current space weather, how long until it falls back into Earth's atmosphere?**

Physics-based reentry predictions are hard because the upper atmosphere keeps changing. When the Sun is active, it heats and expands the atmosphere, which increases drag on everything in low orbit. This project uses machine learning on tens of thousands of historical reentries to learn how orbit, object size and solar activity relate to orbital lifetime. All data is public, from [CelesTrak](https://celestrak.org).

> **Status:** the first days-until-reentry models are trained and tested on reentries from 2024 onward (see [Results](#results-days-until-reentry)). Next: a physics-based drag model and a physics + ML hybrid (see [Roadmap](#roadmap)).

## What's here

| Script | What it does |
|---|---|
| [`src/lookup_satellite.py`](src/lookup_satellite.py) | Interactive lookup: search any of the ~71,000 tracked objects by name or NORAD ID and see its launch, orbit, size and lifetime |
| [`src/explore_data.py`](src/explore_data.py) | Exploratory analysis of the catalog and space weather data; builds the cleaned `data/lifetime_dataset.csv` used for modeling |
| [`src/collect_snapshots.py`](src/collect_snapshots.py) | Daily collector that saves current orbits of low-Earth-orbit objects, building a dataset to predict days until reentry |
| [`src/spacetrack.py`](src/spacetrack.py) | Logs in to [Space-Track.org](https://www.space-track.org) and makes rate-limited queries |
| [`src/download_history.py`](src/download_history.py) | Downloads the final year of orbits for every object that reentered since 2000 |
| [`src/build_reentry_dataset.py`](src/build_reentry_dataset.py) | Turns those orbits into a training table: one row per object per day, with decay rates, space weather and days left |
| [`src/train_reentry_model.py`](src/train_reentry_model.py) | Trains and compares four models for days until reentry |

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

## Results: days until reentry

Trained on 12,661 objects that reentered before 2024, tested on 4,897 objects that reentered from 2024 onward. Every test object is one the models never saw, from a later time. The numbers are the median error relative to the true time left (lower is better).

| Model | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Rule of thumb (current sinking rate down to 120 km) | 534% | 388% | 396% | 450% | 430% | 6% |
| Linear regression | 178% | 38% | 27% | 36% | 36% | 35% |
| XGBoost | **23%** | 22% | 25% | 26% | 26% | 49% |
| Gaussian process (3,000 training rows) | 32% | **17%** | **24%** | **22%** | **22%** | **56%** |

- The Gaussian process's 90% ranges contain the true answer 90% of the time, so its uncertainty is well calibrated.
- XGBoost is about 20% off for debris, rocket bodies and ordinary payloads, but 40% for Starlink, whose reentries are controlled by the operator.
- The most useful inputs are `MEAN_MOTION_DOT` (how fast drag is speeding up the orbit) and the 7-day perigee sinking rate.

Space-Track's terms don't allow republishing its data, so the downloaded orbits (`data/spacetrack/`) and trained models (`models/`) are not in this repo. To reproduce the results, create a free Space-Track account and put your login in a `.env` file:

```
SPACETRACK_USER=you@example.com
SPACETRACK_PASSWORD=your-password
```

Then run `download_history.py` (about a day, because of Space-Track's rate limits), `build_reentry_dataset.py` and `train_reentry_model.py`.

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

**Part 2: Days until reentry**
- [x] Download the final year of orbits for every reentry since 2000 from Space-Track
- [x] Features: altitude, rate of altitude loss, eccentricity, B*, F10.7, Ap
- [x] Train and compare models, tested on reentries from 2024 onward
- [ ] Physics model: simulate atmospheric drag with the NRLMSIS atmosphere
- [ ] Physics + ML hybrid: train ML to correct the physics model's errors
- [ ] Compare against the official US Space Force reentry predictions
- [ ] Year-by-year backtests and conformal prediction intervals
- [ ] Show predicted reentry in the lookup tool

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
