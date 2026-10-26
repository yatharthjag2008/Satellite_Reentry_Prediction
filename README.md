# Machine Learning for Satellite Reentry Prediction Using Orbital and Space Weather Data

**Question:** Given an object's orbit and current space weather, how long until it reenters Earth's atmosphere?

Physics-based reentry predictions struggle because solar activity, which heats and expands the upper atmosphere, is hard to model. This project trains ML models on historical reentries and compares linear regression, XGBoost, and a Gaussian process (which also estimates uncertainty).

All data comes from [CelesTrak](https://celestrak.org) and is public. No account or approval is needed.

## Data

| File | Source | Contents |
|---|---|---|
| `data/satcat.csv` | https://celestrak.org/pub/satcat.csv | Catalog of about 71,000 objects. About 35,000 have a `DECAY_DATE` (the ground truth). Also includes apogee, perigee, inclination, period, radar cross-section (RCS), and object type (payload, rocket body, debris). |
| `data/SW-All.csv` | https://celestrak.org/SpaceData/SW-All.csv | Daily space weather since 1957: F10.7 solar flux (observed, plus 81-day averages), Kp and Ap geomagnetic indices, and sunspot number. |
| Daily TLE snapshots (Part 2) | https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv (and other groups) | Current orbit elements for each object, including the B* drag term. |

**Caveat:** The orbit values in SATCAT are a single snapshot for each object, not a history. For some decayed objects the snapshot is from close to reentry, for others it's much older. Check this during EDA.

## Approach

**Part 1: Orbital lifetime (historical, can start now)**
- Target: lifetime in days, computed as `DECAY_DATE - LAUNCH_DATE`. Start with objects that reached orbit soon after launch.
- Features: perigee, apogee, eccentricity (from apogee and perigee), inclination, RCS (a proxy for size and drag), object type, and solar activity at launch (F10.7, its 81-day average, Kp).
- Avoid leakage: only use solar data from on or before the prediction date.

**Part 2: Days until reentry (data you collect yourself)**
- Download CelesTrak orbit snapshots every day for low-perigee objects.
- When an object shows up in SATCAT with a `DECAY_DATE`, its earlier snapshots become labeled training rows.
- Features: altitude, how fast the altitude is dropping, eccentricity, B*, F10.7, and Kp. This matches the original project question.
- More than 5,000 objects have reentered since 2024, so a few weeks of collection already gives you plenty of examples.

## Setup

```
pip install -r requirements.txt
```
