# Predicting satellite reentry with drag physics and machine learning

*Yatharth Jaggi · November 2026 · code, data sources and every experiment: [github.com/yatharthjag2008/Satellite_Reentry_Prediction](https://github.com/yatharthjag2008/Satellite_Reentry_Prediction) · live forecasts: [yatharthjag2008.github.io/Satellite_Reentry_Prediction](https://yatharthjag2008.github.io/Satellite_Reentry_Prediction/)*

## Abstract

Thousands of satellites, spent rocket stages and pieces of debris fall back into the atmosphere every year, and the number is rising quickly with large constellations. When an object will reenter is hard to predict, because the drag that brings it down depends on the density of the upper atmosphere, which the Sun can change several-fold within days. This project predicts the days until reentry from an object's recent orbits and the space weather. It derives the orbital decay caused by drag from first principles, drives it with the NRLMSIS 2.1 atmosphere model, fits each object's drag parameter from how its orbit actually sank, and trains a machine learning model to correct the physics where it is systematically wrong. Tested on 4,897 objects that reentered from 2024 on, which no model saw during training, the physics + ML hybrid has a median error of **16%** of the time left, against 24% for the physics alone, 19% for tuned XGBoost and over 400% for the common rule of thumb. It is the most accurate model in 20 of 22 years of a backtest over two solar cycles, its 90% prediction ranges contain the truth 89% of the time, and, given equally fresh orbits, the physics is 28% more accurate than the official US Space Force predictions from four or more days out. The model now runs every six hours on a public website, forecasting real reentries before they happen and scoring itself afterwards.

## 1. The problem

An object in low Earth orbit is slowed by the thin air it passes through. Losing energy, it sinks into denser air, which slows it faster, so the decay accelerates until it falls in a few final hours. Predicting the day it falls matters for avoiding collisions, for warning where debris may land, and for managing an orbit that now holds more than 10,000 active satellites.

Two things make it hard. First, the air density at 200–600 km depends on the Sun: extreme ultraviolet light and geomagnetic storms heat and expand the upper atmosphere, changing the density several-fold over the 11-year solar cycle and by tens of percent within days. Second, every object's response to the air depends on its drag parameter B = C_D·A/m (drag coefficient × area / mass), which is unknown for almost all debris and changes as an object tumbles.

The common rule of thumb, extrapolating today's rate of sinking, ignores that decay accelerates, and is off by a factor of four to five.

## 2. Data

- **Orbits.** Space-Track.org (US Space Force) keeps every orbit ever published for every tracked object. For each of the 17,558 objects that reentered between 2000 and November 2026, I downloaded the orbits of its final year and turned them into one row per object per day (3.7 million rows), with the rates of sinking, the solar flux (F10.7) and geomagnetic activity (Ap) known that day, and the true time left until the catalog's reentry date.
- **Catalog and space weather.** CelesTrak's satellite catalog (launch and decay dates, object type, radar size) and its record of daily solar and geomagnetic indices since 1957.
- **Official predictions.** About 13,000 of the US Space Force's Tracking and Impact Prediction (TIP) messages for objects in their final days, used as a benchmark.

**Splits.** Every model trains on objects that reentered before 2024 and is tested on those that reentered from 2024 on, so the test objects are both unseen and later in time. Every modeling choice (settings, fit windows, rules) was made on a validation year (objects that reentered in 2023), never on the test set. Each choice is logged in a numbered experiment log (26 experiments so far).

## 3. Method

### 3.1 Physics: how drag shrinks an orbit

Drag does negative work on the object, and the orbital energy E = −μ/2a depends only on the semi-major axis a, so

$$\frac{da}{dt} = -B\,\rho(h)\,\sqrt{\mu a}$$

for a circular orbit, where ρ is the air density at height h and μ is Earth's gravitational parameter. For a stretched orbit, a second equation for the eccentricity, de/dt = −Bρv(e + cos ν), shows that drag at perigee lowers the apogee while leaving the perigee in place, so orbits first become circular and then sink. Both rates are averaged over one orbit by integrating over the eccentric anomaly. The air density comes from NRLMSIS 2.1, pre-computed as a table over altitude, solar flux, geomagnetic activity and season. A fourth-order Runge–Kutta integrator then follows the orbit forward until it falls. The full derivation is in [`docs/derivation.md`](derivation.md).

The code is tested against exact answers: in an exponential atmosphere, the lifetime has a closed form involving the Dawson function, and the simulation matches it to 0.2%.

### 3.2 Measuring each object's drag

Because every rate is proportional to B, the decay of the observed orbit is a straight line in the integrated unit-drag rate G(t): a(t) − a(t₀) = B·G(t). A least-squares fit over the past 30 days gives each object's B, using only data available at the time. Two kinds of fit are rejected: a negative B (the orbit rose, so the object is maneuvering) and a B above 1 m²/kg (more area per kilogram than any solid object has, which means something other than drag, such as the Sun's and Moon's gravity on a stretched orbit, is moving it; experiments 021–022).

### 3.3 Machine learning that corrects the physics

The hybrid model is XGBoost with shallow, strongly regularized trees, trained to predict log(actual time left / physics prediction): a correction factor rather than the answer itself. Its inputs are the orbit, the rates of sinking, the space weather, the object type and the physics outputs. Where the physics has no answer (an object holding its altitude with thrusters), the hybrid takes over. Experiment 016 shows what it learned: it leaves the physics alone where it is already right and corrects three specific weaknesses, at the solar peak, in the final weeks and for Starlink satellites.

### 3.4 Honest uncertainty

Mondrian conformal prediction turns past errors into a 90% range around each prediction, separately for each band of predicted time and each object group, and recalibrated every month on the past year's reentries, because the population changes (far more Starlink reentries since 2024).

## 4. Results

### 4.1 Days until reentry

Median error relative to the true time left, on the 4,897 test objects:

| Model | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Rule of thumb | 522% | 389% | 395% | 444% | 427% | 7% |
| Physics only | 22% | 17% | 21% | 25% | 24% | 52% |
| XGBoost (tuned) | 27% | 15% | 19% | 19% | 19% | 63% |
| Gaussian process | 24% | 18% | 25% | 23% | 23% | 54% |
| **Physics + ML hybrid** | **19%** | **15%** | **18%** | **16%** | **16%** | **68%** |

The hybrid is the most accurate model for every object type: 15% for debris and rocket bodies, 11% for ordinary satellites and 28% for Starlink, whose reentries the operator chooses. The physics alone, with one fitted number per object and no training, comes close to XGBoost.

**Across the solar cycle.** Retrained and tested year by year since 2005, the hybrid is best or tied in 20 of 22 years (median 22%, against 24% for XGBoost and 27% for the physics). The physics alone struggles exactly when the Sun is waking from a minimum (2010–2011, 2021), because holding today's low solar flux fixed underestimates future drag; the hybrid learns this from earlier cycles.

**Ranges.** The rolling 90% ranges contain the truth 89% of the time overall, 90% for debris and rocket bodies and 87% for Starlink.

### 4.2 Against the official predictions

For about 10,500 official TIP messages on objects that reentered from 2024 on, the models predicted from our latest orbit at the moment each message was issued. Median error of the predicted reentry time:

| Made before reentry | Official | Physics, one orbit a day | Physics, fresh orbits, 8-day drag fit |
|---|---|---|---|
| Under 1 day | **0.7 h** | 4.1 h | 1.6 h |
| 1–2 days | **3.7 h** | 9.0 h | 6.4 h |
| 2–4 days | **7.8 h** | 13.6 h | 9.5 h |
| 4+ days | 37.1 h | 37.3 h | **26.7 h** |

With fresh orbits and the drag measured over only the last eight days, which captures recent changes such as tumbling, the physics beats the official predictions from four or more days out. In the final day the official forecasts, made with dense radar tracking, are far better. Experiment 023 gives these final-days predictions calibrated 90% ranges, which held for 87–95% of later reentries; experiment 024 found that even a perfect forecast of geomagnetic storms would improve them by only about an hour.

### 4.3 Whole lifetimes from the first month

From only an object's orbits 10–40 days after launch, the same physics with a forecast of the solar cycle predicts its whole lifetime, often years ahead. On 2,982 satellites and rocket bodies launched since 2000, including the Moon's and Sun's gravity for transfer orbits (below), the median error is 26% (19% for lifetimes of 3–10 years), against 69% with today's solar flux held fixed. Tested on a later period of launches, the physics held up while ML models trained on earlier launches did not (experiment 019).

**Transfer orbits.** Rocket bodies left in stretched orbits toward geostationary altitude reach tens of thousands of kilometers, where the Moon's and Sun's gravity slowly move the perigee, and drag alone predicts most of them to stay up for decades (median error 503%). Adding the orbit-averaged tides of the Sun and Moon at their actual positions, and Earth's flattening, in the vector form of Milankovitch's equations, brings the median error on 261 transfer orbits down to **28%**, the same as for ordinary orbits, with 97% of them predicted to fall within 30 years instead of 66% (experiment 025). Part 1 now simulates every transfer orbit this way, and the hybrid built on it improves from 71% to 37% on them. The same holds in an object's final year: on 1,943 weekly test rows of transfer orbits, the median error drops from 39% to 10%, and every row gets an answer instead of 80% (experiment 026), so the lookup and the live forecasts use it too.

## 5. Live forecasts

All the results above are backtests: the answers were known when the models were scored. To test the model the hard way, it now runs every six hours on GitHub Actions. It collects the latest orbits of about 12,000 low-orbit objects from CelesTrak, forecasts each one with the physics model and a calibrated 90% range, publishes them on the project's website with a map of where each imminent reentry could happen, and scores every forecast once the catalog lists the object as reentered. The physics runs alone here, because the hybrid was trained on Space-Track data that cannot be republished. Replayed on 809 past reentries, this pipeline had a median error of 20%, with 96% of its ranges containing the truth. The live track record started on 27 November 2026.

## 6. Limitations and next steps

- **The atmosphere is the main uncertainty.** A perfect forecast of solar activity would improve the physics by only two points (experiment 013), and a perfect forecast of storms by about an hour in the final days (024); what limits the predictions is how well each object's drag is known.
- **Operators decide when Starlink satellites come down**, so for them every model is less accurate and the ranges are wider.
- **Reentry dates in the catalog are only to the day**, which limits how finely the final day can be scored.
- Next: keep scoring the live forecasts as they come true, and compare them with the official predictions over the same period.

## References

- D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013.
- D. G. King-Hele, *Satellite Orbits in an Atmosphere: Theory and Applications*, 1987.
- J. T. Emmert et al., "NRLMSIS 2.0: A whole-atmosphere empirical model of temperature and neutral species densities", *Earth and Space Science* 8, 2021.
- S. Tremaine, J. Touma and F. Namouni, "Satellite dynamics on the Laplace surface", *Astronomical Journal* 137, 3706, 2009.
- F. R. Hoots and R. L. Roehrich, *Spacetrack Report No. 3*, 1980.
- V. Vovk, A. Gammerman and G. Shafer, *Algorithmic Learning in a Random World*, 2005.
- T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system", *KDD*, 2016.
