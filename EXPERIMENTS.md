# Experiment log

Numbered experiments on predicting days until reentry. Each entry records what changed, what happened and what I concluded. Unless noted, models train on objects that reentered before 2024-01-01 and are tested on objects that reentered from 2024-01-01 onward, and errors are the median of |predicted / actual − 1|.

## 001: Baseline models

**Setup:** `train_reentry_model.py` on 3.7 million daily orbits (17,558 objects). Four models: a rule of thumb (current sinking rate extrapolated to 120 km), linear regression, XGBoost and a Gaussian process (3,000-row sample, 6 features), all predicting log10(days left).

| Model | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Rule of thumb | 534% | 388% | 396% | 450% | 430% | 6% |
| Linear regression | 178% | 38% | 27% | 36% | 36% | 35% |
| XGBoost | 23% | 22% | 25% | 26% | 26% | 49% |
| Gaussian process | 32% | 17% | 24% | 22% | 22% | 56% |

**Conclusion:** both nonlinear models beat the rule of thumb by more than 10×, because orbits sink faster as they reach denser air. A Gaussian process on 3,000 rows beating XGBoost on 2.3 million suggests XGBoost learns patterns that don't carry over to 2024–2026.

## 002: Where XGBoost goes badly low

**Setup:** `analyze_errors.py`. Rows with 100+ days left where the prediction is under half the true value.

**Result:** 23% of far-out predictions are badly low. Starlink is 35% of those rows but 56% of the badly low ones (37% of Starlink rows are badly low, against 10% for debris and rocket bodies). The badly low rows have a median 81-day solar flux of 197, against 163 for all far-out rows, and only 2% of training rows have solar flux that high.

**Conclusion:** two causes. (1) Satellites that steer, mainly Starlink, sit at a fixed altitude and then get lowered on the operator's schedule, which no orbit-based model can see coming. (2) 2024–2025 was a strong solar maximum, and the model learned "high flux now → fast decay", which overstates decay when the flux drops over the following months. A physics model that holds today's space weather fixed would make the same mistake, so the forecast of future solar flux matters (tested in a later experiment).

## 003: Physics model alone

**Setup:** `physics_model.py`. For each object and day, fit B by least squares over the past 30 days of orbits (derivation section 6), then simulate forward with global-average NRLMSIS 2.1 density, holding the latest 81-day solar flux and 7-day Ap fixed. Simulated one orbit a week per object (541,031 orbits, about 7 minutes). Where the physics has no answer (B ≤ 0, too few orbits, or more than 10 years), it falls back to 365 days. All models in this and later experiments are scored on the same weekly rows; on these rows XGBoost scores 23% rather than 26%.

| Model | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Physics | 22% | 17% | 21% | 25% | 24% | 52% |
| XGBoost | 18% | 20% | 22% | 24% | 23% | 53% |
| Gaussian process | 24% | 18% | 25% | 23% | 23% | 54% |

The physics has its own answer for 71% of test rows. On those rows its median error is 20%, and on payloads other than Starlink it is **14%, better than every ML model**. It is worst on Starlink (49%), where the fitted B is near zero or negative because the satellites hold their altitude.

**Conclusion:** with no training at all, a drag simulation plus one fitted number per object matches XGBoost trained on 336,000 rows. Where the physics fails, it fails for a clear reason (thrusters), which tells the hybrid where to step in.

## 004: Physics + ML hybrid

**Setup:** XGBoost (same settings) trained on log10(actual / physics), with the 19 usual features plus log10(physics days), log10(B) and the number of orbits in the fit. Prediction = physics × 10^(correction).

| Model | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Hybrid | 22% | 22% | 23% | **22%** | **22%** | **54%** |

By object type: debris and rocket bodies 17% (XGBoost 19%, physics 23%), payloads 15%, Starlink 38%. The physics prediction is by far the hybrid's most used input (importance 0.34, next 0.09).

**Conclusion:** the hybrid is the best overall and on debris, but only by 1–2 points, and it is worse than XGBoost in the final week and worse than the physics alone on payloads (14%). The residual model seems to be undoing good physics predictions as often as fixing bad ones. Next: regularize the hybrid and check whether simply averaging models helps.

## 005: Ways to combine physics and ML (validation year)

**Setup:** from here on, choices are made on a **validation year** so the test set is only used once at the end: fit on objects that reentered before 2023 (10,656), score on objects that reentered during 2023 (2,005). `experiments.py hybrid_variants`.

| Variant | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Physics | 19% | 13% | 16% | 26% | 23% | 53% |
| XGBoost | 17% | 16% | 25% | 23% | 23% | 55% |
| Hybrid, residual, default trees (depth 8) | 20% | 20% | 28% | 21% | 22% | 56% |
| **Hybrid, residual, shallow trees** (depth 4, min_child_weight 50, L2 10, 400 trees) | **17%** | **12%** | 16% | **17%** | **17%** | **69%** |
| XGBoost with physics as an extra input | 20% | 18% | 26% | 23% | 23% | 54% |
| Physics where it has an answer, else XGBoost | 18% | 12% | 15% | 26% | 23% | 54% |
| Average of the three ML models | 19% | 17% | 26% | 22% | 22% | 55% |

**Conclusion:** how the physics is combined matters more than whether it's used. Given the physics prediction as just another input, XGBoost mostly ignores it (23%, same as without). Learning a *correction* to the physics works, but only if the correction is kept simple: deep trees overfit the residuals and wreck good physics predictions at 1–3 months (28% vs 16%), while shallow, regularized trees keep them and fix the far horizon (17% vs 26%). The shallow residual hybrid is better than every other option in every horizon but one.

## 006: Tuning the hybrid (validation year)

**Setup:** `experiments.py hybrid_tuning`. Grid over tree depth (2, 3, 4, 6), min_child_weight (50, 500) and 400 trees × 0.05 vs 1,000 trees × 0.02, all with L2 regularization 10.

**Result:** every setting with depth 3–6 scores 17% overall and 67–69% within 25%; depth 2 scores 18%. The setting from experiment 005 (depth 4, min_child_weight 50, 400 trees) is among the best and is kept.

**Conclusion:** the gain comes from regularization (min_child_weight and L2), not from a lucky depth. The result is not sensitive to the exact settings, which makes it more believable.

## 007: Tuning plain XGBoost the same way (validation year)

**Setup:** `experiments.py xgboost_tuning`. The default XGBoost (depth 8, no regularization) against depth 3/4/6 with min_child_weight 50/500 and L2 10.

| XGBoost | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Default (depth 8) | 17% | 16% | 25% | 23% | 23% | 55% |
| **Depth 3, min_child_weight 50** | 20% | 14% | 20% | 18% | **18%** | 66% |
| Shallow hybrid (from 005) | 17% | 12% | 16% | 17% | **17%** | **69%** |

**Conclusion:** an important correction to experiments 001 and 005. Plain XGBoost was overfitting too, and regularizing it cuts its error from 23% to 18%. So most of the hybrid's apparent advantage was really "regularized vs. not". Against a fairly tuned baseline the physics still helps, but less: 1 point overall, 3 points within 25%, and 4 points at 1–3 months (16% vs 20%), where the drag simulation is most reliable. Both tuned settings are now the defaults in `train_reentry_model.py`.

## 008: Final test of the tuned models

**Setup:** the settings chosen on the validation year (006–007), retrained on every object that reentered before 2024 and scored once on the 4,897 test objects that reentered from 2024 on.

| Model | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Rule of thumb | 522% | 389% | 395% | 444% | 427% | 7% |
| Physics | 22% | 17% | 21% | 25% | 24% | 52% |
| Linear regression | 164% | 38% | 27% | 37% | 36% | 35% |
| XGBoost (tuned) | 27% | **15%** | 19% | 19% | 19% | 63% |
| Gaussian process | 24% | 18% | 25% | 23% | 23% | 54% |
| **Physics + ML hybrid (tuned)** | **21%** | 16% | **18%** | **16%** | **16%** | **67%** |

| Object type | Physics | XGBoost | Hybrid |
|---|---|---|---|
| Debris + rocket bodies | 23% | 17% | **15%** |
| Payloads (not Starlink) | 14% | 14% | **11%** |
| Starlink | 49% | 29% | **28%** |

**Conclusion:** the validation-year ranking held on the test set. The hybrid is the best model overall and for every object type, and its 16% compares with 26% for the original XGBoost (experiment 001). The tuned XGBoost is weak in the final week (27%), where the physics carries the hybrid. Starlink is still the hardest case, as expected for satellites whose reentry is chosen by the operator.

## 009: Conformal 90% ranges, calibrated on 2023

**Setup:** `uncertainty.py`. Hybrid fit on reentries before 2023; its errors on 2023 reentries set the range width. Two methods: Mondrian split conformal (one width per band of predicted days) and conformalized quantile regression (CQR: XGBoost predicts the 5th and 95th percentiles, then the 2023 errors widen them as needed).

| Test rows | Mondrian: contains truth | Mondrian: high/low | CQR: contains truth | CQR: high/low |
|---|---|---|---|---|
| Under 1 week | 85% | 2.1× | 93% | 6.1× |
| 1–4 weeks | 78% | 2.0× | 93% | 2.7× |
| 1–3 months | 77% | 2.4× | 87% | 2.8× |
| 3–12 months | 82% | 2.5× | 85% | 2.8× |
| **Overall** | **81%** | 2.5× | **86%** | 2.8× |
| Debris + rocket bodies | 90% | 2.5× | 90% | 2.8× |
| Payloads | 94% | 2.5× | 88% | 2.3× |
| Starlink | 59% | 2.5× | 81% | 4.2× |

**Conclusion:** conformal prediction guarantees 90% only if the future looks like the calibration data. For debris and rocket bodies it does, and coverage is exactly 90%. For Starlink it doesn't: 2024–2026 brought far more Starlink reentries, at higher solar activity, than 2023. CQR adapts better because its widths depend on each object's features (Starlink ranges are 4.2× wide against 2.3× for payloads), but still falls short. Next: recalibrate as time goes on, using only objects that had already reentered before each prediction.

## 010: Rolling recalibration of the ranges

**Setup:** same hybrid and scores as 009, but the range width is recalibrated every month from the objects that reentered in the previous 365 days (all already known at the time), separately for each object group (debris + rocket bodies, payloads, Starlink) and band of predicted days. Cells with fewer than 200 rows fall back to the band across all groups.

| Test rows | Static Mondrian | **Rolling Mondrian** | Static CQR | Rolling CQR |
|---|---|---|---|---|
| Overall | 81% (2.5×) | **89% (2.5×)** | 86% (2.8×) | 88% (2.8×) |
| Debris + rocket bodies | 90% (2.5×) | 90% (2.5×) | 90% (2.8×) | 89% (2.7×) |
| Payloads | 94% (2.5×) | 89% (2.0×) | 88% (2.3×) | 87% (2.3×) |
| Starlink | 59% (2.5×) | **88% (4.1×)** | 81% (4.2×) | 87% (4.4×) |

(Each cell: share of ranges containing the truth, and the median ratio of the top of the range to the bottom.)

**Conclusion:** keeping the calibration recent and per group almost restores the 90% target (89%) without making ranges wider on average: they get wider for Starlink (4.1×) and narrower for ordinary payloads (2.0×), which is where the uncertainty really is. The simpler Mondrian method does as well as CQR here, so the lookup tool uses it, calibrated on the latest year of reentries.

## 011: Against the official US Space Force predictions

**Setup:** `compare_official.py`. Space-Track's TIP messages give the official predicted reentry time, issued at a known moment, for objects in their final days: 10,647 messages for 1,941 test objects. For each message, our models predict from our latest orbit before it (9,779 messages have one within 2 days). The truth is each object's final TIP message, issued after reentry.

| Made before reentry | Messages | Our orbit's age | Official | Physics | XGBoost | Hybrid |
|---|---|---|---|---|---|---|
| Under 1 day | 5,246 | 11.9 h | **0.7 h (8%)** | 4.1 h (51%) | 13.2 h (179%) | 6.0 h (83%) |
| 1–2 days | 1,666 | 9.6 h | **3.6 h (10%)** | 9.0 h (25%) | 27.1 h (71%) | 9.7 h (27%) |
| 2–4 days | 2,372 | 8.1 h | **7.7 h (11%)** | 13.6 h (20%) | 36.6 h (53%) | 15.5 h (23%) |
| 4+ days | 489 | 9.9 h | 36.4 h (30%) | 37.3 h (30%) | 39.7 h (29%) | 41.7 h (32%) |

(Median error in hours, and as a share of the time left.) The true reentry fell inside the official ± window for 84% of messages.

**Conclusion:** in the final four days, the official predictions are 2–6 times more accurate. That isn't surprising: they use dense radar tracking right up to the message, while our latest orbit is typically 8–12 hours old (the training table keeps one orbit per day), which in the final day is about as long as the time left. Further out (4+ days) all four are equally good at about 30%. Among our models the physics is clearly best close to reentry; XGBoost, which never sees how density changes along the way, is poor. Next: give our models the same fresh orbits.

## 012: Year-by-year backtest, 2005–2026

**Setup:** `backtest.py`. For each year Y, train XGBoost and the hybrid (tuned settings) on every object that reentered before Y, and test on the objects that reentered during Y. The physics needs no training. This covers two full solar cycles.

**Result:** median error over the 22 years: **hybrid 22%**, XGBoost 24%, physics 27%. The hybrid is best or tied in 20 of the 22 years (the exceptions are 2008 and 2022, by 1–2 points). See `results/backtest_by_year.png`.

The physics alone is erratic: 18–27% in most years, but 42–48% in 2010–2011 and 2021. Those are exactly the years when solar activity **rose** out of a minimum. Holding today's low solar flux fixed then underestimates future drag, so the physics predicts lifetimes that are too long.

**Conclusion:** the 2024+ test result (experiment 008) isn't a lucky period: the hybrid wins across quiet and active Suns. The physics model's weak point is the solar forecast, and the hybrid learns to correct it, because the training data contains earlier solar cycles. Next: measure how much a perfect solar forecast would help the physics.

## 013: How much would a perfect solar forecast help the physics?

**Setup:** `experiments.py solar_forecast`. 5,000 random test rows where the physics has a fitted B. The same fitted B and starting orbit, with four assumptions about the future atmosphere. No training is involved, so test rows are fair to use.

| Future weather | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Today's held fixed, year-average season (baseline) | 22% | 17% | 18% | 27% | 24% | 51% |
| Today's held fixed, with the seasons | 19% | 14% | 20% | 29% | 25% | 50% |
| Perfect forecast, centered 81-day flux | 22% | 17% | 20% | 27% | 24% | 51% |
| Perfect forecast, trailing 81-day flux | **19%** | **13%** | 19% | **24%** | **22%** | **54%** |

**Conclusion:** two lessons.
1. **The forecast has to match how B was fitted.** B was fitted using the trailing 81-day flux. Switching to the "better" centered flux for the forecast gains nothing, because B has already absorbed the lag between the two. Used consistently, the perfect forecast helps.
2. **Even a perfect forecast only improves the physics from 24% to 22%.** In 2024–2026 the main error isn't the weather but B itself: satellites change their attitude, raise their orbits, or start tumbling, so the drag measured over the past month doesn't hold for the coming months. That's the part the hybrid learns to correct, and it suggests a good solar forecast is worth less than better handling of B (for example, how much B varied over the fit window).

The seasonal cycle helps in the final month (19% vs 22%, 14% vs 17%) but not further out, so the baseline keeps the year-average density.

## 014: Which window to fit B over, and does drag stability help?

**Setup:** `experiments.py drag_stability` on the 2023 validation year. B fitted over the past 7, 14, 30 or 60 days. With the weather held fixed, the lifetime scales exactly as 1/B (derivation section 2), so the physics prediction for another window is the 30-day prediction × B₃₀/B_w, with no new simulation. Separately, the hybrid gets three extra inputs, log10(B_w/B₃₀) for w = 7, 14, 60, which are near 0 when the drag is steady.

| | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Physics, 7-day B | 21% | 17% | 29% | 53% | 44% | 37% |
| Physics, 14-day B | 20% | 15% | 19% | 35% | 29% | 46% |
| **Physics, 30-day B** | 19% | 13% | 16% | 26% | **23%** | 53% |
| Physics, 60-day B | 19% | 13% | 15% | 27% | 23% | 53% |
| Hybrid | 17% | 12% | 16% | 17% | **17%** | 69% |
| Hybrid + drag stability | 18% | 13% | 15% | 17% | 17% | 69% |

**Conclusion:** short windows are much worse far from reentry. At 500 km an orbit sinks only tens of meters a day, so a week of decay is mostly measurement noise. 30 and 60 days are equally good, so 30 days stays (it needs less history). The stability inputs get some use (the 60-day ratio has importance 0.07) but don't change the result, so they aren't added: the hybrid already sees the 7- and 30-day sinking rates, which carry the same information.

## 015: Fresh orbits and a short fit window in the final days

**Setup:** experiment 011 used one orbit a day, typically 8–12 hours old. `download_final_orbits.py` fetched every orbit from each object's final 40 days (213,372 orbits, 2,006 objects), and `compare_official.py --fresh` repeats the comparison with our latest orbit, now a median 1.7 hours old. Then `experiments.py final_days_window` refits B over the last 1–30 days using every orbit (physics only, no training, so test objects are fair).

| Made before reentry | Official | Physics, daily orbits (011) | Physics, fresh orbits, 30-day B | **Physics, fresh orbits, 8-day B** | Hybrid, fresh orbits |
|---|---|---|---|---|---|
| Under 1 day | **0.7 h** | 4.1 h | 2.3 h | 1.6 h | 5.4 h |
| 1–2 days | **3.7 h** | 9.0 h | 7.8 h | 6.4 h | 8.3 h |
| 2–4 days | **7.8 h** | 13.6 h | 12.3 h | 9.5 h | 13.9 h |
| 4+ days | 37.1 h | 37.3 h | 34.3 h | **26.7 h** | 40.5 h |

A 4-day window does about as well as 8 days; 2 days gets worse further out, and 1 day often has too few orbits to fit.

**Conclusion:** experiment 014 found that a short window is noise far from reentry, but near the end the orbit sinks kilometers per day, so a few days of tracking measure B precisely and capture recent changes (tumbling, attitude) that a 30-day average blurs. With fresh orbits and an 8-day B, the physics model **beats the official prediction from 4+ days out by 28%** and comes within 1.2× of it at 2–4 days. In the final day the official forecast, with dense radar tracking and higher-fidelity models, stays far ahead. The hybrid hurts here, since its correction was learned from one-orbit-a-week rows and almost never saw the last hours. So the live tool now uses the short-window physics close to reentry.

## 016: What the hybrid learned to correct

**Setup:** `experiments.py hybrid_corrections`. On test rows where the physics has an answer, compare the correction the hybrid applies (its predicted actual/physics factor) with the factor that was actually needed, grouped three ways. See `results/hybrid_corrections.png`.

| Condition | Needed | Learned |
|---|---|---|
| Solar flux 100–190 | 0.98–1.03 | 0.99 |
| Solar flux above 190 | 1.18 | 1.15 |
| Physics says under 7 days | 1.71 | 1.45 |
| Physics says 7–30 days | 1.25 | 1.11 |
| Physics says 90–365 days | 1.02 | 1.00 |
| Physics says over a year | 0.55 | 0.55 |
| Starlink | 1.52 | 1.15 |
| Debris + rocket bodies, payloads | 1.01–1.03 | 1.00–1.04 |

**Conclusion:** the hybrid leaves the physics alone where it's already right (moderate solar flux, passive objects) and corrects three specific weaknesses:
1. **At the solar peak** (flux above 190) objects lived 18% longer than the physics said, because the flux fell over the following months. This is experiment 002's problem, now handled by a learned correction instead of a solar forecast.
2. **In the final weeks** the physics is too short by a factor of 1.25–1.7. Part of this is the reentry definition (the simulation stops when the apogee reaches 120 km, while the catalog date can be up to a day later) and part is that the 30-day B lags behind a fast-changing final orbit (experiment 015).
3. **Starlink** lives 1.5× longer than its drag suggests, because satellites keep holding altitude for a while. The hybrid only learns part of this (1.15×), as the timing is the operator's choice.

The 0.55 for physics predictions over a year is partly an artifact: every row in the table is within a year of reentry, so a long physics prediction can only ever be too long. The hybrid learns that too, which is harmless here but means it shouldn't be used beyond a year (the lookup tool doesn't).
