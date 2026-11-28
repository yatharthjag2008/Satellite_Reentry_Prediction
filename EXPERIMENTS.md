# Experiment log

Numbered experiments on predicting reentry. Each entry records what changed, what happened and what I concluded. Unless noted (Part 1 is experiments 017–019), models train on objects that reentered before 2024-01-01 and are tested on objects that reentered from 2024-01-01 onward, and errors are the median of |predicted / actual − 1|.

| # | Question | Answer in one line |
|---|---|---|
| [001](#001-baseline-models) | How good are standard ML models? | XGBoost 26%, Gaussian process 22%; rule of thumb 430% |
| [002](#002-where-xgboost-goes-badly-low) | Where does XGBoost go wrong? | Starlink and the 2024–25 solar peak |
| [003](#003-physics-model-alone) | How good is the drag simulation alone? | 24%, with no training at all; 14% on payloads |
| [004](#004-physics--ml-hybrid) | Does a physics + ML hybrid help? | Slightly (22%), with default settings |
| [005](#005-ways-to-combine-physics-and-ml-validation-year) | How should physics and ML be combined? | Learn a simple correction to the physics |
| [006](#006-tuning-the-hybrid-validation-year) | How sensitive is the hybrid to its settings? | Not much: regularization is what matters |
| [007](#007-tuning-plain-xgboost-the-same-way-validation-year) | Is the baseline fairly tuned? | It wasn't; tuned, XGBoost improves to 18% |
| [008](#008-final-test-of-the-tuned-models) | Final test | Hybrid 16%, XGBoost 19%, physics 24% |
| [009](#009-conformal-90-ranges-calibrated-on-2023) | Are 90% ranges reliable? | Only 81%: the population changed |
| [010](#010-rolling-recalibration-of-the-ranges) | Can recalibration fix that? | Yes: 89%, Starlink from 59% to 88% |
| [011](#011-against-the-official-us-space-force-predictions) | How do we compare with official forecasts? | Equal from 4+ days out, worse close in |
| [012](#012-year-by-year-backtest-20052026) | Does it hold across solar cycles? | Hybrid best in 20 of 22 years |
| [013](#013-how-much-would-a-perfect-solar-forecast-help-the-physics) | Is the solar forecast the bottleneck? | No: a perfect one gains 2 points |
| [014](#014-which-window-to-fit-b-over-and-does-drag-stability-help) | What fit window for B? | 30 days; shorter is noise far out |
| [015](#015-fresh-orbits-and-a-short-fit-window-in-the-final-days) | Can fresh orbits close the gap with official forecasts? | Beats them 4+ days out; 8-day B near the end |
| [016](#016-what-the-hybrid-learned-to-correct) | What does the hybrid correct? | Solar peak, final weeks and Starlink |
| [017](#017-part-1-physics-from-the-first-40-days-to-reentry) | Lifetime from the first 40 days? | 29% with a solar-cycle forecast (84% without) |
| [018](#018-part-1-physics-vs-ml-vs-hybrid) | Does ML help for lifetimes? | Seems to (26%) under cross-validation by launch... |
| [019](#019-part-1-with-a-fair-time-split) | ...and on a later period? | No: physics 35%, ML 55–69% |
| [020](#020-does-the-solar-cycle-forecast-help-within-a-year) | Does the solar-cycle forecast help Part 2? | No: it only matters over years |
| [021](#021-drag-fits-too-large-to-be-drag) | Can the physics tell when it's wrong? | Yes: a fitted B over 1 m²/kg isn't drag; 90% ranges 3.5× narrower |
| [022](#022-the-drag-cap-in-the-hybrid) | Does the hybrid gain from the same rule? | A little (hybrid under 1 week 21% → 19%); now used everywhere |
| [023](#023-90-ranges-for-the-final-days) | Can the final-days physics get honest ranges? | Yes: 87–95% coverage on later reentries |
| [024](#024-would-a-storm-forecast-help-in-the-final-week) | Would a storm forecast help in the final week? | Barely: even a perfect one saves about an hour, only during storms |
| [025](#025-part-1-the-moon-and-the-sun-on-transfer-orbits) | Do the Moon's and Sun's gravity fix transfer orbits? | Yes: 503% → 28% median error on 261 transfer orbits |
| [026](#026-the-moon-and-the-sun-on-transfer-orbits-in-their-final-year) | And in their final year? | Yes: 39% → 10% on the test years, and 100% get an answer instead of 80% |

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

## 017: Part 1 physics, from the first 40 days to reentry

**Setup:** `lifetime_model.py`. For 4,948 satellites and rocket bodies launched since 2000 that have reentered, fit B from orbits 10–40 days after launch, then simulate from about day 40 to reentry under three assumptions about the Sun. Debris is excluded (its catalog launch date is its parent's). Of these, 2,982 (60%) get a positive B; the rest were still raising their orbits or the decay was too small to measure. Scores below are for those 2,982.

| Future solar flux | < 1 year | 1–3 years | 3–10 years | 10+ years | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Today's held fixed | 57% | 54% | 247% | 62% | 84% | 23% |
| **Solar-cycle forecast** (`solar_cycle.py`) | 56% | 31% | 20% | 30% | **29%** | 46% |
| Actual flux (perfect forecast) | 53% | 16% | 20% | 18% | 20% | 58% |

By group (solar-cycle forecast): near-circular rocket bodies 21%, near-circular satellites 25%, rocket bodies in transfer orbits 528%, Starlink 96%.

**Conclusion:** for multi-year lifetimes the solar cycle is everything. Holding today's flux fixed gives 84% error (247% for 3–10 years, where an object launched near solar minimum sees a whole maximum); the cycle forecast cuts that to 29%, close to the 20% a perfect forecast would give. With it, the physics predicts the lifetime of near-circular objects to about 20–25% from a single month of tracking, years in advance. It fails for transfer orbits (lunisolar perturbations, derivation section 9) and for Starlink, whose lifetime is set by the operator.

## 018: Part 1 physics vs. ML vs. hybrid

**Setup:** `train_lifetime_model.py`, all 4,948 objects (no-answer physics falls back to 30 years). XGBoost and the hybrid use the same tuned settings as Part 2, with the early orbit, BSTAR, radar size, object type, fitted B and (hybrid) the solar-cycle physics prediction. Because every object here has reentered, a time split would be biased (only short-lived objects from recent launches are down yet), so the models are **cross-validated by launch**: 5 folds, with all objects from one launch date in the same fold.

| Model | < 1 year | 1–3 years | 3–10 years | 10+ years | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Physics, solar-cycle forecast | 749% | 55% | 404% | 49% | 139% | 28% |
| XGBoost | 60% | 24% | 24% | 44% | 29% | 44% |
| **Physics + ML hybrid** | 56% | 23% | 22% | 36% | **26%** | **49%** |

| Group (overall) | Physics | XGBoost | Hybrid |
|---|---|---|---|
| Rocket bodies, near-circular | 24% | 36% | 25% |
| Rocket bodies, transfer orbits | 533% | 68% | 71% |
| Satellites (not Starlink), near-circular | 30% | 26% | **22%** |
| Starlink | 698% | 29% | **28%** |

**Conclusion:** the same pattern as Part 2. The hybrid is best overall (26%) because it takes the physics where the physics works and falls back on learned patterns where it doesn't: for Starlink, the ML learns that operators keep them up about five years, which no drag model can know. For near-circular rocket bodies, which nothing steers, the physics alone is already as good as anything (24%). The caveat is the validation: cross-validating by launch keeps sibling objects apart but doesn't test on a later time period, so these numbers are likely somewhat optimistic for the ML models (not for the physics, which isn't trained).

## 019: Part 1 with a fair time split

**Setup:** `train_lifetime_model.py --time-split`. Experiment 018 cross-validated by launch, which doesn't test on a later period. A plain time split would be biased by censoring, so this keeps only objects launched before 2016 that lived at most 10 years: every such object has reentered by 2026, so training (438 objects launched 2000–2011) and testing (184 launched 2012–2015) are filtered the same way.

| Model (test: launched 2012–2015) | Overall | Within 25% | Near-circular satellites | Near-circular rocket bodies |
|---|---|---|---|---|
| **Physics, solar-cycle forecast** | **35%** | **43%** | **23%** | **28%** |
| XGBoost | 69% | 16% | 70% | 44% |
| Physics + ML hybrid | 55% | 16% | 51% | 34% |

**Conclusion:** an important correction to 018. Tested on a later period, the ML models fall apart (69% and 55%), while the physics, which isn't trained at all, keeps its accuracy (23% for satellites, compared with 25% in 017). Two reasons: the training set is small (438 objects), and the satellite population changed between periods (the CubeSat boom started around 2012), so patterns learned from 2000–2011 don't transfer. Cross-validation by launch let the ML see objects from the same era, which flattered it. **For whole lifetimes, the physics with the solar-cycle forecast is the method to trust**; the hybrid's advantage in 018 shouldn't be relied on. In Part 2 the hybrid did pass a fair time split (008, 012), because there it had 336,000 training rows covering two solar cycles.

## 020: Does the solar-cycle forecast help within a year?

**Setup:** `experiments.py solar_forecast` again, adding the Part 1 solar-cycle forecast (month by month, Ap held at its recent value) to the 5,000-row comparison of experiment 013.

| Future weather | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall |
|---|---|---|---|---|---|
| Today's held fixed (baseline) | 22% | 17% | 18% | 27% | 24% |
| Solar-cycle forecast | 21% | 16% | 18% | 27% | 24% |
| Perfect forecast (trailing 81-day flux) | 19% | 13% | 19% | 24% | 22% |

**Conclusion:** no gain. The forecast blends from today's flux into the cycle shape over about a year, so within a year it hardly differs from holding today's flux fixed, and the cycle shape can't predict the month-to-month swings that make up most of the remaining 2-point gap to a perfect forecast. It matters for multi-year lifetimes (017: 84% → 29%), not for days-until-reentry, so Part 2 keeps the simpler fixed weather and leaves the solar peak to the hybrid (016).

## 021: Drag fits too large to be drag

**Setup:** calibrating 90% ranges for physics-only forecasts (for the website, which can't use the trained hybrid) gave ranges far wider than the median errors suggest: for debris predicted to reenter in 3–10 days, ÷9 to ×9. The worst cases are debris at 550–700 km (Fengyun 1C, CZ-6A and Scout debris) with a fitted B of 15–470 m²/kg, a hundred times more than any solid object, predicted to fall within days but staying up for months. In the example checked, the perigee falls while the apogee *rises*, which drag can't do: gravity (Earth's odd zonal harmonics, the Sun and the Moon) is slowly shifting the eccentricity, and with almost no air at that height the fit blames the tiny change in a on drag. `experiments.py drag_cap` tries rules that flag such fits, chosen on the 2023 validation year. A flagged row gets no forecast; rows are those with a positive B.

| Validation 2023 | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Rows kept | 90% range, predicted 3–10 days | 10–30 days |
|---|---|---|---|---|---|---|---|---|
| No rule | 17% | 12% | 14% | 27% | 22% | 100% | ×5.77 | ×1.96 |
| B > 0.5 m²/kg | 17% | 11% | 13% | 26% | 21% | 95% | ×1.56 | ×1.45 |
| **B > 1 m²/kg** | 17% | 11% | 14% | 27% | 22% | 98% | ×1.64 | ×1.50 |
| Apogee rising over 30 days | 17% | 12% | 14% | 27% | 22% | 100% | ×5.77 | ×1.96 |

On the test set (2024+), B > 1 m²/kg keeps 98% of rows, the overall median error stays at 20% (by horizon each is a point lower or the same), and the 90% range for predictions of 3–10 days narrows from ×19 to ×14 across all groups. That is still wide because of Starlink, whose operator chooses when to deorbit; for debris and rocket bodies in the latest year of reentries it is ×1.6 (`forecast_ranges.py`).

**Conclusion:** B = C_D·A/m above 1 m²/kg means more than about 0.45 m² per kilogram, foil rather than hardware, so a fit that large is the model admitting that drag doesn't explain the orbit. B > 1 is kept over B > 0.5, which narrows the ranges about as much but drops twice as many rows, many of them fine. The rising-apogee rule catches almost nothing, since in daily mean elements the apogee rarely rises for a whole month. The live forecasts on the website say "not decaying by drag alone" instead of forecasting when B > 1. Experiment 022 tests the rule in the hybrid too.

## 022: The drag cap in the hybrid

**Setup:** `experiments.py hybrid_drag_cap` on the 2023 validation year. The hybrid is retrained with the physics giving no answer when B > 1 m²/kg (as for a negative B, its input is then 365 days and the ML takes over), and compared with the hybrid as it was.

| Validation 2023 | < 1 week | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% |
|---|---|---|---|---|---|---|
| Hybrid | 17.5% | 12.3% | 15.9% | 17.3% | 16.8% | 68.7% |
| **Hybrid, capped** | 17.8% | 12.0% | 15.3% | 17.0% | **16.5%** | **69.8%** |

The 1.4% of rows the rule flags were hard for the hybrid either way (29% error with or without it).

**Result:** a small but broad gain on the validation year, so the cap became part of the physics model (`physics_model.forecast` gives no answer for B > 1) and everything downstream was rerun. On the test set (2024+), the hybrid is **19%** under 1 week (was 21%), 15% at 1–4 weeks (was 16%), 16% overall (unchanged) and within 25% for **68%** of rows (was 67%); the physics-only row is unchanged at 24%, because those few flagged rows are scored at the 365-day fallback either way. The rolling 90% ranges still contain the truth 89% of the time. The year-by-year backtest (012) is unchanged in its medians (hybrid 22%, XGBoost 24%, physics 27%) and still has the hybrid best or tied in 20 of 22 years. The largest change is in the final days: against the official predictions with one orbit a day (011), the hybrid's median error falls from 6.0, 9.7, 15.5 and 41.7 hours to 4.8, 6.7, 11.1 and 34.2 hours (under 1 day, 1–2, 2–4 and 4+ days), since near reentry a Starlink lowering itself with thrusters looks like enormous drag. The 8-day physics of 015 is unchanged.

**Conclusion:** the hybrid had mostly learned to distrust these fits already (016), but telling it outright is slightly better, and it makes the physics consistent across the lookup tool, the live forecasts and the experiments. The README's tables now come from this version. The lookup tool says when the physics has no answer for this reason and shows the hybrid's prediction.

## 023: 90% ranges for the final days

**Setup:** within 10 days of reentry the lookup tool uses the physics model with B fitted over the last 8 days of every orbit (015), but it gave no range for that prediction: the hybrid's conformal ranges (009–010) were calibrated on one orbit a week and almost never saw the final days. `experiments.py final_days_ranges` calibrates Mondrian conformal ranges (by object group and band of predicted days) for the 8-day-B prediction on every orbit in the final 40 days (`download_final_orbits.py`), against the reentry time of the final official TIP message, which is good to the hour, instead of the catalog date. It calibrates on objects that reentered before 2025-07-01 and checks on the 886 that reentered after. No model is trained, so the only thing fitted is the widths.

| Predicted | Group | Rows | 90% range contains truth | Median width |
|---|---|---|---|---|
| Under 3 days | Debris and rocket bodies | 1,168 | 90% | 33 h |
| | Other payloads | 798 | 89% | 26 h |
| | Starlink | 7,685 | 87% | 65 h |
| 3–10 days | Debris and rocket bodies | 1,619 | 95% | 95 h |
| | Other payloads | 1,261 | 87% | 98 h |
| | Starlink | 26,448 | 89% | 661 h |

**Conclusion:** the ranges hold on the later period, 87–95% against 90% asked for. They are tight for uncontrolled objects (about ±16 hours around a prediction under 3 days for debris and rocket bodies) and wide for Starlink, whose operator lowers them on its own schedule. `forecast_ranges.py` now also saves these widths, calibrated on the latest year of reentries, and the lookup tool shows the range with its final-days prediction.

## 024: Would a storm forecast help in the final week?

**Setup:** geomagnetic storms heat the upper atmosphere within hours, and in the final days the orbit is low enough to feel it, so a forecast of storms (such as NOAA's 3-day forecast of Ap) might sharpen the final-days prediction. Before testing a real forecast, `experiments.py storm_bound` measures the most any forecast could gain: the 8-day-B physics from every orbit in the final week (015) is rerun with the 3-hourly ap index that actually happened until reentry, instead of the 7-day average held fixed. Errors are against the final official reentry time, on 33,834 orbits of 1,971 objects; 24% of them had ap reach 50 (a minor storm) before reentry. No training is involved, so test objects are used.

| Median error (hours) | Under 1 day | 1–2 days | 2–4 days | 4+ days |
|---|---|---|---|---|
| All, 7-day ap held fixed (now) | 1.6 | 5.3 | 10.6 | 18.5 |
| All, actual ap (perfect forecast) | 1.6 | 5.3 | 10.5 | 18.2 |
| Storm before reentry, held fixed | 1.9 | 6.8 | 12.4 | 20.3 |
| Storm before reentry, perfect forecast | 1.8 | 5.7 | 10.8 | 19.7 |
| No storm, held fixed | 1.6 | 5.1 | 10.1 | 17.5 |
| No storm, perfect forecast | 1.5 | 5.3 | 10.4 | 17.3 |

**Conclusion:** even perfect knowledge of the coming storms helps little: nothing on average, and about an hour (13–16%) at 1–4 days when a storm does come. A real forecast would get only part of that, so none is added. The 8-day fit already absorbs much of the effect: it measures the drag of the last few days, storms included. As in 013, what limits the final days is not the weather but how well the drag itself (B) is known.

## 025: Part 1, the Moon and the Sun on transfer orbits

**Setup:** Part 1's physics failed on transfer orbits (apogee above 5,000 km, mostly rocket bodies that sent satellites toward geostationary orbit): near apogee the Moon and the Sun pull on the orbit and slowly raise or lower the perigee, which drag alone can't follow (017). `lunisolar.py` adds their orbit-averaged tides at their actual positions, and Earth's J₂ flattening, to the drag simulation (`docs/derivation.md`, section 9). `download_transfer_orbits.py` fetched each object's orientation (node and argument of perigee) at its day-40 starting orbit. `experiments.py transfer_orbits` then simulates the same 261 transfer orbits with a fitted B, from the same starting orbits, with the same B and the same solar-cycle forecast as Part 1, three ways. Nothing is fitted, so all of them are scored.

| Forces | Under 1 year | 1–3 years | 3–10 years | 10+ years | Overall | Within 25% | Answer within 30 years |
|---|---|---|---|---|---|---|---|
| Drag only (Part 1 until now) | 1904% | 495% | 229% | 75% | 503% | 10% | 66% |
| Drag + J₂ | 1903% | 495% | 229% | 75% | 503% | 10% | 66% |
| **Drag + J₂ + Sun + Moon** | **26%** | **33%** | **34%** | **28%** | **28%** | **46%** | **97%** |

The middle half of predicted / actual lifetimes goes from 2.0–21.7 to 0.68–1.14. Halving the step from one day to half a day changes the lifetimes by a median of 0.09% (60 objects), so the result isn't a numerical artifact.

**Conclusion:** for transfer orbits the Sun and Moon aren't a correction, they decide the lifetime: with drag alone the model mostly predicts a perigee that never comes down, while the real perigee is pulled down into the air (or lifted out of it) over months. J₂ alone changes nothing, as expected, since it turns the orbit without changing the perigee height, but it matters together with the Sun and Moon by setting which way the orbit faces them. With all three, transfer orbits are predicted about as well as near-circular ones (28% against Part 1's 29%), years ahead, from 40 days of tracking.

**Follow-up:** `lifetime_model.py` now simulates every transfer orbit whose angles are known this way (near-circular orbits stay drag-only, and their predictions are unchanged). Part 1's physics with the solar-cycle forecast goes from 29% to **26%** median error on all 2,982 objects with a fitted B, and from 528% to 29% on the 250 transfer rocket bodies among them. In `train_lifetime_model.py`, counting every object (no fitted B counts as 30 years), the physics on transfer rocket bodies goes from 533% to 32%, and the hybrid built on it from 71% to 37%. XGBoost alone stays at 68%. The hybrid over all 4,948 objects goes from 26% to 25%. The lookup tool now asks Space-Track for the two angles, and uses this physics (with the solar-cycle forecast, and no hybrid, which learned to correct drag alone) for a transfer orbit more than 10 days from reentry.

## 026: The Moon and the Sun on transfer orbits in their final year

**Setup:** 025 fixed transfer orbits years ahead, from 40 days after launch. Does the same help in the final year, which is what the lookup and the live forecasts predict? The main physics (003, with the drag cap from 021) is also poor on these orbits: on the weekly rows with an apogee above 5,000 km, about 4% of all weekly rows, it has an answer for only 73%. `download_final_year_angles.py` fetched the node and argument of perigee for all 24,337 of those rows (1,049 objects), and `experiments.py final_year_transfer` simulates the 19,532 rows with a plausible B (803 objects) from the same orbits, with the same fitted B and the same space weather held fixed, with drag only (as now) and with J₂, the Sun and the Moon. With every force but drag turned off, the second integrator matches the first to a median 0.01%. Nothing is fitted, so every row is scored; the test years are also shown alone.

| Rows | Forces | 1–4 weeks | 1–3 months | 3–12 months | Overall | Within 25% | With an answer |
|---|---|---|---|---|---|---|---|
| All years, both have an answer (14,288) | Drag only (now) | 88% | 52% | 39% | 42% | 39% | |
| | **Drag + J₂ + Sun + Moon** | **53%** | **23%** | **13%** | **15%** | **69%** | |
| All years, every row (19,532; no answer counts as 365 days) | Drag only (now) | 100% | 97% | 49% | 53% | 33% | 73% |
| | **Drag + J₂ + Sun + Moon** | **58%** | **25%** | **14%** | **15%** | **69%** | **99%** |
| Test (reentered 2024+), every row (1,943) | Drag only (now) | 128% | 49% | 38% | 39% | 38% | 80% |
| | **Drag + J₂ + Sun + Moon** | **45%** | **17%** | **10%** | **10%** | **80%** | **100%** |

In the final week (95 rows) both are equally poor (median error 100%): the perigee is already near 100 km, where the simulation stops, and within 10 days the lookup and the live forecasts use the 8-day fit (015) anyway. The 90% ranges the live forecasts use (calibrated on drag-only physics, 023 and `forecast_ranges.py`), put around the new predictions for the 2,489 rows from 2023 on, contained the truth 96.5% of the time: 94% at 1–3 months and 98% at 3–12 months, but only 64% at 1–4 weeks (44 rows).

**Conclusion:** the Sun and Moon matter in the final year too. With them, transfer orbits go from the physics' worst case to better than its average: 10% median error on the test years, against 20% for all objects in the replay of the live forecasts. The live forecasts now simulate transfer orbits this way (`live_forecasts.with_sun_and_moon`), as the lookup already does.

**Follow-up, ranges:** `forecast_ranges.py` now also calibrates "transfer" widths on these predictions (latest 3 years of reentries, 93 objects, since only about 30 transfer orbits reenter a year). Calibrated on reentries before 2025 (82 objects) and checked on later ones (59 objects, 1,178 rows), the transfer widths alone held only 84% (weekly rows of so few objects are far from independent), the usual 30-day widths 96% overall but 79% at 1–4 weeks, and the wider of the two 96% overall and 95% at 1–4 weeks (19 rows). The live forecasts use the wider of the two.
