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
