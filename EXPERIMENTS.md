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
