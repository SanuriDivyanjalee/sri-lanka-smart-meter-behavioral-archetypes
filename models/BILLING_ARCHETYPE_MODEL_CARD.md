# Model Card — Monthly-Billing Archetype-Transfer Research Prototype

## Purpose

This model transfers four locked smart-meter-derived K-Means pseudo-labels
using household-level monthly billing features.

It is a supplementary contemporaneous label-transfer prototype, not an
archetype-discovery, forecasting, causal, or externally validated
operational model.

## Locked data endpoint

- Locked K-Means population: 2,227 households
- Monthly-billing matched population used for fitting: 1,146
- Predictor count: 32
- Model family: Random Forest
- Feature set: Extended reduced

## Primary evaluated procedure

Step 4 provides the primary approximately unbiased internal estimate for the
**adaptive nested workflow-selection procedure**, in which the selected feature
set, model family and parameters may differ across outer folds:

- Accuracy: 0.6824
- Balanced accuracy: 0.5226
- Macro-F1: 0.4727

These values must not be presented as an independent unbiased estimate of the
exact fixed prototype saved in this package.

## Final fitted prototype

The saved Random Forest configuration was selected using
full-dataset repeated cross-validation and then fitted on all matched
households. Its repeated-CV values are secondary post-selection internal
estimates:

- Repeated-CV accuracy: 0.6771
- Repeated-CV balanced accuracy: 0.5359
- Repeated-CV macro-F1: 0.4812

No independent unbiased performance estimate exists for this exact
full-data-fitted prototype.

## Input contract

The estimator requires the exact engineered feature columns listed in the
feature schema. The accompanying `archetype_runtime.py` and Streamlit app
transform validated raw monthly billing records into those features. At least
15 valid monthly observations are required, and at most the latest 25 are used.

## Intended use

- Methodological proof-of-concept
- Supplementary analysis of whether billing features contain information
  about smart-meter-derived archetypes
- Reproducible research inference on correctly engineered feature rows

## Not intended for

- Replacing 15-minute smart-meter clustering
- Discovering new archetypes
- Prospective household forecasting
- Causal, welfare, tariff, or policy eligibility decisions
- Operational deployment without external validation

## Principal limitations

- Pseudo-label rather than external ground truth
- Differential billing-match coverage across archetypes
- Only 21 matched Daytime-Active households
- Monthly totals cannot directly recover time-of-use structure
- All-available billing predictors overlap the smart-meter label period
- Moderate balanced accuracy and macro-F1
- No independent external validation cohort
- Returned probabilities are raw uncalibrated model scores, not validated confidence estimates
- Training coverage was 15–25 observed months; inputs outside this range are not validated
