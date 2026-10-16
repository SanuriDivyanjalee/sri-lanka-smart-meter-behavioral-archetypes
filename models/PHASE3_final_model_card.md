# Phase 3 High-Consumption Profiling Model

## Purpose

This is a supplementary descriptive model for profiling households
with comparatively high observed mean monthly electricity
consumption. It is not a forecasting or causal model.

## Training data

- Households: 1,413
- Predictor variables: 16
- Outcome: observed mean monthly electricity consumption
- Primary high-consumption cut-off:
  133.056 kWh/month
- Definition: sample top one-third

## Final estimator

- Algorithm: Logistic Regression
- Hyperparameters: {'C': 0.1, 'class_weight': 'balanced'}
- Numeric preprocessing: median imputation and RobustScaler
- Categorical preprocessing: most-frequent imputation and
  one-hot encoding with unknown-category handling
- Probability calibration: Sigmoid
- Saved-model decision threshold:
  0.295

## Unbiased nested out-of-fold performance

- ROC-AUC: 0.8207 (95% CI 0.7962-0.8434)
- PR-AUC: 0.7194 (95% CI 0.6839-0.7575)
- Balanced accuracy:
  0.7404 (95% CI 0.7166-0.7649)
- Precision: 0.5901 (95% CI 0.5620-0.6209)
- Recall: 0.7367 (95% CI 0.6985-0.7749)
- Specificity: 0.7442 (95% CI 0.7187-0.7707)
- F1-score: 0.6553 (95% CI 0.6268-0.6844)
- Brier score: 0.1561 (95% CI 0.1465-0.1656)

The operating threshold was selected using training-only out-of-fold
predictions. It improved recall, balanced accuracy, and F1 relative
to threshold 0.50, while reducing precision and specificity.

## Robustness checks

- Nested comparison of Logistic Regression, Random Forest, and
  Gradient Boosting
- Training-only calibration selection
- Training-only decision-threshold selection
- Bootstrap confidence intervals
- Exclusion of nine zero-consumption households
- Alternative high-consumption definitions
- Balanced-accuracy and ROC-AUC learning curves
- Cross-validated permutation importance
- Saved-model reload and unseen-category smoke tests

### Target-definition sensitivity

- Top 40 percent: ROC-AUC 0.8063, balanced accuracy 0.7315, F1 0.6795
- Top one-third - primary: ROC-AUC 0.8212, balanced accuracy 0.7473, F1 0.6633
- Top 25 percent: ROC-AUC 0.8457, balanced accuracy 0.7782, F1 0.6394
- Top 20 percent: ROC-AUC 0.8678, balanced accuracy 0.7959, F1 0.6040

### Strongest cross-validated permutation features

- fan_total_count: mean ROC-AUC decrease 0.0405
- household_size: mean ROC-AUC decrease 0.0340
- ac_total_count: mean ROC-AUC decrease 0.0171
- appliance_total_count: mean ROC-AUC decrease 0.0116
- firewood_cooking: mean ROC-AUC decrease 0.0090

Permutation importance describes contribution to out-of-fold
discrimination. It does not establish a causal effect.

## Intended use

- supplementary research profiling;
- exploratory identification of households with comparatively
  high observed consumption;
- methodological demonstration of a leakage-safe model workflow.

## Prohibited or unsupported use

- future electricity-consumption forecasting;
- electricity-bill estimation;
- tariff eligibility;
- welfare, poverty, or vulnerability classification;
- causal inference;
- national household classification;
- operational deployment without independent external validation.

## Limitations

The outcome is sample-relative. The matched analytical population is
not nationally representative. The model has not been externally
validated on an independent population. Household-profile variables
may be associated with consumption but must not be interpreted as
causes. The final model should be accompanied by its feature schema,
target cut-off, calibration method, decision threshold, and model
metadata.
