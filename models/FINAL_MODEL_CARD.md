# Carbon-XAI Saved Research Prototype

## Purpose
Estimate calculated mean monthly electricity-related operational CO2 emissions
and identify households above the development-set Q75 emission threshold. The
full-cohort model uses five target-excluded temporal load-shape predictors. A
paired Wave-1 household-context matched-cohort model additionally uses eight genuine
household, demographic and appliance variables.

## Data and scope
- Study period: May-November 2024
- Primary cohort: at least 5 quality-passing full months
- Adjacent cross-month allocation limit: 48.0 hours
- Missing months: not imputed
- Emission factor: 0.71 kg CO2/kWh
- Evaluation: internal household-level held-out test
- Design: cross-sectional explanatory modelling, not future forecasting

## Selection safeguards
- Final test split created before model development
- Nested development CV for model-family and hyperparameter selection
- Classification target threshold derived inside each outer training fold
- Hyperparameter-tuning labels derive Q75 again inside each inner training fold
- Classification model-selection folds are not stratified using a global target
- Probability decision threshold selected from honest nested OOF predictions
- Leave-one-feature-out ablation is restricted to development folds
- Held-out residual diagnostics characterize errors without retuning the model
- Matched Wave 1 improvement is accompanied by paired bootstrap confidence intervals
- Logistic Regression and Random Forest compared with both unweighted and
  class-weight-balanced training
- Three split seeds audited as a sensitivity analysis
- Dummy baselines remained eligible if no trained model improved the primary
  nested out-of-fold metric

## Prohibited interpretations
- Not external validation
- Not a complete household carbon footprint
- Not causal evidence
- Not calibrated household risk unless calibration is separately established
- Not operational deployment ready

## Required inputs for the full-cohort model
- evening_peak_ratio
- night_baseload_ratio
- load_uniformity
- load_factor
- leisure_usage_ratio

## Additional inputs for the Wave-1 household-context matched-cohort model
- n_members
- mean_age
- floor_area
- total_monthly_expenditure_of_last_month
- n_acs
- n_appliances
- avg_hours_home
- occupant_density

## Direct-target variables deliberately excluded
- avg_monthly_kwh: directly constructs the target
- avg_log_daily_kwh: directly encodes current-period magnitude
- final_cluster: derived from the temporal features and not independent
