# Research Methodology

## Scope and evidence

This document summarizes the **submitted undergraduate thesis methodology**, not a new or independently reproduced analysis. The source of record is the submitted thesis, *A Data-Driven Approach for Characterizing and Segmenting Temporal Electricity-Use Patterns in Sri Lankan Households Using High-Frequency Smart Meter Data*, and the retained notebook source. The public repository provides curated source-code drafts and selected aggregate result artifacts, not restricted research records or a complete rerunnable pipeline.

## Research question and analysis period

The central question is whether 15-minute residential electricity measurements reveal interpretable **when-and-how** consumption patterns that are concealed by monthly totals. The primary analysis uses the **LIRNEasia Sri Lankan Residential Electricity Consumption Dataset**, restricting the final comparison window to **May–November 2024**. After the documented eligibility, quality, export-activity, and near-zero-activity checks, **2,227 households** were retained for clustering.

The analysis uses household-level feature vectors; individual 15-minute measurements are not publicly redistributed. Monthly billing, six-hour cumulative registers, and household survey data are used as supporting evidence and for separate supplementary studies, not as ground-truth labels for the primary clustering.

## Preparation and eligibility

The thesis describes a fixed preprocessing chain that: parses and orders interval readings; applies timing, nonnegative-consumption, physical plausibility and coverage rules; excludes export-affected households to focus the main analysis on electricity-import behavior; and uses a common May–November window. After engineering the features, near-zero-activity households with `log_daily_kwh <= 0.05` are excluded. Each retained feature is clipped at the 1st and 99th percentiles, followed by `StandardScaler` normalization for the distance-based clustering. These choices are study-specific; this document does not assert that they are universal household screening thresholds.

**Privacy boundary:** the public notebooks may still read or create record-level files when executed. They must not be run against confidential datasets in a public workspace. The complete original private workflow, including optional household-specific investigations, is not distributed.

## Six behavioral features

| Feature | Operational interpretation in the thesis |
| --- | --- |
| `evening_peak_ratio` | Fraction of daily imported electricity in the 18:30–22:30 peak block |
| `night_baseload_ratio` | Fraction of daily imported electricity in the 22:30–05:30 overnight block |
| `load_uniformity` | Shannon entropy of the 96-interval daily load profile |
| `load_factor` | Mean 15-minute electricity import divided by the 15-minute peak |
| `leisure_usage_ratio` | `log1p` of non-working-day (weekend/public holiday) to working-day consumption ratio |
| `log_daily_kwh` | `log1p` of mean daily imported electricity consumption |

The behavioral features use observed interval demand; they do not establish appliance-level causation, income status, or future consumption.

## Clustering and K selection

The main model is **K-Means with K = 4, random_state = 42, and n_init = 20** on the standardized six-feature household matrix. The candidate K values were assessed using silhouette/inertia and other internal diagnostics, alternative clustering methods, initialization sensitivity, bootstrap stability, and substantive interpretation.

K = 4 was selected as the **smallest interpretable solution preserving four distinct temporal patterns**, not because all numerical criteria had a common optimum. The thesis reports that unrestricted compactness measures often favored K = 2, elbow behavior approximately K = 3, and K = 5 could split already recognizable patterns. The final groups are *Peak-Exposed, Typical Household, Heavy Continuous,* and *Daytime-Active*.

## Validation and limitations

The thesis examines internal cluster separation, changes across random initializations, bootstrap resampling, temporal-state persistence, alternative algorithms, and relationships with separately collected billing, surveys and six-hour registers. Those external sources supply **contextual consistency checks**, not definitive ground-truth archetype labels. The observed silhouette score and bootstrap variation require a cautious interpretation of cluster boundaries.

Supplementary analyses have different populations, outcomes and estimators:

- **Billing-to-archetype transfer:** a Random Forest predicts **smart-meter-derived pseudo-labels** from billing features in a 1,146-household matched sample. Reported nested out-of-fold scores represent an adaptive evaluation process rather than guaranteed performance for a final saved artifact.
- **High-consumption profiling:** a sigmoid-calibrated Logistic Regression with 16 profile features examines a study-relative **top-one-third consumption** target across 1,413 households, assessed using nested out-of-fold predictions.
- **Electricity-related operational CO₂:** reconstructed six-hour electricity imports are multiplied by a **fixed 0.71 kg CO₂/kWh** coefficient. Regression and study-relative high-emission classification use five temporal features within 1,656 qualified households and include held-out evaluation. These are **retrospective electricity-related estimates**, not total household footprints or future forecasting.

### Source availability and execution status

The 10 included notebooks have cleared outputs. Two additional notebooks (EDA and survey high-consumption analysis) were withheld pending review of their file-deletion and output-replacement operations. Some household-specific sections in the main and survey-validation public copies have been withheld; this is disclosed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

No notebook sequence or application has been independently rerun end to end from the public package. There is no verified repository-wide environment lockfile, and raw records, trained model binaries and upstream derived private data are absent. Consult [DATA.md](DATA.md) for source rights and [RESULTS.md](RESULTS.md) for the recorded aggregate findings.
