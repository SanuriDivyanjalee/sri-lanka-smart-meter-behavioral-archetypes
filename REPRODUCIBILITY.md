# Reproducibility and execution boundaries

**Status:** Research portfolio draft; this repository is **not** a complete, independently reproducible or deployment-ready release.

This document distinguishes the evidence that readers can inspect from the experiments and applications that cannot currently be reproduced using the files in this repository alone. It reflects the thesis and the selected artifact files reviewed during portfolio preparation; no clean-environment end-to-end execution has been performed for this public draft.

## What is included

| Material | Location | What it supports |
| --- | --- | --- |
| Research notebook source, with stored outputs cleared | [`notebooks/`](notebooks/) | Inspection of the analysis methodology and documented checks; **not** proof of a successful rerun |
| Two Streamlit application source drafts | [`app/`](app/) | Inspection of research-prototype interfaces and code paths; **not** a deployable application |
| Aggregate clustering figures | [`figures/`](figures/) | Illustration of reported behavioral patterns |
| Four aggregate result tables | [`results/`](results/) | Inspection of selected monthly feature/trend and carbon hold-out results |
| Three model cards and three metadata documents | [`models/`](models/) | Model purpose, evaluation scope, configurations, and limitations |

The 10 notebooks and the two applications are curated **source drafts**, not packaged or verified end-user software. The notebook outputs have been cleared; earlier execution output should not be inferred from an empty `outputs` field.

### Public-review notebook changes

The main clustering and survey-validation notebooks were **replaced by revised public-review drafts** after the original cleaned drafts were placed in a private review folder:

- [`SmartMeter_Master_Pipeline_Public_Review_Draft.ipynb`](notebooks/SmartMeter_Master_Pipeline_Public_Review_Draft.ipynb) retains the main analysis while replacing nine case-specific diagnostic sections with explanatory Markdown notes and redacting household-specific references in two additional sections.
- [`SmartMeter_Survey_Validation_Public_Review_Draft.ipynb`](notebooks/SmartMeter_Survey_Validation_Public_Review_Draft.ipynb) preserves ten original analysis/setup code cells and replaces an optional three-household sensitivity investigation with a notice that it was withheld.

These changes mean that **not all original source cells are publicly available**. The complete original notebooks and their investigations remain in the separate private source project. Earlier cleaned drafts are held in a private review folder. Neither is included in this public repository. Additionally, the EDA and survey-based high-consumption analysis notebooks have been **withheld** pending review of their file-deletion and output-replacement operations. Their omission does not invalidate the selected aggregate evidence or model documentation, but it limits the source workflows available for independent inspection. This curation has **not** been validated by an end-to-end rerun. Some remaining cells still depend on restricted inputs or write household-level records.

## Restricted or omitted material

The following are intentionally excluded from the public draft:

- Raw or household-level **15-minute smart-meter readings**, monthly billing information, **six-hour cumulative-register readings**, and Wave 1–3 household survey data.
- Household IDs, record-level merges, personally sensitive diagnostics, and intermediate files derived from restricted inputs.
- The full `SmartMeter_Exploratory_Data_Analysis_Clean_Draft.ipynb` and `SmartMeter_Survey_High_Consumption_Analysis_Clean_Draft.ipynb` drafts, pending a separate public-execution-safety review of file deletion and output rebuilding.
- Serialized `.joblib` / `.pkl` models, unreviewed binary artifacts, application caches, and full internal output directories.
- Credentials, secret configuration, and unreviewed third-party files. Some notebook source still contains local-path references, so the final source-level publication audit remains necessary.

The principal research source is the **LIRNEasia Sri Lankan Residential Electricity Consumption Dataset**. Naming that source is not a grant of access or redistribution rights. Readers wishing to reproduce the analysis must establish lawful access to the required source datasets and applicable use permissions independently.

## Research workflow and dependencies

The thesis describes the following general analytical sequence:

1. Quality-control and eligibility filtering of 15-minute household electricity readings for May–November 2024.
2. Derivation of six household-level electricity-use features and selection of four K-Means behavioral archetypes.
3. Internal validity, seed/bootstrap, temporal and alternative-method checks.
4. Cross-source interpretation using monthly billing, six-hour readings and surveys.
5. Separate, supplementary models for billing-to-archetype label transfer, survey-based high-consumption profiling, and estimated operational electricity-related CO₂.
6. Streamlit research interfaces and result visualization.

Some notebooks expect locally generated upstream feature files, intermediate data products, and private inputs. **The filenames in the public draft do not establish a tested, one-command execution order.** Do not interpret notebook numbering, file ordering, or displayed figures as proof of fully reproducible end-to-end execution.

## Environment status

The saved model documentation records portions of the original training environment. For example, `models/PHASE3_final_model_metadata.json` reports Python 3.11.8 and scikit-learn 1.4.2 for the survey high-consumption model. These are **artifact-specific provenance details**, not a confirmed compatible environment for every notebook and both Streamlit applications.

There is currently **no verified, repository-wide dependency lockfile or clean-environment install procedure** for this draft. Do not assume that a new environment can run the code from the listed metadata alone. Installation instructions and exact launch commands should be added only after the dependency set and all relevant application routes have been tested.

## Verification levels

| Review activity | Status for this draft |
| --- | --- |
| Notebook source curation | Eight output-cleared drafts plus two privacy-revised notebooks; two further notebook drafts withheld pending execution-safety review |
| Stored outputs removed from all 10 included notebook drafts | Checked during notebook preparation |
| Notebook Python syntax / application Python syntax | Checked statically; does not establish runtime correctness |
| Main reported findings against thesis and selected result/metadata files | Compared for selected published metrics |
| Private-data requirements, household-level outputs and some file-writing/deletion risks | Identified; **complete publication safety and runtime audit still pending** |
| Installation in a fresh environment | **Not tested** |
| All notebooks successfully executed in dependency order | **Not tested** |
| Both apps launched and every page exercised with approved public artifacts | **Not tested** |
| Saved models independently loaded and predictions checked in an isolated environment | **Not tested** |
| Independent population / external validation | **Not conducted** |

An artifact's own metadata may report a historical saved-model smoke test. That report is **not** an independent test of the GitHub draft. Do not load untrusted `.joblib` or `.pkl` files: Python pickle-based deserialization can run code.

## Safe inspection guidance

You can read the `.ipynb` source as text or inspect it within a notebook editor without executing cells. Some existing code paths write, delete, or replace intermediate outputs and may display or export household identifiers. Do not run the notebooks or apps against restricted source data in a public, shared, or synchronized workspace without reviewing every input/output path and permission first.

If an authorized researcher later attempts independent reproduction, they should first create a **separate, private** working copy, document legitimate data access, pin an environment, run explicit security/privacy checks, review destructive operations, validate upstream dependencies, and compare regenerated **aggregate** metrics with the recorded research results. This is a future verification procedure, **not** a claim that these checks have been completed here.

## Interpretation limits

The main result is a sample-specific, descriptive four-cluster segmentation of **2,227 households**, not a causal or national household classifier. Billing-to-archetype labels come from clustering and are pseudo-labels. Survey high-consumption and carbon high-emission cutoffs are defined relative to their respective study populations. Operational electricity-related CO₂ was **estimated with a constant 0.71 kg CO₂/kWh factor**; it is not full household carbon-footprint measurement or future emissions forecasting.

See the [README](README.md) and model cards under [`models/`](models/) for metric-specific evaluation populations, limitations, and supported claims.

## Before a public release

The draft still requires a complete check for sensitive data or identifiers embedded in **all remaining** code/metadata, absolute or private file paths, destructive operations, dependency and licensing terms, and any non-public intermediate outputs. The case-specific sections removed from two notebooks do not establish that the rest of the repository is safe to publish. The inclusion of `.gitignore` reduces accidental staging but **does not make unreviewed files safe to publish**.
