# Data Sources, Access, and Privacy Boundaries

This document describes the evidence sources used in the thesis research on temporal electricity-use patterns in Sri Lankan households. **The repository contains selected aggregate results and figures, not the underlying household records.** This is a research portfolio, not a distributable dataset or a ready-to-run replication package.

## Primary and supporting data

| Source | Role in the research | Public-repository status |
| --- | --- | --- |
| LIRNEasia Sri Lankan Residential Electricity Consumption Dataset: 15-minute smart-meter load profiles | Main analysis source for household-level temporal features and K-Means behavioral segmentation | Raw records and household-level derived tables withheld |
| Monthly electricity-consumption/billing records | Supporting comparison of consumption magnitude, billing consistency, and supplementary billing-to-archetype transfer | Household-level records and model inputs withheld |
| Six-hour cumulative smart-meter register records | Cross-source consistency checks, reconstruction of electricity imports, and supplementary estimated operational CO₂ analysis | Raw and reconstructed household-level records withheld |
| Household surveys, including Wave 1 | Contextual comparison using household, dwelling, and appliance characteristics; supplementary high-consumption profiling | Household-level responses and matched linkage tables withheld |
| Historical weather, holiday, and electricity-tariff information | Contextual supplementary analyses, not inputs defining the four main behavioral archetypes | Only selected code descriptions are shared; source permissions and reproducibility are not established by this repository |

## Official dataset source and licence status

The primary dataset was downloaded from LIRNEasia’s official Kaggle listing:

- **Dataset:** [Sri Lankan Residential Electricity Consumption — LIRNEasia](https://www.kaggle.com/datasets/lirneasia/sri-lankan-residential-electricity-consumption)
- **Platform:** Kaggle; the dataset page showed **Version 2 (5.79 GB)** when checked on 15 October 2026.
- **Kaggle licence field:** **“Other (specified in description)”**. This is not a verified standard open-data licence. Specific permissions for reuse and redistribution have **not** been independently established for this repository.

The dataset is linked here for source attribution and independent access, **not rehosted**. Public download availability does not by itself grant unrestricted redistribution rights or remove the need to protect household-level information. Anyone accessing the source should review the publisher’s terms and applicable research/privacy requirements.

The data sources above have different coverage periods, sampling conditions, and linkage availability. **Their sample sizes must not be treated as interchangeable.**

## Primary analysis population and derived features

The main clustering analysis retained **2,227 households** for the **May–November 2024** analysis window, following the project's eligibility and data-quality checks. The six standardized clustering features were:

1. Evening peak ratio
2. Night baseload ratio
3. Load uniformity
4. Load factor
5. Leisure-usage ratio
6. Log daily electricity consumption

K-Means with **K = 4** produced the four sample-specific electricity-use archetypes: Peak-Exposed, Typical Household, Heavy Continuous, and Daytime-Active. See the [README](README.md) for the validated cluster sizes and evaluation results.

## Different samples and targets in supplementary studies

- **Billing-to-archetype transfer:** 1,146 billing-matched households; the supervised target consists of the archetype labels assigned by the main clustering workflow. It does **not** independently observe or recover time-of-use behavior.
- **Survey-based high-consumption profiling:** 1,413 households; the target is a study-relative top-one-third consumption definition, **not** an official national or tariff category.
- **Electricity-related operational CO₂ analysis:** 1,656 eligible households; calculated using electricity imports and a **fixed 0.71 kg CO₂/kWh factor**. This is a study-specific estimate of electricity-related operational emissions, **not** a complete household carbon footprint or a time-varying electricity-grid emission estimate.

The samples, outcomes, training/evaluation designs, and periods vary. Do not compare their predictive scores as though they were measurements of the same task. Consult the corresponding documents in [`models/`](models/) for model-level scope and limitations.

## What is included here

- Selected **aggregate** evaluation tables in [`results/`](results/).
- Two group-level figures in [`figures/`](figures/).
- Source-code notebook and application **drafts** for methodological inspection, with notebook execution outputs removed.
- Model documentation and non-executable metadata in [`models/`](models/).

The source code is **not evidence that the underlying datasets are public**. Certain notebook cells still describe or generate household-level analyses. Some cells may write intermediate output files. None of the notebooks or applications has been independently rerun as a complete, privacy-safe workflow from this repository alone.

## Access, confidentiality, and reuse

The repository intentionally does **not** provide:

- Original household electricity readings, monthly billing tables, six-hour register records, or survey responses.
- Household identifiers, matching keys, household-level predictions, or linked administrative/survey records.
- Serialized model files (`.pkl`, `.joblib`) or private intermediate feature datasets.
- Authentication credentials or permissions for the source datasets.

**Do not reconstruct, distribute, or upload these restricted records from the code examples.** Anyone who wishes to reproduce the research must independently establish legitimate data access, applicable permissions, privacy/ethics requirements, and the original processing environment. Mentioning LIRNEasia as a data source does not imply that this repository grants access or redistribution rights.

No licence for the underlying third-party data is asserted here. No open-source software licence has been assigned in this draft. The absence of a licence does not make the code or data freely reusable.

For execution limits, missing dependencies, and withheld notebook sections, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). For research claims and their boundaries, see the [README](README.md).
