
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Residential Electricity Research Insights",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
SEARCH_ROOTS = []
for candidate in (APP_DIR, APP_DIR.parent):
    if candidate not in SEARCH_ROOTS:
        SEARCH_ROOTS.append(candidate)


# ============================================================
# EXACT NOTEBOOK OUTPUTS USED BY THIS DASHBOARD
# ============================================================

FILES = {
    "clusters": [
        "clean_fm_clusters.pkl",
        "clean_fm_clusters.csv",
    ],
    "transition_summary": [
        "tables_outputs/authoritative_transition_summary.csv",
        "authoritative_transition_summary.csv",
    ],
    "billing_percentiles": [
        "monthly_billing_outputs/monthly_billing_golden_percentiles_by_cluster.csv",
        "monthly_billing_golden_percentiles_by_cluster.csv",
    ],
    "survey_tests": [
        "survey_outputs/survey_final_tests_ROBUST_FDR_corrected.csv",
        "survey_final_tests_ROBUST_FDR_corrected.csv",
    ],
    "high_use_performance": [
        "model_outputs/phase3_final/PHASE3_pooled_oof_performance.csv",
        "PHASE3_pooled_oof_performance.csv",
    ],
    "carbon_archetype": [
        "carbon_sixhour_xai_outputs/15_archetype_carbon_summary.csv",
        "15_archetype_carbon_summary.csv",
    ],
    "carbon_regression": [
        "carbon_sixhour_xai_outputs/21_primary_continuous_emission_regression_heldout_test_metrics.csv",
        "21_primary_continuous_emission_regression_heldout_test_metrics.csv",
    ],
    "carbon_regression_importance": [
        "carbon_sixhour_xai_outputs/24_regression_heldout_permutation_importance.csv",
        "24_regression_heldout_permutation_importance.csv",
    ],
    "emission_factor_audit": [
        "carbon_sixhour_xai_outputs/12_emission_factor_audit.csv",
        "12_emission_factor_audit.csv",
    ],
}

IMAGES = {
    # Final EDA notebook outputs
    "eda_band_households": [
        "complete_eda_safe_outputs/figures/12_unit_band_household_share.png",
        "12_unit_band_household_share.png",
    ],
    "eda_band_energy": [
        "complete_eda_safe_outputs/figures/13_unit_band_energy_contribution.png",
        "13_unit_band_energy_contribution.png",
    ],
    "eda_load_shape": [
        "complete_eda_safe_outputs/figures/73_smart15_hourly_load_shape.png",
        "73_smart15_hourly_load_shape.png",
    ],

    # Final master clustering outputs
    "load_profiles": [
        "figures/fig_load_profiles_color.png",
        "fig_load_profiles_bw.png",
    ],
    "feature_heatmap": [
        "figures/cluster_feature_heatmap.png",
        "cluster_feature_heatmap.png",
    ],

    # K-selection notebook output
    "k_selection": [
        "K4_justification_dashboard.png",
    ],

    # Final survey/high-consumption notebook output
    "high_use_importance": [
        "model_outputs/phase3_final/figures/Figure_24_FINAL_cross_validated_permutation_importance.png",
        "Figure_24_FINAL_cross_validated_permutation_importance.png",
    ],
}


# ============================================================
# FILE RESOLUTION
# ============================================================

def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_file(candidates: list[str]) -> Path | None:
    # 1) Exact expected relative locations first.
    for relative in candidates:
        for root in SEARCH_ROOTS:
            path = root / relative
            if path.is_file():
                return path

    # 2) Fallback by basename. This keeps the app robust if an output folder
    #    was copied one level up/down without changing the notebook outputs.
    matches: list[Path] = []
    for relative in candidates:
        basename = Path(relative).name
        for root in SEARCH_ROOTS:
            try:
                matches.extend(path for path in root.rglob(basename) if path.is_file())
            except Exception:
                pass

    matches = _dedupe_paths(matches)
    if not matches:
        return None

    # Prefer shallower paths, then newer copies.
    matches.sort(
        key=lambda p: (
            len(p.parts),
            -p.stat().st_mtime if p.exists() else 0,
        )
    )
    return matches[0]


def latest_notebook(pattern: str) -> Path | None:
    matches: list[Path] = []
    for root in SEARCH_ROOTS:
        try:
            matches.extend(path for path in root.rglob(pattern) if path.is_file())
        except Exception:
            pass

    matches = _dedupe_paths(matches)
    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


# ============================================================
# LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def read_table(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        value = pd.read_pickle(path)
        if isinstance(value, pd.Series):
            return value.to_frame()
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{path.name} does not contain a pandas DataFrame.")
        return value
    return pd.read_csv(path)


def load_table(key: str) -> pd.DataFrame | None:
    path = resolve_file(FILES[key])
    if path is None:
        return None
    try:
        return read_table(str(path))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def notebook_output_text(path_text: str) -> str:
    """Read only executed notebook outputs, not code/source text."""
    notebook = json.loads(Path(path_text).read_text(encoding="utf-8"))
    parts: list[str] = []

    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text_value = output.get("text")
            if isinstance(text_value, list):
                parts.extend(str(x) for x in text_value)
            elif isinstance(text_value, str):
                parts.append(text_value)

            data = output.get("data", {})
            if isinstance(data, dict):
                plain = data.get("text/plain")
                if isinstance(plain, list):
                    parts.extend(str(x) for x in plain)
                elif isinstance(plain, str):
                    parts.append(plain)

    return "\n".join(parts)


def get_notebook_text(pattern: str) -> str:
    path = latest_notebook(pattern)
    if path is None:
        return ""
    try:
        return notebook_output_text(str(path))
    except Exception:
        return ""


# ============================================================
# SMALL DISPLAY HELPERS
# ============================================================

def show_image(key: str, caption: str | None = None) -> bool:
    path = resolve_file(IMAGES[key])
    if path is None:
        return False
    st.image(str(path), use_container_width=True)
    if caption:
        st.caption(caption)
    return True


def as_bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def first_column_containing(
    columns: list[str],
    required_tokens: tuple[str, ...],
    forbidden_tokens: tuple[str, ...] = (),
) -> str | None:
    for column in columns:
        name = str(column).lower()
        if all(token in name for token in required_tokens) and not any(
            token in name for token in forbidden_tokens
        ):
            return column
    return None


def numeric_value(row: pd.Series, candidates: list[str]) -> float | None:
    for candidate in candidates:
        if candidate in row.index:
            value = pd.to_numeric(pd.Series([row[candidate]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)

    lower_index = {str(c).lower(): c for c in row.index}
    for candidate in candidates:
        actual = lower_index.get(candidate.lower())
        if actual is not None:
            value = pd.to_numeric(pd.Series([row[actual]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def metric_or_dash(column, label: str, value: float | None, fmt: str = ".3f", suffix: str = ""):
    if value is None:
        column.metric(label, "—")
    else:
        column.metric(label, f"{value:{fmt}}{suffix}")


def section_title(title: str, subtitle: str | None = None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


# ============================================================
# CORE DATA-BACKED SUMMARIES
# ============================================================

def cluster_summary() -> tuple[int | None, pd.DataFrame | None]:
    clusters = load_table("clusters")
    if clusters is None or clusters.empty:
        return None, None

    total = int(len(clusters))

    name_col = first_existing(
        list(clusters.columns),
        ["cluster_name", "archetype", "Archetype"],
    )
    if name_col is None:
        return total, None

    counts = (
        clusters[name_col]
        .astype(str)
        .value_counts()
        .rename_axis("Archetype")
        .reset_index(name="Households")
    )
    counts["Share (%)"] = counts["Households"] / counts["Households"].sum() * 100
    counts["Share (%)"] = counts["Share (%)"].round(1)

    order = [
        "Peak-Exposed",
        "Typical Household",
        "Heavy Continuous",
        "Daytime-Active",
    ]
    counts["_order"] = counts["Archetype"].map(
        {name: i for i, name in enumerate(order)}
    ).fillna(99)
    counts = counts.sort_values(["_order", "Archetype"]).drop(columns="_order")

    return total, counts


def persistence_summary() -> tuple[float | None, int | None]:
    frame = load_table("transition_summary")
    if frame is None or frame.empty:
        return None, None

    row = frame.iloc[0]
    persistence = numeric_value(
        row,
        ["same_archetype_persistence_percent"],
    )
    transitions = numeric_value(
        row,
        ["valid_consecutive_transitions"],
    )
    return persistence, int(transitions) if transitions is not None else None


def clustering_validation_metrics():
    text = get_notebook_text("Master_Pipeline*.ipynb")

    seed_mean = seed_min = bootstrap_mean = silhouette = None

    seed_match = re.search(
        r"SEED STABILITY.*?Mean ARI:\s*([0-9.]+)\s*\|\s*Min ARI:\s*([0-9.]+)",
        text,
        flags=re.S | re.I,
    )
    if seed_match:
        seed_mean = float(seed_match.group(1))
        seed_min = float(seed_match.group(2))

    boot_match = re.search(
        r"BOOTSTRAP ROBUSTNESS.*?Mean ARI:\s*([0-9.]+)",
        text,
        flags=re.S | re.I,
    )
    if boot_match:
        bootstrap_mean = float(boot_match.group(1))

    sil_match = re.search(
        r"Overall K=4 Silhouette:\s*([0-9.]+)",
        text,
        flags=re.S | re.I,
    )
    if sil_match:
        silhouette = float(sil_match.group(1))

    return seed_mean, seed_min, bootstrap_mean, silhouette


def billing_global_test():
    text = get_notebook_text("monthly_validation*.ipynb")
    match = re.search(
        r"Golden-window billing household median.*?"
        r"Kruskal-Wallis H\s*=\s*([0-9.]+)\s*,\s*p\s*=\s*([0-9.eE+-]+)",
        text,
        flags=re.S | re.I,
    )
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def high_use_selected_row() -> pd.Series | None:
    frame = load_table("high_use_performance")
    if frame is None or frame.empty:
        return None

    # Prefer the model's selected threshold row rather than the default 0.50 row.
    for column in frame.columns:
        if frame[column].dtype == object:
            mask = (
                frame[column]
                .astype(str)
                .str.contains("training-selected|selected threshold", case=False, regex=True, na=False)
            )
            if mask.any():
                return frame.loc[mask].iloc[0]

    return frame.iloc[0]


def carbon_regression_metrics() -> tuple[float | None, float | None]:
    frame = load_table("carbon_regression")
    if frame is None or frame.empty:
        return None, None

    row = frame.iloc[0]
    r2 = numeric_value(
        row,
        ["r2", "r_squared", "r2_score", "heldout_r2", "test_r2"],
    )
    mae = numeric_value(
        row,
        ["mae", "mean_absolute_error", "heldout_mae", "test_mae"],
    )

    # Generic fallback by column-name tokens.
    if r2 is None:
        col = first_column_containing(list(frame.columns), ("r2",))
        if col is not None:
            r2 = numeric_value(row, [col])

    if mae is None:
        col = first_column_containing(
            list(frame.columns),
            ("mae",),
            ("med", "median"),
        )
        if col is not None:
            mae = numeric_value(row, [col])

    return r2, mae


def emission_factor_value() -> float | None:
    frame = load_table("emission_factor_audit")
    if frame is None or frame.empty:
        return None

    # Find a likely factor column.
    candidates = [
        c for c in frame.columns
        if "factor" in str(c).lower() and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not candidates:
        candidates = [
            c for c in frame.columns
            if pd.to_numeric(frame[c], errors="coerce").notna().any()
        ]

    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            value = float(values.iloc[0])
            # Plausibility guard for kg CO2 / kWh.
            if 0 < value < 5:
                return value
    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Research Insights")
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "EDA",
        "Behavioural archetypes",
        "Validation",
        "Household context",
        "Carbon & XAI",
    ],
)

st.sidebar.caption(
    "Final thesis outputs only. Interactive household tools are kept in the separate Models & Profilers app."
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    section_title(
        "Residential Electricity Research Insights",
        "Final thesis findings from the locked analysis outputs.",
    )

    total, counts = cluster_summary()
    persistence, transitions = persistence_summary()

    c1, c2, c3 = st.columns(3)
    c1.metric("Final clustered households", f"{total:,}" if total is not None else "—")
    c2.metric("Behavioural archetypes", f"{len(counts):,}" if counts is not None else "—")
    c3.metric(
        "Consecutive-month persistence",
        f"{persistence:.2f}%" if persistence is not None else "—",
    )

    if transitions is not None:
        st.caption(
            f"Persistence is based on {transitions:,} valid consecutive household-month transitions."
        )

    if counts is not None:
        st.subheader("Final archetype population")
        st.dataframe(
            counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Share (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    st.info(
        "The archetypes are sample-specific behavioural electricity-use groups, not national, tariff, welfare, poverty or efficiency categories."
    )


# ============================================================
# EDA
# ============================================================

elif page == "EDA":
    section_title(
        "Exploratory Data Analysis",
        "Only the headline descriptive outputs used to motivate the behavioural analysis.",
    )

    st.subheader("Monthly-consumption concentration")
    left, right = st.columns(2)

    with left:
        if not show_image("eda_band_households"):
            st.caption("Household-share EDA figure was not found.")

    with right:
        if not show_image("eda_band_energy"):
            st.caption("Energy-contribution EDA figure was not found.")

    st.subheader("Overall 15-minute load shape")
    if not show_image(
        "eda_load_shape",
        "Household-normalized golden-window load shape from the final EDA notebook.",
    ):
        st.caption("15-minute load-shape figure was not found.")


# ============================================================
# BEHAVIOURAL ARCHETYPES
# ============================================================

elif page == "Behavioural archetypes":
    section_title(
        "Behavioural Archetypes",
        "Final K-means cluster profiles from the six-feature household representation.",
    )

    st.subheader("Normalized 24-hour archetype profiles")
    if not show_image(
        "load_profiles",
        "Profiles are normalized within archetype; the figure compares temporal shape rather than absolute consumption.",
    ):
        st.caption("Final load-profile figure was not found.")

    st.subheader("Behavioural-feature comparison")
    if not show_image("feature_heatmap"):
        st.caption("Final feature-profile heatmap was not found.")


# ============================================================
# VALIDATION
# ============================================================

elif page == "Validation":
    section_title(
        "Validation",
        "The minimum evidence needed to defend the final clustering solution.",
    )

    seed_mean, seed_min, bootstrap_mean, silhouette = clustering_validation_metrics()
    persistence, transitions = persistence_summary()

    cols = st.columns(5)
    metric_or_dash(cols[0], "Seed mean ARI", seed_mean)
    metric_or_dash(cols[1], "Minimum seed ARI", seed_min)
    metric_or_dash(cols[2], "Bootstrap mean ARI", bootstrap_mean)
    metric_or_dash(cols[3], "Silhouette", silhouette)
    metric_or_dash(cols[4], "Monthly persistence", persistence, ".2f", "%")

    st.caption(
        "Seed results assess initialization reproducibility; bootstrap ARI assesses resampling robustness; "
        "silhouette = 0.251 indicates modest geometric separation with overlap."
        if silhouette is not None
        else
        "Seed results assess initialization reproducibility; bootstrap ARI assesses resampling robustness."
    )

    if transitions is not None:
        st.caption(f"Consecutive-month persistence uses {transitions:,} valid transitions.")

    st.subheader("Why K = 4 was retained")
    if not show_image("k_selection"):
        st.info(
            "K=4 was retained as an interpretation-led minimum sufficient solution; simpler numerical solutions existed, while higher K mainly subdivided existing behavioural patterns."
        )
    else:
        st.caption(
            "K=4 was retained as an interpretation-led minimum sufficient solution, not as a universal numerical optimum."
        )

    st.divider()
    st.subheader("Matched monthly-billing consistency")

    billing = load_table("billing_percentiles")
    if billing is not None and not billing.empty:
        display = billing.copy()

        rename = {}
        if "cluster" in display.columns:
            rename["cluster"] = "Archetype"
        if "cluster_name" in display.columns:
            rename["cluster_name"] = "Archetype"
        if "n" in display.columns:
            rename["n"] = "Households"
        if "p50_median" in display.columns:
            rename["p50_median"] = "Median monthly kWh"
        display = display.rename(columns=rename)

        wanted = [
            c for c in ["Archetype", "Households", "Median monthly kWh"]
            if c in display.columns
        ]
        if wanted:
            st.dataframe(
                display[wanted],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(display, use_container_width=True, hide_index=True)

        if "Households" in display.columns:
            valid_n = pd.to_numeric(display["Households"], errors="coerce").sum()
            if pd.notna(valid_n):
                st.caption(f"Valid golden-window household medians: {int(valid_n):,}.")

    billing_h, billing_p = billing_global_test()
    b1, b2 = st.columns(2)
    metric_or_dash(b1, "Kruskal-Wallis H", billing_h)
    if billing_p is None:
        b2.metric("Overall p-value", "—")
    elif billing_p < 0.001:
        b2.metric("Overall p-value", "< 0.001")
    else:
        b2.metric("Overall p-value", f"{billing_p:.3g}")

    st.caption(
        "Billing was used only after clustering as matched-source contextual validation; it does not provide ground-truth archetype labels."
    )

    st.divider()
    st.subheader("Wave 1 survey associations surviving FDR correction")

    survey = load_table("survey_tests")
    if survey is not None and not survey.empty:
        display = survey.copy()

        flag_col = first_existing(
            list(display.columns),
            ["significant_after_fdr", "fdr_significant", "significant_fdr"],
        )
        if flag_col is not None:
            display = display.loc[as_bool_mask(display[flag_col])].copy()

        rename_map = {
            "variable": "Variable",
            "Variable": "Variable",
            "test": "Test",
            "effect_size": "Effect size",
            "effect_size_name": "Effect-size measure",
            "p_fdr_bh": "FDR-adjusted p",
            "n": "N",
        }
        display = display.rename(
            columns={k: v for k, v in rename_map.items() if k in display.columns}
        )

        wanted = [
            c for c in [
                "Variable",
                "Test",
                "Effect size",
                "Effect-size measure",
                "FDR-adjusted p",
                "N",
            ]
            if c in display.columns
        ]
        st.dataframe(
            display[wanted] if wanted else display,
            use_container_width=True,
            hide_index=True,
        )
        st.caption("These post-clustering associations are observational and do not establish causation.")
    else:
        st.caption("Final FDR-corrected survey table was not found.")


# ============================================================
# HOUSEHOLD CONTEXT
# ============================================================

elif page == "Household context":
    section_title(
        "Household Context",
        "Final high-consumption profiling result from the survey-based supplementary model.",
    )

    st.subheader("Cross-validated permutation importance")
    if not show_image(
        "high_use_importance",
        "Permutation importance measures predictive dependence of the fitted model; it is not a causal effect.",
    ):
        st.caption("Final high-consumption permutation-importance figure was not found.")

    st.subheader("High-consumption profiling performance")
    row = high_use_selected_row()

    if row is None:
        st.caption("Final high-consumption performance table was not found.")
    else:
        roc_auc = numeric_value(row, ["roc_auc", "roc-auc"])
        pr_auc = numeric_value(row, ["pr_auc", "average_precision", "pr-auc"])
        bal_acc = numeric_value(row, ["balanced_accuracy", "balanced_acc"])
        f1 = numeric_value(row, ["f1", "f1_score"])

        cols = st.columns(4)
        metric_or_dash(cols[0], "ROC-AUC", roc_auc)
        metric_or_dash(cols[1], "PR-AUC", pr_auc)
        metric_or_dash(cols[2], "Balanced accuracy", bal_acc)
        metric_or_dash(cols[3], "F1", f1)

    st.caption(
        "The high-consumption target is relative to the study sample and is not a national household-consumption category."
    )


# ============================================================
# CARBON & XAI
# ============================================================

elif page == "Carbon & XAI":
    section_title(
        "Electricity-Related CO₂ and XAI",
        "Fixed-factor electricity-related operational CO₂ only; this is not a complete household carbon footprint.",
    )

    carbon = load_table("carbon_archetype")
    r2, mae = carbon_regression_metrics()
    factor = emission_factor_value()

    cohort = None
    if carbon is not None and not carbon.empty:
        count_col = first_existing(
            list(carbon.columns),
            ["n_households", "households", "eligible_households", "n"],
        )
        if count_col is None:
            count_col = first_column_containing(list(carbon.columns), ("household",))
        if count_col is not None:
            values = pd.to_numeric(carbon[count_col], errors="coerce")
            if values.notna().any():
                cohort = int(values.sum())

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Carbon-qualified households", f"{cohort:,}" if cohort is not None else "—")
    top2.metric(
        "Study emission factor",
        f"{factor:.2f} kg CO₂/kWh" if factor is not None else "—",
    )
    metric_or_dash(top3, "Held-out regression R²", r2)
    metric_or_dash(top4, "Held-out regression MAE", mae, ".1f", " kg/month")

    st.subheader("Archetype electricity-related CO₂")

    if carbon is not None and not carbon.empty:
        display = carbon.copy()

        archetype_col = first_existing(
            list(display.columns),
            ["cluster_name", "archetype", "cluster"],
        )
        count_col = first_existing(
            list(display.columns),
            ["n_households", "households", "eligible_households", "n"],
        )

        median_co2_col = None
        for column in display.columns:
            name = str(column).lower()
            if "median" in name and ("emission" in name or "co2" in name):
                median_co2_col = column
                break

        selected = []
        rename = {}
        if archetype_col is not None:
            selected.append(archetype_col)
            rename[archetype_col] = "Archetype"
        if count_col is not None and count_col not in selected:
            selected.append(count_col)
            rename[count_col] = "Households"
        if median_co2_col is not None and median_co2_col not in selected:
            selected.append(median_co2_col)
            rename[median_co2_col] = "Median electricity-related CO₂ (kg/month)"

        if selected:
            st.dataframe(
                display[selected].rename(columns=rename),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption("Final archetype carbon summary was not found.")

    st.subheader("Global model explanation")
    importance = load_table("carbon_regression_importance")

    if importance is not None and not importance.empty:
        feature_col = first_existing(
            list(importance.columns),
            ["feature", "Feature"],
        )
        importance_col = first_existing(
            list(importance.columns),
            [
                "permutation_importance_mean",
                "importance_mean",
                "mean_importance",
            ],
        )

        if feature_col is not None and importance_col is not None:
            view = importance[[feature_col, importance_col]].copy()
            view[importance_col] = pd.to_numeric(view[importance_col], errors="coerce")
            view = view.sort_values(importance_col, ascending=False).head(5)
            view = view.rename(
                columns={
                    feature_col: "Feature",
                    importance_col: "Held-out MAE increase after permutation",
                }
            )
            st.dataframe(view, use_container_width=True, hide_index=True)

            if not view.empty:
                st.success(
                    f"The strongest held-out regression predictor was {view.iloc[0]['Feature']}."
                )
        else:
            st.dataframe(importance, use_container_width=True, hide_index=True)
    else:
        st.caption("Final held-out regression permutation-importance table was not found.")

    st.caption(
        "Permutation importance describes fitted-model dependence. It does not establish a causal effect on electricity use or emissions."
    )


st.markdown("---")
st.caption(
    "Research dashboard: final analysis outputs only. Results are descriptive/model-based and should be interpreted within the study sample and methodological scope."
)
