from __future__ import annotations

# ========================================================================
# PUBLIC REPOSITORY DRAFT — NOT AN APPROVED DEPLOYMENT
# ========================================================================
# This script is a source-only research prototype. It is not self-contained.
# The original app expects several fitted model and private study-data files.
# Do not publish raw household files, reference CSVs, or unreviewed pickle/
# joblib artifacts. Loading those artifacts is trusted-local-only behavior.
# Review missing dependencies, user-data/privacy handling, and independent
# execution before any release or hosting. No model runs were tested here.
# ========================================================================

from pathlib import Path
from typing import Any, Iterable
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


# ============================================================
# FINAL DEMAND-SIDE THESIS DASHBOARD
# ============================================================
#
# Core user-facing tools:
# 1. 15-minute smart-meter data -> locked KMeans behavioural archetype
# 2. Monthly billing -> archetype label-transfer classifier
# 3. 15-minute smart-meter data -> direct electricity/CO2 accounting + carbon regression + high-emission classifier
# 4. Wave-1-style survey inputs -> high-consumption profile + reference benchmark
#
# Together these routes expose five fitted research models:
# KMeans, archetype label-transfer classifier, high-consumption classifier,
# carbon regression, and high-emission classifier. The monthly-consumption
# benchmark and similar-household reference are descriptive tools, not ML models.
#
# The carbon feature construction below follows the locked master notebook:
# - valid day: >= 90 valid 15-minute intervals
# - household eligibility: >= 30 valid days
# - Peak: 18:30 <= time < 22:30
# - Off-Peak: time >= 22:30 or time < 05:30
# - Shannon entropy on the 96-position mean profile
# - load factor = mean interval kWh / max interval kWh
# - LUR = log1p(mean non-working-day kWh / mean working-day kWh)
# - raw activity gate: log_daily_kwh > 0.05
# - no thesis-wide percentile clipping before the carbon model
#
# The 2024 May-November public-holiday dates are locked exactly to the
# master notebook. For other years, the "holidays" package resolves
# Sri Lankan public holidays from the uploaded timestamps.
# ============================================================


APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_ROWS = 500_000
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
PHYSICAL_LIMIT_KWH_15MIN = 6.0
MIN_VALID_INTERVALS_PER_DAY = 90
MIN_VALID_DAYS = 30
GRID_FACTOR_KG_CO2_PER_KWH = 0.71

CARBON_FEATURES = [
    "evening_peak_ratio",
    "night_baseload_ratio",
    "load_uniformity",
    "load_factor",
    "leisure_usage_ratio",
]

CLUSTER_FEATURES = CARBON_FEATURES + ["log_daily_kwh"]

LOCKED_2024_PUBLIC_HOLIDAYS = pd.to_datetime([
    "2024-05-01",
    "2024-05-23",
    "2024-05-24",
    "2024-06-21",
    "2024-07-20",
    "2024-08-19",
    "2024-09-16",
    "2024-09-17",
    "2024-10-17",
    "2024-10-31",
    "2024-11-15",
]).normalize()

ARCHETYPE_ORDER = [
    "Peak-Exposed",
    "Typical Household",
    "Heavy Continuous",
    "Daytime-Active",
]

ARCHETYPE_COUNTS = {
    "Peak-Exposed": 145,
    "Typical Household": 1426,
    "Heavy Continuous": 515,
    "Daytime-Active": 141,
}

ARCHETYPE_DESCRIPTIONS = {
    "Peak-Exposed": (
        "Electricity use is comparatively concentrated during the evening peak period."
    ),
    "Typical Household": (
        "The largest and comparatively balanced pattern in the study sample."
    ),
    "Heavy Continuous": (
        "Higher sustained demand with the strongest night-baseload ratio and higher load factor."
    ),
    "Daytime-Active": (
        "Electricity use is comparatively more daytime-oriented with very low night share."
    ),
}

ARCHETYPE_PEAK_TIMES = {
    "Peak-Exposed": "19:45",
    "Typical Household": "20:15",
    "Heavy Continuous": "21:45",
    "Daytime-Active": "11:45",
}

# Sri Lanka-specific household demand-side guidance. These recommendations are
# intentionally conservative and are grounded in official SLSEA/PUCSL guidance.
# They are general actions, not guaranteed bill-saving estimates.
ARCHETYPE_RECOMMENDATIONS = {
    "Peak-Exposed": [
        "Prioritise moving flexible high-power activities such as ironing, laundry and electric water heating (if used) away from the 18:30–22:30 evening peak.",
        "If appliances support timers or smart scheduling, schedule flexible loads for daytime or off-peak hours instead of stacking them in the evening.",
        "Keep essential refrigeration running; reduce evening demand through flexible loads rather than switching the refrigerator off.",
        "Use efficient LED lighting and energy-labelled fans/appliances when replacing equipment.",
    ],
    "Heavy Continuous": [
        "Start with continuously operating loads: check refrigerator door seals, ventilation and settings, and consider replacing very old inefficient refrigerators when practical.",
        "For air conditioning, use it only when needed, set around 26°C, keep filters clean and keep doors/windows closed while cooling.",
        "Use efficient fans and lighting; Sri Lanka has appliance energy-labelling/efficiency programmes for common household equipment.",
        "Shift flexible laundry and ironing away from 18:30–22:30 so high all-day demand is not compounded by evening-peak use.",
    ],
    "Typical Household": [
        "Maintain efficient everyday habits: LEDs, efficient labelled appliances, sensible refrigerator settings and regular maintenance.",
        "Run washing machines with useful/full loads and avoid energy-intensive drying when sun drying is practical.",
        "Batch ironing rather than repeatedly reheating the iron, and move flexible chores outside 18:30–22:30 when convenient.",
        "If air conditioning is used, use fans first when comfortable and keep AC near 26°C with clean filters.",
    ],
    "Daytime-Active": [
        "Your pattern already avoids much of the evening peak; keep flexible chores in daytime rather than moving them into 18:30–22:30.",
        "Use daylight where practical and choose efficient LEDs for rooms that still need daytime lighting.",
        "For daytime cooling, use fans first when comfortable; if AC is needed, keep it near 26°C and maintain clean filters.",
        "When buying or replacing appliances, use Sri Lankan energy labels/efficiency information and choose the right appliance size for the household.",
    ],
}

SLSEA_HOME_URL = "https://www.energy.gov.lk/index.php/en/knowledge/resources/your-home"
SLSEA_LABEL_URL = "https://www.energy.gov.lk/en/energy-management/labelling-programme"
PUCSL_TARIFF_URL = "https://www.pucsl.gov.lk/electricity/tariff/electricity-tariff-and-charges/"

PUCSL_BAND_ORDER = [
    "0–30 kWh",
    "31–60 kWh",
    "61–90 kWh",
    "91–120 kWh",
    "121–180 kWh",
    ">180 kWh",
]

# Exact approved artefact locations. The application never recursively searches
# arbitrary parent directories, which prevents a stale model from being loaded
# silently. Same-folder fallbacks make the final app easy to package for a viva.
PROJECT_ROOT = APP_DIR

ARCHETYPE_MODEL_CANDIDATES = [
    PROJECT_ROOT
    / "archetype_classifier_final_outputs"
    / "streamlit_archetype_app"
    / "archetype_classifier_bundle.joblib",
    PROJECT_ROOT
    / "archetype_classifier_final_outputs"
    / "final_research_prototype"
    / "archetype_classifier_bundle.joblib",
    PROJECT_ROOT / "archetype_classifier_bundle.joblib",
]

CARBON_MODEL_CANDIDATES = [
    PROJECT_ROOT
    / "carbon_sixhour_xai_outputs"
    / "FINAL_carbon_xai_model_bundle.joblib",
    PROJECT_ROOT / "FINAL_carbon_xai_model_bundle.joblib",
]

HIGH_USE_MODEL_CANDIDATES = [
    PROJECT_ROOT
    / "model_outputs"
    / "phase3_final"
    / "PHASE3_final_high_consumption_model.joblib",
    PROJECT_ROOT / "PHASE3_final_high_consumption_model.joblib",
]

SURVEY_REFERENCE_CANDIDATES = [
    PROJECT_ROOT
    / "survey_analysis_main_outputs"
    / "ANALYSIS2_consumption_group_profile_dataset.csv",
]
KMEANS_RAW_FEATURES_CANDIDATES = [
    PROJECT_ROOT / "raw_active_features_unclipped.pkl",
]

KMEANS_LOCKED_LABELS_CANDIDATES = [
    PROJECT_ROOT / "clean_fm_clusters.pkl",
]

SURVEY_REFERENCE_PERIOD_START = pd.Period("2023-04", freq="M")
SURVEY_REFERENCE_PERIOD_END = pd.Period("2024-03", freq="M")
SURVEY_REFERENCE_MONTHS = 12


def _first_existing_path(candidates: Iterable[Path]) -> Path | None:
    """Return the first existing approved path without recursive discovery."""
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    return None


# Resolve KMeans source paths once. If missing, keep the canonical expected path
# so error messages remain concrete and helpful.
KMEANS_RAW_FEATURES_PATH = (
    _first_existing_path(KMEANS_RAW_FEATURES_CANDIDATES)
    or KMEANS_RAW_FEATURES_CANDIDATES[0]
)
KMEANS_LOCKED_LABELS_PATH = (
    _first_existing_path(KMEANS_LOCKED_LABELS_CANDIDATES)
    or KMEANS_LOCKED_LABELS_CANDIDATES[0]
)

# ============================================================
# SHARED UI / FILE HELPERS
# ============================================================


def _configure_page() -> None:
    st.set_page_config(
        page_title="Residential Electricity Analysis",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Presentation layer only. These rules deliberately force a light,
    # high-contrast research UI so browser/Streamlit dark-mode preferences
    # cannot produce white-on-white or dark uploader components.
    st.markdown(
        """
        <style>
        /* ------------------------------------------------------------
           FIXED LIGHT RESEARCH THEME
           ------------------------------------------------------------ */
        :root {
            color-scheme: light !important;
        }

        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #FFFFFF !important;
            color: #172033 !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
        }

        header[data-testid="stHeader"] {
            background: rgba(255,255,255,0.98) !important;
            border-bottom: 1px solid #E7EAF0 !important;
        }

        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] span,
        [data-testid="stToolbar"] p {
            color: #344054 !important;
        }

        .block-container {
            max-width: 1160px !important;
            padding-top: 1.75rem !important;
            padding-bottom: 4rem !important;
        }

        /* ------------------------------------------------------------
           TYPOGRAPHY / CONTENT CONTRAST
           ------------------------------------------------------------ */
        .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp h5, .stApp h6 {
            color: #172033 !important;
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMarkdownContainer"] {
            color: #475467 !important;
        }

        a {
            color: #245A8D !important;
            text-decoration-thickness: 1px !important;
            text-underline-offset: 2px !important;
        }

        /* ------------------------------------------------------------
           APPLICATION HEADER
           ------------------------------------------------------------ */
        .app-header {
            margin: 0 0 2rem 0 !important;
            padding: 0 0 1.35rem 0 !important;
            border-bottom: 1px solid #E7EAF0 !important;
        }
        .app-kicker {
            margin: 0 0 .45rem 0 !important;
            color: #245A8D !important;
            font-size: .70rem !important;
            font-weight: 700 !important;
            letter-spacing: .105em !important;
            text-transform: uppercase !important;
        }
        .app-title {
            margin: 0 !important;
            color: #172033 !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            letter-spacing: -.025em !important;
            line-height: 1.14 !important;
        }
        .app-subtitle {
            margin: .55rem 0 0 0 !important;
            max-width: 760px !important;
            color: #667085 !important;
            font-size: .96rem !important;
            line-height: 1.55 !important;
        }

        /* ------------------------------------------------------------
           TOOL HEADINGS
           ------------------------------------------------------------ */
        .tool-head {
            margin: 0 0 1.45rem 0 !important;
        }
        .tool-label {
            margin: 0 0 .38rem 0 !important;
            color: #245A8D !important;
            font-size: .68rem !important;
            font-weight: 700 !important;
            letter-spacing: .105em !important;
            text-transform: uppercase !important;
        }
        .tool-title {
            margin: 0 !important;
            color: #172033 !important;
            font-size: 1.52rem !important;
            font-weight: 700 !important;
            letter-spacing: -.018em !important;
            line-height: 1.22 !important;
        }
        .tool-copy {
            margin: .5rem 0 0 0 !important;
            max-width: 860px !important;
            color: #667085 !important;
            font-size: .94rem !important;
            line-height: 1.55 !important;
        }

        /* ------------------------------------------------------------
           SIDEBAR
           ------------------------------------------------------------ */
        [data-testid="stSidebar"] {
            background: #F7F9FC !important;
            border-right: 1px solid #E4E7EC !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: #F7F9FC !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.45rem !important;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #344054 !important;
        }

        .side-brand {
            margin-bottom: 1.25rem !important;
        }
        .side-brand-title {
            margin: 0 !important;
            color: #172033 !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }
        .side-brand-copy {
            margin: .22rem 0 0 0 !important;
            color: #667085 !important;
            font-size: .78rem !important;
            line-height: 1.35 !important;
        }

        .readiness-ok,
        .readiness-warn {
            display: inline-flex !important;
            align-items: center !important;
            gap: .42rem !important;
            padding: .38rem .58rem !important;
            margin: .15rem 0 .75rem 0 !important;
            border-radius: 7px !important;
            font-size: .78rem !important;
            font-weight: 650 !important;
        }
        .readiness-ok {
            background: #ECF7F0 !important;
            color: #256244 !important;
            border: 1px solid #D5ECDE !important;
        }
        .readiness-warn {
            background: #FFF7E8 !important;
            color: #8A5A16 !important;
            border: 1px solid #F4E3BE !important;
        }
        .status-dot {
            width: .46rem !important;
            height: .46rem !important;
            border-radius: 50% !important;
            background: currentColor !important;
            display: inline-block !important;
            flex: 0 0 auto !important;
        }

        /* Sidebar radio navigation */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: .20rem !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            border-radius: 8px !important;
            padding: .46rem .52rem !important;
            transition: background-color .12s ease, color .12s ease !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: #EEF3F8 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label p {
            color: #344054 !important;
            font-size: .90rem !important;
            font-weight: 520 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: #E9F0F7 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
            color: #1F4E79 !important;
            font-weight: 700 !important;
        }

        /* Sidebar expanders */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #E4E7EC !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
            color: #344054 !important;
        }

        /* ------------------------------------------------------------
           FORM LABELS / CHECKBOXES / INPUTS
           ------------------------------------------------------------ */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCheckbox"] p,
        [data-testid="stFileUploader"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label {
            color: #344054 !important;
            font-weight: 520 !important;
        }

        [data-testid="stCheckbox"] label span,
        [data-testid="stCheckbox"] label p {
            color: #344054 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {
            background: #FFFFFF !important;
            border-color: #D0D5DD !important;
            color: #172033 !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] input {
            color: #172033 !important;
            caret-color: #172033 !important;
        }
        div[data-baseweb="select"] span {
            color: #172033 !important;
        }

        /* ------------------------------------------------------------
           FILE UPLOADER — FORCE LIGHT COMPONENT
           ------------------------------------------------------------ */
        [data-testid="stFileUploader"] {
            color: #344054 !important;
        }
        [data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"] {
            background: #FFFFFF !important;
            border: 1px dashed #B8C1CC !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #475467 !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            background: #FFFFFF !important;
            color: #344054 !important;
            border: 1px solid #D0D5DD !important;
            border-radius: 7px !important;
            box-shadow: none !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            border-color: #245A8D !important;
            color: #1F4E79 !important;
            background: #F8FAFC !important;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            color: #667085 !important;
            fill: currentColor !important;
        }

        /* ------------------------------------------------------------
           FORMS / METRICS / TABLES
           ------------------------------------------------------------ */
        [data-testid="stForm"] {
            background: #FFFFFF !important;
            border: 1px solid #E4E7EC !important;
            border-radius: 10px !important;
            padding: 1rem 1rem .45rem 1rem !important;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #E4E7EC !important;
            border-radius: 10px !important;
            padding: .88rem 1rem !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, .025) !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #667085 !important;
            font-size: .80rem !important;
            font-weight: 560 !important;
        }
        [data-testid="stMetricValue"] {
            color: #172033 !important;
            font-weight: 700 !important;
        }

        [data-testid="stDataFrame"] {
            background: #FFFFFF !important;
            border: 1px solid #E4E7EC !important;
            border-radius: 9px !important;
            overflow: hidden !important;
        }

        [data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #E4E7EC !important;
            border-radius: 9px !important;
        }
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #344054 !important;
        }

        /* ------------------------------------------------------------
           BUTTONS
           ------------------------------------------------------------ */
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stDownloadButton"] button {
            min-height: 2.45rem !important;
            border-radius: 7px !important;
            font-weight: 620 !important;
            box-shadow: none !important;
            transition: background-color .12s ease, border-color .12s ease !important;
        }

        .stDownloadButton > button,
        [data-testid="stDownloadButton"] button {
            background: #FFFFFF !important;
            color: #344054 !important;
            border: 1px solid #D0D5DD !important;
        }
        .stDownloadButton > button:hover,
        [data-testid="stDownloadButton"] button:hover {
            background: #F8FAFC !important;
            color: #1F4E79 !important;
            border-color: #7EA2C4 !important;
        }

        .stButton > button[kind="primary"],
        button[kind="primary"] {
            background: #1F4E79 !important;
            color: #FFFFFF !important;
            border: 1px solid #1F4E79 !important;
        }
        .stButton > button[kind="primary"] p,
        button[kind="primary"] p {
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:hover,
        button[kind="primary"]:hover {
            background: #183D61 !important;
            border-color: #183D61 !important;
        }

        /* ------------------------------------------------------------
           ALERTS — retain semantic colours but enforce readable text
           ------------------------------------------------------------ */
        [data-testid="stAlert"] {
            border-radius: 8px !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {
            color: #344054 !important;
        }

        /* Plotly charts sit on the same white research surface. */
        [data-testid="stPlotlyChart"] {
            background: #FFFFFF !important;
            border-radius: 8px !important;
        }

        footer {
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-kicker">Thesis research application</div>
            <h1 class="app-title">Residential Electricity Analysis</h1>
            <p class="app-subtitle">
                Apply the final research models to smart-meter, monthly billing and
                household-survey inputs, with separate descriptive reference tools.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tool_header(label: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="tool-head">
            <div class="tool-label">{label}</div>
            <h2 class="tool-title">{title}</h2>
            <p class="tool-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )




@st.cache_resource
def _load_locked_kmeans_from_sources() -> dict[str, Any]:
    """Rebuild and verify the exact locked KMeans study segmentation."""
    if not KMEANS_RAW_FEATURES_PATH.is_file():
        raise FileNotFoundError(
            "Missing KMeans source file: raw_active_features_unclipped.pkl. "
            "Place it beside the app or in the thesis project root."
        )
    if not KMEANS_LOCKED_LABELS_PATH.is_file():
        raise FileNotFoundError(
            "Missing KMeans source file: clean_fm_clusters.pkl. "
            "Place it beside the app or in the thesis project root."
        )

    raw = pd.read_pickle(KMEANS_RAW_FEATURES_PATH)
    locked = pd.read_pickle(KMEANS_LOCKED_LABELS_PATH)

    if not isinstance(raw, pd.DataFrame) or not isinstance(locked, pd.DataFrame):
        raise TypeError("The locked KMeans source artefacts must contain pandas DataFrames.")

    if "household_ID" in raw.columns:
        raw = raw.set_index("household_ID")
    if "household_ID" in locked.columns:
        locked = locked.set_index("household_ID")

    raw.index = raw.index.astype(str).str.strip()
    locked.index = locked.index.astype(str).str.strip()

    if raw.index.duplicated().any() or locked.index.duplicated().any():
        raise ValueError("Duplicate household_ID values were found in KMeans source artefacts.")

    required_locked = set(CLUSTER_FEATURES + ["final_cluster", "cluster_name"])
    missing_locked = sorted(required_locked - set(locked.columns))
    missing_raw = sorted(set(CLUSTER_FEATURES) - set(raw.columns))
    if missing_raw:
        raise ValueError(
            "raw_active_features_unclipped.pkl is missing: " + ", ".join(missing_raw)
        )
    if missing_locked:
        raise ValueError(
            "clean_fm_clusters.pkl is missing: " + ", ".join(missing_locked)
        )

    active = raw.loc[raw.index.intersection(locked.index), CLUSTER_FEATURES].copy()
    active = active.apply(pd.to_numeric, errors="coerce")
    active = active.replace([np.inf, -np.inf], np.nan).dropna()
    active = active.loc[active["log_daily_kwh"] > 0.05].copy()

    if len(active) != 2227 or set(active.index) != set(locked.index):
        raise RuntimeError(
            "The supplied KMeans source files do not reproduce the locked "
            "2,227-household study population."
        )

    clip_bounds: dict[str, tuple[float, float]] = {}
    clipped = active.copy()
    for feature in CLUSTER_FEATURES:
        lower = float(active[feature].quantile(0.01))
        upper = float(active[feature].quantile(0.99))
        clip_bounds[feature] = (lower, upper)
        clipped[feature] = clipped[feature].clip(lower=lower, upper=upper)

    scaler = StandardScaler()
    X = scaler.fit_transform(clipped[CLUSTER_FEATURES])
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20,
    )
    rebuilt_labels = kmeans.fit_predict(X).astype(int)

    expected = locked.loc[clipped.index, "final_cluster"].astype(int).to_numpy()
    if not np.array_equal(rebuilt_labels, expected):
        mismatches = int(np.sum(rebuilt_labels != expected))
        raise RuntimeError(
            f"Locked KMeans verification failed: {mismatches} of 2,227 labels differ."
        )

    mapping_table = (
        locked.loc[clipped.index, ["final_cluster", "cluster_name"]]
        .drop_duplicates()
    )
    if mapping_table["final_cluster"].duplicated().any():
        raise RuntimeError("A KMeans cluster maps to more than one archetype name.")

    cluster_name_map = {
        int(row.final_cluster): str(row.cluster_name)
        for row in mapping_table.itertuples(index=False)
    }
    if set(cluster_name_map.values()) != set(ARCHETYPE_ORDER):
        raise RuntimeError(
            "Locked KMeans archetype names do not match the four final study archetypes."
        )

    # Distribution of training assignment distances, used only as an application-
    # time caution flag. It does not alter or reject the KMeans assignment.
    full_distances = kmeans.transform(X)
    assigned_distances = full_distances[np.arange(len(X)), rebuilt_labels]
    cluster_distance_p95 = {
        int(cluster_id): float(
            np.quantile(assigned_distances[rebuilt_labels == cluster_id], 0.95)
        )
        for cluster_id in sorted(np.unique(rebuilt_labels))
    }

    return {
        "feature_columns": list(CLUSTER_FEATURES),
        "clip_bounds": clip_bounds,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_name_map": cluster_name_map,
        "cluster_distance_p95": cluster_distance_p95,
    }



@st.cache_resource
def _load_joblib(path_text: str) -> Any:
    path = Path(path_text).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Model artefact not found: {path}")

    return joblib.load(path)


@st.cache_data
def _load_reference_csv(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


@st.cache_data
def _build_synthetic_smartmeter_demo(days: int = 35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic synthetic CSV examples that satisfy the input contract.

    The values are only for testing the web application and are never presented as
    observed research data.
    """
    timestamps = pd.date_range(
        "2024-08-01 00:00:00",
        periods=int(days) * 96,
        freq="15min",
    )
    minute = timestamps.hour * 60 + timestamps.minute
    hour = minute / 60.0
    weekend = (timestamps.dayofweek >= 5).astype(float)

    # Smooth positive demand pattern with an evening rise and a smaller daytime rise.
    base = 0.055 + 0.010 * np.cos(2 * np.pi * (hour - 2) / 24)
    evening = 0.085 * np.exp(-0.5 * ((hour - 20.0) / 1.6) ** 2)
    daytime = 0.025 * np.exp(-0.5 * ((hour - 12.5) / 2.5) ** 2)
    leisure = 0.010 * weekend
    consumption = np.clip(base + evening + daytime + leisure, 0.005, None)

    interval = pd.DataFrame({
        "household_ID": "DEMO_001",
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "consumption_kwh": np.round(consumption, 6),
    })

    cumulative = np.cumsum(consumption) + 1000.0
    register = pd.DataFrame({
        "household_ID": "DEMO_001",
        "obis": "LP",
        "date": timestamps.strftime("%Y-%m-%d"),
        "time": timestamps.strftime("%H:%M:%S"),
        "importkwh(kwh)": np.round(cumulative, 6),
        "exportkwh(kwh)": 0.0,
    })
    return interval, register

def _safe_positive_score(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_)

    positive = np.where(classes == 1)[0]
    if positive.size != 1:
        raise RuntimeError("Positive class 1 could not be identified.")

    return probabilities[:, int(positive[0])]


def _read_csv(upload: Any, label: str) -> pd.DataFrame:
    try:
        upload_size = int(getattr(upload, "size", 0) or 0)
    except Exception:
        upload_size = 0

    if upload_size and upload_size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{label} exceeds the {MAX_UPLOAD_BYTES / (1024**2):.0f} MB upload limit."
        )

    try:
        upload.seek(0)
    except Exception:
        pass

    data = pd.read_csv(upload, nrows=MAX_UPLOAD_ROWS + 1)

    if len(data) > MAX_UPLOAD_ROWS:
        raise ValueError(
            f"{label} exceeds the {MAX_UPLOAD_ROWS:,}-row dashboard limit."
        )

    return data


def _parse_local_naive_timestamp(series: pd.Series, label: str) -> pd.Series:
    """
    Parse timestamps under the dashboard's deployment contract:
    timestamps must represent Sri Lanka local clock time and must not carry
    timezone offsets. This prevents holiday/date shifts and tz-aware/naive
    comparison failures.
    """
    raw = series.astype(str).str.strip()
    tz_marker = raw.str.contains(
        r"(?:Z|[+-]\d{2}:?\d{2})\s*$",
        regex=True,
        na=False,
    )
    if tz_marker.any():
        examples = raw.loc[tz_marker].head(5).tolist()
        raise ValueError(
            f"{label} contains timezone-aware values. Upload Sri Lanka local "
            f"time without a timezone suffix/offset. Examples: {examples}"
        )

    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.isna().any():
        examples = raw.loc[parsed.isna()].head(10).tolist()
        raise ValueError(
            f"Invalid {label} values were found. Examples: {examples}"
        )

    # Defensive check in case pandas inferred a timezone despite no string marker.
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            raise ValueError(
                f"{label} must be timezone-naive Sri Lanka local time."
            )
    except AttributeError:
        raise ValueError(
            f"{label} could not be parsed into a single timezone-naive datetime format."
        )

    return parsed


def _validate_interval_cadence(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that interval-kWh uploads really represent the 15-minute grid used
    by the thesis. Missing 15-minute slots are allowed, but 5-minute/10-minute
    data and multiple readings within the same nominal quarter-hour are rejected.
    """
    minute_ok = data["meter_dt"].dt.minute.isin([0, 15, 30, 45])
    second_ok = data["meter_dt"].dt.second.eq(0) & data["meter_dt"].dt.microsecond.eq(0)

    if not (minute_ok & second_ok).all():
        bad = data.loc[
            ~(minute_ok & second_ok),
            ["household_ID", "meter_dt"],
        ].head(10).to_dict("records")
        raise ValueError(
            "Interval-kWh timestamps must lie on the 15-minute grid "
            "(HH:00, HH:15, HH:30 or HH:45 with zero seconds). "
            f"Examples outside the grid: {bad}"
        )

    nominal_slot = data["meter_dt"].dt.floor("15min")
    duplicate_slot = pd.DataFrame({
        "household_ID": data["household_ID"],
        "nominal_slot": nominal_slot,
    }).duplicated(["household_ID", "nominal_slot"], keep=False)

    if duplicate_slot.any():
        bad = data.loc[
            duplicate_slot,
            ["household_ID", "meter_dt"],
        ].head(10).to_dict("records")
        raise ValueError(
            "More than one interval-kWh reading was supplied for the same "
            f"15-minute slot. Examples: {bad}"
        )

    gaps = (
        data.groupby("household_ID", observed=True)["meter_dt"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    positive_gaps = gaps.dropna()
    if (positive_gaps <= 0).any():
        raise ValueError("Timestamps must increase within each household.")

    # Because every accepted timestamp lies on the quarter-hour grid, any
    # positive gap is an integer multiple of 15 minutes. Keep this explicit
    # audit to make the deployment contract transparent.
    off_grid_gap = ((positive_gaps / 15.0) % 1.0).abs() > 1e-9
    if off_grid_gap.any():
        raise ValueError(
            "Detected a timestamp gap that is not an integer multiple of 15 minutes."
        )

    return data



# ============================================================
# EXACT ARCHETYPE RUNTIME EXTRACTED FROM LATEST CLASSIFIER NOTEBOOK
# ============================================================

from typing import Tuple


MIN_VALID_MONTHS = 15
MAX_VALID_MONTHS = 25

CORE_FEATURES = [
    "billing_median_kwh",
    "billing_mean_kwh",
    "billing_std_kwh",
    "billing_iqr_kwh",
    "billing_cv",
    "billing_p10_kwh",
    "billing_p90_kwh",
    "billing_robust_range_kwh",
    "recent_3_month_median_kwh",
    "recent_6_month_median_kwh",
    "recent_vs_overall_log_ratio",
    "early_6_month_median_kwh",
    "late_6_month_median_kwh",
    "late_vs_early_log_ratio",
    "monthly_trend_slope_log1p",
    "median_abs_log_change",
    "near_zero_month_rate",
    "above_120_month_rate",
]

EXTENDED_ONLY_FEATURES = [
    "billing_min_kwh",
    "billing_max_kwh",
    "billing_range_kwh",
    "billing_log_std",
    "mean_abs_monthly_change_kwh",
    "maximum_monthly_increase_kwh",
    "maximum_monthly_decrease_kwh",
    "low_30_month_rate",
    "above_90_month_rate",
    "above_180_month_rate",
    "above_240_month_rate",
    "median_kwh_calendar_q1",
    "median_kwh_calendar_q2",
    "median_kwh_calendar_q3",
    "median_kwh_calendar_q4",
]

AUDIT_COLUMNS = [
    "months_observed",
    "first_month",
    "last_month",
    "observation_span_months",
]


def _normalise_household_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _calculate_month_span(
    start_series: pd.Series,
    end_series: pd.Series,
) -> pd.Series:
    return (
        (end_series.dt.year - start_series.dt.year) * 12
        + (end_series.dt.month - start_series.dt.month)
        + 1
    )


def _calculate_household_trend(group: pd.DataFrame) -> float:
    group = group.sort_values("month")

    if len(group) < 3:
        return np.nan

    month_index = (
        group["month"].dt.year.to_numpy() * 12
        + group["month"].dt.month.to_numpy()
    ).astype(float)

    month_index = month_index - month_index.min()

    if np.unique(month_index).size < 3:
        return np.nan

    log_consumption = np.log1p(
        group["consumption"].to_numpy(dtype=float)
    )

    return float(
        np.polyfit(month_index, log_consumption, 1)[0]
    )


def prepare_monthly_records(
    raw_monthly: pd.DataFrame,
    minimum_months: int = MIN_VALID_MONTHS,
    maximum_months: int = MAX_VALID_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate raw monthly billing rows and enforce the deployment
    coverage contract used by the research prototype.

    Required columns:
    - household_ID
    - month
    - consumption

    Households need at least 15 valid monthly observations. When more
    than 25 are supplied, only the most recent 25 observations are used
    to stay close to the training coverage range.
    """

    required = {"household_ID", "month", "consumption"}
    missing = sorted(required - set(raw_monthly.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Expected household_ID, month and consumption."
        )

    data = raw_monthly[
        ["household_ID", "month", "consumption"]
    ].copy()

    data["household_ID"] = _normalise_household_id(
        data["household_ID"]
    )

    invalid_ids = data["household_ID"].isin(
        ["", "nan", "None", "<NA>"]
    )

    if invalid_ids.any():
        raise ValueError(
            "Blank or invalid household_ID values were found."
        )

    original_month = data["month"].copy()

    data["month"] = pd.to_datetime(
        data["month"],
        errors="coerce",
    )

    invalid_months = data["month"].isna()

    if invalid_months.any():
        examples = (
            original_month[invalid_months]
            .head(10)
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Invalid month values were found. "
            f"Examples: {examples}"
        )

    data["month"] = (
        data["month"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    data["consumption"] = pd.to_numeric(
        data["consumption"]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    missing_consumption_count = int(
        data["consumption"].isna().sum()
    )

    data = data.dropna(
        subset=["consumption"]
    ).copy()

    if (data["consumption"] < 0).any():
        examples = (
            data.loc[
                data["consumption"] < 0,
                ["household_ID", "month", "consumption"],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "Negative monthly consumption is not supported. "
            f"Examples: {examples}"
        )

    duplicate_mask = data.duplicated(
        subset=["household_ID", "month"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            data.loc[
                duplicate_mask,
                ["household_ID", "month", "consumption"],
            ]
            .head(20)
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate household-month rows were found. "
            "Resolve them before prediction. "
            f"Examples: {examples}"
        )

    data = data.sort_values(
        ["household_ID", "month"]
    ).reset_index(drop=True)

    original_counts = (
        data.groupby("household_ID")["month"]
        .nunique()
        .rename("months_supplied")
    )

    too_short = original_counts[
        original_counts < minimum_months
    ]

    if not too_short.empty:
        raise ValueError(
            f"At least {minimum_months} valid monthly observations "
            "are required per household. Insufficient households: "
            f"{too_short.to_dict()}"
        )

    # Preserve the most recent months when a longer history is supplied.
    data = (
        data.groupby(
            "household_ID",
            group_keys=False,
        )
        .tail(maximum_months)
        .reset_index(drop=True)
    )

    retained_counts = (
        data.groupby("household_ID")["month"]
        .nunique()
        .rename("months_used")
    )

    audit = (
        pd.concat(
            [original_counts, retained_counts],
            axis=1,
        )
        .reset_index()
    )

    audit["rows_dropped_for_missing_consumption_total"] = (
        missing_consumption_count
    )

    audit["history_truncated_to_latest_25"] = (
        audit["months_supplied"] > maximum_months
    )

    return data, audit


def build_household_billing_features(
    validated_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Create one engineered feature row per household."""

    df = validated_monthly.copy()

    df = df.sort_values(
        ["household_ID", "month"]
    ).reset_index(drop=True)

    df["log_consumption"] = np.log1p(
        df["consumption"]
    )

    grouped = df.groupby(
        "household_ID",
        sort=False,
    )

    features = grouped.agg(
        billing_median_kwh=("consumption", "median"),
        billing_mean_kwh=("consumption", "mean"),
        billing_std_kwh=("consumption", "std"),
        billing_min_kwh=("consumption", "min"),
        billing_max_kwh=("consumption", "max"),
        billing_p10_kwh=(
            "consumption",
            lambda values: values.quantile(0.10),
        ),
        billing_p25_kwh=(
            "consumption",
            lambda values: values.quantile(0.25),
        ),
        billing_p75_kwh=(
            "consumption",
            lambda values: values.quantile(0.75),
        ),
        billing_p90_kwh=(
            "consumption",
            lambda values: values.quantile(0.90),
        ),
        billing_log_std=("log_consumption", "std"),
        months_observed=("month", "nunique"),
        first_month=("month", "min"),
        last_month=("month", "max"),
    ).reset_index()

    features["billing_iqr_kwh"] = (
        features["billing_p75_kwh"]
        - features["billing_p25_kwh"]
    )

    features["billing_robust_range_kwh"] = (
        features["billing_p90_kwh"]
        - features["billing_p10_kwh"]
    )

    features["billing_range_kwh"] = (
        features["billing_max_kwh"]
        - features["billing_min_kwh"]
    )

    features["billing_cv"] = np.where(
        features["billing_mean_kwh"] > 0,
        features["billing_std_kwh"]
        / features["billing_mean_kwh"],
        np.nan,
    )

    features["observation_span_months"] = (
        _calculate_month_span(
            features["first_month"],
            features["last_month"],
        )
    )

    rate_source = df.assign(
        near_zero_month=(df["consumption"] < 10).astype(float),
        low_30_month=(df["consumption"] <= 30).astype(float),
        above_90_month=(df["consumption"] > 90).astype(float),
        above_120_month=(df["consumption"] > 120).astype(float),
        above_180_month=(df["consumption"] > 180).astype(float),
        above_240_month=(df["consumption"] > 240).astype(float),
    )

    rate_features = (
        rate_source.groupby("household_ID")
        .agg(
            near_zero_month_rate=("near_zero_month", "mean"),
            low_30_month_rate=("low_30_month", "mean"),
            above_90_month_rate=("above_90_month", "mean"),
            above_120_month_rate=("above_120_month", "mean"),
            above_180_month_rate=("above_180_month", "mean"),
            above_240_month_rate=("above_240_month", "mean"),
        )
        .reset_index()
    )

    features = features.merge(
        rate_features,
        on="household_ID",
        how="left",
        validate="one_to_one",
    )

    def window_median(
        frame: pd.DataFrame,
        count: int,
        take_first: bool,
        output_name: str,
    ) -> pd.DataFrame:
        grouped_rows = frame.groupby(
            "household_ID",
            group_keys=False,
        )

        selected = (
            grouped_rows.head(count)
            if take_first
            else grouped_rows.tail(count)
        )

        return (
            selected.groupby("household_ID")["consumption"]
            .median()
            .rename(output_name)
            .reset_index()
        )

    early_6 = window_median(
        df,
        6,
        True,
        "early_6_month_median_kwh",
    )

    late_6 = window_median(
        df,
        6,
        False,
        "late_6_month_median_kwh",
    )

    recent_3 = window_median(
        df,
        3,
        False,
        "recent_3_month_median_kwh",
    )

    recent_6 = late_6.rename(
        columns={
            "late_6_month_median_kwh":
            "recent_6_month_median_kwh"
        }
    )

    for window_table in [
        early_6,
        late_6,
        recent_3,
        recent_6,
    ]:
        features = features.merge(
            window_table,
            on="household_ID",
            how="left",
            validate="one_to_one",
        )

    features["recent_vs_overall_log_ratio"] = (
        np.log1p(
            features["recent_3_month_median_kwh"]
        )
        - np.log1p(features["billing_median_kwh"])
    )

    features["late_vs_early_log_ratio"] = (
        np.log1p(
            features["late_6_month_median_kwh"]
        )
        - np.log1p(
            features["early_6_month_median_kwh"]
        )
    )

    df["calendar_month_number"] = (
        df["month"].dt.year * 12
        + df["month"].dt.month
    )

    df["month_gap"] = (
        df.groupby("household_ID")[
            "calendar_month_number"
        ]
        .diff()
    )

    raw_change = (
        df.groupby("household_ID")["consumption"]
        .diff()
    )

    raw_log_change = (
        df.groupby("household_ID")["log_consumption"]
        .diff()
    )

    df["monthly_change_kwh"] = raw_change.where(
        df["month_gap"].eq(1)
    )

    df["log_change"] = raw_log_change.where(
        df["month_gap"].eq(1)
    )

    dynamic_features = (
        df.groupby("household_ID")
        .agg(
            mean_abs_monthly_change_kwh=(
                "monthly_change_kwh",
                lambda values: values.abs().mean(),
            ),
            maximum_monthly_increase_kwh=(
                "monthly_change_kwh",
                lambda values: (
                    max(float(values.max(skipna=True)), 0.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            maximum_monthly_decrease_kwh=(
                "monthly_change_kwh",
                lambda values: (
                    max(-float(values.min(skipna=True)), 0.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_abs_log_change=(
                "log_change",
                lambda values: values.abs().median(),
            ),
        )
        .reset_index()
    )

    features = features.merge(
        dynamic_features,
        on="household_ID",
        how="left",
        validate="one_to_one",
    )

    df["calendar_quarter"] = (
        (df["month"].dt.month - 1) // 3 + 1
    )

    quarter_features = (
        df.pivot_table(
            index="household_ID",
            columns="calendar_quarter",
            values="consumption",
            aggfunc="median",
        )
        .rename(
            columns={
                1: "median_kwh_calendar_q1",
                2: "median_kwh_calendar_q2",
                3: "median_kwh_calendar_q3",
                4: "median_kwh_calendar_q4",
            }
        )
        .reset_index()
    )

    quarter_columns = [
        "median_kwh_calendar_q1",
        "median_kwh_calendar_q2",
        "median_kwh_calendar_q3",
        "median_kwh_calendar_q4",
    ]

    for column in quarter_columns:
        if column not in quarter_features.columns:
            quarter_features[column] = np.nan

    features = features.merge(
        quarter_features[
            ["household_ID"] + quarter_columns
        ],
        on="household_ID",
        how="left",
        validate="one_to_one",
    )

    trend_features = pd.DataFrame(
        [
            {
                "household_ID": household_id,
                "monthly_trend_slope_log1p":
                    _calculate_household_trend(group),
            }
            for household_id, group
            in df.groupby("household_ID")
        ]
    )

    features = features.merge(
        trend_features,
        on="household_ID",
        how="left",
        validate="one_to_one",
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features


def load_model_package(
    package_path: str | Path,
) -> dict:
    package = joblib.load(package_path)

    required = {
        "estimator",
        "feature_order",
        "index_to_class",
        "workflow_model_family",
    }

    missing = sorted(required - set(package))

    if missing:
        raise ValueError(
            f"Invalid model package. Missing keys: {missing}"
        )

    return package


def predict_from_monthly_records(
    raw_monthly: pd.DataFrame,
    package_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate raw monthly rows, engineer household features and
    predict the locked archetype pseudo-label.
    """

    package = load_model_package(package_path)

    validated, coverage_audit = prepare_monthly_records(
        raw_monthly
    )

    features = build_household_billing_features(
        validated
    )

    feature_order = list(package["feature_order"])

    missing_features = [
        feature
        for feature in feature_order
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Feature-engineering output is missing model "
            f"predictors: {missing_features}"
        )

    X = (
        features[feature_order]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    if X.isna().all(axis=1).any():
        bad_households = features.loc[
            X.isna().all(axis=1),
            "household_ID",
        ].tolist()

        raise ValueError(
            "All model predictors are missing for households: "
            f"{bad_households}"
        )

    estimator = package["estimator"]
    predicted_indices = estimator.predict(X).astype(int)
    probabilities = estimator.predict_proba(X)

    index_to_class = {
        int(key): value
        for key, value in package["index_to_class"].items()
    }

    estimator_classes = np.asarray(
        estimator.classes_,
        dtype=int,
    )

    if probabilities.shape[1] != len(estimator_classes):
        raise RuntimeError(
            "Probability columns do not match estimator classes."
        )

    order = np.argsort(
        probabilities,
        axis=1,
    )[:, ::-1]

    top_1_columns = order[:, 0]
    top_2_columns = order[:, 1]

    results = pd.DataFrame({
        "household_ID": features["household_ID"],
        "predicted_class_index": predicted_indices,
        "predicted_archetype": [
            index_to_class[index]
            for index in predicted_indices
        ],
        "top_model_score": probabilities[
            np.arange(len(probabilities)),
            top_1_columns,
        ],
        "second_archetype": [
            index_to_class[
                int(estimator_classes[column])
            ]
            for column in top_2_columns
        ],
        "second_model_score": probabilities[
            np.arange(len(probabilities)),
            top_2_columns,
        ],
    })

    for probability_column, class_index in enumerate(
        estimator_classes
    ):
        safe_name = (
            index_to_class[int(class_index)]
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace("/", "_")
        )

        results[
            f"model_score_{safe_name}"
        ] = probabilities[:, probability_column]

    results["score_interpretation"] = (
        "Raw uncalibrated relative model score; "
        "not a validated probability of correctness."
    )

    results = results.merge(
        coverage_audit,
        on="household_ID",
        how="left",
        validate="one_to_one",
    )

    return results, features, coverage_audit



# ============================================================
# SMART-METER -> CARBON FEATURE ENGINEERING
# ============================================================

def _get_sl_public_holidays(
    years: Iterable[int],
) -> tuple[set[pd.Timestamp], str]:
    years = sorted({int(year) for year in years})
    holiday_dates: set[pd.Timestamp] = set()
    sources: list[str] = []

    for year in years:
        if year == 2024:
            holiday_dates.update(
                pd.Timestamp(value).normalize()
                for value in LOCKED_2024_PUBLIC_HOLIDAYS
            )
            sources.append("2024 thesis public-holiday calendar")
            continue

        try:
            import holidays
        except ImportError as error:
            raise RuntimeError(
                "Automatic Sri Lankan holiday lookup for years other than 2024 "
                "requires the 'holidays' package. Install requirements_FINAL_REBUILT.txt."
            ) from error

        try:
            calendar = holidays.country_holidays(
                "LK",
                years=[year],
                language="en_US",
            )
        except TypeError:
            # Compatibility with older versions that do not accept language.
            calendar = holidays.country_holidays(
                "LK",
                years=[year],
            )

        for holiday_date in calendar.keys():
            holiday_dates.add(
                pd.Timestamp(holiday_date).normalize()
            )

        sources.append(f"{year} Sri Lanka public-holiday calendar")

    return holiday_dates, "; ".join(sources)


def _detect_smartmeter_format(data: pd.DataFrame) -> str:
    columns = set(data.columns)

    interval_required = {
        "household_ID",
        "timestamp",
        "consumption_kwh",
    }

    cumulative_required = {
        "household_ID",
        "date",
        "time",
        "importkwh(kwh)",
        "exportkwh(kwh)",
    }

    cumulative_timestamp_required = {
        "household_ID",
        "timestamp",
        "importkwh(kwh)",
        "exportkwh(kwh)",
    }

    if interval_required.issubset(columns):
        return "interval"

    if cumulative_required.issubset(columns):
        return "cumulative_date_time"

    if cumulative_timestamp_required.issubset(columns):
        return "cumulative_timestamp"

    raise ValueError(
        "Smart-meter CSV format not recognised. Use one of the two templates "
        "provided in the app."
    )


def _prepare_interval_smartmeter(
    raw: pd.DataFrame,
    confirm_non_export: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not confirm_non_export:
        raise ValueError(
            "The clustering and carbon models were developed for non-export households. "
            "For interval-kWh uploads, confirm that the supplied household(s) "
            "do not have active solar/export behaviour."
        )

    data = raw[
        ["household_ID", "timestamp", "consumption_kwh"]
    ].copy()

    data["household_ID"] = data["household_ID"].astype(str).str.strip()
    invalid_ids = data["household_ID"].isin(["", "nan", "None", "<NA>"])
    if invalid_ids.any():
        raise ValueError("Blank or invalid household_ID values were found.")

    data["meter_dt"] = _parse_local_naive_timestamp(
        data["timestamp"],
        "timestamp",
    )

    data["consumption_kwh"] = pd.to_numeric(
        data["consumption_kwh"],
        errors="coerce",
    )

    if data["consumption_kwh"].isna().any():
        raise ValueError("Missing or non-numeric consumption_kwh values were found.")

    duplicate_mask = data.duplicated(
        ["household_ID", "meter_dt"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate household-timestamp rows were found. Resolve them before upload."
        )

    data = data.sort_values(
        ["household_ID", "meter_dt"]
    ).reset_index(drop=True)

    data = _validate_interval_cadence(data)

    valid_energy = (
        data["consumption_kwh"].ge(0)
        & data["consumption_kwh"].le(PHYSICAL_LIMIT_KWH_15MIN)
    )

    data["true_15min_kwh"] = data[
        "consumption_kwh"
    ].where(valid_energy)

    quality = (
        data.groupby("household_ID", as_index=False)
        .agg(
            rows_supplied=("meter_dt", "size"),
            first_timestamp=("meter_dt", "min"),
            last_timestamp=("meter_dt", "max"),
            valid_interval_energy_rows=("true_15min_kwh", "count"),
        )
    )

    quality["invalid_energy_rows"] = (
        quality["rows_supplied"]
        - quality["valid_interval_energy_rows"]
    )
    quality["timestamp_contract"] = "15-minute Sri Lanka local time"

    return (
        data[["household_ID", "meter_dt", "true_15min_kwh"]].copy(),
        quality,
    )


def _prepare_cumulative_smartmeter(
    raw: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = raw.copy()

    if "obis" in data.columns:
        data = data.loc[
            data["obis"].astype(str).str.strip().eq("LP")
        ].copy()

        if data.empty:
            raise ValueError("No OBIS='LP' rows remained after filtering.")

    data["household_ID"] = (
        data["household_ID"]
        .astype(str)
        .str.strip()
    )

    if data["household_ID"].eq("").any():
        raise ValueError("Blank household_ID values were found.")

    if mode == "cumulative_date_time":
        combined_timestamp = (
            data["date"].astype(str).str.strip()
            + " "
            + data["time"].astype(str).str.strip()
        )
        data["meter_dt"] = _parse_local_naive_timestamp(
            combined_timestamp,
            "date/time",
        )
    else:
        data["meter_dt"] = _parse_local_naive_timestamp(
            data["timestamp"],
            "timestamp",
        )

    for column in ["importkwh(kwh)", "exportkwh(kwh)"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[["importkwh(kwh)", "exportkwh(kwh)"]].isna().any().any():
        raise ValueError(
            "Missing/non-numeric import or export cumulative-register values were found."
        )

    duplicate_mask = data.duplicated(
        ["household_ID", "meter_dt"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate household-timestamp rows were found. Resolve them before upload."
        )

    data = data.sort_values(
        ["household_ID", "meter_dt"]
    ).reset_index(drop=True)

    export_delta = (
        data.groupby("household_ID")["exportkwh(kwh)"].max()
        - data.groupby("household_ID")["exportkwh(kwh)"].min()
    )

    solar_ids = set(
        export_delta[
            export_delta > 1.0
        ].index.astype(str)
    )

    data["excluded_active_export"] = (
        data["household_ID"].isin(solar_ids)
    )

    non_export = data.loc[
        ~data["excluded_active_export"]
    ].copy()

    non_export["time_delta_mins"] = (
        non_export.groupby("household_ID")["meter_dt"]
        .diff()
        .dt.total_seconds()
        / 60.0
    )

    non_export["raw_kwh_diff"] = (
        non_export.groupby("household_ID")["importkwh(kwh)"]
        .diff()
    )

    valid_time = non_export["time_delta_mins"].between(
        14.5,
        15.5,
        inclusive="both",
    )

    valid_energy = (
        non_export["raw_kwh_diff"].ge(0)
        & non_export["raw_kwh_diff"].le(PHYSICAL_LIMIT_KWH_15MIN)
    )

    non_export["true_15min_kwh"] = np.where(
        valid_time & valid_energy,
        non_export["raw_kwh_diff"],
        np.nan,
    )

    all_households = (
        data[["household_ID"]]
        .drop_duplicates()
        .copy()
    )

    quality_core = (
        non_export.groupby("household_ID", as_index=False)
        .agg(
            rows_supplied=("meter_dt", "size"),
            first_timestamp=("meter_dt", "min"),
            last_timestamp=("meter_dt", "max"),
            valid_interval_energy_rows=("true_15min_kwh", "count"),
        )
    )

    quality = all_households.merge(
        quality_core,
        on="household_ID",
        how="left",
        validate="one_to_one",
    )

    quality["excluded_active_export"] = (
        quality["household_ID"].isin(solar_ids)
    )

    quality["export_register_delta_kwh"] = (
        quality["household_ID"].map(export_delta)
    )

    return (
        non_export[
            ["household_ID", "meter_dt", "true_15min_kwh"]
        ].copy(),
        quality,
    )


def _build_carbon_features(
    interval_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if interval_data.empty:
        raise ValueError("No non-export interval rows are available for feature construction.")

    df = interval_data.copy()

    df["household_ID"] = (
        df["household_ID"]
        .astype(str)
        .str.strip()
    )

    df["meter_dt"] = pd.to_datetime(
        df["meter_dt"],
        errors="coerce",
    )

    df["true_15min_kwh"] = pd.to_numeric(
        df["true_15min_kwh"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["household_ID", "meter_dt"]
    ).copy()

    df["date_only"] = df["meter_dt"].dt.normalize()

    daily_counts = (
        df.groupby(
            ["household_ID", "date_only"],
            observed=True,
        )["true_15min_kwh"]
        .count()
        .rename("valid_intervals")
        .reset_index()
    )

    valid_days = daily_counts.loc[
        daily_counts["valid_intervals"]
        >= MIN_VALID_INTERVALS_PER_DAY
    ].copy()

    valid_day_counts = (
        valid_days.groupby("household_ID")
        .size()
        .rename("valid_days")
    )

    eligible_ids = valid_day_counts.loc[
        valid_day_counts >= MIN_VALID_DAYS
    ].index.astype(str)

    valid_day_keys = valid_days.loc[
        valid_days["household_ID"].astype(str).isin(
            set(eligible_ids)
        ),
        ["household_ID", "date_only"],
    ].drop_duplicates()

    feature_df = df.merge(
        valid_day_keys,
        on=["household_ID", "date_only"],
        how="inner",
        validate="many_to_one",
    )

    if feature_df.empty:
        raise ValueError(
            "No household passed the requirement of at least 30 valid days "
            "with at least 90 valid 15-minute intervals per day."
        )

    years = sorted(
        feature_df["meter_dt"]
        .dt.year
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    holiday_dates, holiday_source = _get_sl_public_holidays(years)

    feature_df["is_non_working_day"] = (
        (feature_df["meter_dt"].dt.dayofweek >= 5)
        | feature_df["date_only"].isin(holiday_dates)
    )

    minute_of_day = (
        feature_df["meter_dt"].dt.hour * 60
        + feature_df["meter_dt"].dt.minute
    )

    feature_df["ceb_block"] = pd.Categorical(
        np.select(
            [
                (
                    (minute_of_day >= 5 * 60 + 30)
                    & (minute_of_day < 18 * 60 + 30)
                ),
                (
                    (minute_of_day >= 18 * 60 + 30)
                    & (minute_of_day < 22 * 60 + 30)
                ),
            ],
            ["Day", "Peak"],
            default="Off-Peak",
        ),
        categories=["Day", "Peak", "Off-Peak"],
        ordered=False,
    )

    daily_block_usage = (
        feature_df.groupby(
            ["household_ID", "date_only", "ceb_block"],
            observed=True,
        )["true_15min_kwh"]
        .sum()
        .unstack(fill_value=0)
        .reindex(
            columns=["Day", "Peak", "Off-Peak"],
            fill_value=0,
        )
    )

    avg_daily_block = (
        daily_block_usage
        .groupby("household_ID", observed=True)
        .mean()
    )

    total_avg_daily = (
        avg_daily_block.sum(axis=1)
    )

    features = pd.DataFrame(
        index=avg_daily_block.index
    )

    features["evening_peak_ratio"] = (
        avg_daily_block["Peak"]
        / total_avg_daily.clip(lower=0.01)
    )

    features["night_baseload_ratio"] = (
        avg_daily_block["Off-Peak"]
        / total_avg_daily.clip(lower=0.01)
    )

    feature_df["interval_id"] = (
        feature_df["meter_dt"].dt.hour * 4
        + feature_df["meter_dt"].dt.minute // 15
    ).astype("uint8")

    interval_profiles = (
        feature_df.groupby(
            ["household_ID", "interval_id"],
            observed=True,
        )["true_15min_kwh"]
        .mean()
        .unstack()
        .reindex(columns=range(96))
    )

    complete_profiles = interval_profiles.notna().all(axis=1)

    complete_interval_profiles = (
        interval_profiles.loc[complete_profiles]
    )

    features["load_uniformity"] = (
        complete_interval_profiles.apply(
            lambda row: entropy(
                row.to_numpy(dtype=float) + 1e-9
            ),
            axis=1,
        )
    )

    household_stats = (
        feature_df.groupby(
            "household_ID",
            observed=True,
        )["true_15min_kwh"]
        .agg(["mean", "max"])
    )

    features["load_factor"] = (
        household_stats["mean"]
        / household_stats["max"].clip(lower=0.01)
    )

    daily_totals = (
        feature_df.groupby(
            [
                "household_ID",
                "date_only",
                "is_non_working_day",
            ],
            observed=True,
        )["true_15min_kwh"]
        .sum()
        .reset_index()
    )

    day_type_avg = (
        daily_totals.groupby(
            ["household_ID", "is_non_working_day"],
            observed=True,
        )["true_15min_kwh"]
        .mean()
        .unstack()
        .reindex(columns=[False, True])
    )

    features["leisure_usage_ratio"] = np.log1p(
        day_type_avg[True]
        / day_type_avg[False].clip(lower=0.01)
    )

    # Activity variable is calculated exactly as in the master notebook
    # but is NOT passed to the carbon model.
    features["log_daily_kwh"] = np.log1p(
        total_avg_daily
    )

    features = (
        features
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if features.empty:
        raise ValueError(
            "Households passed the valid-day rule, but none had enough information "
            "to construct every behavioural feature. The upload must cover all 96 "
            "quarter-hour positions and include both working and non-working days."
        )

    features["activity_eligible"] = (
        features["log_daily_kwh"] > 0.05
    )

    eligible_features = features.loc[
        features["activity_eligible"]
    ].copy()

    if eligible_features.empty:
        raise ValueError(
            "No household passed the final activity criterion (log_daily_kwh > 0.05)."
        )

    coverage = (
        daily_counts.groupby("household_ID", as_index=False)
        .agg(
            observed_days=("date_only", "nunique"),
            minimum_valid_intervals_in_observed_day=(
                "valid_intervals",
                "min",
            ),
            median_valid_intervals_in_observed_day=(
                "valid_intervals",
                "median",
            ),
        )
    )

    coverage = coverage.merge(
        valid_day_counts.reset_index(),
        on="household_ID",
        how="left",
    )

    coverage["valid_days"] = (
        coverage["valid_days"]
        .fillna(0)
        .astype(int)
    )

    working_counts = (
        daily_totals.loc[
            ~daily_totals["is_non_working_day"]
        ]
        .groupby("household_ID")
        .size()
        .rename("working_days_used")
    )

    non_working_counts = (
        daily_totals.loc[
            daily_totals["is_non_working_day"]
        ]
        .groupby("household_ID")
        .size()
        .rename("non_working_days_used")
    )

    coverage = (
        coverage
        .merge(
            working_counts.reset_index(),
            on="household_ID",
            how="left",
        )
        .merge(
            non_working_counts.reset_index(),
            on="household_ID",
            how="left",
        )
    )

    for column in [
        "working_days_used",
        "non_working_days_used",
    ]:
        coverage[column] = (
            coverage[column]
            .fillna(0)
            .astype(int)
        )

    coverage["passes_30_valid_day_rule"] = (
        coverage["valid_days"] >= MIN_VALID_DAYS
    )

    coverage["has_all_96_profile_positions"] = (
        coverage["household_ID"]
        .map(complete_profiles)
        .fillna(False)
        .astype(bool)
    )

    coverage["passes_activity_gate"] = (
        coverage["household_ID"]
        .map(features["activity_eligible"])
        .fillna(False)
        .astype(bool)
    )

    return (
        eligible_features[
            CARBON_FEATURES + ["log_daily_kwh"]
        ].copy(),
        coverage,
        holiday_source,
    )


def _prepare_smartmeter_upload(
    raw: pd.DataFrame,
    confirm_non_export: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Validate a raw smart-meter CSV and return cleaned 15-minute intervals."""
    mode = _detect_smartmeter_format(raw)

    if mode == "interval":
        intervals, source_quality = _prepare_interval_smartmeter(
            raw,
            confirm_non_export=confirm_non_export,
        )
        input_description = "Interval-kWh format"
    else:
        intervals, source_quality = _prepare_cumulative_smartmeter(
            raw,
            mode=mode,
        )
        input_description = "Cumulative-register format"

    if intervals.empty:
        raise ValueError(
            "No eligible non-export 15-minute intervals remained after input validation."
        )

    return intervals, source_quality, input_description

def _process_smartmeter_upload(
    raw: pd.DataFrame,
    confirm_non_export: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    intervals, source_quality, input_description = _prepare_smartmeter_upload(
        raw,
        confirm_non_export=confirm_non_export,
    )

    features, coverage, holiday_source = _build_carbon_features(intervals)
    if features.empty:
        raise ValueError(
            "No household had sufficient coverage to construct all six behavioural "
            "features required by the final study models."
        )

    combined_audit = coverage.merge(
        source_quality,
        on="household_ID",
        how="left",
        validate="one_to_one",
    )

    return (
        features,
        combined_audit,
        intervals,
        holiday_source,
        input_description,
    )



# ============================================================
# SURVEY TOOL HELPERS
# ============================================================

def _extract_encoder_categories(
    estimator: Any,
    categorical_features: list[str],
) -> dict[str, list[str]]:
    if not categorical_features:
        return {}

    visited: set[int] = set()
    stack: list[Any] = [estimator]

    while stack:
        current = stack.pop()

        if current is None or id(current) in visited:
            continue

        visited.add(id(current))

        categories = getattr(
            current,
            "categories_",
            None,
        )

        if (
            categories is not None
            and len(categories) == len(categorical_features)
        ):
            return {
                feature: [
                    str(value)
                    for value in values.tolist()
                ]
                for feature, values
                in zip(categorical_features, categories)
            }

        named_steps = getattr(
            current,
            "named_steps",
            None,
        )

        if named_steps is not None:
            stack.extend(
                list(named_steps.values())
            )

        transformers = getattr(
            current,
            "transformers_",
            None,
        )

        if transformers is not None:
            for _, transformer, _ in transformers:
                if transformer not in {"drop", "passthrough"}:
                    stack.append(transformer)

        calibrated = getattr(
            current,
            "calibrated_classifiers_",
            None,
        )

        if calibrated is not None:
            for calibrated_classifier in calibrated:
                stack.append(
                    getattr(
                        calibrated_classifier,
                        "estimator",
                        None,
                    )
                )

        for attribute in [
            "estimator",
            "base_estimator",
            "classifier",
        ]:
            child = getattr(
                current,
                attribute,
                None,
            )

            if child is not None and child is not current:
                stack.append(child)

    return {}


def _survey_defaults(
    reference: pd.DataFrame | None,
) -> dict[str, Any]:
    defaults = {
        "household_size": 4,
        "room_count": 4,
        "appliance_total_count": 10,
        "appliance_type_count": 7,
        "fan_total_count": 3,
        "light_total_count": 8,
        "ac_total_count": 0,
        "electricity_meter_count": 1,
        "has_business": 0,
        "has_backup_generator": 0,
        "grid_cooking": 0,
        "gas_cooking": 1,
        "firewood_cooking": 0,
    }

    if reference is None:
        return defaults

    for column in [
        "household_size",
        "room_count",
        "appliance_total_count",
        "appliance_type_count",
        "fan_total_count",
        "light_total_count",
        "ac_total_count",
        "electricity_meter_count",
    ]:
        if column in reference.columns:
            values = pd.to_numeric(
                reference[column],
                errors="coerce",
            ).dropna()

            if not values.empty:
                defaults[column] = int(
                    max(
                        0,
                        round(float(values.median())),
                    )
                )

    return defaults


def _survey_reference_benchmark(
    reference: pd.DataFrame,
    input_row: pd.DataFrame,
    neighbours: int = 30,
) -> tuple[pd.DataFrame, dict[str, float]]:
    numeric_binary_features = [
        "household_size",
        "room_count",
        "appliance_total_count",
        "appliance_type_count",
        "fan_total_count",
        "light_total_count",
        "ac_total_count",
        "electricity_meter_count",
        "has_ac",
        "has_business",
        "has_backup_generator",
        "grid_cooking",
        "gas_cooking",
        "firewood_cooking",
    ]
    categorical_features = ["house_type", "meter_type"]

    required = set(numeric_binary_features + ["mean_monthly_kwh"])
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError(
            "Survey profiler reference file is missing: "
            + ", ".join(missing)
        )

    history = reference.copy()
    query = input_row.copy()

    for column in numeric_binary_features:
        history[column] = pd.to_numeric(history[column], errors="coerce")
        query[column] = pd.to_numeric(query[column], errors="coerce")

        median = history[column].median()
        if not np.isfinite(median):
            median = 0.0
        history[column] = history[column].fillna(median)
        query[column] = query[column].fillna(median)

    scaler = StandardScaler()
    history_numeric = scaler.fit_transform(history[numeric_binary_features])
    query_numeric = scaler.transform(query[numeric_binary_features])

    history_parts = [history_numeric]
    query_parts = [query_numeric]

    # Include house type and meter type when the reference dataset contains them.
    # One-hot encoding is fitted jointly only for this descriptive lookup; this
    # is not part of the predictive Logistic Regression pipeline.
    available_categories = [
        c for c in categorical_features
        if c in history.columns and c in query.columns
    ]
    for column in available_categories:
        history_values = history[column].fillna("Unknown").astype(str)
        query_value = str(query[column].iloc[0])
        levels = sorted(set(history_values.tolist()) | {query_value})
        hist_onehot = np.column_stack(
            [(history_values == level).astype(float).to_numpy() for level in levels]
        )
        query_onehot = np.array(
            [[1.0 if query_value == level else 0.0 for level in levels]],
            dtype=float,
        )
        history_parts.append(hist_onehot)
        query_parts.append(query_onehot)

    history_matrix = np.column_stack(history_parts)
    query_matrix = np.column_stack(query_parts)

    history["similarity_distance"] = euclidean_distances(
        query_matrix,
        history_matrix,
    )[0]

    similar = (
        history
        .nsmallest(min(neighbours, len(history)), "similarity_distance")
        .copy()
    )

    if "consumption_tertile" in similar.columns:
        group_column = "consumption_tertile"
    else:
        lower = float(reference["mean_monthly_kwh"].quantile(1 / 3))
        upper = float(reference["mean_monthly_kwh"].quantile(2 / 3))
        similar["consumption_tertile"] = pd.cut(
            pd.to_numeric(similar["mean_monthly_kwh"], errors="coerce"),
            bins=[-np.inf, lower, upper, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)
        group_column = "consumption_tertile"

    metrics = {
        "similar_households": float(len(similar)),
        "median_monthly_kwh": float(
            pd.to_numeric(similar["mean_monthly_kwh"], errors="coerce").median()
        ),
        "mean_monthly_kwh": float(
            pd.to_numeric(similar["mean_monthly_kwh"], errors="coerce").mean()
        ),
        "high_group_share": float(
            similar[group_column].astype(str).eq("High").mean()
        ),
    }

    return similar, metrics



# ============================================================
# DASHBOARD INTERPRETATION / PROFILER HELPERS
# ============================================================

@st.cache_data
def _locked_archetype_medians() -> pd.DataFrame:
    if not KMEANS_LOCKED_LABELS_PATH.is_file():
        return pd.DataFrame()
    locked = pd.read_pickle(KMEANS_LOCKED_LABELS_PATH).copy()
    if "cluster_name" not in locked.columns:
        return pd.DataFrame()
    available = [column for column in CLUSTER_FEATURES if column in locked.columns]
    if not available:
        return pd.DataFrame()
    return locked.groupby("cluster_name", observed=True)[available].median()


def _render_archetype_guidance(archetype: str, transfer_model: bool = False) -> None:
    st.markdown("#### Study-archetype reference")
    peak_time = ARCHETYPE_PEAK_TIMES.get(archetype, "—")
    description = ARCHETYPE_DESCRIPTIONS.get(
        archetype,
        "Study-sample behavioural archetype.",
    )

    medians = _locked_archetype_medians()
    cols = st.columns(3)
    cols[0].metric("Study-profile peak time", peak_time)

    if archetype in medians.index and "log_daily_kwh" in medians.columns:
        daily_kwh = float(np.expm1(medians.loc[archetype, "log_daily_kwh"]))
        cols[1].metric("Study median daily use", f"{daily_kwh:.1f} kWh/day")
    else:
        cols[1].metric("Study median daily use", "—")

    if archetype in medians.index and "evening_peak_ratio" in medians.columns:
        peak_share = float(medians.loc[archetype, "evening_peak_ratio"]) * 100.0
        cols[2].metric("Study median evening share", f"{peak_share:.1f}%")
    else:
        cols[2].metric("Study median evening share", "—")

    st.caption(description)
    st.caption(
        "These values describe the assigned study archetype; they are reference "
        "characteristics and are not direct measurements for the uploaded household."
    )

    if transfer_model:
        st.caption(
            "Billing-based output transfers the KMeans study labels from monthly billing "
            "patterns; it does not directly observe the household's 15-minute load shape."
        )
    else:
        st.caption(
            "Smart-meter output assigns the household to the nearest centroid of the final "
            "K=4 study segmentation; it is not a supervised class prediction."
        )

    with st.expander("Optional general energy guidance"):
        for recommendation in ARCHETYPE_RECOMMENDATIONS.get(archetype, []):
            st.markdown(f"- {recommendation}")
        st.caption(
            "General demand-side guidance only; these actions are not household-specific "
            "diagnoses and do not guarantee bill or emissions savings."
        )
        st.markdown(
            f"[SLSEA household guidance]({SLSEA_HOME_URL}) · "
            f"[SLSEA energy labelling]({SLSEA_LABEL_URL}) · "
            f"[PUCSL tariff information]({PUCSL_TARIFF_URL})"
        )


def _assign_pucsl_band(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < 0:
        return "Invalid"
    if value <= 30:
        return "0–30 kWh"
    if value <= 60:
        return "31–60 kWh"
    if value <= 90:
        return "61–90 kWh"
    if value <= 120:
        return "91–120 kWh"
    if value <= 180:
        return "121–180 kWh"
    return ">180 kWh"


def _dominant_pucsl_band(values: pd.Series) -> str:
    series = pd.Series(values).dropna().astype(str)
    series = series[series.isin(PUCSL_BAND_ORDER)]
    if series.empty:
        return "Unknown"
    counts = series.value_counts().reindex(PUCSL_BAND_ORDER, fill_value=0)
    return str(counts.idxmax())


def _assign_reference_group(value: float, low_cut: float, high_cut: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if float(value) <= low_cut:
        return "Low"
    if float(value) <= high_cut:
        return "Medium"
    return "High"


def _load_monthly_profiler_reference(reference_path: Path | None) -> pd.DataFrame | None:
    if reference_path is None or not reference_path.is_file():
        return None
    reference = _load_reference_csv(str(reference_path)).copy()
    required = {"household_ID", "mean_monthly_kwh"}
    if not required.issubset(reference.columns):
        return None
    reference["household_ID"] = reference["household_ID"].astype(str).str.strip()
    reference["mean_monthly_kwh"] = pd.to_numeric(
        reference["mean_monthly_kwh"], errors="coerce"
    )
    reference = reference.dropna(subset=["household_ID", "mean_monthly_kwh"])
    return reference if not reference.empty else None


def _has_consecutive_months(months: pd.Series, expected_count: int) -> bool:
    periods = pd.PeriodIndex(pd.to_datetime(months, errors="coerce"), freq="M").dropna().unique()
    periods = periods.sort_values()
    if len(periods) != int(expected_count):
        return False
    expected = pd.period_range(periods[0], periods[-1], freq="M")
    return len(expected) == int(expected_count) and periods.equals(expected)


def _matches_survey_reference_period(months: pd.Series) -> bool:
    periods = pd.PeriodIndex(pd.to_datetime(months, errors="coerce"), freq="M").dropna().unique()
    periods = periods.sort_values()
    expected = pd.period_range(
        SURVEY_REFERENCE_PERIOD_START,
        SURVEY_REFERENCE_PERIOD_END,
        freq="M",
    )
    return periods.equals(expected)

def _summarise_profiler_values(
    values: np.ndarray,
    reference: pd.DataFrame,
    allow_study_reference: bool,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Monthly consumption values must all be finite numbers.")
    if (values < 0).any():
        raise ValueError("Negative monthly consumption is not supported.")

    mean_value = float(values.mean())
    median_value = float(np.median(values))
    bands = pd.Series([_assign_pucsl_band(value) for value in values])

    result: dict[str, Any] = {
        "mean": mean_value,
        "median": median_value,
        "total": float(values.sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "dominant_band": _dominant_pucsl_band(bands),
        "group": None,
        "study_percentile": np.nan,
        "low_cut": np.nan,
        "high_cut": np.nan,
    }

    # The reference dataset was built from one complete 12-month window
    # (2023-04 to 2024-03). Do not assign a study tertile to shorter histories.
    if allow_study_reference:
        low_cut = float(reference["mean_monthly_kwh"].quantile(1 / 3))
        high_cut = float(reference["mean_monthly_kwh"].quantile(2 / 3))
        reference_values = reference["mean_monthly_kwh"].to_numpy(dtype=float)
        percentile = float((reference_values <= mean_value).mean() * 100.0)
        result.update({
            "group": _assign_reference_group(mean_value, low_cut, high_cut),
            "study_percentile": percentile,
            "low_cut": low_cut,
            "high_cut": high_cut,
        })

    return result



def _render_monthly_consumption_profiler(reference_path: Path | None) -> None:
    _tool_header(
        "Monthly consumption",
        "12-month consumption benchmark",
        "Enter 3–12 monthly kWh values. Descriptive statistics are shown for every "
        "accepted history; Low/Medium/High comparison is shown only for 12 consecutive months.",
    )

    reference = _load_monthly_profiler_reference(reference_path)
    if reference is None:
        st.error(
            "Reference dataset not found. Place "
            "ANALYSIS2_consumption_group_profile_dataset.csv beside the app."
        )
        return

    low_cut = float(reference["mean_monthly_kwh"].quantile(1 / 3))
    high_cut = float(reference["mean_monthly_kwh"].quantile(2 / 3))
    st.caption(
        "Reference population: household mean monthly consumption calculated over "
        "April 2023–March 2024. "
        f"Low ≤ {low_cut:.1f} kWh/month; Medium ≤ {high_cut:.1f} kWh/month; "
        "above that = High."
    )

    manual_tab, csv_tab = st.tabs(["Enter monthly values", "Upload CSV"])

    with manual_tab:
        with st.form("monthly_profiler_manual_form"):
            months_count = st.slider(
                "Number of monthly values",
                min_value=3,
                max_value=12,
                value=12,
                step=1,
            )
            values: list[float] = []
            columns = st.columns(3)
            for index in range(months_count):
                with columns[index % 3]:
                    values.append(
                        st.number_input(
                            f"Month {index + 1} kWh",
                            min_value=0.0,
                            max_value=5000.0,
                            value=90.0,
                            step=1.0,
                            key=f"monthly_profile_{index}",
                        )
                    )
            submitted = st.form_submit_button(
                "Profile consumption",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            array = np.asarray(values, dtype=float)
            reference_eligible = months_count == SURVEY_REFERENCE_MONTHS
            summary = _summarise_profiler_values(
                array,
                reference,
                allow_study_reference=reference_eligible,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean monthly use", f"{summary['mean']:.1f} kWh")
            c2.metric("Median monthly use", f"{summary['median']:.1f} kWh")
            if reference_eligible:
                c3.metric("Study-sample group", str(summary["group"]))
                c4.metric(
                    "Percentile vs study sample",
                    f"{summary['study_percentile']:.0f}th",
                )
            else:
                c3.metric("Study-sample group", "Not assigned")
                c4.metric("Dominant PUCSL band", summary["dominant_band"])

            monthly = pd.DataFrame({
                "Month": np.arange(1, months_count + 1),
                "Consumption (kWh)": array,
            })
            monthly["PUCSL consumption band"] = monthly["Consumption (kWh)"].apply(
                _assign_pucsl_band
            )
            chart = px.line(
                monthly,
                x="Month",
                y="Consumption (kWh)",
                markers=True,
                title="Entered monthly electricity use",
            )
            chart.update_layout(height=330, showlegend=False)
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(monthly, use_container_width=True, hide_index=True)

            if reference_eligible:
                st.caption(
                    "Low/Medium/High compares the entered 12-month mean with the study "
                    "household distribution from April 2023–March 2024. Manual entries "
                    "have no dates, so the app does not assume that they come from that period."
                )
            else:
                st.warning(
                    "Low/Medium/High is intentionally not assigned because the study "
                    "reference was constructed from complete 12-month household means."
                )

            if summary["mean"] < 1:
                st.warning(
                    "Mean consumption is below 1 kWh/month; check account activity and data quality."
                )

    with csv_tab:
        template = pd.DataFrame({
            "household_ID": ["DEMO_001"] * 12,
            "month": pd.period_range(
                SURVEY_REFERENCE_PERIOD_START,
                SURVEY_REFERENCE_PERIOD_END,
                freq="M",
            ).to_timestamp().strftime("%Y-%m-%d"),
            "consumption": [82, 88, 91, 95, 93, 90, 97, 99, 101, 96, 94, 98],
        })
        st.download_button(
            "Download 12-month CSV example",
            template.to_csv(index=False).encode("utf-8"),
            file_name="monthly_consumption_12month_example.csv",
            mime="text/csv",
            key="monthly_profiler_template",
        )
        upload = st.file_uploader(
            "Monthly CSV · household_ID, month, consumption",
            type=["csv"],
            key="monthly_profiler_upload",
        )
        if upload is not None:
            try:
                data = _read_csv(upload, "Monthly consumption CSV")
                required = {"household_ID", "month", "consumption"}
                missing = sorted(required - set(data.columns))
                if missing:
                    raise ValueError("Missing required columns: " + ", ".join(missing))

                data = data[["household_ID", "month", "consumption"]].copy()
                data["household_ID"] = data["household_ID"].astype(str).str.strip()
                invalid_ids = data["household_ID"].isin(["", "nan", "None", "<NA>"])
                if invalid_ids.any():
                    raise ValueError("Blank or invalid household_ID values were found.")

                original_month = data["month"].copy()
                data["month"] = pd.to_datetime(data["month"], errors="coerce")
                if data["month"].isna().any():
                    examples = original_month.loc[data["month"].isna()].head(5).tolist()
                    raise ValueError(f"Invalid month values were found. Examples: {examples}")

                raw_consumption = data["consumption"].copy()
                data["consumption"] = pd.to_numeric(
                    data["consumption"].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
                if data["consumption"].isna().any():
                    examples = raw_consumption.loc[data["consumption"].isna()].head(5).tolist()
                    raise ValueError(
                        f"Missing or non-numeric consumption values were found. Examples: {examples}"
                    )
                if (data["consumption"] < 0).any():
                    raise ValueError("Negative monthly consumption is not supported.")

                data["_month_period"] = data["month"].dt.to_period("M")
                if data.duplicated(subset=["household_ID", "_month_period"]).any():
                    raise ValueError("Duplicate household-month rows were found.")
                data = data.drop(columns=["_month_period"])

                counts = data.groupby("household_ID")["month"].nunique()
                invalid = counts[(counts < 3) | (counts > 12)]
                if not invalid.empty:
                    raise ValueError(
                        "The monthly consumption tool accepts 3–12 unique months per household. "
                        f"Invalid households: {invalid.to_dict()}"
                    )

                rows: list[dict[str, Any]] = []
                data["PUCSL consumption band"] = data["consumption"].apply(_assign_pucsl_band)
                for household_id, group in data.groupby("household_ID", observed=True):
                    group = group.sort_values("month")
                    months = int(group["month"].nunique())
                    twelve_consecutive = _has_consecutive_months(
                        group["month"],
                        SURVEY_REFERENCE_MONTHS,
                    )
                    summary = _summarise_profiler_values(
                        group["consumption"].to_numpy(dtype=float),
                        reference,
                        allow_study_reference=twelve_consecutive,
                    )
                    rows.append({
                        "household_ID": household_id,
                        "months": months,
                        "mean_monthly_kwh": summary["mean"],
                        "median_monthly_kwh": summary["median"],
                        "study_reference_group": (
                            summary["group"] if twelve_consecutive else "Not assigned"
                        ),
                        "study_percentile": (
                            summary["study_percentile"] if twelve_consecutive else np.nan
                        ),
                        "dominant_pucsl_band": summary["dominant_band"],
                        "twelve_consecutive_months": bool(twelve_consecutive),
                        "matches_2023_04_to_2024_03_reference_period": bool(
                            twelve_consecutive
                            and _matches_survey_reference_period(group["month"])
                        ),
                    })

                output = pd.DataFrame(rows)
                st.dataframe(output, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download benchmark results",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="monthly_benchmark_results.csv",
                    mime="text/csv",
                    key="monthly_benchmark_results_download",
                )

                if len(output) == 1:
                    household_id = str(output.iloc[0]["household_ID"])
                    plot_data = data.loc[data["household_ID"] == household_id].sort_values("month")
                    chart = px.line(
                        plot_data,
                        x="month",
                        y="consumption",
                        markers=True,
                        title="Uploaded monthly electricity use",
                    )
                    chart.update_layout(height=330, showlegend=False)
                    st.plotly_chart(chart, use_container_width=True)

                if (~output["twelve_consecutive_months"]).any():
                    st.warning(
                        "At least one household does not have 12 consecutive months. "
                        "Its Low/Medium/High study-sample comparison is therefore not assigned."
                    )
                comparable = output["twelve_consecutive_months"]
                if comparable.any() and (
                    ~output.loc[comparable, "matches_2023_04_to_2024_03_reference_period"]
                ).any():
                    st.caption(
                        "For 12-consecutive-month histories outside April 2023–March 2024, "
                        "the tertile and percentile are historical-reference comparisons, "
                        "not contemporaneous national categories."
                    )
            except Exception as error:
                st.error(str(error))

    st.caption(
        "PUCSL bands shown here are kWh consumption bands for a 30-day domestic billing "
        "context. This tool does not calculate a live bill or tariff cost."
    )
    st.markdown(f"[Check current PUCSL tariff information]({PUCSL_TARIFF_URL})")





# ============================================================
# PAGES
# ============================================================



def _render_kmeans_tool() -> None:
    _tool_header(
        "Smart meter",
        "Behavioural archetype assignment",
        "Upload 15-minute smart-meter data. The app calculates the six final behavioural "
        "features, applies the final preprocessing pipeline, and assigns the nearest K=4 centroid.",
    )
    st.caption(
        "Eligibility: ≥30 valid days; each valid day has ≥90 valid 15-minute intervals. "
        "For interval-kWh input, confirm that the household has no active solar/export."
    )

    interval_demo, register_demo = _build_synthetic_smartmeter_demo()
    a, b = st.columns(2)
    with a:
        st.download_button(
            "Download eligible interval-kWh demo CSV",
            interval_demo.to_csv(index=False).encode("utf-8"),
            file_name="smartmeter_interval_kwh_DEMO_35days.csv",
            mime="text/csv",
            key="kmeans_interval_template",
        )
    with b:
        st.download_button(
            "Download eligible cumulative-register demo CSV",
            register_demo.to_csv(index=False).encode("utf-8"),
            file_name="smartmeter_cumulative_register_DEMO_35days.csv",
            mime="text/csv",
            key="kmeans_register_template",
        )
    st.caption("Demo files are synthetic and are provided only to demonstrate the required schema.")

    confirm_non_export = st.checkbox(
        "Interval-kWh upload only: I confirm the household has no active solar/export.",
        key="kmeans_non_export",
    )

    upload = st.file_uploader(
        "Smart-meter CSV",
        type=["csv"],
        key="kmeans_upload",
    )
    if upload is None:
        return

    try:
        raw = _read_csv(upload, "Smart-meter CSV")
        features, audit, intervals, _, input_description = _process_smartmeter_upload(
            raw,
            confirm_non_export=confirm_non_export,
        )
        bundle = _load_locked_kmeans_from_sources()

        raw_X = features[CLUSTER_FEATURES].copy()
        clipped_X = raw_X.copy()
        clipped_count = pd.Series(0, index=raw_X.index, dtype=int)

        for feature in CLUSTER_FEATURES:
            lower, upper = bundle["clip_bounds"][feature]
            outside = (raw_X[feature] < lower) | (raw_X[feature] > upper)
            clipped_count = clipped_count + outside.astype(int)
            clipped_X[feature] = clipped_X[feature].clip(lower=lower, upper=upper)

        transformed = bundle["scaler"].transform(clipped_X[CLUSTER_FEATURES])
        cluster_ids = bundle["kmeans"].predict(transformed).astype(int)
        distances = bundle["kmeans"].transform(transformed)
        assigned_distances = distances[np.arange(len(cluster_ids)), cluster_ids]
        distance_p95 = np.array([
            bundle["cluster_distance_p95"][int(cluster_id)]
            for cluster_id in cluster_ids
        ])

        result = pd.DataFrame({
            "household_ID": raw_X.index.astype(str),
            "assigned_study_archetype": [
                bundle["cluster_name_map"][int(cluster_id)]
                for cluster_id in cluster_ids
            ],
            "cluster_id": cluster_ids,
            "centroid_distance": assigned_distances,
            "study_cluster_distance_p95": distance_p95,
            "outside_study_cluster_distance_95pct": assigned_distances > distance_p95,
            "clipped_features": clipped_count.to_numpy(),
        })
    except Exception as error:
        st.error(str(error))
        return

    if len(result) == 1:
        row = result.iloc[0]
        archetype = str(row["assigned_study_archetype"])
        c1, c2 = st.columns(2)
        c1.metric("Assigned study archetype", archetype)
        c2.metric(
            "Distance vs training cluster",
            "Beyond 95th percentile" if bool(row["outside_study_cluster_distance_95pct"]) else "Within 95th percentile",
        )
        if archetype in ARCHETYPE_DESCRIPTIONS:
            st.caption(ARCHETYPE_DESCRIPTIONS[archetype])
        _render_archetype_guidance(archetype, transfer_model=False)
    else:
        st.dataframe(
            result[[
                "household_ID",
                "assigned_study_archetype",
                "outside_study_cluster_distance_95pct",
            ]],
            use_container_width=True,
            hide_index=True,
        )

    if (result["clipped_features"] > 0).any():
        st.warning(
            "Some engineered features fell outside the 1st–99th percentile range observed "
            "in the training households and were clipped using the final pipeline bounds."
        )

    if result["outside_study_cluster_distance_95pct"].any():
        st.warning(
            "At least one assignment is farther from its assigned centroid than 95% "
            "of training households in that cluster. The assignment is still produced, "
            "but it should be treated as an extrapolative study-profile match."
        )

    date_min = intervals["meter_dt"].min()
    date_max = intervals["meter_dt"].max()
    if pd.notna(date_min) and pd.notna(date_max) and (
        date_min < pd.Timestamp("2024-05-01") or date_max >= pd.Timestamp("2024-12-01")
    ):
        st.warning(
            "The uploaded period is outside the May–November 2024 model-development "
            "window. The assignment is therefore outside the development period and should "
            "be interpreted cautiously."
        )

    st.download_button(
        "Download archetype assignments",
        result.to_csv(index=False).encode("utf-8"),
        file_name="smartmeter_archetype_assignments.csv",
        mime="text/csv",
        key="kmeans_result_download",
    )

    with st.expander("Data and model details"):
        st.caption("Detected input: " + input_description)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.markdown("**Engineered behavioural features**")
        st.dataframe(
            features[CLUSTER_FEATURES].reset_index(),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Coverage and input-quality audit**")
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.caption(
            "Cluster ID and centroid distance are technical diagnostics. Neither is a "
            "probability that the household's archetype is correct."
        )



def _render_archetype_tool(model_path: Path | None) -> None:
    _tool_header(
        "Monthly billing",
        "Billing-based archetype label transfer",
        "Upload 15–25 valid monthly records. The trained label-transfer model estimates "
        "which of the four KMeans study archetypes is most strongly supported by the billing history.",
    )

    template = pd.DataFrame({
        "household_ID": ["DEMO_001"] * 15,
        "month": pd.date_range("2023-08-01", periods=15, freq="MS").strftime("%Y-%m-%d"),
        "consumption": [
            92, 95, 101, 98, 106,
            111, 109, 115, 118, 121,
            117, 123, 128, 126, 131,
        ],
    })
    st.download_button(
        "Download 15-month billing example",
        template.to_csv(index=False).encode("utf-8"),
        file_name="monthly_billing_15month_example.csv",
        mime="text/csv",
        key="billing_template",
    )

    upload = st.file_uploader(
        "Monthly billing CSV · household_ID, month, consumption",
        type=["csv"],
        key="billing_upload",
    )
    if upload is None:
        return

    if model_path is None:
        st.error(
            "Archetype label-transfer model not found. Place "
            "archetype_classifier_bundle.joblib beside the app or in its final model folder."
        )
        return

    try:
        raw = _read_csv(upload, "Monthly billing CSV")
        predictions, engineered, audit = predict_from_monthly_records(
            raw_monthly=raw,
            package_path=model_path,
        )
    except Exception as error:
        st.error(str(error))
        return

    if len(predictions) == 1:
        row = predictions.iloc[0]
        predicted = str(row["predicted_archetype"])
        top_score = float(row["top_model_score"])
        b1, b2 = st.columns(2)
        b1.metric("Predicted study archetype", predicted)
        b2.metric("Highest model score", f"{top_score:.3f}")
        st.caption(
            "The score ranks the model's class outputs. It is not calibrated as the "
            "probability that the assigned archetype is correct."
        )
        if predicted in ARCHETYPE_DESCRIPTIONS:
            st.caption(ARCHETYPE_DESCRIPTIONS[predicted])
        _render_archetype_guidance(predicted, transfer_model=True)
    else:
        st.dataframe(
            predictions[["household_ID", "predicted_archetype", "top_model_score"]],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "This model transfers KMeans study labels from monthly billing patterns. It does "
        "not directly observe 15-minute load shape, is not an external ground-truth "
        "classifier, and is not a future-demand forecast."
    )

    st.download_button(
        "Download label-transfer results",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="billing_archetype_label_transfer_results.csv",
        mime="text/csv",
        key="billing_result_download",
    )

    with st.expander("Coverage and engineered-feature details"):
        display_columns = [
            column
            for column in [
                "household_ID",
                "predicted_archetype",
                "top_model_score",
                "second_archetype",
                "second_model_score",
                "months_observed",
                "first_month",
                "last_month",
            ]
            if column in predictions.columns
        ]
        st.dataframe(predictions[display_columns], use_container_width=True, hide_index=True)
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.dataframe(engineered, use_container_width=True, hide_index=True)




def _calculate_uploaded_electricity_carbon(
    intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate direct electricity/CO2 summaries from model-valid uploaded days.

    This is carbon accounting from the uploaded interval energy, not an ML output.
    The main period total uses only days meeting the thesis >=90-valid-interval rule.
    A calendar month is reported separately only when every calendar day contains
    all 96 valid 15-minute positions, avoiding a misleading monthly total from
    partial coverage.
    """
    data = intervals[["household_ID", "meter_dt", "true_15min_kwh"]].copy()
    data["household_ID"] = data["household_ID"].astype(str).str.strip()
    data["meter_dt"] = pd.to_datetime(data["meter_dt"], errors="coerce")
    data["true_15min_kwh"] = pd.to_numeric(
        data["true_15min_kwh"], errors="coerce"
    )
    data = data.dropna(subset=["household_ID", "meter_dt"]).copy()
    data["date_only"] = data["meter_dt"].dt.normalize()
    data["month"] = data["meter_dt"].dt.to_period("M")

    daily = (
        data.groupby(["household_ID", "date_only", "month"], observed=True)
        .agg(
            valid_intervals=("true_15min_kwh", "count"),
            observed_kwh=("true_15min_kwh", "sum"),
        )
        .reset_index()
    )

    valid_days = daily.loc[
        daily["valid_intervals"] >= MIN_VALID_INTERVALS_PER_DAY
    ].copy()

    if valid_days.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        valid_days.groupby("household_ID", as_index=False)
        .agg(
            valid_days_used=("date_only", "nunique"),
            first_valid_day=("date_only", "min"),
            last_valid_day=("date_only", "max"),
            electricity_on_valid_days_kwh=("observed_kwh", "sum"),
        )
    )
    summary["co2_on_valid_days_kg"] = (
        summary["electricity_on_valid_days_kwh"]
        * GRID_FACTOR_KG_CO2_PER_KWH
    )

    complete_rows: list[dict[str, Any]] = []
    for (household_id, month), group in daily.groupby(
        ["household_ID", "month"], observed=True
    ):
        expected_days = int(month.days_in_month)
        complete = (
            group["date_only"].nunique() == expected_days
            and len(group) == expected_days
            and group["valid_intervals"].eq(96).all()
        )
        if not complete:
            continue

        month_kwh = float(group["observed_kwh"].sum())
        complete_rows.append(
            {
                "household_ID": str(household_id),
                "month": str(month),
                "electricity_kwh": month_kwh,
                "electricity_related_co2_kg": (
                    month_kwh * GRID_FACTOR_KG_CO2_PER_KWH
                ),
            }
        )

    complete_months = pd.DataFrame(complete_rows)
    if complete_months.empty:
        summary["complete_calendar_months"] = 0
        summary["mean_complete_month_kwh"] = np.nan
        summary["mean_complete_month_co2_kg"] = np.nan
    else:
        monthly_summary = (
            complete_months.groupby("household_ID", as_index=False)
            .agg(
                complete_calendar_months=("month", "count"),
                mean_complete_month_kwh=("electricity_kwh", "mean"),
                mean_complete_month_co2_kg=(
                    "electricity_related_co2_kg", "mean"
                ),
            )
        )
        summary = summary.merge(
            monthly_summary,
            on="household_ID",
            how="left",
            validate="one_to_one",
        )
        summary["complete_calendar_months"] = (
            summary["complete_calendar_months"].fillna(0).astype(int)
        )

    return summary, complete_months


def _carbon_feature_ood_audit(
    X: pd.DataFrame,
    model_bundle: dict,
) -> pd.DataFrame:
    """Compare uploaded behavioural features with the carbon-model cohort.

    The model bundle stores the development and held-out household IDs. When
    available, a central-95% reference range and percentile rank are calculated
    on that complete carbon-model cohort using the unclipped feature export.
    This is an application-time caution flag only; it does not alter predictions.
    """
    if not KMEANS_RAW_FEATURES_PATH.is_file():
        return pd.DataFrame()

    reference = pd.read_pickle(KMEANS_RAW_FEATURES_PATH)
    if "household_ID" in reference.columns:
        reference = reference.set_index("household_ID")
    reference.index = reference.index.astype(str).str.strip()

    missing = sorted(set(CARBON_FEATURES) - set(reference.columns))
    if missing:
        return pd.DataFrame()

    cohort_ids: list[str] = []
    for key in ["development_household_ids", "heldout_test_household_ids"]:
        values = model_bundle.get(key, [])
        if values is not None:
            cohort_ids.extend(str(value).strip() for value in values)

    cohort_ids = list(dict.fromkeys(cohort_ids))
    if cohort_ids:
        available_ids = [value for value in cohort_ids if value in reference.index]
        if available_ids:
            reference = reference.loc[available_ids, CARBON_FEATURES].copy()
        else:
            reference = reference[CARBON_FEATURES].copy()
    else:
        reference = reference[CARBON_FEATURES].copy()

    reference = (
        reference.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if reference.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for household_id, row in X.iterrows():
        for feature in CARBON_FEATURES:
            lower = float(reference[feature].quantile(0.025))
            upper = float(reference[feature].quantile(0.975))
            value = float(row[feature])
            percentile = float((reference[feature] <= value).mean() * 100.0)
            rows.append(
                {
                    "household_ID": str(household_id),
                    "feature": feature,
                    "value": value,
                    "study_percentile": percentile,
                    "study_p025": lower,
                    "study_p975": upper,
                    "outside_study_central_95pct": bool(
                        value < lower or value > upper
                    ),
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# LOCAL SHAP EXPLANATION FOR CARBON REGRESSION
# ============================================================

def _carbon_regression_shap(
    fitted_pipeline: Any,
    X: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, float]:
    """
    Explain the fitted carbon regression prediction using SHAP.

    This does NOT retrain or modify the model.
    It only explains the prediction produced by the existing fitted pipeline.
    """
    import shap

    # Apply exactly the preprocessing already fitted inside the pipeline
    transformed = X.copy()

    for _, transformer in list(
        fitted_pipeline.named_steps.items()
    )[:-1]:
        transformed = transformer.transform(transformed)

    transformed = np.asarray(transformed, dtype=float)

    if transformed.shape[1] != len(feature_columns):
        raise RuntimeError(
            "SHAP transformed-feature count does not match "
            "the five carbon predictors."
        )

    # Extract the already-trained final regression model
    model = fitted_pipeline.named_steps["model"]

    # Tree-based SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    shap_values = np.asarray(shap_values, dtype=float)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    expected_shape = (
        len(X),
        len(feature_columns),
    )

    if shap_values.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected SHAP shape {shap_values.shape}; "
            f"expected {expected_shape}."
        )

    base_value = float(
        np.asarray(explainer.expected_value)
        .reshape(-1)[0]
    )

    rows = []

    for row_number, (household_id, row) in enumerate(
        X.iterrows()
    ):
        for feature_number, feature in enumerate(feature_columns):
            contribution = float(
                shap_values[row_number, feature_number]
            )

            rows.append({
                "household_ID": str(household_id),
                "feature": feature,
                "feature_value": float(row[feature]),
                "shap_value": contribution,
                "absolute_shap_value": abs(contribution),
                "direction": (
                    "Pushes estimate higher"
                    if contribution > 0
                    else "Pushes estimate lower"
                    if contribution < 0
                    else "No material push"
                ),
            })

    return pd.DataFrame(rows), base_value

def _render_carbon_tool(carbon_path: Path | None) -> None:
    _tool_header(
        "Smart meter",
        "Electricity and CO₂ analysis",
        "Calculate electricity and electricity-related CO₂ from observed valid intervals. "
        "If model eligibility is met, the app also runs the temporal-shape regression and "
        "high-emission screening models.",
    )

    interval_demo, register_demo = _build_synthetic_smartmeter_demo()
    a, b = st.columns(2)
    with a:
        st.download_button(
            "Download eligible interval-kWh demo CSV",
            interval_demo.to_csv(index=False).encode("utf-8"),
            file_name="smartmeter_interval_kwh_DEMO_35days.csv",
            mime="text/csv",
            key="carbon_interval_template",
        )
    with b:
        st.download_button(
            "Download eligible cumulative-register demo CSV",
            register_demo.to_csv(index=False).encode("utf-8"),
            file_name="smartmeter_cumulative_register_DEMO_35days.csv",
            mime="text/csv",
            key="carbon_register_template",
        )
    st.caption("Demo files are synthetic and are provided only to demonstrate the required schema.")

    confirm_non_export = st.checkbox(
        "Interval-kWh upload only: I confirm the household has no active solar/export.",
        key="carbon_non_export",
    )

    upload = st.file_uploader(
        "Smart-meter CSV",
        type=["csv"],
        key="carbon_upload",
    )
    if upload is None:
        return

    # Stage 1: input validation and interval reconstruction. This stage is shared
    # by direct accounting and the ML models.
    try:
        raw = _read_csv(upload, "Smart-meter CSV")
        intervals, source_quality, input_description = _prepare_smartmeter_upload(
            raw,
            confirm_non_export=confirm_non_export,
        )
    except Exception as error:
        st.error(str(error))
        return

    # Stage 2: deterministic electricity/CO2 accounting. This remains usable even
    # if the household cannot construct all model features.
    try:
        direct_summary, complete_months = _calculate_uploaded_electricity_carbon(intervals)
    except Exception as error:
        st.error("Electricity/CO₂ calculation failed: " + str(error))
        return

    st.markdown("### Calculated from uploaded electricity")
    if direct_summary.empty:
        st.warning(
            "No day met the minimum 90-valid-interval coverage rule, so a direct "
            "valid-day electricity/CO₂ summary cannot be reported."
        )
    elif len(direct_summary) == 1:
        row = direct_summary.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Observed electricity across model-valid days",
            f"{float(row['electricity_on_valid_days_kwh']):.1f} kWh",
        )
        c2.metric(
            "Calculated electricity-related CO₂",
            f"{float(row['co2_on_valid_days_kg']):.1f} kg",
        )
        c3.metric("Model-valid days", f"{int(row['valid_days_used'])}")

        complete_count = int(row.get("complete_calendar_months", 0) or 0)
        if complete_count > 0 and pd.notna(row.get("mean_complete_month_co2_kg")):
            st.info(
                f"Complete calendar months: {complete_count}. Mean complete-month "
                f"electricity = {float(row['mean_complete_month_kwh']):.1f} kWh; "
                f"mean complete-month electricity-related CO₂ = "
                f"{float(row['mean_complete_month_co2_kg']):.1f} kg."
            )
        else:
            st.caption(
                "No fully observed calendar month contains all 96 valid intervals on "
                "every day, so the app does not invent a complete-month accounting value."
            )
    else:
        st.dataframe(direct_summary, use_container_width=True, hide_index=True)

    st.caption(
        "Direct CO₂ = observed electricity × 0.71 kg CO₂/kWh. The factor is the fixed "
        "study scenario, not an exact time-varying grid-intensity measurement. A model-"
        "valid day can contain 90–96 valid intervals; no missing-interval energy is imputed."
    )

    if not direct_summary.empty:
        st.download_button(
            "Download direct electricity/CO₂ summary",
            direct_summary.to_csv(index=False).encode("utf-8"),
            file_name="direct_electricity_co2_summary.csv",
            mime="text/csv",
            key="carbon_direct_download",
        )

    # Stage 3: model feature construction. Failure here must not invalidate the
    # direct accounting result already shown above.
    try:
        features, coverage, holiday_source = _build_carbon_features(intervals)
        audit = coverage.merge(
            source_quality,
            on="household_ID",
            how="left",
            validate="one_to_one",
        )
    except Exception as error:
        st.warning(
            "Direct accounting completed, but the behavioural carbon models cannot be "
            "run for this upload: " + str(error)
        )
        with st.expander("Input-quality details"):
            st.caption("Detected input: " + input_description)
            st.dataframe(source_quality, use_container_width=True, hide_index=True)
        return

    date_min = intervals["meter_dt"].min()
    date_max = intervals["meter_dt"].max()
    outside_model_period = bool(
        pd.notna(date_min)
        and pd.notna(date_max)
        and (
            date_min < pd.Timestamp("2024-05-01")
            or date_max >= pd.Timestamp("2024-12-01")
        )
    )
    if outside_model_period:
        st.warning(
            "The behavioural carbon models were developed using May–November 2024 data. "
            "This upload is outside that development period, so the model outputs should "
            "be interpreted cautiously. The direct electricity calculation above is not a model prediction."
        )

    if carbon_path is None:
        st.warning(
            "The carbon model bundle is not available, so only direct accounting can be "
            "shown. Place FINAL_carbon_xai_model_bundle.joblib beside the app or in its final model folder."
        )
        with st.expander("Input-quality and engineered-feature details"):
            st.dataframe(features.reset_index(), use_container_width=True, hide_index=True)
            st.dataframe(audit, use_container_width=True, hide_index=True)
        return

    try:
        bundle = _load_joblib(str(carbon_path))
        required_keys = {
            "feature_columns",
            "regression_deployment_pipeline",
            "classification_deployment_pipeline",
            "classification_decision_threshold",
        }
        if not isinstance(bundle, dict):
            raise ValueError("Carbon artefact is not the expected model bundle.")
        missing_keys = sorted(required_keys - set(bundle))
        if missing_keys:
            raise ValueError(
                "Carbon model bundle is missing keys: " + ", ".join(missing_keys)
            )

        bundle_features = list(bundle["feature_columns"])
        if bundle_features != CARBON_FEATURES:
            raise ValueError(
                "Carbon model feature schema does not match the final five-feature "
                f"contract. Bundle features: {bundle_features}"
            )

        X = features[bundle_features].copy().replace([np.inf, -np.inf], np.nan)
        if X.isna().any().any():
            raise ValueError("A required carbon predictor is missing after feature construction.")

        regression = bundle["regression_deployment_pipeline"]
        classification = bundle["classification_deployment_pipeline"]
        shape_based_estimate = np.clip(
            np.asarray(regression.predict(X), dtype=float),
            0.0,
            None,
        )
        high_scores = _safe_positive_score(classification, X)
        decision_threshold = float(bundle["classification_decision_threshold"])
        high_labels = (high_scores >= decision_threshold).astype(int)

        ml_results = pd.DataFrame({
            "household_ID": X.index.astype(str),
            "estimated_mean_monthly_electricity_related_co2_kg": shape_based_estimate,
            "study_relative_high_emission_screening": np.where(
                high_labels == 1,
                "High",
                "Not high",
            ),
            "high_emission_model_score": high_scores,
            "decision_threshold": decision_threshold,
        })
        ood_audit = _carbon_feature_ood_audit(X, bundle)
    except Exception as error:
        st.error("Carbon-model analysis failed: " + str(error))
        return

    st.markdown("### Behavioural research-model outputs")

    ood_household_ids: set[str] = set()
    if not ood_audit.empty and ood_audit["outside_study_central_95pct"].any():
        ood_household_ids = set(
            ood_audit.loc[
                ood_audit["outside_study_central_95pct"],
                "household_ID",
            ].astype(str)
        )
        affected = len(ood_household_ids)
        st.warning(
            f"{affected} household(s) contain behavioural features outside the central "
            "95% range of the study carbon-model cohort. As a conservative application "
            "safeguard, ML regression and high-emission screening outputs are withheld for "
            "those households. This does not change the trained models or the thesis analysis."
        )

    display_ml_results = ml_results.loc[
        ~ml_results["household_ID"].astype(str).isin(ood_household_ids)
    ].copy()

    if display_ml_results.empty:
        st.info(
            "No behavioural ML output is reported because the uploaded household profile "
            "falls outside the study model's central 95% behavioural-feature range. "
            "The direct electricity and CO₂ calculation above remains available."
        )
    elif len(display_ml_results) == 1:
        row = display_ml_results.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Estimated mean monthly electricity-related CO₂",
            f"{float(row['estimated_mean_monthly_electricity_related_co2_kg']):.1f} kg/month",
        )
        m2.metric(
            "High-emission screening (study threshold)",
            str(row["study_relative_high_emission_screening"]),
        )
        m3.metric(
            "High-emission model score",
            f"{float(row['high_emission_model_score']):.3f}",
        )
    else:
        st.dataframe(display_ml_results, use_container_width=True, hide_index=True)

    if not display_ml_results.empty:
        st.caption(
            "The regression uses only the five temporal load-shape predictors. High/Not high "
            "uses the threshold defined during model development. The classifier score is not "
            "a calibrated probability. These models estimate the study outcome for the uploaded "
            "profile; they do not forecast future emissions or define a national emissions category."
        )

        # ------------------------------------------------------------
        # Local SHAP explanation of the carbon regression prediction
        # ------------------------------------------------------------
        st.markdown("#### Why this CO₂ estimate?")

        try:
            # Explain only households for which the ML output was NOT
            # withheld by the OOD safeguard.
            explain_ids = (
                display_ml_results["household_ID"]
                .astype(str)
                .tolist()
            )

            X_explain = X.copy()
            X_explain.index = X_explain.index.astype(str)

            X_explain = X_explain.loc[
                explain_ids,
                bundle_features,
            ].copy()

            shap_rows, shap_base_value = (
                _carbon_regression_shap(
                    regression,
                    X_explain,
                    bundle_features,
                )
            )

            feature_labels = {
                "evening_peak_ratio": "Evening peak ratio",
                "night_baseload_ratio": "Night baseload ratio",
                "load_uniformity": "Load uniformity",
                "load_factor": "Load factor",
                "leisure_usage_ratio": "Leisure usage ratio",
            }

            shap_rows["feature_label"] = (
                shap_rows["feature"]
                .map(feature_labels)
                .fillna(shap_rows["feature"])
            )

            # --------------------------------------------------------
            # Single-household explanation
            # --------------------------------------------------------
            if len(X_explain) == 1:
                household_id = str(X_explain.index[0])

                local_shap = (
                    shap_rows.loc[
                        shap_rows["household_ID"].eq(
                            household_id
                        )
                    ]
                    .sort_values(
                        "shap_value",
                        ascending=True,
                    )
                    .copy()
                )

                shap_chart = px.bar(
                    local_shap,
                    x="shap_value",
                    y="feature_label",
                    orientation="h",
                    hover_data={
                        "feature_value": ":.4f",
                        "shap_value": ":.3f",
                    },
                    labels={
                        "shap_value":
                            "SHAP contribution to CO₂ estimate",
                        "feature_label":
                            "Temporal feature",
                    },
                    title="Local SHAP explanation",
                )

                shap_chart.add_vline(
                    x=0,
                    line_dash="dash",
                )

                shap_chart.update_layout(
                    height=350,
                    showlegend=False,
                )

                st.plotly_chart(
                    shap_chart,
                    use_container_width=True,
                )

                # Raw prediction before the application's non-negative clip
                raw_prediction = float(
                    regression.predict(X_explain)[0]
                )

                reconstructed_prediction = float(
                    shap_base_value
                    + local_shap["shap_value"].sum()
                )

                st.caption(
                    f"SHAP baseline = {shap_base_value:.1f}; "
                    f"SHAP-reconstructed model output = "
                    f"{reconstructed_prediction:.1f} kg CO₂/month; "
                    f"raw model output = "
                    f"{raw_prediction:.1f} kg CO₂/month."
                )

                explanation_table = (
                    local_shap
                    .sort_values(
                        "absolute_shap_value",
                        ascending=False,
                    )
                    [
                        [
                            "feature_label",
                            "feature_value",
                            "shap_value",
                            "direction",
                        ]
                    ]
                    .rename(
                        columns={
                            "feature_label": "Feature",
                            "feature_value": "Value",
                            "shap_value": "SHAP contribution",
                            "direction": "Effect on prediction",
                        }
                    )
                )

                st.dataframe(
                    explanation_table,
                    use_container_width=True,
                    hide_index=True,
                )

                if raw_prediction < 0:
                    st.caption(
                        "The application clips negative regression "
                        "predictions to zero. SHAP explains the raw "
                        "model output before that clipping rule."
                    )

            # --------------------------------------------------------
            # Multiple-household upload
            # --------------------------------------------------------
            else:
                top_shap = (
                    shap_rows
                    .sort_values(
                        [
                            "household_ID",
                            "absolute_shap_value",
                        ],
                        ascending=[True, False],
                    )
                    .groupby(
                        "household_ID",
                        as_index=False,
                    )
                    .head(3)
                )

                st.dataframe(
                    top_shap[
                        [
                            "household_ID",
                            "feature_label",
                            "feature_value",
                            "shap_value",
                            "direction",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "Positive SHAP values push the model's CO₂ estimate "
                "higher; negative values push it lower. Larger absolute "
                "values indicate stronger influence on this prediction. "
                "SHAP explains the fitted model and does not establish causation."
            )

        except ImportError:
            st.info(
                "SHAP explanation is unavailable because the "
                "'shap' package is not installed."
            )

        except Exception as error:
            st.warning(
                "SHAP explanation could not be generated: "
                + str(error)
            )

        st.download_button(
            "Download carbon-model results",
            display_ml_results.to_csv(index=False).encode("utf-8"),
            file_name="carbon_research_model_results.csv",
            mime="text/csv",
            key="carbon_model_download",
        )

    with st.expander("Model and data details"):
        st.markdown("**Engineered temporal features**")
        feature_display = features[CARBON_FEATURES + ["log_daily_kwh"]].reset_index()
        st.dataframe(feature_display, use_container_width=True, hide_index=True)

        if not ood_audit.empty:
            st.markdown("**Study-range check**")
            st.dataframe(ood_audit, use_container_width=True, hide_index=True)

        if not complete_months.empty:
            st.markdown("**Direct complete-calendar-month calculations**")
            st.dataframe(complete_months, use_container_width=True, hide_index=True)

        st.markdown("**Coverage and input-quality checks**")
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.caption("Calendar handling: " + holiday_source)
        st.caption("Detected input: " + input_description)
        st.info(
            "The regression and high-emission classifier are two representations of the "
            "same electricity-derived outcome and are not independent confirmation. Direct "
            "CO₂ calculated from this upload is a separate deterministic accounting summary."
        )


def _render_survey_tool(
    high_use_path: Path | None,
    reference_path: Path | None,
) -> None:
    _tool_header(
        "Household survey",
        "High-consumption screening",
        "Enter household, dwelling and appliance characteristics. The model identifies "
        "whether the profile falls in the study sample's upper one-third consumption class. "
        "The similar-household comparison is optional and descriptive.",
    )

    if high_use_path is None:
        st.error(
            "High-consumption model not found. Place "
            "PHASE3_final_high_consumption_model.joblib beside the app or in its final model folder."
        )
        return

    try:
        package = _load_joblib(str(high_use_path))
    except Exception as error:
        st.error("Could not load the high-consumption model: " + str(error))
        return

    if not isinstance(package, dict):
        st.error("The high-consumption artefact is not the expected model package.")
        return

    required = {
        "model",
        "feature_columns",
        "decision_threshold",
        "binary_features",
        "categorical_features",
    }
    missing = sorted(required - set(package))
    if missing:
        st.error("High-consumption model package is missing: " + ", ".join(missing))
        return

    reference = None
    if reference_path is not None:
        try:
            reference = _load_reference_csv(str(reference_path))
        except Exception:
            reference = None

    defaults = _survey_defaults(reference)
    categorical_features = list(package.get("categorical_features", []))
    fitted_categories = _extract_encoder_categories(
        package["model"],
        categorical_features,
    )

    house_options = fitted_categories.get("house_type", [])
    meter_options = fitted_categories.get("meter_type", [])

    if not house_options and reference is not None and "house_type" in reference.columns:
        house_options = sorted(reference["house_type"].dropna().astype(str).unique().tolist())
    if not meter_options and reference is not None and "meter_type" in reference.columns:
        meter_options = sorted(reference["meter_type"].dropna().astype(str).unique().tolist())
    if not house_options:
        house_options = ["Unknown"]
    if not meter_options:
        meter_options = ["Unknown"]

    with st.form("survey_household_form"):
        a, b, c, d = st.columns(4)
        with a:
            household_size = st.number_input(
                "Household members",
                min_value=1,
                max_value=30,
                value=max(1, int(defaults["household_size"])),
                step=1,
            )
            room_count = st.number_input(
                "Rooms",
                min_value=1,
                max_value=50,
                value=max(1, int(defaults["room_count"])),
                step=1,
            )
            meter_count = st.number_input(
                "Electricity meters",
                min_value=1,
                max_value=10,
                value=max(1, int(defaults["electricity_meter_count"])),
                step=1,
            )
        with b:
            appliance_count = st.number_input(
                "Total appliances",
                min_value=0,
                max_value=150,
                value=max(0, int(defaults["appliance_total_count"])),
                step=1,
            )
            appliance_type_count = st.number_input(
                "Appliance types",
                min_value=0,
                max_value=60,
                value=max(0, int(defaults["appliance_type_count"])),
                step=1,
            )
            fan_count = st.number_input(
                "Fans",
                min_value=0,
                max_value=50,
                value=max(0, int(defaults["fan_total_count"])),
                step=1,
            )
        with c:
            light_count = st.number_input(
                "Lights",
                min_value=0,
                max_value=200,
                value=max(0, int(defaults["light_total_count"])),
                step=1,
            )
            ac_count = st.number_input(
                "Air conditioners",
                min_value=0,
                max_value=20,
                value=max(0, int(defaults["ac_total_count"])),
                step=1,
            )
            house_type = st.selectbox("House type", house_options)
        with d:
            meter_type = st.selectbox("Meter type", meter_options)
            has_business = st.checkbox("Business at household")
            has_backup_generator = st.checkbox("Backup generator")

        e, f, g = st.columns(3)
        with e:
            grid_cooking = st.checkbox("Electric cooking")
        with f:
            gas_cooking = st.checkbox("Gas cooking", value=True)
        with g:
            firewood_cooking = st.checkbox("Firewood cooking")

        submitted = st.form_submit_button(
            "Analyse household",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    input_row = pd.DataFrame([{
        "household_size": household_size,
        "room_count": room_count,
        "appliance_total_count": appliance_count,
        "appliance_type_count": appliance_type_count,
        "fan_total_count": fan_count,
        "light_total_count": light_count,
        "ac_total_count": ac_count,
        "electricity_meter_count": meter_count,
        "has_ac": int(ac_count > 0),
        "has_business": int(has_business),
        "has_backup_generator": int(has_backup_generator),
        "grid_cooking": int(grid_cooking),
        "gas_cooking": int(gas_cooking),
        "firewood_cooking": int(firewood_cooking),
        "house_type": str(house_type),
        "meter_type": str(meter_type),
    }])

    if reference is not None:
        numeric_profile_columns = [
            "household_size",
            "room_count",
            "appliance_total_count",
            "appliance_type_count",
            "fan_total_count",
            "light_total_count",
            "ac_total_count",
            "electricity_meter_count",
        ]
        outside_reference = []
        for column in numeric_profile_columns:
            if column in reference.columns and column in input_row.columns:
                ref_values = pd.to_numeric(reference[column], errors="coerce").dropna()
                if not ref_values.empty:
                    value = float(input_row[column].iloc[0])
                    if value < float(ref_values.min()) or value > float(ref_values.max()):
                        outside_reference.append(column)
        if outside_reference:
            st.warning(
                "Some entered values fall outside the observed survey reference range: "
                + ", ".join(outside_reference)
                + ". The model can still produce a score, but this is an extrapolative case."
            )

    feature_columns = list(package["feature_columns"])
    missing_input = [feature for feature in feature_columns if feature not in input_row.columns]
    if missing_input:
        st.error("Input form is missing model features: " + ", ".join(missing_input))
        return

    X = input_row[feature_columns]
    try:
        score = float(_safe_positive_score(package["model"], X)[0])
        threshold = float(package["decision_threshold"])
        predicted_high = bool(score >= threshold)
    except Exception as error:
        st.error("Survey model prediction failed: " + str(error))
        return

    r1, r2 = st.columns(2)
    r1.metric(
        "High-consumption screening (study upper third)",
        "High" if predicted_high else "Not high",
    )
    r2.metric("Model score", f"{score:.3f}")
    st.caption(
        "The target is the study-sample upper one-third of observed mean monthly "
        "electricity consumption. This is a screening result, not a national consumption "
        "class, tariff category, causal result or future-consumption forecast."
    )

    result_row = input_row.copy()
    result_row.insert(0, "study_relative_high_consumption_screening", "High" if predicted_high else "Not high")
    result_row.insert(1, "model_score", score)
    result_row.insert(2, "decision_threshold", threshold)
    if "consumption_target_threshold_kwh" in package:
        result_row.insert(
            3,
            "training_sample_high_consumption_cutoff_kwh",
            float(package["consumption_target_threshold_kwh"]),
        )
    st.download_button(
        "Download screening result",
        result_row.to_csv(index=False).encode("utf-8"),
        file_name="survey_high_consumption_screening_result.csv",
        mime="text/csv",
        key="survey_result_download",
    )

    with st.expander("Optional exploratory similar-household reference"):
        st.caption(
            "This descriptive lookup is intentionally separated from the fitted ML model."
        )
        if reference is None:
            st.info(
                "Survey reference dataset was not found, so the exploratory reference cannot be shown."
            )
        else:
            try:
                similar, benchmark = _survey_reference_benchmark(
                    reference,
                    input_row,
                    neighbours=30,
                )
                s1, s2, s3 = st.columns(3)
                s1.metric("Nearest observed reference households", int(benchmark["similar_households"]))
                s2.metric("Neighbour median monthly use", f"{benchmark['median_monthly_kwh']:.1f} kWh")
                s3.metric("High reference-group share", f"{benchmark['high_group_share'] * 100:.1f}%")

                if "consumption_tertile" in similar.columns:
                    counts = (
                        similar["consumption_tertile"]
                        .astype(str)
                        .value_counts()
                        .reindex(["Low", "Medium", "High"], fill_value=0)
                        .rename_axis("Consumption group")
                        .reset_index(name="Households")
                    )
                    chart = px.bar(
                        counts,
                        x="Consumption group",
                        y="Households",
                        text="Households",
                        title="Observed consumption-group mix among nearest study households",
                    )
                    chart.update_layout(height=300, showlegend=False)
                    st.plotly_chart(chart, use_container_width=True)

                st.caption(
                    "Nearest households are selected using standardized feature distance "
                    "within the observed study sample. This is not a fitted predictive model, "
                    "and the distance has not been validated as a measure of true household similarity."
                )
            except Exception as error:
                st.warning("Exploratory reference unavailable: " + str(error))




def _render_decision_tools(
    archetype_path: Path | None,
    carbon_path: Path | None,
    high_use_path: Path | None,
    reference_path: Path | None,
) -> None:
    st.sidebar.markdown("#### Analysis tools")
    selected = st.sidebar.radio(
        "Select tool",
        [
            "Behavioural archetype",
            "Billing label transfer",
            "Monthly consumption",
            "Electricity & CO₂",
            "High-consumption screening",
        ],
        label_visibility="collapsed",
        key="primary_navigation",
    )

    st.sidebar.markdown("---")

    if selected == "Behavioural archetype":
        _render_kmeans_tool()
    elif selected == "Billing label transfer":
        _render_archetype_tool(archetype_path)
    elif selected == "Monthly consumption":
        _render_monthly_consumption_profiler(reference_path)
    elif selected == "Electricity & CO₂":
        _render_carbon_tool(carbon_path)
    else:
        _render_survey_tool(high_use_path, reference_path)






def _render_status(artefacts: dict[str, Path | None]) -> None:
    st.sidebar.markdown(
        """
        <div class="side-brand">
            <p class="side-brand-title">Residential Electricity</p>
            <p class="side-brand-copy">Thesis research application</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = {
        "kmeans_raw": "KMeans raw features",
        "kmeans_labels": "KMeans study labels",
        "archetype": "Billing model",
        "carbon": "Carbon model",
        "high_use": "High-consumption model",
        "survey_reference": "Survey reference",
    }

    missing = [
        label
        for key, label in labels.items()
        if artefacts.get(key) is None or not artefacts[key].is_file()
    ]

    if missing:
        st.sidebar.markdown(
            f"""
            <div class="readiness-warn">
                <span class="status-dot"></span>
                {len(missing)} required file{'s' if len(missing) != 1 else ''} missing
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            """
            <div class="readiness-ok">
                <span class="status-dot"></span>
                Models ready
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar.expander("Model files", expanded=False):
        for key, label in labels.items():
            path = artefacts.get(key)
            state = "Ready" if path is not None and path.is_file() else "Missing"
            st.write(f"{label} — {state}")

    with st.sidebar.expander("Input guide", expanded=False):
        st.markdown(
            "**Smart meter**  \n"
            "15-minute interval-kWh or cumulative-register CSV; at least 30 model-valid days."
        )
        st.markdown(
            "**Billing**  \n"
            "`household_ID`, `month`, `consumption`; 15–25 valid months."
        )
        st.markdown(
            "**Monthly consumption**  \n"
            "3–12 months; study group only for 12 consecutive months."
        )
        st.markdown(
            "**Survey**  \n"
            "Enter household and appliance details in the form."
        )

    st.sidebar.caption(
        "Research application · archetype and screening labels refer to the study sample; "
        "they are not official national household categories."
    )





def main() -> None:
    _configure_page()

    archetype_path = _first_existing_path(ARCHETYPE_MODEL_CANDIDATES)
    carbon_path = _first_existing_path(CARBON_MODEL_CANDIDATES)
    high_use_path = _first_existing_path(HIGH_USE_MODEL_CANDIDATES)
    reference_path = _first_existing_path(SURVEY_REFERENCE_CANDIDATES)
    kmeans_raw_path = _first_existing_path(KMEANS_RAW_FEATURES_CANDIDATES)
    kmeans_labels_path = _first_existing_path(KMEANS_LOCKED_LABELS_CANDIDATES)

    artefacts = {
        "kmeans_raw": kmeans_raw_path,
        "kmeans_labels": kmeans_labels_path,
        "archetype": archetype_path,
        "carbon": carbon_path,
        "high_use": high_use_path,
        "survey_reference": reference_path,
    }

    _render_status(artefacts)
    _render_app_header()

    _render_decision_tools(
        archetype_path=archetype_path,
        carbon_path=carbon_path,
        high_use_path=high_use_path,
        reference_path=reference_path,
    )




if __name__ == "__main__":
    main()
