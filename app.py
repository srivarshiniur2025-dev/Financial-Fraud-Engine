"""
app.py
------
Premium fintech Streamlit UI for the Financial Fraud Intelligence Engine.
Prediction and CSV logic are unchanged — UI/UX only.
Run with: streamlit run app.py
"""

import io
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model import load_saved_model, predict_transaction
from utils import (
    DEMO_DATA_PATH,
    DATA_PATH,
    calculate_risk_score,
    get_dataset_summary,
)

DEMO_DATASET_BANNER = (
    "Demo dataset loaded. Upload the full Kaggle dataset for complete analysis."
)
UPLOAD_BYTES_KEY = "creditcard_upload_bytes"

# Must match the feature order used when training the Random Forest in model.py
FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]

PROJECT_TITLE = "Financial Fraud Intelligence Engine"
FULL_NAME = "Srivarshini U R"
REGISTERED_EMAIL = "srivarshini.ur2025@vitstudent.ac.in"
PROJECT_TOPIC = "AI-powered Credit Card Fraud Detection"
PROJECT_SUBTITLE = (
    "AI-powered fraud detection and transaction risk analysis using Machine Learning."
)

PRIMARY = "#B77466"
ACCENT = "#FFE1AF"
SECONDARY = "#E2B59A"
DARK_NEUTRAL = "#957C62"
BACKGROUND = "#FFFFFF"
LIGHT_BG = "#F8F7F5"
PRIMARY_TEXT = "#2C2C2C"
SUCCESS = "#22C55E"
DANGER = "#DC2626"

PAGES = [
    "🏠 Home",
    "🔍 Fraud Detection",
    "📂 Batch Analysis",
    "📊 Dashboard",
    "ℹ About",
]

st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'DM Sans', sans-serif;
        }}
        .stApp {{
            background: {BACKGROUND};
            color: {PRIMARY_TEXT};
        }}
        .block-container {{
            padding-top: 1.75rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }}
        [data-testid="stSidebar"] {{
            background: {LIGHT_BG};
            border-right: 1px solid {SECONDARY};
        }}
        [data-testid="stSidebar"] .stRadio > div {{
            gap: 0.35rem;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            background: {BACKGROUND};
            border: 1px solid transparent;
            border-radius: 12px;
            padding: 0.65rem 0.85rem !important;
            transition: all 0.2s ease;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            border-color: {SECONDARY};
            box-shadow: 0 4px 14px rgba(149, 124, 98, 0.12);
            transform: translateY(-1px);
        }}
        [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div,
        [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
            background: linear-gradient(135deg, {PRIMARY}, {DARK_NEUTRAL}) !important;
            color: white !important;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            background: linear-gradient(90deg, {PRIMARY}22, {ACCENT}55) !important;
            border: 1px solid {PRIMARY} !important;
            box-shadow: 0 4px 16px rgba(183, 116, 102, 0.18);
        }}
        h1, h2, h3 {{
            color: {PRIMARY_TEXT};
            letter-spacing: -0.02em;
        }}
        .premium-hero {{
            background: linear-gradient(135deg, {PRIMARY} 0%, {DARK_NEUTRAL} 100%);
            border-radius: 24px;
            padding: 2.75rem 2.25rem;
            box-shadow: 0 16px 40px rgba(183, 116, 102, 0.28);
            margin-bottom: 1.75rem;
            position: relative;
            overflow: hidden;
        }}
        .premium-hero::after {{
            content: "";
            position: absolute;
            right: -40px;
            top: -40px;
            width: 180px;
            height: 180px;
            background: {ACCENT}33;
            border-radius: 50%;
        }}
        .premium-hero h1 {{
            color: #fff !important;
            font-size: 2.45rem;
            font-weight: 700;
            margin: 0 0 0.6rem 0;
            position: relative;
            z-index: 1;
        }}
        .premium-hero p {{
            color: {ACCENT};
            font-size: 1.12rem;
            margin: 0;
            max-width: 720px;
            position: relative;
            z-index: 1;
        }}
        .feature-card, .kpi-rich, .metric-card, .panel-card, .chart-card, .info-card {{
            background: {LIGHT_BG};
            border: 1px solid {SECONDARY};
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(149, 124, 98, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .feature-card:hover, .kpi-rich:hover, .metric-card:hover, .panel-card:hover, .chart-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 14px 32px rgba(149, 124, 98, 0.14);
        }}
        .feature-card {{
            padding: 1.35rem 1.2rem;
            min-height: 150px;
            height: 100%;
        }}
        .feature-card .icon {{ font-size: 1.5rem; margin-bottom: 0.55rem; }}
        .feature-card h4 {{
            color: {PRIMARY};
            margin: 0 0 0.4rem 0;
            font-size: 1.05rem;
        }}
        .feature-card p {{
            color: {DARK_NEUTRAL};
            margin: 0;
            font-size: 0.92rem;
            line-height: 1.45;
        }}
        .kpi-rich {{
            padding: 1.2rem 1.1rem;
            text-align: left;
            height: 100%;
        }}
        .kpi-rich .icon {{ font-size: 1.35rem; margin-bottom: 0.45rem; }}
        .kpi-rich .value {{
            font-size: 1.7rem;
            font-weight: 700;
            color: {PRIMARY_TEXT};
            line-height: 1.2;
        }}
        .kpi-rich .label {{
            color: {PRIMARY};
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 0.2rem;
        }}
        .kpi-rich .desc {{
            color: {DARK_NEUTRAL};
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }}
        .page-header h1 {{
            font-size: 2rem;
            margin-bottom: 0.35rem;
        }}
        .page-header p {{
            color: {DARK_NEUTRAL};
            margin: 0 0 1.25rem 0;
            font-size: 1.02rem;
        }}
        .section-divider {{
            height: 1px;
            background: linear-gradient(90deg, {SECONDARY}, transparent);
            margin: 1.5rem 0;
            border: none;
        }}
        .info-card {{
            padding: 1rem 1.2rem;
            border-left: 5px solid {PRIMARY};
            background: linear-gradient(90deg, {ACCENT}55, {LIGHT_BG});
            color: {PRIMARY_TEXT};
            margin: 0.75rem 0 1.1rem 0;
        }}
        .panel-card {{
            padding: 1.35rem 1.35rem 1.1rem 1.35rem;
            margin-bottom: 1rem;
        }}
        .panel-card h3 {{
            margin: 0 0 0.85rem 0;
            color: {PRIMARY_TEXT};
            font-size: 1.15rem;
        }}
        .status-genuine {{
            background: #ECFDF5;
            border: 2px solid {SUCCESS};
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 10px 28px rgba(34, 197, 94, 0.12);
        }}
        .status-fraud {{
            background: #FEF2F2;
            border: 2px solid {DANGER};
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 10px 28px rgba(220, 38, 38, 0.12);
        }}
        .status-genuine .badge {{
            display: inline-block;
            background: {SUCCESS};
            color: white;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
        }}
        .status-fraud .badge {{
            display: inline-block;
            background: {DANGER};
            color: white;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
        }}
        .status-genuine h2 {{ color: {SUCCESS}; margin: 0.2rem 0; font-size: 1.55rem; }}
        .status-fraud h2 {{ color: {DANGER}; margin: 0.2rem 0; font-size: 1.55rem; }}
        .status-genuine p, .status-fraud p {{
            margin: 0.35rem 0 0 0;
            color: {PRIMARY_TEXT};
        }}
        .metric-card {{
            padding: 1.1rem 1rem;
            text-align: center;
            height: 100%;
        }}
        .metric-card .icon {{ font-size: 1.3rem; margin-bottom: 0.35rem; }}
        .metric-card .value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {PRIMARY_TEXT};
        }}
        .metric-card .subtitle {{
            color: {DARK_NEUTRAL};
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }}
        .recommend-card {{
            background: {LIGHT_BG};
            border: 1px solid {SECONDARY};
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            margin-top: 1rem;
            box-shadow: 0 8px 22px rgba(149, 124, 98, 0.08);
        }}
        .recommend-card h4 {{
            margin: 0 0 0.4rem 0;
            color: {PRIMARY};
        }}
        .upload-zone {{
            background: {LIGHT_BG};
            border: 2px dashed {SECONDARY};
            border-radius: 20px;
            padding: 2rem 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: inset 0 0 0 6px {BACKGROUND};
        }}
        .upload-zone .icon {{ font-size: 2.4rem; }}
        .upload-zone h3 {{ margin: 0.5rem 0 0.25rem 0; color: {PRIMARY_TEXT}; }}
        .upload-zone p {{ margin: 0; color: {DARK_NEUTRAL}; }}
        .chart-card {{
            padding: 1rem 1rem 0.5rem 1rem;
            margin-bottom: 1rem;
        }}
        .chart-card h4 {{
            margin: 0 0 0.5rem 0.25rem;
            color: {PRIMARY_TEXT};
            font-size: 1rem;
        }}
        .sidebar-brand {{
            padding: 0.4rem 0.2rem 1rem 0.2rem;
        }}
        .sidebar-brand .title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {PRIMARY};
            line-height: 1.3;
        }}
        .sidebar-brand .caption {{
            color: {DARK_NEUTRAL};
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }}
        .sidebar-meta {{
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 14px;
            background: {BACKGROUND};
            border: 1px solid {SECONDARY};
            font-size: 0.82rem;
            color: {DARK_NEUTRAL};
            line-height: 1.55;
        }}
        .sidebar-meta strong {{ color: {PRIMARY_TEXT}; }}
        .detail-card {{
            background: {LIGHT_BG};
            border: 1px solid {SECONDARY};
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 8px 24px rgba(149, 124, 98, 0.08);
            margin-bottom: 1rem;
        }}
        .stButton > button {{
            border-radius: 12px !important;
            font-weight: 600;
            transition: all 0.2s ease;
            border: 1px solid {SECONDARY};
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(183, 116, 102, 0.2);
            border-color: {PRIMARY};
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {PRIMARY}, {DARK_NEUTRAL});
            border: none;
            color: white;
        }}
        /* Demo sample loader button colors (adjacent sibling pattern) */
        div.random-btn + div button {{
            background: {PRIMARY} !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
        }}
        div.genuine-btn + div button {{
            background: {SUCCESS} !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
        }}
        div.fraud-btn + div button {{
            background: {DANGER} !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
        }}
        .sample-meta-card {{
            background: {LIGHT_BG};
            border: 1px solid {SECONDARY};
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 8px 22px rgba(149, 124, 98, 0.08);
            margin: 0.85rem 0 1.1rem 0;
        }}
        .sample-meta-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
        }}
        .sample-meta-item .label {{
            color: {DARK_NEUTRAL};
            font-size: 0.78rem;
            margin-bottom: 0.2rem;
        }}
        .sample-meta-item .value {{
            color: {PRIMARY_TEXT};
            font-weight: 700;
            font-size: 1.02rem;
        }}
        .sample-badge-fraud {{
            display: inline-block;
            background: {DANGER};
            color: white;
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            font-weight: 700;
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }}
        .sample-badge-genuine {{
            display: inline-block;
            background: {SUCCESS};
            color: white;
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            font-weight: 700;
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }}
        div[data-testid="stFileUploader"] {{
            background: {LIGHT_BG};
            border-radius: 16px;
            padding: 0.75rem;
            border: 1px solid {SECONDARY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Dataset loading — single entry point used by every page
# ---------------------------------------------------------------------------
@st.cache_data
def _read_csv_cached(file_bytes: bytes):
    """Cache a CSV loaded from raw bytes (uploaded or otherwise)."""
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data
def _read_csv_from_path(path: str):
    """Cache a CSV loaded from a filesystem path."""
    return pd.read_csv(path)


def _offer_full_dataset_upload(uploader_key):
    """Optional uploader to replace the demo/full disk dataset for this session."""
    uploaded = st.file_uploader(
        "Upload full creditcard.csv (optional)",
        type=["csv"],
        key=uploader_key,
    )
    if uploaded is not None:
        try:
            file_bytes = uploaded.getvalue()
            _read_csv_cached(file_bytes)
            st.session_state[UPLOAD_BYTES_KEY] = file_bytes
            get_dashboard_score_sample.clear()
            st.rerun()
        except Exception as error:
            st.error(f"Could not load dataset: {error}")


def get_dataset(show_uploader=True, uploader_key="dataset_uploader"):
    """
    Return the credit-card fraud dataset, or None if nothing is available.

    Priority:
    1. Uploaded creditcard.csv (session, cached via @st.cache_data)
    2. data/creditcard.csv if present
    3. data/demo_creditcard.csv (bundled demo — always preferred over crashing)

    Never raises FileNotFoundError.
    """
    uploaded_bytes = st.session_state.get(UPLOAD_BYTES_KEY)
    if uploaded_bytes:
        return _read_csv_cached(uploaded_bytes)

    if Path(DATA_PATH).is_file():
        return _read_csv_from_path(DATA_PATH)

    if Path(DEMO_DATA_PATH).is_file():
        if show_uploader:
            st.info(DEMO_DATASET_BANNER)
            _offer_full_dataset_upload(uploader_key)
        return _read_csv_from_path(DEMO_DATA_PATH)

    if show_uploader:
        st.warning(
            "Dataset not found.\nPlease upload creditcard.csv using the uploader below."
        )
        _offer_full_dataset_upload(uploader_key)

    return None


def hour_to_dataset_time(hour):
    return float(hour) * 3600.0


def sample_transaction(mode="random"):
    """
    Load a complete real transaction from the active dataset.
    mode: "random" | "genuine" | "fraud"
    Returns None if the dataset is unavailable.
    """
    df = get_dataset(show_uploader=False)
    if df is None:
        return None

    if mode == "genuine":
        subset = df[df["Class"] == 0]
        sample_type = "Genuine"
    elif mode == "fraud":
        subset = df[df["Class"] == 1]
        sample_type = "Fraud"
    else:
        subset = df
        sample_type = "Random"

    if subset.empty:
        subset = df
        sample_type = "Random"

    row = subset.sample(1).iloc[0]
    original_time = float(row["Time"])
    original_amount = float(row["Amount"])
    original_class = int(row["Class"])

    return {
        "sample_type": sample_type,
        "v_features": {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)},
        "amount": float(min(max(original_amount, 0.0), 30000.0)),
        "hour": int((original_time % 86400) // 3600),
        "original_time": original_time,
        "original_amount": original_amount,
        "row_index": int(row.name),
        "original_class": original_class,
        "full_row": row.to_dict(),
    }


def load_sample_into_session(mode="random"):
    """Store a complete sample in session state for demo prediction."""
    sample = sample_transaction(mode=mode)
    if sample is None:
        return False
    st.session_state["sample_type"] = sample["sample_type"]
    st.session_state["sample_v_features"] = sample["v_features"]
    st.session_state["sample_row_index"] = sample["row_index"]
    st.session_state["sample_original_class"] = sample["original_class"]
    st.session_state["sample_original_time"] = sample["original_time"]
    st.session_state["sample_original_amount"] = sample["original_amount"]
    st.session_state["sample_original_hour"] = sample["hour"]
    st.session_state["sample_full_row"] = sample["full_row"]
    st.session_state["amount_input"] = sample["amount"]
    st.session_state["hour_input"] = sample["hour"]
    # Clear previous prediction when a new sample is loaded
    st.session_state.pop("last_demo_result", None)
    return True


def ensure_sample_loaded():
    if (
        "sample_v_features" not in st.session_state
        or "sample_row_index" not in st.session_state
        or "sample_original_class" not in st.session_state
        or "sample_type" not in st.session_state
    ):
        load_sample_into_session(mode="random")


def build_transaction(hour, amount_value, v_features):
    transaction = {"Time": hour_to_dataset_time(hour)}
    transaction.update(v_features)
    transaction["Amount"] = float(amount_value)
    return transaction


def process_batch_dataframe(upload_df):
    missing = [col for col in FEATURE_COLUMNS if col not in upload_df.columns]
    if missing:
        return None, missing

    model = load_saved_model()
    features = upload_df[FEATURE_COLUMNS]
    predictions = model.predict(features)
    fraud_probabilities = model.predict_proba(features)[:, 1]

    result = upload_df.copy()
    result["Prediction"] = ["Fraud" if p == 1 else "Genuine" for p in predictions]
    result["Fraud Probability"] = [round(float(p) * 100, 2) for p in fraud_probabilities]
    result["Risk Score"] = [calculate_risk_score(p) for p in fraud_probabilities]
    return result, []


@st.cache_data
def get_dashboard_score_sample(_df, sample_size=3000, random_state=42):
    sample = _df.sample(n=min(sample_size, len(_df)), random_state=random_state)
    scored, missing = process_batch_dataframe(sample)
    if missing:
        return None
    return scored


def append_recent_prediction(amount, hour, prediction, risk_score):
    if "recent_predictions" not in st.session_state:
        st.session_state["recent_predictions"] = []
    entry = {
        "Time": f"{int(hour):02d}:00",
        "Amount": round(float(amount), 2),
        "Prediction": prediction,
        "Risk Score": risk_score,
        "Logged At": datetime.now().strftime("%H:%M:%S"),
    }
    history = st.session_state["recent_predictions"]
    history.insert(0, entry)
    st.session_state["recent_predictions"] = history[:10]


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rich_kpi(icon, label, value, description):
    st.markdown(
        f"""
        <div class="kpi-rich">
            <div class="icon">{icon}</div>
            <div class="value">{value}</div>
            <div class="label">{label}</div>
            <div class="desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(icon, title, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="icon">{icon}</div>
            <div class="value">{value}</div>
            <div class="subtitle">{title} · {subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axes(ax):
    ax.set_facecolor(LIGHT_BG)
    ax.figure.patch.set_facecolor(BACKGROUND)
    ax.tick_params(colors=PRIMARY_TEXT)
    ax.xaxis.label.set_color(PRIMARY_TEXT)
    ax.yaxis.label.set_color(PRIMARY_TEXT)
    ax.title.set_color(PRIMARY_TEXT)
    for spine in ax.spines.values():
        spine.set_color(SECONDARY)


def chart_fraud_vs_genuine(genuine_count, fraud_count):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(
        ["Genuine", "Fraud"],
        [genuine_count, fraud_count],
        color=[SUCCESS, DANGER],
        edgecolor="white",
        width=0.55,
    )
    ax.set_ylabel("Count")
    style_axes(ax)
    fig.tight_layout()
    return fig


def chart_amount_distribution(amounts):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(amounts, bins=40, color=PRIMARY, edgecolor="white")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    style_axes(ax)
    fig.tight_layout()
    return fig


def chart_probability_distribution(probabilities):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(probabilities, bins=30, color=SECONDARY, edgecolor="white")
    ax.set_xlabel("Fraud Probability (%)")
    ax.set_ylabel("Frequency")
    style_axes(ax)
    fig.tight_layout()
    return fig


def chart_risk_distribution(risk_scores):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(risk_scores, bins=30, color=DARK_NEUTRAL, edgecolor="white")
    ax.set_xlabel("Risk Score (/100)")
    ax.set_ylabel("Frequency")
    style_axes(ax)
    fig.tight_layout()
    return fig


def render_chart_card(title, figure):
    st.markdown(f'<div class="chart-card"><h4>{title}</h4>', unsafe_allow_html=True)
    st.pyplot(figure, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def home_page():
    st.markdown(
        f"""
        <div class="premium-hero">
            <h1>{PROJECT_TITLE}</h1>
            <p>{PROJECT_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        ("🛡", "Fraud Detection", "Score individual transactions with AI risk analysis."),
        ("📂", "Batch CSV Analysis", "Upload and analyze many transactions at once."),
        ("📈", "Interactive Dashboard", "Explore KPIs and fraud intelligence charts."),
        ("⚡", "Risk Scoring", "View fraud probability and risk score out of 100."),
    ]
    cols = st.columns(4)
    for col, (icon, title, text) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="icon">{icon}</div>
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

    df = get_dataset(uploader_key="home_dataset_upload")
    if df is None:
        return

    summary = get_dataset_summary(df)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        rich_kpi("💳", "Total Transactions", f"{summary['num_rows']:,}", "Records in the Kaggle dataset")
    with k2:
        rich_kpi("⚠", "Fraud Cases", f"{summary['fraudulent_transactions']:,}", "Confirmed fraudulent labels")
    with k3:
        rich_kpi("🎯", "Model Accuracy", "99.95%", "Random Forest test-set accuracy")
    with k4:
        rich_kpi("⚡", "Risk Engine", "Active", "Live scoring with fraud_model.pkl")

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="detail-card">
            <p><strong>Project Topic:</strong> {PROJECT_TOPIC}</p>
            <p><strong>Full Name:</strong> {FULL_NAME}</p>
            <p style="margin-bottom:0;"><strong>Registered Email:</strong> {REGISTERED_EMAIL}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_results(prediction, fraud_probability, risk_score, confidence):
    if prediction == "Fraud":
        st.markdown(
            """
            <div class="status-fraud">
                <div class="badge">⚠ HIGH RISK</div>
                <h2>⚠ Fraud Detected</h2>
                <p>Immediate review recommended.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        recommendation = "⚠ Hold the transaction and perform manual verification."
    else:
        st.markdown(
            """
            <div class="status-genuine">
                <div class="badge">✅ SAFE</div>
                <h2>✅ Genuine Transaction</h2>
                <p>This transaction appears safe.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        recommendation = "✔ Transaction can be processed safely."

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("🛡", "Prediction", prediction, "Model decision")
    with m2:
        metric_card("📈", "Fraud Probability", f"{fraud_probability * 100:.2f}%", "Class 1 probability")
    with m3:
        metric_card("⚠", "Risk Score", f"{risk_score} / 100", "Scaled risk")
    with m4:
        metric_card("🎯", "Confidence", f"{confidence * 100:.2f}%", "Decision confidence")

    st.progress(min(max(confidence, 0.0), 1.0))

    st.markdown(
        f"""
        <div class="recommend-card">
            <h4>Recommended Action</h4>
            <p style="margin:0; color:{PRIMARY_TEXT};">{recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_predictions():
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.markdown("### Recent Predictions")
    history = st.session_state.get("recent_predictions", [])
    if not history:
        st.caption("No predictions yet. Run a demo prediction to populate this table.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def render_demo_prediction():
    st.markdown(
        """
        <div class="info-card">
            This demonstration uses real transactions from the Kaggle Credit Card
            Fraud Detection dataset. Hidden anonymized features (V1–V28) are loaded
            automatically from the selected sample while you may modify the
            transaction amount and hour before prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if get_dataset(uploader_key="demo_dataset_upload") is None:
        return

    ensure_sample_loaded()

    st.markdown("#### Load Sample from Dataset")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown('<div class="random-btn"></div>', unsafe_allow_html=True)
        if st.button("🎲 Load Random Sample", use_container_width=True, key="load_random_btn"):
            load_sample_into_session(mode="random")
            st.rerun()
    with b2:
        st.markdown('<div class="genuine-btn"></div>', unsafe_allow_html=True)
        if st.button("✅ Load Genuine Sample", use_container_width=True, key="load_genuine_btn"):
            load_sample_into_session(mode="genuine")
            st.rerun()
    with b3:
        st.markdown('<div class="fraud-btn"></div>', unsafe_allow_html=True)
        if st.button("🚨 Load Fraud Sample", use_container_width=True, key="load_fraud_btn"):
            load_sample_into_session(mode="fraud")
            st.rerun()

    sample_type = st.session_state.get("sample_type", "Random")
    original_class = int(st.session_state.get("sample_original_class", 0))
    original_amount = float(st.session_state.get("sample_original_amount", 0.0))
    original_time = float(st.session_state.get("sample_original_time", 0.0))
    class_label = "Fraud" if original_class == 1 else "Genuine"

    if original_class == 1:
        badge_html = '<div class="sample-badge-fraud">🚨 Real Fraud Sample Loaded</div>'
    else:
        badge_html = '<div class="sample-badge-genuine">✅ Genuine Sample Loaded</div>'

    st.markdown(
        f"""
        <div class="sample-meta-card">
            {badge_html}
            <div class="sample-meta-grid">
                <div class="sample-meta-item">
                    <div class="label">Sample Type</div>
                    <div class="value">{sample_type}</div>
                </div>
                <div class="sample-meta-item">
                    <div class="label">Original Dataset Class</div>
                    <div class="value">{class_label}</div>
                </div>
                <div class="sample-meta-item">
                    <div class="label">Dataset Row Index</div>
                    <div class="value">{st.session_state.get("sample_row_index", "—")}</div>
                </div>
                <div class="sample-meta-item">
                    <div class="label">Original Amount</div>
                    <div class="value">{original_amount:,.2f}</div>
                </div>
                <div class="sample-meta-item">
                    <div class="label">Original Time</div>
                    <div class="value">{original_time:,.1f}s</div>
                </div>
                <div class="sample-meta-item">
                    <div class="label">Original Hour</div>
                    <div class="value">{int(st.session_state.get("sample_original_hour", 0)):02d}:00</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel-card"><h3>Transaction Details</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    amount_value = col1.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        max_value=30000.0,
        step=10.0,
        format="%.2f",
        help="Editable amount. V1–V28 stay from the loaded sample.",
        key="amount_input",
    )
    hour_value = col2.number_input(
        "🕒 Transaction Hour (0–23)",
        min_value=0,
        max_value=23,
        step=1,
        help="Editable hour. Converted to Time in seconds; V1–V28 stay unchanged.",
        key="hour_input",
    )

    cur1, cur2 = st.columns(2)
    cur1.caption(f"Current Amount: **{float(amount_value):,.2f}**")
    cur2.caption(
        f"Current Time: **{hour_to_dataset_time(hour_value):,.1f}s** "
        f"(from hour {int(hour_value)})"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⚡ Predict Transaction", type="primary", key="demo_predict_btn"):
        try:
            # Complete vector: sampled V1–V28 + user Amount/Time only
            transaction = build_transaction(
                hour_value,
                amount_value,
                st.session_state["sample_v_features"],
            )
            result = predict_transaction(transaction)

            prediction = result["prediction"]
            fraud_probability = result["fraud_probability"]
            risk_score = result["risk_score"]
            confidence = (
                fraud_probability if prediction == "Fraud" else 1.0 - fraud_probability
            )

            append_recent_prediction(amount_value, hour_value, prediction, risk_score)
            st.session_state["last_demo_result"] = {
                "prediction": prediction,
                "fraud_probability": fraud_probability,
                "risk_score": risk_score,
                "confidence": confidence,
            }
        except FileNotFoundError:
            st.error(
                "Model file `fraud_model.pkl` not found. "
                "Train once with: `python model.py`"
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    last = st.session_state.get("last_demo_result")
    if last:
        render_prediction_results(
            last["prediction"],
            last["fraud_probability"],
            last["risk_score"],
            last["confidence"],
        )

    render_recent_predictions()


def render_batch_analysis(key_prefix="batch"):
    st.markdown(
        """
        <div class="upload-zone">
            <div class="icon">📂</div>
            <h3>Upload Transaction CSV</h3>
            <p>Drag and drop a file, or browse to upload. Required model feature columns must be present.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional: unlock shared dataset for pages that need it; batch scoring itself
    # uses the transaction CSV uploaded below and does not require creditcard.csv.
    get_dataset(show_uploader=False)

    with st.expander("Required columns"):
        st.code(", ".join(FEATURE_COLUMNS))

    uploaded = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key=f"{key_prefix}_uploader",
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.caption("Waiting for a CSV file…")
        return

    try:
        upload_df = _read_csv_cached(uploaded.getvalue())
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        return

    st.markdown('<div class="panel-card"><h3>File Preview</h3>', unsafe_allow_html=True)
    st.dataframe(upload_df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("▶ Run Batch Prediction", type="primary", key=f"{key_prefix}_run"):
        with st.spinner("Scoring transactions…"):
            result_df, missing = process_batch_dataframe(upload_df)

        if missing:
            st.error("Missing required columns:")
            st.write(missing)
            return

        st.session_state[f"{key_prefix}_results"] = result_df

    result_df = st.session_state.get(f"{key_prefix}_results")
    if result_df is None:
        return

    total = len(result_df)
    fraud_n = int((result_df["Prediction"] == "Fraud").sum())
    genuine_n = int((result_df["Prediction"] == "Genuine").sum())
    fraud_pct = (fraud_n / total * 100) if total else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        rich_kpi("💳", "Total Transactions", f"{total:,}", "Rows scored in this file")
    with k2:
        rich_kpi("⚠", "Fraud Detected", f"{fraud_n:,}", "Predicted as Fraud")
    with k3:
        rich_kpi("✅", "Genuine Transactions", f"{genuine_n:,}", "Predicted as Genuine")
    with k4:
        rich_kpi("📈", "Fraud Percentage", f"{fraud_pct:.2f}%", "Share of fraud predictions")

    st.write("")
    st.markdown('<div class="panel-card"><h3>Processed Results</h3>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇ Download Results",
        data=csv_buffer.getvalue(),
        file_name="fraud_analysis_results.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def fraud_detection_page():
    page_header(
        "🔍 Fraud Detection",
        "Analyze a financial transaction using an AI-powered fraud detection model.",
    )
    mode = st.radio(
        "Mode",
        ["Demo Prediction", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    if mode == "Demo Prediction":
        render_demo_prediction()
    else:
        render_batch_analysis(key_prefix="fraud_upload")


def batch_analysis_page():
    page_header(
        "📂 Batch Analysis",
        "Upload a CSV and score every transaction with the trained Random Forest model.",
    )
    render_batch_analysis(key_prefix="batch_page")


def dashboard_page():
    page_header(
        "📊 Dashboard",
        "Professional analytics overview of dataset volume and model risk signals.",
    )

    df = get_dataset(uploader_key="dashboard_dataset_upload")
    if df is None:
        return

    summary = get_dataset_summary(df)
    total = summary["num_rows"]
    genuine = summary["genuine_transactions"]
    fraud = summary["fraudulent_transactions"]
    fraud_rate = (fraud / total * 100) if total else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        rich_kpi("💳", "Total Transactions", f"{total:,}", "Full dataset volume")
    with k2:
        rich_kpi("⚠", "Fraud Transactions", f"{fraud:,}", "Labeled fraud cases")
    with k3:
        rich_kpi("✅", "Genuine Transactions", f"{genuine:,}", "Labeled genuine cases")
    with k4:
        rich_kpi("📈", "Fraud Rate", f"{fraud_rate:.3f}%", "Fraud share of dataset")

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    scored = get_dashboard_score_sample(df)

    c1, c2 = st.columns(2)
    with c1:
        render_chart_card("Fraud vs Genuine", chart_fraud_vs_genuine(genuine, fraud))
    with c2:
        render_chart_card("Amount Distribution", chart_amount_distribution(df["Amount"]))

    if scored is not None:
        c3, c4 = st.columns(2)
        with c3:
            render_chart_card(
                "Fraud Probability Distribution",
                chart_probability_distribution(scored["Fraud Probability"]),
            )
        with c4:
            render_chart_card(
                "Risk Score Distribution",
                chart_risk_distribution(scored["Risk Score"]),
            )
    else:
        st.warning("Could not compute probability charts. Ensure fraud_model.pkl exists.")

    with st.expander("Dataset snapshot"):
        st.dataframe(df.head(10), use_container_width=True)


def about_page():
    page_header(
        "ℹ About",
        "Internship-ready overview of the Financial Fraud Intelligence Engine.",
    )

    st.markdown(
        f"""
        <div class="detail-card">
            <h3 style="color:{PRIMARY}; margin-top:0;">Project Overview</h3>
            <p style="color:{PRIMARY_TEXT}; margin-bottom:0;">
                A premium analytics interface for detecting fraudulent credit card
                transactions with a Random Forest model trained on the Kaggle
                Credit Card Fraud Detection dataset.
            </p>
        </div>
        <div class="detail-card">
            <h3 style="color:{PRIMARY}; margin-top:0;">Purpose</h3>
            <p style="color:{PRIMARY_TEXT}; margin-bottom:0;">
                Demonstrate end-to-end fraud intelligence: single-transaction demos,
                batch CSV scoring, risk recommendations, and an interactive dashboard
                suitable for internship presentations.
            </p>
        </div>
        <div class="detail-card">
            <h3 style="color:{PRIMARY}; margin-top:0;">Machine Learning Model</h3>
            <p style="color:{PRIMARY_TEXT}; margin-bottom:0;">
                Random Forest Classifier loaded from <code>fraud_model.pkl</code>.
                Predictions return class labels, fraud probability, and risk score.
            </p>
        </div>
        <div class="detail-card">
            <h3 style="color:{PRIMARY}; margin-top:0;">Dataset</h3>
            <p style="color:{PRIMARY_TEXT}; margin-bottom:0;">
                Kaggle Credit Card Fraud Detection Dataset — anonymized features
                V1–V28, Time, Amount, and Class labels.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Technologies Used")
    tech_cols = st.columns(3)
    technologies = [
        ("🐍", "Python"),
        ("🐼", "Pandas"),
        ("🧠", "Scikit-learn"),
        ("🌐", "Streamlit"),
        ("📊", "Matplotlib"),
        ("🌲", "Random Forest"),
    ]
    for idx, (icon, tech) in enumerate(technologies):
        with tech_cols[idx % 3]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="icon">{icon}</div>
                    <h4>{tech}</h4>
                    <p>Part of the fraud intelligence stack.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="detail-card">
            <h3 style="color:{PRIMARY}; margin-top:0;">Developer Information</h3>
            <p><strong>Name:</strong> {FULL_NAME}</p>
            <p><strong>Email:</strong> {REGISTERED_EMAIL}</p>
            <p style="margin-bottom:0;"><strong>Project Type:</strong> Internship Project</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
inject_styles()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="caption">PREMIUM ANALYTICS</div>
            <div class="title">🛡 Fraud Intelligence Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown(
        """
        <div class="sidebar-meta">
            ------------------------<br/>
            <strong>Model</strong><br/>Random Forest<br/><br/>
            <strong>Dataset</strong><br/>Kaggle Credit Card Fraud<br/><br/>
            <strong>Version</strong><br/>1.0<br/>
            ------------------------
        </div>
        """,
        unsafe_allow_html=True,
    )

if page.endswith("Home"):
    home_page()
elif page.endswith("Fraud Detection"):
    fraud_detection_page()
elif page.endswith("Batch Analysis"):
    batch_analysis_page()
elif page.endswith("Dashboard"):
    dashboard_page()
else:
    about_page()
