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

from model import MODEL_UNAVAILABLE_MSG, get_model as resolve_model, predict_transaction
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
    "AI-powered fraud detection and transaction risk intelligence using Machine Learning."
)

# ColorHunt palette (light mode): https://colorhunt.co/palette/3e3232503c3c7e6363a87c7c
INK = "#3E3232"
GRAPHITE = "#503C3C"
TEAL = "#7E6363"
MIST = "#F7F4F4"

DEEP_NAVY = INK
SLATE_GRAY = GRAPHITE
TERRACOTTA = TEAL
WARM_SAND = "#A87C7C"
PRIMARY = INK
PRIMARY_HOVER = GRAPHITE
SECONDARY = TEAL
ACCENT = WARM_SAND
BACKGROUND = MIST
CARD_BG = "#FFFFFF"
BORDER = "rgba(62, 50, 50, 0.14)"
PRIMARY_TEXT = INK
SECONDARY_TEXT = GRAPHITE
SUCCESS = "#16A34A"
SUCCESS_SOFT = "#15803D"
DANGER = "#DC2626"
DANGER_SOFT = "#B91C1C"
INFO = "#F0EAEA"
GRID = "#E8E0E0"
LIGHT_BG = CARD_BG
DARK_NEUTRAL = GRAPHITE
SURFACE_ELEVATED = "#F5EEEE"
HERO_START = INK
HERO_END = GRAPHITE

PAGES = [
    "Home",
    "Fraud Detection",
    "Batch Analysis",
    "Dashboard",
    "About",
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
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fillBar {{
            from {{ width: 0 !important; }}
        }}

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}
        .stApp {{
            background: {BACKGROUND};
            color: {SECONDARY_TEXT};
        }}
        .stApp, .stApp [data-testid="stAppViewContainer"] {{
            background: {BACKGROUND} !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(247, 244, 244, 0.9) !important;
        }}
        .block-container {{
            padding-top: 1.75rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }}
        h1, h2, h3, h4 {{
            color: {PRIMARY_TEXT} !important;
        }}
        p, label, .stMarkdown, .stCaption, span {{
            color: {SECONDARY_TEXT};
        }}
        [data-testid="stMetricValue"], [data-testid="stWidgetLabel"] label {{
            color: {PRIMARY_TEXT} !important;
        }}
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 14px;
        }}
        .stExpander {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 14px;
        }}
        hr {{
            border-color: {BORDER} !important;
        }}

        /* ---------- Sidebar / Professional Nav ---------- */
        [data-testid="stSidebar"] {{
            background: {INK} !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.25rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }}
        [data-testid="stSidebar"] * {{
            color: rgba(255, 255, 255, 0.88);
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: rgba(255, 255, 255, 0.78) !important;
        }}

        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.35rem 0.35rem 1.35rem 0.35rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 1.15rem;
        }}
        .sidebar-brand .logo-mark {{
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: {WARM_SAND};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 800;
            color: #fff !important;
            flex-shrink: 0;
        }}
        .sidebar-brand .brand-copy {{
            min-width: 0;
        }}
        .sidebar-brand .caption {{
            color: {WARM_SAND} !important;
            font-size: 0.65rem;
            letter-spacing: 0.14em;
            margin: 0 0 0.2rem 0;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .sidebar-brand .title {{
            font-size: 0.98rem;
            font-weight: 700;
            color: #fff !important;
            line-height: 1.25;
            margin: 0;
            letter-spacing: -0.01em;
        }}

        .nav-section-label {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.42) !important;
            margin: 0 0 0.55rem 0.45rem;
        }}

        [data-testid="stSidebar"] .stRadio > div {{
            gap: 0.3rem;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.72rem 0.9rem 0.72rem 0.85rem !important;
            transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
            color: rgba(255, 255, 255, 0.82) !important;
            position: relative;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(255, 255, 255, 0.06) !important;
            border-color: rgba(255, 255, 255, 0.06) !important;
        }}
        [data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio label span {{
            color: rgba(255, 255, 255, 0.86) !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.01em;
        }}
        /* Hide default radio circles for a cleaner nav look */
        [data-testid="stSidebar"] .stRadio label > div:first-child {{
            display: none !important;
        }}
        div[role="radiogroup"] label:has(input:checked) {{
            background: rgba(168, 124, 124, 0.16) !important;
            border: 1px solid rgba(168, 124, 124, 0.4) !important;
            box-shadow: inset 3px 0 0 {WARM_SAND};
        }}
        div[role="radiogroup"] label:has(input:checked) p,
        div[role="radiogroup"] label:has(input:checked) span {{
            color: #fff !important;
            font-weight: 700 !important;
        }}

        .sidebar-meta {{
            margin-top: 1.75rem;
            padding: 0.95rem 1rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.78rem;
            color: rgba(255, 255, 255, 0.7) !important;
            line-height: 1.5;
        }}
        .sidebar-meta .meta-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            padding: 0.42rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }}
        .sidebar-meta .meta-row:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}
        .sidebar-meta .meta-row:first-child {{
            padding-top: 0;
        }}
        .sidebar-meta .meta-key {{
            color: rgba(255, 255, 255, 0.42) !important;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.66rem;
        }}
        .sidebar-meta .meta-val {{
            color: #fff !important;
            font-weight: 600;
            text-align: right;
        }}
        .sidebar-meta .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(168, 124, 124, 0.18);
            color: {WARM_SAND} !important;
            border: 1px solid rgba(168, 124, 124, 0.35);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }}
        .sidebar-meta .status-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: {WARM_SAND};
        }}
        .sidebar-footer {{
            margin-top: 0.85rem;
            padding: 0 0.35rem;
            font-size: 0.66rem;
            color: rgba(255, 255, 255, 0.32) !important;
            letter-spacing: 0.04em;
        }}

        .section-label {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {TEAL} !important;
            margin: 0.15rem 0 0.4rem 0;
        }}
        .section-title {{
            font-size: 1.3rem;
            font-weight: 750;
            color: {INK} !important;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.02em;
        }}
        .section-desc {{
            color: {GRAPHITE} !important;
            font-size: 0.94rem;
            margin: 0 0 1.1rem 0;
            line-height: 1.5;
        }}

        h1, h2, h3, h4 {{
            color: {PRIMARY_TEXT};
            letter-spacing: -0.02em;
            font-weight: 700;
        }}
        p, label, .stMarkdown {{
            color: {SECONDARY_TEXT};
        }}

        /* ---------- Hero ---------- */
        .premium-hero {{
            background: {INK};
            border-radius: 20px;
            padding: 2.4rem 2.5rem 2.2rem 2.5rem;
            box-shadow: 0 12px 32px rgba(62, 50, 50, 0.14);
            border: 1px solid rgba(62, 50, 50, 0.2);
            margin-bottom: 1.85rem;
            position: relative;
            overflow: hidden;
            animation: fadeIn 0.45s ease both;
        }}
        .premium-hero::before {{
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: {WARM_SAND};
        }}
        .premium-hero .hero-content {{
            position: relative;
            z-index: 2;
            max-width: 760px;
            padding-left: 0.35rem;
        }}
        .premium-hero .hero-eyebrow {{
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {WARM_SAND} !important;
            margin-bottom: 0.9rem;
        }}
        .premium-hero .hero-title,
        .premium-hero h1.hero-title,
        div[data-testid="stMarkdownContainer"] .premium-hero .hero-title,
        div[data-testid="stMarkdownContainer"] .premium-hero h1 {{
            color: #FFFFFF !important;
            font-size: clamp(1.9rem, 3vw, 2.45rem);
            font-weight: 800;
            margin: 0 0 0.7rem 0;
            letter-spacing: -0.03em;
            line-height: 1.18;
        }}
        .premium-hero .hero-subtitle,
        .premium-hero p.hero-subtitle,
        div[data-testid="stMarkdownContainer"] .premium-hero p {{
            color: rgba(255, 255, 255, 0.82) !important;
            font-size: 1.02rem;
            margin: 0 0 1.25rem 0;
            max-width: 620px;
            line-height: 1.6;
            font-weight: 400;
        }}
        .premium-hero .hero-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .premium-hero .hero-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.38rem 0.72rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: rgba(255, 255, 255, 0.92) !important;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .premium-hero .hero-chip .dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: {WARM_SAND};
        }}

        /* ---------- Cards ---------- */
        .feature-card, .kpi-rich, .metric-card, .panel-card, .chart-card,
        .info-card, .detail-card, .sample-meta-card, .recommend-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 14px;
            box-shadow: 0 4px 16px rgba(62, 50, 50, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            animation: fadeIn 0.45s ease both;
        }}
        .feature-card:hover, .kpi-rich:hover, .metric-card:hover,
        .panel-card:hover, .chart-card:hover, .detail-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(62, 50, 50, 0.1);
            border-color: rgba(168, 124, 124, 0.3);
        }}
        .feature-card {{
            padding: 1.35rem 1.2rem;
            min-height: 148px;
            height: 100%;
        }}
        .feature-card .icon {{
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            color: {WARM_SAND} !important;
            margin-bottom: 0.65rem;
        }}
        .feature-card h4 {{
            color: {PRIMARY_TEXT};
            margin: 0 0 0.4rem 0;
            font-size: 1.02rem;
            font-weight: 700;
        }}
        .feature-card p {{
            color: {SECONDARY_TEXT};
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.5;
        }}

        /* ---------- Metric / KPI cards ---------- */
        .kpi-rich {{
            padding: 1.25rem 1.15rem;
            text-align: left;
            height: 100%;
        }}
        .kpi-rich .icon {{
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            margin-bottom: 0.55rem;
            color: {TEAL};
        }}
        .kpi-rich .value {{
            font-size: 1.7rem;
            font-weight: 800;
            color: {PRIMARY_TEXT};
            line-height: 1.2;
        }}
        .kpi-rich .label {{
            color: {PRIMARY_TEXT};
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 0.2rem;
        }}
        .kpi-rich .desc {{
            color: {SECONDARY_TEXT};
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }}
        .kpi-rich.kpi-total .icon {{ color: {INK}; }}
        .kpi-rich.kpi-fraud .icon,
        .kpi-rich.kpi-fraud .value {{ color: {TEAL}; }}
        .kpi-rich.kpi-genuine .icon,
        .kpi-rich.kpi-genuine .value {{ color: {SUCCESS}; }}
        .kpi-rich.kpi-rate .icon,
        .kpi-rich.kpi-rate .value {{ color: {WARM_SAND}; }}

        .page-header {{
            margin-bottom: 0.35rem;
        }}
        .page-header h1 {{
            font-size: 1.85rem;
            margin-bottom: 0.35rem;
            color: {PRIMARY_TEXT};
            letter-spacing: -0.02em;
        }}
        .page-header p {{
            color: {SECONDARY_TEXT};
            margin: 0 0 1.25rem 0;
            font-size: 1rem;
            line-height: 1.5;
        }}
        .section-divider {{
            height: 1px;
            background: {BORDER};
            margin: 1.5rem 0;
            border: none;
        }}
        .info-card {{
            padding: 1rem 1.25rem;
            border-left: 4px solid {WARM_SAND};
            background: {INFO};
            border: 1px solid {BORDER};
            border-left: 4px solid {WARM_SAND};
            border-radius: 12px;
            color: {SECONDARY_TEXT};
            margin: 0.75rem 0 1.1rem 0;
            font-size: 0.94rem;
            line-height: 1.55;
        }}
        .mode-row {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }}
        .tech-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.55rem 0.95rem;
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 10px;
            color: {INK} !important;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0.25rem 0.35rem 0.25rem 0;
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

        /* ---------- Prediction status ---------- */
        .status-genuine {{
            background: {SUCCESS};
            border: none;
            border-radius: 16px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 10px 24px rgba(22, 163, 74, 0.18);
            color: #fff;
            animation: fadeIn 0.45s ease both;
        }}
        .status-fraud {{
            background: {DANGER};
            border: none;
            border-radius: 16px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 10px 24px rgba(220, 38, 38, 0.18);
            color: #fff;
            animation: fadeIn 0.45s ease both;
        }}
        .status-genuine .badge,
        .status-fraud .badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.22);
            color: white;
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
            letter-spacing: 0.04em;
        }}
        .status-genuine h2,
        .status-fraud h2 {{
            color: #fff !important;
            margin: 0.2rem 0;
            font-size: 1.55rem;
        }}
        .status-genuine p,
        .status-fraud p {{
            margin: 0.35rem 0 0 0;
            color: rgba(255, 255, 255, 0.92) !important;
        }}

        .metric-card {{
            padding: 1.15rem 1rem;
            text-align: left;
            height: 100%;
        }}
        .metric-card .value {{
            font-size: 1.4rem;
            font-weight: 800;
            color: {PRIMARY_TEXT};
        }}
        .metric-card .subtitle {{
            color: {SECONDARY_TEXT};
            font-size: 0.8rem;
            margin-top: 0.35rem;
            letter-spacing: 0.01em;
        }}

        .risk-meter, .prob-meter {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1rem 1.15rem;
            margin: 0.65rem 0;
            box-shadow: 0 8px 22px rgba(62, 50, 50, 0.06);
            animation: fadeIn 0.5s ease both;
        }}
        .risk-meter .meter-label,
        .prob-meter .meter-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.88rem;
            font-weight: 600;
            color: {PRIMARY_TEXT};
            margin-bottom: 0.55rem;
        }}
        .risk-meter .track,
        .prob-meter .track {{
            height: 12px;
            border-radius: 999px;
            background: {GRID};
            overflow: hidden;
        }}
        .risk-meter .fill {{
            height: 100%;
            border-radius: 999px;
            background: {WARM_SAND};
            animation: fillBar 0.85s ease;
        }}
        .prob-meter .fill {{
            height: 100%;
            border-radius: 999px;
            background: {INK};
            animation: fillBar 0.85s ease;
        }}

        .recommend-card {{
            padding: 1.1rem 1.25rem;
            margin-top: 1rem;
        }}
        .recommend-card h4 {{
            margin: 0 0 0.4rem 0;
            color: {PRIMARY_TEXT};
        }}

        /* ---------- Upload ---------- */
        .upload-zone {{
            background: {CARD_BG};
            border: 1px dashed rgba(168, 124, 124, 0.55);
            border-radius: 14px;
            padding: 1.75rem 1.5rem;
            text-align: left;
            margin-bottom: 1rem;
            transition: background 0.2s ease, border-color 0.2s ease;
        }}
        .upload-zone:hover {{
            background: {SURFACE_ELEVATED};
            border-color: {TEAL};
        }}
        .upload-zone h3 {{ margin: 0 0 0.35rem 0; color: {PRIMARY_TEXT}; font-size: 1.05rem; }}
        .upload-zone p {{ margin: 0; color: {SECONDARY_TEXT}; font-size: 0.92rem; }}

        .chart-card {{
            padding: 1rem 1rem 0.5rem 1rem;
            margin-bottom: 1rem;
        }}
        .chart-card h4 {{
            margin: 0 0 0.5rem 0.25rem;
            color: {PRIMARY_TEXT};
            font-size: 1rem;
        }}

        .detail-card {{
            padding: 1.35rem 1.4rem;
            margin-bottom: 1rem;
        }}

        /* ---------- Buttons & inputs ---------- */
        .stButton > button {{
            border-radius: 12px !important;
            font-weight: 600;
            transition: all 0.2s ease;
            border: 1px solid {BORDER} !important;
            background: {CARD_BG} !important;
            color: {PRIMARY_TEXT} !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px) scale(1.02);
            box-shadow: 0 10px 22px rgba(62, 50, 50, 0.14);
            border-color: {INK} !important;
        }}
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        button[kind="primary"] {{
            background: {INK} !important;
            border: none !important;
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"] *,
        .stButton > button[data-testid="baseButton-primary"] *,
        button[kind="primary"] p,
        button[kind="primary"] span,
        button[data-testid="baseButton-primary"] p,
        button[data-testid="baseButton-primary"] span {{
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {{
            background: {GRAPHITE} !important;
            transform: translateY(-1px) scale(1.02);
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"]:hover *,
        .stButton > button[data-testid="baseButton-primary"]:hover * {{
            color: #FFFFFF !important;
        }}
        div.random-btn + div button {{
            background: {INK} !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
        }}
        div.genuine-btn + div button {{
            background: {TEAL} !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
        }}
        div.fraud-btn + div button {{
            background: {WARM_SAND} !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
        }}

        .stTextInput input, .stNumberInput input, .stSelectbox > div > div {{
            border-radius: 12px !important;
            border: 1px solid {BORDER} !important;
            background: {CARD_BG} !important;
            color: {PRIMARY_TEXT} !important;
        }}
        .stTextInput input:focus, .stNumberInput input:focus {{
            border-color: {INK} !important;
            box-shadow: 0 0 0 2px rgba(62, 50, 50, 0.15) !important;
        }}

        .sample-meta-card {{
            padding: 1.15rem 1.25rem;
            margin: 0.85rem 0 1.1rem 0;
        }}
        .sample-meta-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
        }}
        .sample-meta-item .label {{
            color: {SECONDARY_TEXT};
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
            background: {CARD_BG};
            border-radius: 16px;
            padding: 0.85rem;
            border: 2px dashed {WARM_SAND};
            transition: background 0.2s ease;
        }}
        div[data-testid="stFileUploader"]:hover {{
            background: {SURFACE_ELEVATED};
        }}

        /* Notifications */
        div[data-testid="stNotification"],
        .stAlert {{
            border-radius: 14px !important;
        }}
        div[data-baseweb="notification"] {{
            border-radius: 14px;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: rgba(255, 255, 255, 0.85) !important;
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
            get_model.clear()
            # Allow retrain against the newly uploaded dataset if no pickle yet
            import model as model_module

            model_module._MODEL_CACHE["model"] = None
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


@st.cache_resource
def get_model():
    """
    Streamlit-cached model accessor.

    Loads fraud_model.pkl when present; otherwise trains a RandomForest on the
    currently available dataset (full, uploaded, or demo) and caches the result.
    """
    df = get_dataset(show_uploader=False)
    return resolve_model(df=df)


def process_batch_dataframe(upload_df):
    missing = [col for col in FEATURE_COLUMNS if col not in upload_df.columns]
    if missing:
        return None, missing

    model = get_model()
    if model is None:
        st.warning(MODEL_UNAVAILABLE_MSG)
        return None, ["__MODEL_UNAVAILABLE__"]

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
            <div class="section-label">Workspace</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rich_kpi(label, value, description, variant="default", tag=""):
    variant_class = f" kpi-{variant}" if variant != "default" else ""
    tag_html = f'<div class="icon">{tag}</div>' if tag else ""
    st.markdown(
        f"""
        <div class="kpi-rich{variant_class}">
            {tag_html}
            <div class="value">{value}</div>
            <div class="label">{label}</div>
            <div class="desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="value">{value}</div>
            <div class="subtitle">{title} · {subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axes(ax):
    ax.set_facecolor(CARD_BG)
    ax.figure.patch.set_facecolor(BACKGROUND)
    ax.tick_params(colors=SECONDARY_TEXT)
    ax.xaxis.label.set_color(PRIMARY_TEXT)
    ax.yaxis.label.set_color(PRIMARY_TEXT)
    ax.title.set_color(PRIMARY_TEXT)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#D6CBCB")


def chart_fraud_vs_genuine(genuine_count, fraud_count):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(
        ["Genuine", "Fraud"],
        [genuine_count, fraud_count],
        color=[INK, WARM_SAND],
        edgecolor="white",
        width=0.55,
    )
    ax.set_ylabel("Count")
    style_axes(ax)
    fig.tight_layout()
    return fig


def chart_amount_distribution(amounts):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(amounts, bins=40, color=INK, edgecolor="white")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    style_axes(ax)
    fig.tight_layout()
    return fig


def chart_probability_distribution(probabilities):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(probabilities, bins=30, color=WARM_SAND, edgecolor="white")
    ax.set_xlabel("Fraud Probability (%)")
    ax.set_ylabel("Frequency")
    style_axes(ax)
    fig.tight_layout()
    return fig


def chart_risk_distribution(risk_scores):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(risk_scores, bins=30, color=TEAL, edgecolor="white")
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
            <div class="hero-content">
                <div class="hero-eyebrow">Enterprise Fraud Analytics</div>
                <div class="hero-title">{PROJECT_TITLE}</div>
                <p class="hero-subtitle">{PROJECT_SUBTITLE}</p>
                <div class="hero-meta">
                    <span class="hero-chip"><span class="dot"></span>Random Forest</span>
                    <span class="hero-chip">Real-time Risk Scoring</span>
                    <span class="hero-chip">Batch CSV Analysis</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-label">Capabilities</div>
        <div class="section-title">What you can do</div>
        <p class="section-desc">Core workflows for detecting, scoring, and reviewing credit card fraud.</p>
        """,
        unsafe_allow_html=True,
    )

    features = [
        ("01", "Fraud Detection", "Score a single transaction with live probability and risk."),
        ("02", "Batch Analysis", "Upload a CSV and score every row in one pass."),
        ("03", "Dashboard", "Inspect volume, fraud rate, and model score distributions."),
        ("04", "Risk Scoring", "Convert fraud probability into an actionable 0–100 score."),
    ]
    cols = st.columns(4)
    for col, (num, title, text) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="icon">{num}</div>
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-label">Dataset Overview</div>
        <div class="section-title">Live metrics</div>
        <p class="section-desc">Summary statistics from the active credit-card dataset.</p>
        """,
        unsafe_allow_html=True,
    )

    df = get_dataset(uploader_key="home_dataset_upload")
    if df is None:
        return

    summary = get_dataset_summary(df)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        rich_kpi(
            "Total Transactions",
            f"{summary['num_rows']:,}",
            "Active dataset volume",
            variant="total",
            tag="VOL",
        )
    with k2:
        rich_kpi(
            "Fraud Cases",
            f"{summary['fraudulent_transactions']:,}",
            "Labeled fraud rows",
            variant="fraud",
            tag="FRD",
        )
    with k3:
        rich_kpi(
            "Model Accuracy",
            "99.95%",
            "Random Forest test accuracy",
            variant="genuine",
            tag="ACC",
        )
    with k4:
        rich_kpi(
            "Risk Engine",
            "Active",
            "Model ready for scoring",
            variant="rate",
            tag="ENG",
        )

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-label">Project</div>
        <div class="section-title">Submission details</div>
        <div class="detail-card" style="margin-top:0.85rem;">
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Topic</div>
                    <div style="color:{INK};font-weight:600;">{PROJECT_TOPIC}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Author</div>
                    <div style="color:{INK};font-weight:600;">{FULL_NAME}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Email</div>
                    <div style="color:{INK};font-weight:600;word-break:break-all;">{REGISTERED_EMAIL}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_results(prediction, fraud_probability, risk_score, confidence):
    if prediction == "Fraud":
        st.markdown(
            """
            <div class="status-fraud">
                <div class="badge">HIGH RISK</div>
                <h2>Fraud Detected</h2>
                <p>Immediate review recommended.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        recommendation = "Hold the transaction and perform manual verification."
    else:
        st.markdown(
            """
            <div class="status-genuine">
                <div class="badge">SAFE</div>
                <h2>Genuine Transaction</h2>
                <p>This transaction appears safe.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        recommendation = "Transaction can be processed safely."

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Prediction", prediction, "Model decision")
    with m2:
        metric_card("Fraud Probability", f"{fraud_probability * 100:.2f}%", "Class 1 probability")
    with m3:
        metric_card("Risk Score", f"{risk_score} / 100", "Scaled risk")
    with m4:
        metric_card("Confidence", f"{confidence * 100:.2f}%", "Decision confidence")

    risk_width = min(max(float(risk_score), 0.0), 100.0)
    prob_width = min(max(float(fraud_probability) * 100.0, 0.0), 100.0)
    st.markdown(
        f"""
        <div class="risk-meter">
            <div class="meter-label"><span>Risk Score</span><span>{risk_score} / 100</span></div>
            <div class="track"><div class="fill" style="width:{risk_width}%;"></div></div>
        </div>
        <div class="prob-meter">
            <div class="meter-label"><span>Fraud Probability</span><span>{fraud_probability * 100:.2f}%</span></div>
            <div class="track"><div class="fill" style="width:{prob_width}%;"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="recommend-card">
            <h4>Recommended Action</h4>
            <p style="margin:0; color:{SECONDARY_TEXT};">{recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_predictions():
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-label">History</div>
        <div class="section-title">Recent predictions</div>
        """,
        unsafe_allow_html=True,
    )
    history = st.session_state.get("recent_predictions", [])
    if not history:
        st.caption("No predictions yet. Run a demo prediction to populate this table.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def render_demo_prediction():
    st.markdown(
        """
        <div class="section-label">Demo</div>
        <div class="section-title">Single transaction scoring</div>
        <p class="section-desc">Load a real sample from the dataset, optionally adjust amount and hour, then score.</p>
        <div class="info-card">
            Hidden anonymized features (V1–V28) are loaded from the selected sample.
            You can edit transaction amount and hour before prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if get_dataset(uploader_key="demo_dataset_upload") is None:
        return

    ensure_sample_loaded()

    st.markdown(
        """
        <div class="section-label">Samples</div>
        <div class="section-title">Load from dataset</div>
        """,
        unsafe_allow_html=True,
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown('<div class="random-btn"></div>', unsafe_allow_html=True)
        if st.button("Load Random Sample", use_container_width=True, key="load_random_btn"):
            load_sample_into_session(mode="random")
            st.rerun()
    with b2:
        st.markdown('<div class="genuine-btn"></div>', unsafe_allow_html=True)
        if st.button("Load Genuine Sample", use_container_width=True, key="load_genuine_btn"):
            load_sample_into_session(mode="genuine")
            st.rerun()
    with b3:
        st.markdown('<div class="fraud-btn"></div>', unsafe_allow_html=True)
        if st.button("Load Fraud Sample", use_container_width=True, key="load_fraud_btn"):
            load_sample_into_session(mode="fraud")
            st.rerun()

    sample_type = st.session_state.get("sample_type", "Random")
    original_class = int(st.session_state.get("sample_original_class", 0))
    original_amount = float(st.session_state.get("sample_original_amount", 0.0))
    original_time = float(st.session_state.get("sample_original_time", 0.0))
    class_label = "Fraud" if original_class == 1 else "Genuine"

    if original_class == 1:
        badge_html = '<div class="sample-badge-fraud">Fraud sample loaded</div>'
    else:
        badge_html = '<div class="sample-badge-genuine">Genuine sample loaded</div>'

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
                    <div class="label">Original Class</div>
                    <div class="value">{class_label}</div>
                </div>
                <div class="sample-meta-item">
                    <div class="label">Dataset Row</div>
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

    st.markdown(
        """
        <div class="section-label">Inputs</div>
        <div class="section-title">Transaction details</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    amount_value = col1.number_input(
        "Transaction Amount",
        min_value=0.0,
        max_value=30000.0,
        step=10.0,
        format="%.2f",
        help="Editable amount. V1–V28 stay from the loaded sample.",
        key="amount_input",
    )
    hour_value = col2.number_input(
        "Transaction Hour (0–23)",
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

    if st.button("Predict Transaction", type="primary", key="demo_predict_btn"):
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
            st.warning(MODEL_UNAVAILABLE_MSG)
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    last = st.session_state.get("last_demo_result")
    if last:
        st.markdown(
            """
            <div class="section-label">Result</div>
            <div class="section-title">Model output</div>
            """,
            unsafe_allow_html=True,
        )
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
        <div class="section-label">Upload</div>
        <div class="section-title">Transaction CSV</div>
        <p class="section-desc">Score every row in one pass. Required model feature columns must be present.</p>
        <div class="upload-zone">
            <h3>Drop or browse a CSV file</h3>
            <p>Columns must include Time, Amount, and V1–V28.</p>
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

    st.markdown(
        """
        <div class="section-label">Preview</div>
        <div class="section-title">File preview</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.dataframe(upload_df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Batch Prediction", type="primary", key=f"{key_prefix}_run"):
        with st.spinner("Scoring transactions…"):
            result_df, missing = process_batch_dataframe(upload_df)

        if missing:
            if missing == ["__MODEL_UNAVAILABLE__"]:
                return
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

    st.markdown(
        """
        <div class="section-label">Summary</div>
        <div class="section-title">Batch results</div>
        """,
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        rich_kpi("Total Transactions", f"{total:,}", "Rows scored in this file", variant="total", tag="VOL")
    with k2:
        rich_kpi("Fraud Detected", f"{fraud_n:,}", "Predicted as Fraud", variant="fraud", tag="FRD")
    with k3:
        rich_kpi("Genuine Transactions", f"{genuine_n:,}", "Predicted as Genuine", variant="genuine", tag="GEN")
    with k4:
        rich_kpi("Fraud Percentage", f"{fraud_pct:.2f}%", "Share of fraud predictions", variant="rate", tag="RATE")

    st.write("")
    st.markdown('<div class="panel-card"><h3>Processed Results</h3>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download Results",
        data=csv_buffer.getvalue(),
        file_name="fraud_analysis_results.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def fraud_detection_page():
    page_header(
        "Fraud Detection",
        "Score a single transaction or upload a CSV for batch analysis.",
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
        "Batch Analysis",
        "Upload a CSV and score every transaction with the trained Random Forest model.",
    )
    render_batch_analysis(key_prefix="batch_page")


def dashboard_page():
    page_header(
        "Dashboard",
        "Dataset volume, fraud rate, and model score distributions.",
    )

    df = get_dataset(uploader_key="dashboard_dataset_upload")
    if df is None:
        return

    summary = get_dataset_summary(df)
    total = summary["num_rows"]
    genuine = summary["genuine_transactions"]
    fraud = summary["fraudulent_transactions"]
    fraud_rate = (fraud / total * 100) if total else 0.0

    st.markdown(
        """
        <div class="section-label">Overview</div>
        <div class="section-title">Key metrics</div>
        """,
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        rich_kpi("Total Transactions", f"{total:,}", "Full dataset volume", variant="total", tag="VOL")
    with k2:
        rich_kpi("Fraud Transactions", f"{fraud:,}", "Labeled fraud cases", variant="fraud", tag="FRD")
    with k3:
        rich_kpi("Genuine Transactions", f"{genuine:,}", "Labeled genuine cases", variant="genuine", tag="GEN")
    with k4:
        rich_kpi("Fraud Rate", f"{fraud_rate:.3f}%", "Fraud share of dataset", variant="rate", tag="RATE")

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-label">Analytics</div>
        <div class="section-title">Distributions</div>
        <p class="section-desc">Volume split and amount profile, plus model probability and risk charts.</p>
        """,
        unsafe_allow_html=True,
    )
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
        st.warning(
            "Could not compute probability charts. "
            + MODEL_UNAVAILABLE_MSG
        )

    with st.expander("Dataset snapshot"):
        st.dataframe(df.head(10), use_container_width=True)


def about_page():
    page_header(
        "About",
        "Internship project overview for the Financial Fraud Intelligence Engine.",
    )

    st.markdown(
        f"""
        <div class="section-label">Overview</div>
        <div class="section-title">Project details</div>
        <div class="detail-card" style="margin-top:0.85rem;">
            <h3 style="color:{PRIMARY_TEXT}; margin-top:0; font-size:1.05rem;">Purpose</h3>
            <p style="color:{SECONDARY_TEXT}; margin-bottom:0; line-height:1.55;">
                Detect fraudulent credit card transactions with a Random Forest model
                trained on the Kaggle Credit Card Fraud Detection dataset. Supports
                single-transaction demos, batch CSV scoring, risk recommendations,
                and an interactive analytics dashboard.
            </p>
        </div>
        <div class="detail-card">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;">
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Model</div>
                    <div style="color:{INK};font-weight:600;margin-bottom:0.35rem;">Random Forest Classifier</div>
                    <p style="color:{SECONDARY_TEXT};margin:0;font-size:0.92rem;line-height:1.5;">
                        Loaded from <code>fraud_model.pkl</code>. Returns class label,
                        fraud probability, and risk score.
                    </p>
                </div>
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Dataset</div>
                    <div style="color:{INK};font-weight:600;margin-bottom:0.35rem;">Kaggle Credit Card Fraud</div>
                    <p style="color:{SECONDARY_TEXT};margin:0;font-size:0.92rem;line-height:1.5;">
                        Anonymized features V1–V28, Time, Amount, and Class labels.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-label">Stack</div>
        <div class="section-title">Technologies</div>
        """,
        unsafe_allow_html=True,
    )
    technologies = ["Python", "Pandas", "Scikit-learn", "Streamlit", "Matplotlib", "Random Forest"]
    pills = "".join(f'<span class="tech-pill">{tech}</span>' for tech in technologies)
    st.markdown(f"<div style='margin:0.5rem 0 1.25rem 0;'>{pills}</div>", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-label">Author</div>
        <div class="section-title">Developer information</div>
        <div class="detail-card" style="margin-top:0.85rem;">
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Name</div>
                    <div style="color:{INK};font-weight:600;">{FULL_NAME}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Email</div>
                    <div style="color:{INK};font-weight:600;word-break:break-all;">{REGISTERED_EMAIL}</div>
                </div>
                <div>
                    <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};font-weight:700;margin-bottom:0.35rem;">Type</div>
                    <div style="color:{INK};font-weight:600;">Internship Project</div>
                </div>
            </div>
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
            <div class="logo-mark">FI</div>
            <div class="brand-copy">
                <div class="caption">Intelligence Suite</div>
                <div class="title">Fraud Engine</div>
            </div>
        </div>
        <div class="nav-section-label">Navigation</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="main_nav",
    )
    st.markdown(
        """
        <div class="sidebar-meta">
            <div class="meta-row">
                <span class="meta-key">Status</span>
                <span class="status-pill"><span class="status-dot"></span>ONLINE</span>
            </div>
            <div class="meta-row">
                <span class="meta-key">Model</span>
                <span class="meta-val">Random Forest</span>
            </div>
            <div class="meta-row">
                <span class="meta-key">Dataset</span>
                <span class="meta-val">Credit Card</span>
            </div>
            <div class="meta-row">
                <span class="meta-key">Version</span>
                <span class="meta-val">1.0.0</span>
            </div>
        </div>
        <div class="sidebar-footer">Financial Fraud Intelligence</div>
        """,
        unsafe_allow_html=True,
    )

if page == "Home":
    home_page()
elif page == "Fraud Detection":
    fraud_detection_page()
elif page == "Batch Analysis":
    batch_analysis_page()
elif page == "Dashboard":
    dashboard_page()
else:
    about_page()
