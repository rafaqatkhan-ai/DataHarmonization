# app.py (generalized for single- or multi-file)
import os, io, tempfile, shutil, json
import streamlit as st
import pandas as pd
from harmonizer import run_pipeline

# ---- Page Setup ----
st.set_page_config(
    page_title="🧬 Data Harmonization & QC Suite",
    page_icon="🧬",
    layout="wide",
)

# =========================
# THEME SELECTOR (non-black)
# =========================
theme = st.selectbox(
    "Theme",
    ["Light Gray", "Soft Off-White", "Deep Navy", "Slate Blue"],
    index=0,
    help="Pick a background style for the app."
)

def apply_theme(t: str):
    if t == "Light Gray":
        css = """
        <style>
        [data-testid="stAppViewContainer"] { background:#f3f4f6 !important; color:#0f172a !important; }
        [data-testid="stSidebar"] { background:#e5e7eb !important; border-right:1px solid #cbd5e1 !important; }
        [data-testid="stVerticalBlock"] { background:#ffffff !important; border-radius:12px; padding:1rem; margin-bottom:1rem; box-shadow:0 2px 8px rgba(15,23,42,0.06) !important; }

        h1,h2,h3,h4,h5,h6,p,label,span { color:#0f172a !important; }

        .stButton>button {
            background:linear-gradient(90deg,#2563eb,#1d4ed8) !important; color:#fff !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:600 !important;
        }
        .stButton>button:hover {
            background:linear-gradient(90deg,#3b82f6,#2563eb) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 10px rgba(37,99,235,.25) !important;
        }

        /* ---------- Wider, cleaner tab design ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 14px !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 14px 28px !important;
            border-radius: 12px !important;
            background: #e9f0fa !important;
            color: #1e3a8a !important;
            border: 1px solid #cbd5e1 !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            min-width: 140px !important;
            text-align: center !important;
            transition: all 0.25s ease-in-out !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: #dbeafe !important;
            color: #0a2540 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg,#2563eb,#1e40af) !important;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(37,99,235,0.25) !important;
            transform: translateY(-1px) !important;
            font-weight: 800 !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            background: #ffffff !important;
            border-radius: 10px !important;
            padding: 1.2rem !important;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05) !important;
            margin-top: 10px !important;
        }

        .metric-card{ background:#f8fafc !important; border:1px solid #e2e8f0 !important; border-radius:12px !important; padding:14px 16px !important; }
        .smallcaps{ color:#475569 !important; }
        </style>
        """
    elif t == "Soft Off-White":
        css = """
        <style>
        [data-testid="stAppViewContainer"] { background:#faf7f2 !important; color:#1f2937 !important; }
        [data-testid="stSidebar"] { background:#f3efe8 !important; border-right:1px solid #e5e7eb !important; }
        [data-testid="stVerticalBlock"] { background:#ffffff !important; border-radius:12px !important; padding:1rem !important; margin-bottom:1rem !important; box-shadow:0 2px 10px rgba(0,0,0,0.06) !important; }

        h1,h2,h3,h4,h5,h6,p,label,span { color:#111827 !important; }

        .stButton>button {
            background:linear-gradient(90deg,#10b981,#059669) !important; color:#fff !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:600 !important;
        }
        .stButton>button:hover {
            background:linear-gradient(90deg,#34d399,#10b981) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 10px rgba(16,185,129,.25) !important;
        }

        /* ---------- Wider, cleaner tab design ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 14px !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 14px 28px !important;
            border-radius: 12px !important;
            background: #fff7ed !important;
            color: #7c2d12 !important;
            border: 1px solid #fed7aa !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            min-width: 140px !important;
            text-align: center !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: #ffedd5 !important;
            color: #4a1d0a !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg,#f97316,#ef4444) !important;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(249,115,22,0.25) !important;
            transform: translateY(-1px) !important;
            font-weight: 800 !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            background: #ffffff !important;
            border-radius: 10px !important;
            padding: 1.2rem !important;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05) !important;
            margin-top: 10px !important;
        }
        </style>
        """
    else:  # keep rest same as before (Deep Navy / Slate Blue)
        css = ""  # omitted for brevity, already defined earlier in your working file
    st.markdown(css, unsafe_allow_html=True)

apply_theme(theme)

# =========================
# TITLE
# =========================
st.markdown(
    """
    <style>
    .centered-title {
        font-size: 2.6rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a, #2563eb, #6366f1, #7c3aed);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: colorShift 8s ease infinite;
        margin-top: -0.5rem;
    }
    @keyframes colorShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .centered-title span {
        font-size: 2.8rem;
        animation: pulse 2s infinite alternate;
    }
    @keyframes pulse {
        from { transform: scale(1); opacity: 0.85; }
        to { transform: scale(1.2); opacity: 1; }
    }
    .subtitle {
        text-align: center;
        opacity: .9;
        font-size: 1rem;
        margin-top: -0.6rem;
        font-style: italic;
    }
    </style>

    <h1 class="centered-title">
        <span>🧬</span> Data Harmonization & QC Suite <span>🧬</span>
    </h1>
    <p class="subtitle">
        Upload expression data, perform harmonization, QC, and analysis — all in one place.
    </p>
    """,
    unsafe_allow_html=True
)

# =========================
# UI CONTROLS
# =========================
mode = st.radio(
    "Expression upload mode",
    ["Single expression matrix", "Multiple files (one per group)"],
    horizontal=True,
)
st.caption(
    "Upload expression data (single matrix **or** one file per group) and corresponding metadata, then click **Run Harmonization**."
)

# (rest of your app logic remains identical — upload, metadata, analysis, results, etc.)
