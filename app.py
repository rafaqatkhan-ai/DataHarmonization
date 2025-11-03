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
# THEME SELECTOR
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

        /* Buttons */
        .stButton>button {
            background:linear-gradient(90deg,#2563eb,#1d4ed8) !important;
            color:#fff !important;
            border:none !important;
            border-radius:8px !important;
            padding:.5rem 1rem !important;
            font-weight:600 !important;
        }
        .stButton>button:hover {
            background:linear-gradient(90deg,#3b82f6,#2563eb) !important;
            transform:translateY(-2px) !important;
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
        </style>
        """
    elif t == "Soft Off-White":
        css = """
        <style>
        [data-testid="stAppViewContainer"] { background:#faf7f2 !important; color:#1f2937 !important; }
        [data-testid="stSidebar"] { background:#f3efe8 !important; border-right:1px solid #e5e7eb !important; }
        [data-testid="stVerticalBlock"] { background:#ffffff !important; border-radius:12px !important; padding:1rem !important; margin-bottom:1rem !important; box-shadow:0 2px 10px rgba(0,0,0,0.06) !important; }
        h1,h2,h3,h4,h5,h6,p,label,span { color:#111827 !important; }

        /* Buttons */
        .stButton>button {
            background:linear-gradient(90deg,#10b981,#059669) !important;
            color:#fff !important;
            border:none !important;
            border-radius:8px !important;
            padding:.5rem 1rem !important;
            font-weight:600 !important;
        }
        .stButton>button:hover {
            background:linear-gradient(90deg,#34d399,#10b981) !important;
            transform:translateY(-2px) !important;
            box-shadow:0 4px 10px rgba(16,185,129,.25) !important;
        }

        /* Tabs */
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
            min-width: 140px !important;
            text-align: center !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: #ffedd5 !important;
            color: #4a1d0a !important;
            transform: translateY(-1px) !important;
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
    else:
        css = ""  # keep other themes as before
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

# ---------------- Expression upload ----------------
single_expr_file = None
normal_file = atypia_file = hpv_pos_file = hpv_neg_file = None

if mode == "Multiple files (one per group)":
    with st.expander("1) Upload Expression Files (one per group)"):
        col1, col2 = st.columns(2)
        with col1:
            normal_file = st.file_uploader("Normal (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="normal")
            atypia_file = st.file_uploader("Atypia (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="atypia")
        with col2:
            hpv_pos_file = st.file_uploader("HPV Positive (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="hpvp")
            hpv_neg_file = st.file_uploader("HPV Negative (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="hpvn")
else:
    with st.expander("1) Upload Single Expression Matrix (XLSX/CSV/TSV)"):
        single_expr_file = st.file_uploader("Expression matrix", type=["xlsx","csv","tsv","txt"], key="single_expr")

# ---------------- Metadata ----------------
with st.expander("2) Upload Metadata (TSV/CSV/XLSX)"):
    metadata_file = st.file_uploader("Metadata file", type=["tsv","csv","txt","xlsx"], key="meta")

    id_cols = st.text_input(
        "Candidate ID columns (comma-separated)",
        "Id,ID,id,CleanID,sample,Sample"
    )
    grp_cols = st.text_input(
        "Candidate GROUP columns (comma-separated)",
        "group,Group,condition,Condition,phenotype,Phenotype"
    )
    batch_col = st.text_input("Batch column name (optional; leave blank to auto-detect)", "")

    if metadata_file is not None:
        try:
            bio = io.BytesIO(metadata_file.getvalue())
            name = metadata_file.name.lower()
            if name.endswith((".xlsx", ".xls")):
                mprev = pd.read_excel(bio, engine="openpyxl").head(3)
            elif name.endswith((".tsv", ".txt")):
                mprev = pd.read_csv(bio, sep="\t").head(3)
            else:
                mprev = pd.read_csv(bio, sep=None, engine="python").head(3)
            st.caption(f"Detected metadata columns: {list(mprev.columns)}")
            with st.expander("Preview first 3 rows of metadata"):
                st.dataframe(mprev, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not preview metadata columns: {e}")

# ---------------- Optional GSEA ----------------
with st.expander("3) Optional: GSEA gene set (.gmt)"):
    gmt_file = st.file_uploader("Gene set GMT (optional)", type=["gmt"])

# ---------------- Advanced ----------------
with st.expander("Advanced settings"):
    out_dir = st.text_input("Output directory", "out")
    pca_topk = st.number_input("Top variable genes for PCA", min_value=500, max_value=50000, value=5000, step=500)
    do_nonlinear = st.checkbox("Make UMAP/t-SNE (if available)", value=True)

# ---------------- Run ----------------
run = st.button("🚀 Run Harmonization", type="primary", use_container_width=True)

if run:
    if not metadata_file:
        st.error("Please upload a metadata file.")
        st.stop()

    kwargs = {
        "metadata_file": io.BytesIO(metadata_file.getvalue()),
        "metadata_name_hint": metadata_file.name,
        "metadata_id_cols": [c.strip() for c in id_cols.split(",") if c.strip()],
        "metadata_group_cols": [c.strip() for c in grp_cols.split(",") if c.strip()],
        "metadata_batch_col": (batch_col.strip() or None),
        "out_root": out_dir,
        "pca_topk_features": int(pca_topk),
        "make_nonlinear": do_nonlinear,
    }

    gmt_path = None
    if gmt_file:
        tmpdir = tempfile.mkdtemp()
        gmt_path = os.path.join(tmpdir, gmt_file.name)
        with open(gmt_path, "wb") as fh:
            fh.write(gmt_file.getvalue())
        kwargs["gsea_gmt"] = gmt_path

    if mode == "Multiple files (one per group)":
        groups = {}
        if normal_file: groups["Normal"] = io.BytesIO(normal_file.getvalue())
        if atypia_file: groups["Atypia"] = io.BytesIO(atypia_file.getvalue())
        if hpv_pos_file: groups["HPV_Pos"] = io.BytesIO(hpv_pos_file.getvalue())
        if hpv_neg_file: groups["HPV_Neg"] = io.BytesIO(hpv_neg_file.getvalue())
        if len(groups) == 0:
            st.error("Please upload at least one expression file.")
            st.stop()
        kwargs["group_to_file"] = groups
    else:
        if not single_expr_file:
            st.error("Please upload the expression matrix.")
            st.stop()
        kwargs["single_expression_file"] = io.BytesIO(single_expr_file.getvalue())
        kwargs["single_expression_name_hint"] = single_expr_file.name

    try:
        with st.spinner("Running harmonization..."):
            out = run_pipeline(**kwargs)
    except Exception as e:
        st.error(f"Run failed: {e}")
        if gmt_file:
            shutil.rmtree(os.path.dirname(gmt_path), ignore_errors=True)
        st.stop()

    st.success("Done!")

    # ------- UI: Results Tabs -------
    st.subheader("Results")

    report = {}
    try:
        with open(out["report_json"], "r") as fh:
            report = json.load(fh)
    except Exception:
        pass

    tabs = st.tabs(["Overview", "QC", "PCA & Embeddings", "DE & GSEA", "Outliers", "Files"])

    # ---- Overview
    with tabs[0]:
        st.json(report if report else {"info": "report.json not found"})

    # ---- QC
    with tabs[1]:
        st.write("QC visualizations appear here...")

    # ---- PCA
    with tabs[2]:
        st.write("PCA & Embeddings visualizations appear here...")

    # ---- DE & GSEA
    with tabs[3]:
        st.write("DE & GSEA results appear here...")

    # ---- Outliers
    with tabs[4]:
        st.write("Outlier detection results appear here...")

    # ---- Files
    with tabs[5]:
        st.write("Download results here...")
