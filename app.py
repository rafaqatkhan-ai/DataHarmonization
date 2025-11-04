# app.py (generalized for single- or multi-file, with PCA fail-soft messaging)
import os, io, tempfile, shutil, json
import streamlit as st
import pandas as pd
from harmonizer import run_pipeline
# --- Streamlit compatibility shims (older versions may not support some kwargs) ---
# --- Streamlit compatibility shims ---
def safe_button(label, **kwargs):
    import streamlit as st
    try:
        return st.button(label, **kwargs)
    except Exception:
        kwargs.pop("type", None)
        kwargs.pop("use_container_width", None)
        return st.button(label, **kwargs)

def safe_download_button(label, data=None, **kwargs):
    """
    Robust wrapper around st.download_button for older Streamlit versions.
    Strategy:
      1) Try as-is.
      2) Drop 'use_container_width' and retry.
      3) Drop non-essential kwargs and retry with only (label, data).
    """
    import streamlit as st
    try:
        return st.download_button(label=label, data=data, **kwargs)
    except Exception:
        # 2) Retry without newer kwarg
        kwargs.pop("use_container_width", None)
        try:
            return st.download_button(label=label, data=data, **kwargs)
        except Exception:
            # 3) Minimal fallback (strip extras)
            kwargs.pop("mime", None)
            kwargs.pop("file_name", None)
            kwargs.pop("help", None)
            kwargs.pop("key", None)
            return st.download_button(label=label, data=data)

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
    # Each theme block uses !important to override any earlier CSS.
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

        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{
            background:#e9f0fa !important; color:#1e3a8a !important; border:1px solid #cbd5e1 !important;
            border-radius:10px !important; font-weight:700 !important; transition:all .25s ease-in-out !important;
            min-width:130px !important; padding:0.6rem 1.2rem !important;
        }
        .stTabs [data-baseweb="tab"]:hover{ background:#dbeafe !important; color:#0a2540 !important; transform:translateY(-1px) !important; }
        .stTabs [aria-selected="true"]{
            background:linear-gradient(135deg,#2563eb,#1e40af) !important; color:#fff !important;
            box-shadow:0 4px 10px rgba(37,99,235,.25) !important; border:none !important; transform:translateY(-1px) !important;
        }
        .stTabs [data-baseweb="tab-panel"]{ background:#ffffff !important; border-radius:10px !important; padding:1rem !important; box-shadow:0 2px 8px rgba(15,23,42,.05) !important; }

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

        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{
            background:#fff7ed !important; color:#7c2d12 !important; border:1px solid #fed7aa !important;
            border-radius:10px !important; font-weight:700 !important; min-width:130px !important; padding:0.6rem 1.2rem !important;
        }
        .stTabs [data-baseweb="tab"]:hover{ background:#ffedd5 !important; color:#4a1d0a !important; }
        .stTabs [aria-selected="true"]{
            background:linear-gradient(135deg,#f97316,#ef4444) !important; color:#fff !important; border:none !important;
            box-shadow:0 4px 12px rgba(249,115,22,.25) !important; transform:translateY(-1px) !important;
        }
        .stTabs [data-baseweb="tab-panel"]{ background:#ffffff !important; border-radius:10px !important; padding:1rem !important; box-shadow:0 2px 8px rgba(0,0,0,.05) !important; }

        .metric-card{ background:#ffffff !important; border:1px solid #f3f4f6 !important; border-radius:12px !important; padding:14px 16px !important; }
        .smallcaps{ color:#6b7280 !important; }
        </style>
        """
    elif t == "Deep Navy":
        css = """
        <style>
        [data-testid="stAppViewContainer"] { background:#0b1020 !important; color:#e5e7eb !important; }
        [data-testid="stSidebar"] { background:#0f172a !important; color:#f3f4f6 !important; border-right:1px solid #1f2a44 !important; }
        [data-testid="stVerticalBlock"] { background:#0d142a !important; border-radius:12px !important; padding:1rem !important; margin-bottom:1rem !important; box-shadow:0 2px 12px rgba(0,0,0,.5) !important; }

        h1,h2,h3,h4,h5,h6,p,label,span { color:#e5e7eb !important; }

        .stButton>button {
            background:linear-gradient(90deg,#06b6d4,#3b82f6) !important; color:#0b1020 !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:700 !important;
        }
        .stButton>button:hover {
            background:linear-gradient(90deg,#22d3ee,#60a5fa) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 12px rgba(34,211,238,.35) !important;
        }

        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{
            background:#111827 !important; color:#cbd5e1 !important; border:1px solid #1f2937 !important;
            border-radius:10px !important; font-weight:700 !important; min-width:130px !important; padding:0.6rem 1.2rem !important;
        }
        .stTabs [data-baseweb="tab"]:hover{ background:#0b1220 !important; color:#f1f5f9 !important; }
        .stTabs [aria-selected="true"]{
            background:linear-gradient(135deg,#06b6d4,#6366f1) !important; color:#0b1020 !important; border:none !important;
            box-shadow:0 4px 12px rgba(6,182,212,.35) !important; transform:translateY(-1px) !important;
        }
        .stTabs [data-baseweb="tab-panel"]{ background:#0b1020 !important; border-radius:10px !important; padding:1rem !important; box-shadow:inset 0 0 0 1px rgba(99,102,241,.15) !important; }

        .metric-card{ background:#0f172a !important; border:1px solid rgba(99,102,241,.2) !important; border-radius:12px !important; padding:14px 16px !important; }
        .smallcaps{ color:#93c5fd !important; }
        </style>
        """
    else:  # "Slate Blue"
        css = """
        <style>
        [data-testid="stAppViewContainer"] { background:#0f172a !important; color:#e2e8f0 !important; }
        [data-testid="stSidebar"] { background:#111827 !important; border-right:1px solid #1f2937 !important; }
        [data-testid="stVerticalBlock"] { background:#0b1220 !important; border-radius:12px !important; padding:1rem !important; margin-bottom:1rem !important; box-shadow:0 2px 10px rgba(2,6,23,.6) !important; }

        h1,h2,h3,h4,h5,h6,p,label,span { color:#e2e8f0 !important; }

        .stButton>button {
            background:linear-gradient(90deg,#818cf8,#22d3ee) !important; color:#0b1220 !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:700 !important;
        }
        .stButton>button:hover {
            background:linear-gradient(90deg,#a5b4fc,#67e8f9) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 12px rgba(129,140,248,.35) !important;
        }

        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{
            background:#111827 !important; color:#cbd5e1 !important; border:1px solid #1f2937 !important;
            border-radius:10px !important; font-weight:700 !important; min-width:130px !important; padding:0.6rem 1.2rem !important;
        }
        .stTabs [data-baseweb="tab"]:hover{ background:#0b1220 !important; color:#f8fafc !important; }
        .stTabs [aria-selected="true"]{
            background:linear-gradient(135deg,#22d3ee,#818cf8,#a78bfa) !important; color:#0b1220 !important; border:none !important;
            box-shadow:0 4px 12px rgba(34,211,238,.35) !important; transform:translateY(-1px) !important;
        }
        .stTabs [data-baseweb="tab-panel"]{ background:#0f172a !important; border-radius:10px !important; padding:1rem !important; box-shadow:inset 0 0 0 1px rgba(148,163,184,.12) !important; }

        .metric-card{ background:#0b1220 !important; border:1px solid rgba(148,163,184,.25) !important; border-radius:12px !important; padding:14px 16px !important; }
        .smallcaps{ color:#94a3b8 !important; }
        </style>
        """
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
        "Id,ID,id,CleanID,sample,Sample,sample_id,Sample_ID,SampleID"
    )
    grp_cols = st.text_input(
        "Candidate GROUP columns (comma-separated)",
        "group,Group,condition,Condition,phenotype,Phenotype"
    )
    batch_col = st.text_input("Batch column name (optional; leave blank to auto-detect)", "")

    # Preview detected metadata columns
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
# ---------------- Run ----------------
run = safe_button("🚀 Run Harmonization", type="primary", use_container_width=True)

if run:
    if not metadata_file:
        st.error("Please upload a metadata file.")
        st.stop()

    # ---- Build kwargs for the pipeline ----
    kwargs = {
        "metadata_file": io.BytesIO(metadata_file.getvalue()),
        "metadata_name_hint": metadata_file.name,
        "metadata_id_cols": [c.strip() for c in id_cols.split(",") if c.strip()],
        "metadata_group_cols": [c.strip() for c in grp_cols.split(",") if c.strip()],
        "metadata_batch_col": (batch_col.strip() or None),
        "out_root": out_dir,                    # will be replaced by timestamped subfolder below
        "pca_topk_features": int(pca_topk),
        "make_nonlinear": do_nonlinear,
    }

    # Optional GSEA GMT
    gmt_path = None
    if gmt_file:
        tmpdir = tempfile.mkdtemp()
        gmt_path = os.path.join(tmpdir, gmt_file.name)
        with open(gmt_path, "wb") as fh:
            fh.write(gmt_file.getvalue())
        kwargs["gsea_gmt"] = gmt_path

    # Expression inputs
    if mode == "Multiple files (one per group)":
        groups = {}
        if normal_file: groups["Normal"] = io.BytesIO(normal_file.getvalue())
        if atypia_file: groups["Atypia"] = io.BytesIO(atypia_file.getvalue())
        if hpv_pos_file: groups["HPV_Pos"] = io.BytesIO(hpv_pos_file.getvalue())
        if hpv_neg_file: groups["HPV_Neg"] = io.BytesIO(hpv_neg_file.getvalue())
        if not groups:
            st.error("Please upload at least one expression file.")
            st.stop()
        kwargs["group_to_file"] = groups
    else:
        if not single_expr_file:
            st.error("Please upload the expression matrix.")
            st.stop()
        kwargs["single_expression_file"] = io.BytesIO(single_expr_file.getvalue())
        kwargs["single_expression_name_hint"] = single_expr_file.name

    # ---- Run pipeline with timestamped out_root ----
    try:
        with st.spinner("Running harmonization..."):
            # Initialize session state
            if "run_id" not in st.session_state:
                st.session_state.run_id = None
            if "out" not in st.session_state:
                st.session_state.out = None

            # Unique run folder
            import datetime as _dt
            run_id = _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
            kwargs["out_root"] = os.path.join(out_dir, run_id)

            # Execute
            out = run_pipeline(**kwargs)

            # Persist current run
            st.session_state.run_id = run_id
            st.session_state.out = out

    except Exception as e:
        st.error(f"Run failed: {e}")
        if gmt_file:
            shutil.rmtree(os.path.dirname(gmt_path), ignore_errors=True)
        st.stop()

    # ---- Load report for KPIs ----
    report = {}
    try:
        with open(out["report_json"], "r") as fh:
            report = json.load(fh)
    except Exception:
        pass



    # Show success, then a gentle warning if PCA was skipped
    st.success("Done!")
    if report.get("notes", {}).get("pca_skipped_reason"):
        st.warning("PCA/UMAP skipped: " + str(report["notes"]["pca_skipped_reason"]))

    # ------- UI: Results Tabs -------
    st.subheader("Results")

    # KPI Cards
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    qc = report.get("qc", {})
    shp = report.get("shapes", {})
    with kcol1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Samples", shp.get("samples", "—"))
        st.markdown('<div class="smallcaps">Total samples</div></div>', unsafe_allow_html=True)
    with kcol2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Genes", shp.get("genes", "—"))
        st.markdown('<div class="smallcaps">Features detected</div></div>', unsafe_allow_html=True)
    with kcol3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Zero fraction", f'{qc.get("zero_fraction", 0):.2f}')
        st.markdown('<div class="smallcaps">Approx. sparsity</div></div>', unsafe_allow_html=True)
    with kcol4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        sil_batch = qc.get("silhouette_batch", None)
        st.metric("Silhouette (batch)", f'{sil_batch:.2f}' if isinstance(sil_batch, (int,float)) else "—")
        st.markdown('<div class="smallcaps">Lower is better</div></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "QC", "PCA & Embeddings", "DE & GSEA", "Outliers", "Files"])

    # ---- Overview
    with tabs[0]:
        st.json(report if report else {"info":"report.json not found"})
        fig_dir = out["figdir"]
        previews = ["dist_pre_vs_post_log2.png", "pca_clean_groups.png", "enhanced_pca_analysis.png"]
        show = [f for f in previews if os.path.exists(os.path.join(fig_dir, f))]
        if show:
            st.write("### Key Figures")
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for i, f in enumerate(show):
                with cols[i % 3]:
                    st.image(os.path.join(fig_dir, f), caption=f, use_column_width=True)

    # ---- QC
    with tabs[1]:
        fig_dir = out["figdir"]
        qc_figs = [
            "qc_library_size.png",
            "qc_zero_rate_hist.png",
            "group_density_post_log2.png",
            "dist_zscore.png",
            "sample_correlation_heatmap.png",
            "hk_cv.png",
            "sex_marker_concordance.png",
        ]
        st.write("### QC Figures")
        for f in qc_figs:
            p = os.path.join(fig_dir, f)
            if os.path.exists(p):
                st.image(p, caption=f, use_column_width=True)

    # ---- PCA & Embeddings
    with tabs[2]:
        fig_dir = out["figdir"]
        pcs = [
            "pca_clean_groups.png",
            "enhanced_pca_analysis.png",
            "pca_loadings_pc1.png",
            "pca_loadings_pc2.png",
            "umap_by_group.png",
            "tsne_by_group.png",
        ]
        st.write("### PCA / UMAP / t-SNE")
        for f in pcs:
            p = os.path.join(fig_dir, f)
            if os.path.exists(p):
                st.image(p, caption=os.path.basename(p), use_column_width=True)
        # If nothing exists, the warning above already explains why.

    # ---- DE & GSEA
    with tabs[3]:
        fig_dir = out["figdir"]
        de_dir = os.path.join(out["outdir"], "de")
        de_files = [f for f in os.listdir(de_dir)] if os.path.isdir(de_dir) else []
        contrasts = [f.replace("DE_","").replace(".tsv","") for f in de_files if f.startswith("DE_")]
        pick = st.selectbox("Select contrast", contrasts) if contrasts else None
        if pick:
            st.write(f"### Differential Expression: {pick}")
            vol = os.path.join(fig_dir, f"volcano_{pick}.png")
            ma = os.path.join(fig_dir, f"ma_{pick}.png")
            hm = os.path.join(fig_dir, f"heatmap_top_50_{pick}.png")
            for p in [vol, ma, hm]:
                if os.path.exists(p):
                    st.image(p, caption=os.path.basename(p), use_column_width=True)
            # show table preview
            tsv = os.path.join(de_dir, f"DE_{pick}.tsv")
            try:
                df = pd.read_csv(tsv, sep="\t", index_col=0).head(50)
                st.dataframe(df, use_container_width=True)
                with open(tsv, "rb") as fh:
                    safe_download_button("⬇️ Download full DE table", fh.read(), file_name=f"DE_{pick}.tsv", mime="text/tab-separated-values")
            except Exception:
                pass
        # GSEA if any
        gsea_dir = os.path.join(out["outdir"], "gsea")
        if os.path.isdir(gsea_dir):
            st.write("### GSEA Results")
            for f in sorted(os.listdir(gsea_dir)):
                if f.endswith(".tsv"):
                    st.write(f)
                    try:
                        df = pd.read_csv(os.path.join(gsea_dir, f), sep="\t").head(30)
                        st.dataframe(df, use_container_width=True)
                    except Exception:
                        pass

# ---- Outliers
with tabs[4]:
    # Always reference the latest run
    out_curr = st.session_state.get("out") or out
    try:
        outliers_path = os.path.join(out_curr["outdir"], "outliers.tsv")
        meta_path = os.path.join(out_curr["outdir"], "metadata.tsv")

        if os.path.exists(outliers_path):
            df = pd.read_csv(outliers_path, sep="\t", index_col=0)

            # Join metadata for readability if available
            if os.path.exists(meta_path):
                meta_df = pd.read_csv(meta_path, sep="\t", index_col=0)
                display_df = (
                    df.copy()
                    .assign(sample=df.index)
                    .join(meta_df[["bare_id", "group"]], how="left")
                    .set_index("sample")
                    .rename(
                        columns={
                            "IsolationForest": "IsolationForest_flag",
                            "LOF": "LOF_flag",
                        }
                    )[
                        ["bare_id", "group", "IsolationForest_flag", "LOF_flag"]
                    ]
                )
            else:
                display_df = df

            st.write("### Outlier flags (1 = outlier)")
            st.dataframe(
                display_df,
                use_container_width=True,
                key=f"outliers_df_{st.session_state.get('run_id','noid')}",
            )

            with open(outliers_path, "rb") as fh:
                safe_download_button(
                    "⬇️ Download outlier table",
                    fh.read(),
                    file_name="outliers.tsv",
                    mime="text/tab-separated-values",
                )
        else:
            st.info("No outlier table found for this run.")
    except Exception as e:
        st.warning(f"Could not load outliers for this run: {e}")


    # ---- Files
    with tabs[5]:
        colA, colB = st.columns(2)
        with colA:
            st.write("**Core Tables**")
            core_files = [
                ("Combined Expression", os.path.join(out["outdir"], "expression_combined.tsv")),
                ("Harmonized Expression", os.path.join(out["outdir"], "expression_harmonized.tsv")),
                ("PCA Scores", os.path.join(out["outdir"], "pca_scores.tsv")),
                ("Metadata (aligned)", os.path.join(out["outdir"], "metadata.tsv")),
                ("Report (JSON)", out["report_json"]),
            ]
            for label, path in core_files:
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        safe_download_button(f"⬇️ {label}", fh.read(), file_name=os.path.basename(path), mime="text/plain", use_column_width=True)
        with colB:
            try:
                with open(out["zip"], "rb") as fh:
                    safe_download_button(
                        label="⬇️ Download ALL results (ZIP)",
                        data=fh.read(),
                        file_name="harmonization_results.zip",
                        mime="application/zip",
                        use_column_width=True,
                    )
            except Exception as e:
                st.warning(f"Could not open ZIP for download: {e}")

    # Cleanup temp GMT if used
    if gmt_file:
        shutil.rmtree(os.path.dirname(gmt_path), ignore_errors=True)












