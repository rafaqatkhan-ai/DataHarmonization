# app.py — always-visible Results tabs (incl. Multi-GEO) + local multi-runner

import os, io, tempfile, shutil, json, re
import streamlit as st
import pandas as pd
import numpy as np
import datetime as _dt
import harmonizer as hz

# =========================
# Streamlit compatibility shims
# =========================
def safe_button(label, **kwargs):
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
    try:
        return st.download_button(label=label, data=data, **kwargs)
    except Exception:
        kwargs.pop("use_container_width", None)
        try:
            return st.download_button(label=label, data=data, **kwargs)
        except Exception:
            kwargs.pop("mime", None)
            kwargs.pop("file_name", None)
            kwargs.pop("help", None)
            kwargs.pop("key", None)
            return st.download_button(label=label, data=data)

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="🧬 Data Harmonization & QC Suite",
    page_icon="🧬",
    layout="wide",
)

# Init session state (used across tabs)
for k, v in [
    ("run_id", None),
    ("out", None),
    ("run_token", None),
    ("multi_geo", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

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
        .stButton>button { background:linear-gradient(90deg,#2563eb,#1d4ed8) !important; color:#fff !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:600 !important; }
        .stButton>button:hover { background:linear-gradient(90deg,#3b82f6,#2563eb) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 10px rgba(37,99,235,.25) !important; }
        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{ background:#e9f0fa !important; color:#1e3a8a !important; border:1px solid #cbd5e1 !important;
            border-radius:10px !important; font-weight:700 !important; transition:all .25s ease-in-out !important; min-width:130px !important; padding:0.6rem 1.2rem !important; }
        .stTabs [data-baseweb="tab"]:hover{ background:#dbeafe !important; color:#0a2540 !important; transform:translateY(-1px) !important; }
        .stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#2563eb,#1e40af) !important; color:#fff !important;
            box-shadow:0 4px 10px rgba(37,99,235,.25) !important; border:none !important; transform:translateY(-1px) !important; }
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
        .stButton>button { background:linear-gradient(90deg,#10b981,#059669) !important; color:#fff !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:600 !important; }
        .stButton>button:hover { background:linear-gradient(90deg,#34d399,#10b981) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 10px rgba(16,185,129,.25) !important; }
        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{ background:#fff7ed !important; color:#7c2d12 !important; border:1px solid #fed7aa !important;
            border-radius:10px !important; font-weight:700 !important; min-width:130px !important; padding:0.6rem 1.2rem !important; }
        .stTabs [data-baseweb="tab"]:hover{ background:#ffedd5 !important; color:#4a1d0a !important; }
        .stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#f97316,#ef4444) !important; color:#fff !important; border:none !important;
            box-shadow:0 4px 12px rgba(249,115,22,.25) !important; transform:translateY(-1px) !important; }
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
        .stButton>button { background:linear-gradient(90deg,#06b6d4,#3b82f6) !important; color:#0b1020 !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:700 !important; }
        .stButton>button:hover { background:linear-gradient(90deg,#22d3ee,#60a5fa) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 12px rgba(34,211,238,.35) !important; }
        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{ background:#111827 !important; color:#cbd5e1 !important; border:1px solid #1f2937 !important;
            border-radius:10px !important; font-weight:700 !important; min-width:130px !important; padding:0.6rem 1.2rem !important; }
        .stTabs [data-baseweb="tab"]:hover{ background:#0b1220 !important; color:#f1f5f9 !important; }
        .stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#06b6d4,#6366f1) !important; color:#0b1020 !important; border:none !important;
            box-shadow:0 4px 12px rgba(6,182,212,.35) !important; transform:translateY(-1px) !important; }
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
        .stButton>button { background:linear-gradient(90deg,#818cf8,#22d3ee) !important; color:#0b1220 !important; border:none !important;
            border-radius:8px !important; padding:.5rem 1rem !important; font-weight:700 !important; }
        .stButton>button:hover { background:linear-gradient(90deg,#a5b4fc,#67e8f9) !important; transform:translateY(-2px) !important;
            box-shadow:0 4px 12px rgba(129,140,248,.35) !important; }
        .stTabs [data-baseweb="tab-list"]{ gap:16px !important; }
        .stTabs [data-baseweb="tab"]{ background:#111827 !important; color:#cbd5e1 !important; border:1px solid #1f2937 !important;
            border-radius:10px !important; font-weight:700 !important; min-width:130px !important; padding:0.6rem 1.2rem !important; }
        .stTabs [data-baseweb="tab"]:hover{ background:#0b1220 !important; color:#f8fafc !important; }
        .stTabs [aria-selected="true"]{ background:linear-gradient(135deg,#22d3ee,#818cf8,#a78bfa) !important; color:#0b1220 !important; border:none !important;
            box-shadow:0 4px 12px rgba(34,211,238,.35) !important; transform:translateY(-1px) !important; }
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
    .centered-title span { font-size: 2.8rem; animation: pulse 2s infinite alternate; }
    @keyframes pulse { from { transform: scale(1); opacity: 0.85; } to { transform: scale(1.2); opacity: 1; } }
    .subtitle { text-align: center; opacity: .9; font-size: 1rem; margin-top: -0.6rem; font-style: italic; }
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

# ============================================================
# Helpers for Multi-GEO runner (self-contained inside app.py)
# ============================================================
def _read_any_table(file_like_or_bytes, name_hint: str|None):
    """Robust table reader for TSV/CSV/XLSX."""
    bio = file_like_or_bytes if isinstance(file_like_or_bytes, io.BytesIO) else io.BytesIO(file_like_or_bytes)
    suffix = (os.path.splitext(name_hint)[1].lower() if name_hint else "")
    try:
        bio.seek(0)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(bio, engine="openpyxl")
        if suffix in (".tsv", ".txt"):
            return pd.read_csv(bio, sep="\t")
        if suffix == ".csv":
            return pd.read_csv(bio)
        # auto
        bio.seek(0)
        try:
            return pd.read_excel(bio, engine="openpyxl")
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio, sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Unable to read table ({name_hint or 'unknown'}): {e}")

def _normalize_gene_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.name is None or df.index.name == "":
        # guess first column is gene id
        df = df.rename(columns={df.columns[0]: "Biomarker"}).set_index("Biomarker")
    df.index = df.index.astype(str).str.strip().str.upper().str.replace(r"\.\d+$", "", regex=True)
    return df

def _resolve_id_column(meta_df: pd.DataFrame, candidates: list[str]) -> str|None:
    lower = {re.sub(r"[^a-z0-9]","", c.lower()): c for c in meta_df.columns}
    for c in candidates:
        key = re.sub(r"[^a-z0-9]","", c.lower())
        if key in lower:
            return lower[key]
    # fallback: best overlap vs sample columns if present in meta
    return None

def _combine_counts(list_of_counts, min_overlap_genes: int) -> tuple[pd.DataFrame,bool]:
    """Intersect by genes; return (combined, combined_ok)."""
    if not list_of_counts:
        return pd.DataFrame(), False
    gene_sets = [set(df.index) for df in list_of_counts]
    inter = set.intersection(*gene_sets) if len(gene_sets) > 1 else gene_sets[0]
    if len(inter) < int(min_overlap_genes):
        return pd.DataFrame(), False
    aligned = [df.loc[sorted(inter)] for df in list_of_counts]
    combined = pd.concat(aligned, axis=1, join="inner")
    return combined, True

def _summarize_multi(mres: dict) -> dict:
    out = {"sizes": {}, "platforms": {}, "points": {"approx_sparsity": {}}}
    runs = (mres or {}).get("runs", {})
    for name, res in runs.items():
        try:
            with open(res["report_json"], "r") as fh:
                rep = json.load(fh)
        except Exception:
            rep = {}
        out["sizes"][name] = rep.get("shapes", {}).get("samples")
        out["platforms"][name] = rep.get("qc", {}).get("platform", "Unknown")
        zf = rep.get("qc", {}).get("zero_fraction", None)
        if zf is not None:
            out["points"]["approx_sparsity"][name] = round(float(zf), 3)
    return out

def run_pipeline_multi_local(
    datasets: list,
    attempt_combine: bool,
    combine_minoverlap_genes: int,
    out_root: str,
    pca_topk_features: int,
    make_nonlinear: bool,
) -> dict:
    """
    Self-contained multi-dataset runner that calls hz.run_pipeline for each dataset,
    and optionally creates a combined run if gene overlap is large enough.
    Each item in `datasets`:
      {
        'geo': str, 'counts': BytesIO, 'meta': BytesIO,
        'meta_id_cols': [...], 'meta_group_cols': [...],
        'meta_batch_col': 'Batch' | None
      }
    """
    os.makedirs(out_root, exist_ok=True)
    per_runs = {}
    counts_list = []
    labels_list = []

    # First: run each dataset individually
    for ds in datasets:
        name = ds["geo"]
        counts = _read_any_table(ds["counts"], name_hint=f"{name}_counts.csv")
        meta = _read_any_table(ds["meta"], name_hint=f"{name}_meta.csv")

        # Normalize counts: gene x sample
        if "Biomarker" in counts.columns:
            counts = counts.set_index("Biomarker")
        counts = _normalize_gene_index(counts)

        # If meta has no explicit batch, add dataset label as batch
        meta_id_col = _resolve_id_column(meta, ds.get("meta_id_cols") or [])
        if meta_id_col is None:
            # leave as-is; hz.run_pipeline will try to align by bare_id later
            meta_id_col = (ds.get("meta_id_cols") or ["sample","Sample","Id","ID"])[0]
            meta[meta_id_col] = meta[meta.columns[0]].astype(str)

        if (ds.get("meta_batch_col") or "").strip() == "" or (ds.get("meta_batch_col") not in meta.columns):
            meta["__dataset_batch__"] = name
            batch_col = "__dataset_batch__"
        else:
            batch_col = ds["meta_batch_col"]

        # Write temporary files to feed into hz.run_pipeline (single-expression mode)
        tmpd = tempfile.mkdtemp()
        counts_path = os.path.join(tmpd, f"{name}_counts.tsv")
        meta_path = os.path.join(tmpd, f"{name}_meta.tsv")
        counts.to_csv(counts_path, sep="\t")
        meta.to_csv(meta_path, sep="\t", index=False)

        out = hz.run_pipeline(
            single_expression_file=counts_path,
            single_expression_name_hint=os.path.basename(counts_path),
            metadata_file=meta_path,
            metadata_name_hint=os.path.basename(meta_path),
            metadata_id_cols=ds.get("meta_id_cols") or ["sample","Sample","Id","ID"],
            metadata_group_cols=ds.get("meta_group_cols") or ["group","Group","condition","Condition","phenotype","Phenotype"],
            metadata_batch_col=batch_col,
            out_root=os.path.join(out_root, name),
            pca_topk_features=pca_topk_features,
            make_nonlinear=make_nonlinear,
        )
        per_runs[name] = out
        counts_list.append(counts)
        labels_list.append(name)

    # Second: try to make a combined run if requested
    decision = {"attempted": attempt_combine, "combined": False, "reason": None}
    combined_out = None

    if attempt_combine and len(counts_list) >= 2:
        combined_counts, ok = _combine_counts(counts_list, combine_minoverlap_genes)
        if not ok:
            decision["reason"] = f"Insufficient overlapping genes (< {combine_minoverlap_genes})."
        else:
            # Build a minimalist combined meta: sample -> bare_id + batch = dataset label
            sample_ids = combined_counts.columns.astype(str)
            meta_comb = pd.DataFrame({
                "bare_id": sample_ids,
                "group": "ALL",
                "batch": [next(lbl for df, lbl in zip(counts_list, labels_list) if c in df.columns) for c in sample_ids]
            })
            tmpd = tempfile.mkdtemp()
            cpath = os.path.join(tmpd, "combined_counts.tsv")
            mpath = os.path.join(tmpd, "combined_meta.tsv")
            combined_counts.to_csv(cpath, sep="\t")
            meta_comb.to_csv(mpath, sep="\t", index=False)

            combined_out = hz.run_pipeline(
                single_expression_file=cpath,
                single_expression_name_hint=os.path.basename(cpath),
                metadata_file=mpath,
                metadata_name_hint=os.path.basename(mpath),
                metadata_id_cols=["bare_id","sample","Sample","Id","ID"],
                metadata_group_cols=["group","Group","condition","Condition","phenotype","Phenotype"],
                metadata_batch_col="batch",
                out_root=os.path.join(out_root, "COMBINED"),
                pca_topk_features=pca_topk_features,
                make_nonlinear=make_nonlinear,
            )
            decision["combined"] = True
            decision["reason"] = f"Combined on {combined_counts.shape[0]} overlapping genes."

    return {
        "runs": per_runs,
        "combined": combined_out,
        "combine_decision": decision,
    }

# =========================
# INPUTS (Single/Multi-file)
# =========================
st.caption("Upload expression data (single matrix **or** one file per group) and corresponding metadata, then click **Run Harmonization**.")

mode = st.radio(
    "Expression upload mode",
    ["Single expression matrix", "Multiple files (one per group)"],
    horizontal=True,
)

single_expr_file = None
normal_file = atypia_file = hpv_pos_file = hpv_neg_file = None

if mode == "Multiple files (one per group)":
    with st.expander("1) Upload Expression Files (one per group)"):
        col1, col2 = st.columns(2)
        with col1:
            normal_file = st.file_uploader("First (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="normal")
            atypia_file = st.file_uploader("Second (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="atypia")
        with col2:
            hpv_pos_file = st.file_uploader("Third (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="hpvp")
            hpv_neg_file = st.file_uploader("Fourth (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="hpvn")
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

# ============================================================
# RESULTS TABS — ALWAYS VISIBLE (incl. Multi-GEO)
# ============================================================
st.subheader("Results")

tabs = st.tabs(["Overview", "QC", "PCA & Embeddings", "DE & GSEA", "Outliers", "Multi-GEO", "Files"])

# Convenience handles to last successful single-dataset run (if any)
out_curr = st.session_state.get("out")
run_id = st.session_state.get("run_id")

# ---- Overview
with tabs[0]:
    if not out_curr:
        st.info("No run loaded yet. Upload data and click **Run Harmonization**.")
    else:
        report = {}
        try:
            with open(out_curr["report_json"], "r") as fh:
                report = json.load(fh)
        except Exception:
            pass

        qc = report.get("qc", {})
        shp = report.get("shapes", {})
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
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
            st.metric("Silhouette (batch)", f'{sil_batch:.2f}' if isinstance(sil_batch, (int, float)) else "—")
            st.markdown('<div class="smallcaps">Lower is better</div></div>', unsafe_allow_html=True)

        st.json(report if report else {"info": "report.json not found"})
        fig_dir = out_curr["figdir"]
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
    if not out_curr:
        st.info("No run loaded yet.")
    else:
        fig_dir = out_curr["figdir"]
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
    if not out_curr:
        st.info("No run loaded yet.")
    else:
        fig_dir = out_curr["figdir"]
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

# ---- DE & GSEA
with tabs[3]:
    if not out_curr:
        st.info("No run loaded yet.")
    else:
        fig_dir = out_curr["figdir"]
        de_dir = os.path.join(out_curr["outdir"], "de")
        de_files = [f for f in os.listdir(de_dir)] if os.path.isdir(de_dir) else []
        contrasts = sorted([f.replace("DE_", "").replace(".tsv", "") for f in de_files if f.startswith("DE_")])
        pick = st.selectbox("Select contrast", contrasts) if contrasts else None
        if pick:
            st.write(f"### Differential Expression: {pick}")
            for pth in [
                os.path.join(fig_dir, f"volcano_{pick}.png"),
                os.path.join(fig_dir, f"ma_{pick}.png"),
                os.path.join(fig_dir, f"heatmap_top_50_{pick}.png"),
            ]:
                if os.path.exists(pth):
                    st.image(pth, caption=os.path.basename(pth), use_column_width=True)
            tsv = os.path.join(de_dir, f"DE_{pick}.tsv")
            try:
                df = pd.read_csv(tsv, sep="\t", index_col=0).head(50)
                st.dataframe(df, use_container_width=True, key=f"de_table__{run_id}__{pick}")
                with open(tsv, "rb") as fh:
                    safe_download_button(
                        "⬇️ Download full DE table",
                        fh.read(),
                        file_name=f"DE_{pick}.tsv",
                        mime="text/tab-separated-values",
                        key=f"dl_de__{run_id}__{pick}",
                    )
            except Exception:
                pass
        gsea_dir = os.path.join(out_curr["outdir"], "gsea")
        if os.path.isdir(gsea_dir):
            st.write("### GSEA Results")
            for f in sorted(os.listdir(gsea_dir)):
                if f.endswith(".tsv"):
                    st.write(f)
                    try:
                        df = pd.read_csv(os.path.join(gsea_dir, f), sep="\t").head(30)
                        st.dataframe(df, use_container_width=True, key=f"gsea_{run_id}__{f}")
                    except Exception:
                        pass

# ---- Outliers
with tabs[4]:
    if not out_curr:
        st.info("No run loaded yet. Please run the pipeline.")
    else:
        outliers_path = os.path.join(out_curr["outdir"], "outliers.tsv")
        meta_path = os.path.join(out_curr["outdir"], "metadata.tsv")
        st.caption(f"Run: **{run_id}**  •  Outdir: `{out_curr['outdir']}`")
        if os.path.exists(outliers_path):
            mtime = int(os.path.getmtime(outliers_path))
            cache_buster = f"{run_id}__{mtime}"
            try:
                df = pd.read_csv(outliers_path, sep="\t", index_col=0)
                if os.path.exists(meta_path):
                    meta_df = pd.read_csv(meta_path, sep="\t", index_col=0)
                    grp_col = "group_raw" if "group_raw" in meta_df.columns else "group"
                    display_df = (
                        df.copy()
                        .assign(sample=df.index)
                        .join(meta_df[["bare_id", grp_col]].rename(columns={grp_col: "group"}), how="left")
                        .set_index("sample")
                        .rename(columns={"IsolationForest": "IsolationForest_flag", "LOF": "LOF_flag"})
                        [["bare_id", "group", "IsolationForest_flag", "LOF_flag"]]
                    )
                else:
                    display_df = df.rename(columns={"IsolationForest": "IsolationForest_flag", "LOF": "LOF_flag"})
                st.write("### Outlier flags (1 = outlier)")
                st.dataframe(display_df, use_container_width=True, key=f"outliers_df__{cache_buster}")
                with open(outliers_path, "rb") as fh:
                    safe_download_button(
                        "⬇️ Download outlier table",
                        fh.read(),
                        file_name="outliers.tsv",
                        mime="text/tab-separated-values",
                        key=f"dl_outliers__{cache_buster}",
                    )
            except Exception as e:
                st.warning(f"Could not load outliers for this run: {e}")
        else:
            st.info("No outlier table found for this run.")

# ---- Multi-GEO (ALWAYS VISIBLE)
with tabs[5]:
    st.write("### Batch-run up to 5 GEO datasets (each with `prep_counts` and `prep_meta`)")
    st.caption("counts = gene × sample; meta = clinical/phenotype table. IDs should map to sample names (or include candidate ID columns).")

    max_sets = 5
    attempt_combine = st.checkbox("Attempt to combine into one run when compatible", value=True, key="mx_attempt")
    min_overlap = st.number_input("Minimum overlapping genes required to combine", 1000, 20000, 3000, step=500, key="mx_overlap")

    geo_rows = []
    for i in range(max_sets):
        with st.expander(f"Dataset {i + 1}", expanded=(i == 0)):
            geo = st.text_input(f"GEO ID (optional, label only) #{i + 1}", key=f"geo_id_{i}", value="" if i > 0 else "GSE273902")
            cfile = st.file_uploader("prep_counts (TSV/CSV/XLSX)", type=["tsv","txt","csv","xlsx","xls"], key=f"counts_{i}")
            mfile = st.file_uploader("prep_meta (TSV/CSV/XLSX)", type=["tsv","txt","csv","xlsx","xls"], key=f"meta_{i}")
            idc = st.text_input("Candidate ID columns in meta (comma-sep)", "bare_id,Id,ID,id,sample,Sample", key=f"idcols_{i}")
            grpc = st.text_input("Candidate GROUP columns in meta (comma-sep)", "group,Group,condition,Condition,phenotype,Phenotype", key=f"grpcols_{i}")
            bcol = st.text_input("Batch column (optional)", "", key=f"bcol_{i}")
            if cfile and mfile:
                geo_rows.append({
                    "geo": (geo.strip() or f"DS{i + 1}"),
                    "counts": io.BytesIO(cfile.getvalue()),
                    "meta": io.BytesIO(mfile.getvalue()),
                    "meta_id_cols": [c.strip() for c in idc.split(",") if c.strip()],
                    "meta_group_cols": [c.strip() for c in grpc.split(",") if c.strip()],
                    "meta_batch_col": (bcol.strip() or None),
                })

    run_multi = safe_button("🚀 Run Multi-GEO Harmonization", use_container_width=True, key="run_multi_geo_btn")

    if run_multi:
        if not geo_rows:
            st.error("Please add at least one dataset with both files.")
            st.stop()
        try:
            with st.spinner("Running multi-dataset analysis..."):
                multi_res = run_pipeline_multi_local(
                    datasets=geo_rows,
                    attempt_combine=attempt_combine,
                    combine_minoverlap_genes=int(min_overlap),
                    out_root=os.path.join(out_dir, "multi_geo"),
                    pca_topk_features=int(pca_topk),
                    make_nonlinear=do_nonlinear,
                )
            st.success("Multi-dataset processing complete.")
            st.session_state.multi_geo = multi_res
        except Exception as e:
            st.error(f"Multi-GEO run failed: {e}")

    # Summary + downloads if available
    if st.session_state.get("multi_geo"):
        multi_res = st.session_state.multi_geo
        st.write("#### Combination decision")
        st.json(multi_res.get("combine_decision", {}))

        st.write("#### Results per dataset")
        cols = st.columns(3)
        i = 0
        for name, res in (multi_res.get("runs") or {}).items():
            pzip = res.get("zip")
            if pzip and os.path.exists(pzip):
                with cols[i % 3]:
                    with open(pzip, "rb") as fh:
                        safe_download_button(f"⬇️ {name} (ZIP)", fh.read(), file_name=f"{name}_results.zip",
                                             mime="application/zip",
                                             key=f"dl_zip_{name}_{st.session_state.get('run_token') or 'tok'}")
            i += 1

        if multi_res.get("combined"):
            st.write("#### Combined run")
            cz = multi_res["combined"].get("zip")
            if cz and os.path.exists(cz):
                with open(cz, "rb") as fh:
                    safe_download_button("⬇️ COMBINED (ZIP)", fh.read(), file_name="COMBINED_results.zip",
                                         mime="application/zip", key="dl_zip_combined")

        # Diabetes-style cross-dataset analysis bullets
        try:
            summary = _summarize_multi(multi_res)
        except Exception:
            summary = {}

        st.write("### Cross-dataset analysis (Diabetes example framing)")
        st.markdown(
            f"""
1. **TISSUE HOMOGENEITY** — All datasets are expected to be **human pancreatic islets** (confirm in uploaded clinical metadata).  
2. **PLATFORM DIVERSITY** — Observed platforms: `{", ".join(sorted(set(summary.get("platforms", {}).values()))) or "n/a"}`.  
   Mixed RNA-seq / microarray demands careful normalization; cross-platform **validation** increases robustness.  
3. **SAMPLE SIZE VARIATION** — Samples per dataset: `{summary.get("sizes", {})}`.  
   Use per-contrast power checks; DE is skipped if groups < 2 samples each.  
4. **CLINICAL HETEROGENEITY** — Harmonize **T2D status, HbA1c, BMI** to unified definitions across GEOs.  
   Consider covariate adjustment or stratified DE if distributions differ.  
5. **ZERO-INFLATION / SPARSITY** — Approx. zero fraction by run: `{summary.get("points", {}).get("approx_sparsity", "n/a")}`.  
6. **BATCH EFFECTS** — Silhouette (batch/group) KPIs are tracked per run; inspect combined run if created.
            """
        )

# ---- Files
with tabs[6]:
    if not out_curr:
        st.info("No run loaded yet.")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.write("**Core Tables**")
            core_files = [
                ("Combined Expression", os.path.join(out_curr["outdir"], "expression_combined.tsv")),
                ("Harmonized Expression", os.path.join(out_curr["outdir"], "expression_harmonized.tsv")),
                ("PCA Scores", os.path.join(out_curr["outdir"], "pca_scores.tsv")),
                ("Metadata (aligned)", os.path.join(out_curr["outdir"], "metadata.tsv")),
                ("Report (JSON)", out_curr["report_json"]),
            ]
            for label, path in core_files:
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        safe_download_button(
                            f"⬇️ {label}",
                            fh.read(),
                            file_name=os.path.basename(path),
                            mime="text/plain",
                            use_container_width=True,
                            key=f"dl_core__{run_id}__{label}",
                        )
        with colB:
            try:
                with open(out_curr["zip"], "rb") as fh:
                    safe_download_button(
                        label="⬇️ Download ALL results (ZIP)",
                        data=fh.read(),
                        file_name="harmonization_results.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key=f"dl_zip__{run_id}",
                    )
            except Exception as e:
                st.warning(f"Could not open ZIP for download: {e}")

# ============================================================
# SINGLE RUN ACTION BUTTON (kept at bottom to avoid confusion)
# ============================================================
run = safe_button("🚀 Run Harmonization (single/multi-file above)", type="primary", use_container_width=True, key="run_single_btn")

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
        "out_root": "out",  # replaced below with timestamped subfolder
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
        if normal_file: groups["First"] = io.BytesIO(normal_file.getvalue())
        if atypia_file: groups["Second"] = io.BytesIO(atypia_file.getvalue())
        if hpv_pos_file: groups["Third"] = io.BytesIO(hpv_pos_file.getvalue())
        if hpv_neg_file: groups["Fourth"] = io.BytesIO(hpv_neg_file.getvalue())
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
            run_id_now = _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
            kwargs["out_root"] = os.path.join(out_dir, run_id_now)
            out = hz.run_pipeline(**kwargs)

            # Persist current run + a fresh token to break widget caches
            st.session_state.run_id = run_id_now
            st.session_state.out = out
            st.session_state.run_token = f"{run_id_now}-{_dt.datetime.now().timestamp():.0f}"

            # >>> IMPORTANT: Clear caches so no old DataFrames/objects linger
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass

        st.success("Done!")
        # (optional) show PCA skipped reason if any
        try:
            with open(out["report_json"], "r") as fh:
                rep = json.load(fh)
            if rep.get("notes", {}).get("pca_skipped_reason"):
                st.warning("PCA/UMAP skipped: " + str(rep["notes"]["pca_skipped_reason"]))
        except Exception:
            pass

    except Exception as e:
        st.error(f"Run failed: {e}")
        if gmt_file:
            shutil.rmtree(os.path.dirname(gmt_path), ignore_errors=True)
        st.stop()

# Cleanup temp GMT if used
# (If single-run failed before we got here, the above handler already tries to remove it)
