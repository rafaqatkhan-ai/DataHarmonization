# app.py — Drive-aware harmonization UI with disease search & per-dataset comparisons
import agent as ag  # NEW
import os, io, tempfile, shutil, json
import datetime as _dt
import itertools
import pandas as pd
import streamlit as st

import harmonizer as hz
import drive_ingest as din

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="🧬 Data Harmonization & QC Suite", page_icon="🧬", layout="wide")

# Cache clear helper
def _clear_caches():
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass

# Init session state
if "run_id" not in st.session_state: st.session_state.run_id = None
if "out" not in st.session_state: st.session_state.out = None
if "run_token" not in st.session_state: st.session_state.run_token = None
if "multi" not in st.session_state: st.session_state.multi = None
if "last_plan" not in st.session_state: st.session_state.last_plan = None
if "input_source" not in st.session_state: st.session_state.input_source = "Manual upload"
if "selected_dataset" not in st.session_state: st.session_state.selected_dataset = None  # NEW: persist chosen dataset

# =========================
# THEME SELECTOR
# =========================
def apply_theme(t: str):
    if t == "Light Gray":
        css = """<style>
        [data-testid="stAppViewContainer"]{background:#f3f4f6!important;color:#0f172a!important}
        [data-testid="stSidebar"]{background:#e5e7eb!important;border-right:1px solid #cbd5e1!important}
        [data-testid="stVerticalBlock"]{background:#ffffff!important;border-radius:12px;padding:1rem;margin-bottom:1rem;box-shadow:0 2px 8px rgba(15,23,42,0.06)!important}
        h1,h2,h3,h4,h5,h6,p,label,span{color:#0f172a!important}
        .stButton>button{background:linear-gradient(90deg,#2563eb,#1d4ed8)!important;color:#fff!important;border:none!important;border-radius:8px!important;padding:.5rem 1rem!important;font-weight:600!important}
        .stButton>button:hover{background:linear-gradient(90deg,#3b82f6,#2563eb)!important;transform:translateY(-2px)!important;box-shadow:0 4px 10px rgba(37,99,235,.25)!important}
        .stTabs [data-baseweb="tab-list"]{gap:16px!important}
        .stTabs [data-baseweb="tab"]{background:#e9f0fa!important;color:#1e3a8a!important;border:1px solid #cbd5e1!important;border-radius:10px!important;font-weight:700!important;min-width:130px!important;padding:0.6rem 1.2rem!important}
        .metric-card{background:#f8fafc!important;border:1px solid #e2e8f0!important;border-radius:12px!important;padding:14px 16px!important}
        .smallcaps{color:#475569!important}
        </style>"""
    elif t == "Soft Off-White":
        css = """<style>
        [data-testid="stAppViewContainer"]{background:#faf7f2!important;color:#1f2937!important}
        [data-testid="stSidebar"]{background:#f3efe8!important;border-right:1px solid #e5e7eb!important}
        [data-testid="stVerticalBlock"]{background:#ffffff!important;border-radius:12px!important;padding:1rem!important;margin-bottom:1rem!important;box-shadow:0 2px 10px rgba(0,0,0,0.06)!important}
        h1,h2,h3,h4,h5,h6,p,label,span{color:#111827!important}
        .stButton>button{background:linear-gradient(90deg,#10b981,#059669)!important;color:#fff!important;border:none!important;border-radius:8px!important;padding:.5rem 1rem!important;font-weight:600!important}
        .stButton>button:hover{background:linear-gradient(90deg,#34d399,#10b981)!important;transform:translateY(-2px)!important;box-shadow:0 4px 10px rgba(16,185,129,.25)!important}
        .stTabs [data-baseweb="tab-list"]{gap:16px!important}
        .stTabs [data-testid="stVerticalBlock"]{overflow:visible!important}
        .stTabs [data-baseweb="tab"]{background:#fff7ed!important;color:#7c2d12!important;border:1px solid #fed7aa!important;border-radius:10px!important;font-weight:700!important;min-width:130px!important;padding:0.6rem 1.2rem!important}
        .metric-card{background:#ffffff!important;border:1px solid #f3f4f6!important;border-radius:12px!important;padding:14px 16px!important}
        .smallcaps{color:#6b7280!important}
        </style>"""
    elif t == "Deep Navy":
        css = """<style>
        [data-testid="stAppViewContainer"]{background:#0b1020!important;color:#e5e7eb!important}
        [data-testid="stSidebar"]{background:#0f172a!important;color:#f3f4f6!important;border-right:1px solid #1f2a44!important}
        [data-testid="stVerticalBlock"]{background:#0d142a!important;border-radius:12px!important;padding:1rem!important;margin-bottom:1rem!important;box-shadow:0 2px 12px rgba(0,0,0,.5)!important}
        h1,h2,h3,h4,h5,h6,p,label,span{color:#e5e7eb!important}
        .stButton>button{background:linear-gradient(90deg,#06b6d4,#3b82f6)!important;color:#0b1020!important;border:none!important;border-radius:8px!important;padding:.5rem 1rem!important;font-weight:700!important}
        .stButton>button:hover{background:linear-gradient(90deg,#22d3ee,#60a5fa)!important;transform:translateY(-2px)!important;box-shadow:0 4px 12px rgba(34,211,238,.35)!important}
        .stTabs [data-baseweb="tab-list"]{gap:16px!important}
        .stTabs [data-baseweb="tab"]{background:#111827!important;color:#cbd5e1!important;border:1px solid #1f2937!important;border-radius:10px!important;font-weight:700!important;min-width:130px!important;padding:0.6rem 1.2rem!important}
        .metric-card{background:#0f172a!important;border:1px solid rgba(99,102,241,.2)!important;border-radius:12px!important;padding:14px 16px!important}
        .smallcaps{color:#93c5fd!important}
        </style>"""
    else:
        css = """<style>
        [data-testid="stAppViewContainer"]{background:#0f172a!important;color:#e2e8f0!important}
        [data-testid="stSidebar"]{background:#111827!important;border-right:1px solid #1f2937!important}
        [data-testid="stVerticalBlock"]{background:#0b1220!important;border-radius:12px!important;padding:1rem!important;margin-bottom:1rem!important;box-shadow:0 2px 10px rgba(2,6,23,.6)!important}
        h1,h2,h3,h4,h5,h6,p,label,span{color:#e2e8f0!important}
        .stButton>button{background:linear-gradient(90deg,#818cf8,#22d3ee)!important;color:#0b1220!important;border:none!important;border-radius:8px!important;padding:.5rem 1rem!important;font-weight:700!important}
        .stButton>button:hover{background:linear-gradient(90deg,#a5b4fc,#67e8f9)!important;transform:translateY(-2px)!important;box-shadow:0 4px 12px rgba(129,140,248,.35)!important}
        .stTabs [data-baseweb="tab-list"]{gap:16px!important}
        .stTabs [data-baseweb="tab"]{background:#111827!important;color:#cbd5e1!important;border:1px solid #1f2937!important;border-radius:10px!important;font-weight:700!important;min-width:130px!important;padding:0.6rem 1.2rem!important}
        .metric-card{background:#0b1220!important;border:1px solid rgba(148,163,184,.25)!important;border-radius:12px!important;padding:14px 16px!important}
        .smallcaps{color:#94a3b8!important}
        </style>"""
    st.markdown(css, unsafe_allow_html=True)

theme = st.selectbox("Theme", ["Light Gray","Soft Off-White","Deep Navy","Slate Blue"], index=0)
apply_theme(theme)

# =========================
# TITLE
# =========================
st.markdown("""
<style>
.centered-title{font-size:2.6rem;font-weight:900;text-align:center;
background:linear-gradient(90deg,#1e3a8a,#2563eb,#6366f1,#7c3aed);
background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:colorShift 8s ease infinite;margin-top:-0.5rem}
@keyframes colorShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.subtitle{text-align:center;opacity:.9;font-size:1rem;margin-top:-0.6rem;font-style:italic}
</style>
<h1 class="centered-title"><span>🧬</span> Data Harmonization & QC Suite <span>🧬</span></h1>
<p class="subtitle">Upload manually or pull from Google Drive by disease keywords—then harmonize, QC, and compare.</p>
""", unsafe_allow_html=True)

# =========================
# Input Source (Manual vs Drive)
# =========================
st.sidebar.header("Input Source")
input_source = st.sidebar.radio(
    "Choose input source",
    ["Manual upload", "Google Drive (deg_data)"],
    index=0,
    help="Use Drive mode to search by disease keywords inside your deg_data folder and auto-run."
)
st.session_state.input_source = input_source

# =========================
# Manual Upload Controls (existing modes)
# =========================
def safe_button(label, **kwargs):
    try:
        return st.button(label, **kwargs)
    except Exception:
        kwargs.pop("type", None); kwargs.pop("use_container_width", None)
        return st.button(label, **kwargs)

def safe_download_button(label, data=None, **kwargs):
    try:
        return st.download_button(label=label, data=data, **kwargs)
    except Exception:
        kwargs.pop("use_container_width", None)
        try:
            return st.download_button(label=label, data=data, **kwargs)
        except Exception:
            for k in ["mime","file_name","help","key"]: kwargs.pop(k, None)
            return st.download_button(label=label, data=data)

mode = None
single_expr_file = None
normal_file = atypia_file = hpv_pos_file = hpv_neg_file = None
metadata_file = None
id_cols = grp_cols = batch_col = ""
gmt_file = None
out_dir = "out"
pca_topk = 5000
do_nonlinear = True
combine_thresh = 3000
multi_datasets = None

if input_source == "Manual upload":
    mode = st.radio(
        "Expression upload mode",
        ["Single expression matrix", "Multiple files (one per group)", "Multiple datasets (each has its own metadata)"],
        horizontal=True
    )
    st.caption("Upload expression data and corresponding metadata, then click **Run Harmonization**.")

    if mode == "Multiple files (one per group)":
        with st.expander("1) Upload Expression Files (one per group)"):
            col1, col2 = st.columns(2)
            with col1:
                normal_file = st.file_uploader("First (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="normal")
                atypia_file = st.file_uploader("Second (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="atypia")
            with col2:
                hpv_pos_file = st.file_uploader("Third (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="hpvp")
                hpv_neg_file = st.file_uploader("Fourth (XLSX/CSV/TSV)", type=["xlsx","csv","tsv","txt"], key="hpvn")
    elif mode == "Single expression matrix":
        with st.expander("1) Upload Single Expression Matrix (XLSX/CSV/TSV)"):
            single_expr_file = st.file_uploader("Expression matrix", type=["xlsx","csv","tsv","txt"], key="single_expr")

    with st.expander("2) Upload Metadata (TSV/CSV/XLSX) [skip this for multi-dataset mode]"):
        metadata_file = st.file_uploader("Metadata file", type=["tsv","csv","txt","xlsx"], key="meta")
        id_cols = st.text_input("Candidate ID columns (comma-separated)",
                                "sample,Sample,Id,ID,id,CleanID,sample_id,Sample_ID,SampleID")
        grp_cols = st.text_input("Candidate GROUP columns (comma-separated)",
                                 "group,Group,condition,Condition,phenotype,Phenotype")
        batch_col = st.text_input("Batch column name (optional; leave blank to auto-detect)", "")
        if metadata_file is not None:
            try:
                bio = io.BytesIO(metadata_file.getvalue())
                name = metadata_file.name.lower()
                if name.endswith((".xlsx", ".xls")):
                    mprev = pd.read_excel(bio, engine="openpyxl")
                elif name.endswith((".tsv", ".txt")):
                    mprev = pd.read_csv(bio, sep="\t")
                else:
                    mprev = pd.read_csv(bio, sep=None, engine="python")
                st.caption(f"Detected metadata columns: {list(mprev.columns)}")
                _BATCH_HINTS = ["batch","Batch","BATCH","center","Center","site","Site","location","Location","series","Series",
                                "geo_series","GEO_series","run","Run","lane","Lane","plate","Plate","sequencer","Sequencer",
                                "flowcell","Flowcell","library","Library","library_prep","LibraryPrep","study","Study","project",
                                "Project","lab","Lab","date","Date","collection_date","CollectionDate","source_name_ch1","title",
                                "characteristics_ch1","characteristics"]
                candidates = [c for c in _BATCH_HINTS if c in mprev.columns]
                if candidates:
                    st.caption(f"Batch-like columns found: {candidates}")
                    pick = candidates[0]
                    vc = mprev[pick].astype(str).value_counts().head(20)
                    with st.expander(f"Preview batch levels from `{pick}` (top 20)"):
                        st.write(vc)
                with st.expander("Preview first 5 rows of metadata"):
                    st.dataframe(mprev.head(5), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not preview metadata columns: {e}")

    with st.expander("3) Optional: GSEA gene set (.gmt)"):
        gmt_file = st.file_uploader("Gene set GMT (optional)", type=["gmt"])

    with st.expander("Advanced settings"):
        out_dir = st.text_input("Output directory", "out")
        pca_topk = st.number_input("Top variable genes for PCA", min_value=500, max_value=50000, value=5000, step=500)
        do_nonlinear = st.checkbox("Make UMAP/t-SNE (if available)", value=True)

    # Multi-dataset (manual)
    if mode == "Multiple datasets (each has its own metadata)":
        with st.expander("1) Upload Datasets (each has its own expression + metadata)"):
            n_ds = st.number_input("How many datasets?", min_value=2, max_value=12, value=3, step=1)
            st.caption("For each dataset, provide an expression matrix (XLSX/CSV/TSV) and a matching metadata file.")
            multi_datasets = []
            for i in range(int(n_ds)):
                with st.container():
                    st.markdown(f"**Dataset {i+1}**")
                    colL, colR = st.columns([2,2])
                    with colL:
                        ds_label = st.text_input(f"Label {i+1}", value=f"DS{i+1}", key=f"ds_label_{i}")
                        expr_file = st.file_uploader(f"Expression {i+1}", type=["xlsx","csv","tsv","txt"], key=f"expr_{i}")
                    with colR:
                        meta_file_i = st.file_uploader(f"Metadata {i+1}", type=["tsv","csv","txt","xlsx"], key=f"meta_{i}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        id_cols_i = st.text_input(f"ID columns {i+1}",
                                                  "sample,Sample,Id,ID,id,CleanID,sample_id,Sample_ID,SampleID",
                                                  key=f"idcols_{i}")
                    with c2:
                        grp_cols_i = st.text_input(f"GROUP columns {i+1}",
                                                   "group,Group,condition,Condition,phenotype,Phenotype",
                                                   key=f"grpcols_{i}")
                    with c3:
                        batch_col_i = st.text_input(f"Batch column (optional) {i+1}", "", key=f"batchcol_{i}")

                    if expr_file and meta_file_i:
                        multi_datasets.append({
                            "geo": ds_label.strip() or f"DS{i+1}",
                            "counts": io.BytesIO(expr_file.getvalue()),
                            "meta": io.BytesIO(meta_file_i.getvalue()),
                            "meta_id_cols": [c.strip() for c in id_cols_i.split(",") if c.strip()],
                            "meta_group_cols": [c.strip() for c in grp_cols_i.split(",") if c.strip()],
                            "meta_batch_col": (batch_col_i.strip() or None),
                        })

        with st.expander("2) Multi-dataset settings"):
            combine_thresh = st.number_input("Minimum overlapping genes to combine", min_value=500, max_value=100000,
                                             value=3000, step=250,
                                             help="If overlap ≥ this, datasets are combined; otherwise analyzed separately.")

# =========================
# Drive Ingestion Controls
# =========================
if input_source == "Google Drive (deg_data)":
    st.subheader("🔗 Google Drive Ingestion (deg_data)")
    st.caption("Provide a **Service Account JSON** and share your deg_data folder with its `client_email` as **Viewer**.")
    json_file = st.file_uploader("Service Account JSON", type=["json"], key="sa_json")
    drive_link = st.text_input("deg_data root folder link (or ID)", value="", help="Example: https://drive.google.com/drive/folders/<ID>")
    disease_query = st.text_input("Disease keywords (space or comma separated)", value="",
                                  help="e.g., acute, myeloid, leukemia — matches any token against disease folder names")
    out_dir = st.text_input("Output directory", "out")
    pca_topk = st.number_input("Top variable genes for PCA", min_value=500, max_value=50000, value=5000, step=500, key="pca_topk_drive")
    do_nonlinear = st.checkbox("Make UMAP/t-SNE (if available)", value=True, key="do_nonlinear_drive")
    combine_thresh = st.number_input("Minimum overlapping genes to combine (multi-dataset)", min_value=500, max_value=100000,
                                     value=3000, step=250, key="combine_drive")

# =========================
# Run Button
# =========================
if input_source == "Manual upload":
    run = safe_button("🚀 Run Harmonization", type="primary", use_container_width=True)
else:
    run = safe_button("🔎 Scan Drive & Run", type="primary", use_container_width=True)

if run:
    _clear_caches()
    if input_source == "Manual upload":
        # ----- Manual paths (unchanged core logic) -----
        if mode == "Multiple datasets (each has its own metadata)":
            if not multi_datasets or len(multi_datasets) < 2:
                st.error("Please provide at least two datasets (each with expression + metadata)."); st.stop()
            kwargs_multi = {
                "datasets": multi_datasets,
                "attempt_combine": True,
                "combine_minoverlap_genes": int(combine_thresh),
                "out_root": out_dir,
                "pca_topk_features": int(pca_topk),
                "make_nonlinear": do_nonlinear,
            }
            try:
                with st.spinner("Running multi-dataset harmonization & meta-analysis..."):
                    run_id = _dt.datetime.now().strftime("multirun_%Y%m%d_%H%M%S")
                    kwargs_multi["out_root"] = os.path.join(out_dir, run_id)
                    multi_out = hz.run_pipeline_multi(**kwargs_multi)
                    st.session_state.run_id = run_id
                    # Keep ALL runs; default selection to Combined (if present) else first dataset
                    st.session_state.out = (multi_out.get("combined") or next(iter(multi_out["runs"].values())))
                    st.session_state.run_token = f"{run_id}-{_dt.datetime.now().timestamp():.0f}"
                    st.session_state.multi = multi_out
                    # NEW: pick default dataset in selector
                    if multi_out.get("combined"): 
                        st.session_state.selected_dataset = "[Combined]"
                    else:
                        st.session_state.selected_dataset = next(iter(multi_out["runs"].keys()))
                _clear_caches()
            except Exception as e:
                st.error(f"Multi-dataset run failed: {e}")
                st.stop()
        else:
            if not metadata_file:
                st.error("Please upload a metadata file."); st.stop()

            kwargs = {
                "metadata_file": io.BytesIO(metadata_file.getvalue()),
                "metadata_name_hint": metadata_file.name,
                "metadata_id_cols": [c.strip() for c in id_cols.split(",") if c.strip()],
                "metadata_group_cols": [c.strip() for c in grp_cols.split(",") if c.strip()],
                "metadata_batch_col": (batch_col.strip() or None),  # None => auto-detect
                "out_root": out_dir,
                "pca_topk_features": int(pca_topk),
                "make_nonlinear": do_nonlinear,
            }

            gmt_path = None
            if gmt_file:
                tmpdir = tempfile.mkdtemp()
                gmt_path = os.path.join(tmpdir, gmt_file.name)
                with open(gmt_path, "wb") as fh: fh.write(gmt_file.getvalue())
                kwargs["gsea_gmt"] = gmt_path

            if mode == "Multiple files (one per group)":
                groups = {}
                if normal_file: groups["First"] = io.BytesIO(normal_file.getvalue())
                if atypia_file: groups["Second"] = io.BytesIO(atypia_file.getvalue())
                if hpv_pos_file: groups["Third"] = io.BytesIO(hpv_pos_file.getvalue())
                if hpv_neg_file: groups["Fourth"] = io.BytesIO(hpv_neg_file.getvalue())
                if not groups:
                    st.error("Please upload at least one expression file."); st.stop()
                kwargs["group_to_file"] = groups
            else:
                if not single_expr_file:
                    st.error("Please upload the expression matrix."); st.stop()
                kwargs["single_expression_file"] = io.BytesIO(single_expr_file.getvalue())
                kwargs["single_expression_name_hint"] = single_expr_file.name

            try:
                with st.spinner("Running harmonization..."):
                    run_id = _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
                    kwargs["out_root"] = os.path.join(out_dir, run_id)
                    out = hz.run_pipeline(**kwargs)
                    st.session_state.run_id = run_id
                    st.session_state.out = out
                    st.session_state.run_token = f"{run_id}-{_dt.datetime.now().timestamp():.0f}"
                    st.session_state.multi = None
                    st.session_state.selected_dataset = None
                _clear_caches()
            except Exception as e:
                st.error(f"Run failed: {e}")
                st.stop()
    else:
        # ----- Drive ingestion path -----
        json_file = st.session_state.get("sa_json") or st.session_state.get("json_file") or json_file
        if not json_file:
            st.error("Please upload your Service Account JSON."); st.stop()
        if not drive_link.strip():
            st.error("Please paste your deg_data root folder link or ID."); st.stop()

        try:
            with st.spinner("Connecting to Drive and building ingestion plan..."):
                drv = din.DriveClient.from_service_account_bytes(json_file.getvalue())
                plan = din.make_ingest_plan(drv, drive_link.strip(), disease_query=disease_query.strip())
                st.session_state.last_plan = plan
        except Exception as e:
            st.error(f"Drive ingest failed: {e}")
            st.stop()

        mode = plan.get("mode")
        if mode == "none":
            st.warning(plan.get("reason", "No content found."))
            st.stop()

        # Build kwargs from plan
        out_root = os.path.join(out_dir, _dt.datetime.now().strftime("drive_%Y%m%d_%H%M%S"))
        if mode == "single":
            single = plan["single"]
            kwargs = {
                "single_expression_file": single["counts"],
                "single_expression_name_hint": single["counts_name"],
                "metadata_file": single["meta"],
                "metadata_name_hint": single["meta_name"],
                "metadata_id_cols": ["sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"],
                "metadata_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                "metadata_batch_col": None,
                "out_root": out_root,
                "pca_topk_features": int(pca_topk),
                "make_nonlinear": do_nonlinear,
            }
            with st.spinner(f"Running SINGLE dataset from Drive: {single.get('label','dataset')}"):
                out = hz.run_pipeline(**kwargs)
                st.session_state.run_id = os.path.basename(out_root)
                st.session_state.out = out
                st.session_state.run_token = f"{st.session_state.run_id}-{_dt.datetime.now().timestamp():.0f}"
                st.session_state.multi = None
                st.session_state.selected_dataset = None
            _clear_caches()

        elif mode == "multi_files_one_meta":
            groups = plan["groups"]
            meta = plan["meta"]
            kwargs = {
                "group_to_file": {k: v[0] for k, v in groups.items()},
                "metadata_file": meta,
                "metadata_name_hint": plan["meta_name"],
                "metadata_id_cols": ["sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"],
                "metadata_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                "metadata_batch_col": None,
                "out_root": out_root,
                "pca_topk_features": int(pca_topk),
                "make_nonlinear": do_nonlinear,
            }
            with st.spinner(f"Running MULTI-FILES-ONE-META from Drive: {plan.get('disease','disease')} / {plan.get('prep_path','prep')}"):
                out = hz.run_pipeline(**kwargs)
                st.session_state.run_id = os.path.basename(out_root)
                st.session_state.out = out
                st.session_state.run_token = f"{st.session_state.run_id}-{_dt.datetime.now().timestamp():.0f}"
                st.session_state.multi = None
                st.session_state.selected_dataset = None
            _clear_caches()

        elif mode == "multi_dataset":
            ds = plan["datasets"]
            datasets_arg = []
            for i, d in enumerate(ds, 1):
                datasets_arg.append({
                    "geo": d["label"],
                    "counts": d["counts"],
                    "counts_name": d["counts_name"],
                    "meta": d["meta"],
                    "meta_name": d["meta_name"],
                    "meta_id_cols": ["sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"],
                    "meta_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                    "meta_batch_col": None,
                })
            kwargs_multi = {
                "datasets": datasets_arg,
                "attempt_combine": True,
                "combine_minoverlap_genes": int(combine_thresh),
                "out_root": out_root,
                "pca_topk_features": int(pca_topk),
                "make_nonlinear": do_nonlinear,
            }
            with st.spinner(f"Running MULTI-DATASET from Drive: {len(datasets_arg)} preps"):
                multi_out = hz.run_pipeline_multi(**kwargs_multi)
                st.session_state.run_id = os.path.basename(out_root)
                st.session_state.out = (multi_out.get("combined") or next(iter(multi_out["runs"].values())))
                st.session_state.run_token = f"{st.session_state.run_id}-{_dt.datetime.now().timestamp():.0f}"
                st.session_state.multi = multi_out
                # NEW: default selected dataset
                if multi_out.get("combined"):
                    st.session_state.selected_dataset = "[Combined]"
                else:
                    st.session_state.selected_dataset = next(iter(multi_out["runs"].keys()))
            _clear_caches()
        else:
            st.error(f"Unexpected plan mode: {mode}")
            st.stop()

# =========================
# RESULTS UI (enhanced for multi-dataset)
# =========================
st.subheader("Results")

# Helper: pick current dataset output based on selector
def _get_current_out():
    multi_ctx = st.session_state.get("multi")
    sel = st.session_state.get("selected_dataset")
    if multi_ctx and isinstance(multi_ctx, dict) and multi_ctx.get("runs"):
        # Build options
        has_combined = bool(multi_ctx.get("combined"))
        run_keys = list(multi_ctx["runs"].keys())
        # Ensure selection is valid
        valid = (["[Combined]"] if has_combined else []) + run_keys
        if not sel or sel not in valid:
            # initialize default
            st.session_state.selected_dataset = "[Combined]" if has_combined else (run_keys[0] if run_keys else None)
            sel = st.session_state.selected_dataset
        if sel == "[Combined]":
            return multi_ctx.get("combined") or st.session_state.get("out")
        return multi_ctx["runs"][sel]
    # single run
    return st.session_state.get("out")

# Sidebar selector (always visible when multi)
multi_ctx = st.session_state.get("multi")
if multi_ctx and isinstance(multi_ctx, dict) and multi_ctx.get("runs"):
    has_combined = bool(multi_ctx.get("combined"))
    run_keys = list(multi_ctx["runs"].keys())
    options = (["[Combined]"] if has_combined else []) + run_keys
    default_index = 0
    if st.session_state.get("selected_dataset") in options:
        default_index = options.index(st.session_state["selected_dataset"])
    st.sidebar.markdown("### 📂 Dataset view")
    selected_key = st.sidebar.selectbox("Select dataset", options, index=default_index, key="selected_dataset")
    with st.sidebar.expander("Per-dataset quick links", expanded=False):
        if has_combined:
            st.write("• **[Combined]**")
            st.caption(multi_ctx["combined"]["outdir"])
        for name in run_keys:
            outdir = multi_ctx["runs"][name]["outdir"]
            st.write(f"• **{name}**")
            st.caption(outdir)

out_curr = _get_current_out()
run_id = st.session_state.get("run_id")

# Build tabs (NEW: added 'Dataset Index')
tabs = st.tabs([
    "Overview","QC","PCA & Embeddings","DE & GSEA","Outliers","Files",
    "Dataset Index","Multi-dataset Summary","Presenter Mode","Comparisons", "Agent"
])

# ---- Overview
with tabs[0]:
    if not out_curr:
        st.info("No run loaded yet. Upload or scan Drive to start.")
    else:
        report = {}
        try:
            with open(out_curr["report_json"], "r") as fh: report = json.load(fh)
        except Exception: pass

        qc = report.get("qc", {}); shp = report.get("shapes", {})
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
            zf = qc.get("zero_fraction", None)
            st.metric("Zero fraction", f'{zf:.2f}' if isinstance(zf, (int, float)) else "—")
            st.markdown('<div class="smallcaps">Approx. sparsity</div></div>', unsafe_allow_html=True)
        with kcol4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            sil_batch = qc.get("silhouette_batch", None)
            st.metric("Silhouette (batch)", f'{sil_batch:.2f}' if isinstance(sil_batch, (int, float)) else "—")
            st.markdown('<div class="smallcaps">Lower is better</div></div>', unsafe_allow_html=True)

        with st.expander("Diagnostics"):
            st.write({
                "harmonization_mode": qc.get("harmonization_mode"),
                "genes_zero_std_after_harmonization": qc.get("genes_zero_std_after_harmonization", "—"),
                "platform": qc.get("platform"),
            })
        st.json(report if report else {"info": "report.json not found"})

        fig_dir = out_curr["figdir"]
        previews = ["dist_pre_vs_post_log2.png","pca_clean_groups.png","enhanced_pca_analysis.png"]
        show = [f for f in previews if os.path.exists(os.path.join(fig_dir, f))]
        if show:
            st.write("### Key Figures")
            c1, c2, c3 = st.columns(3); cols = [c1, c2, c3]
            for i, f in enumerate(show):
                with cols[i % 3]:
                    st.image(os.path.join(fig_dir, f), caption=f, use_column_width=True)

# ---- QC
with tabs[1]:
    if not out_curr: st.info("No run loaded yet.")
    else:
        fig_dir = out_curr["figdir"]
        qc_figs = ["qc_library_size.png","qc_zero_rate_hist.png","group_density_post_log2.png",
                   "dist_zscore.png","sample_correlation_heatmap.png","hk_cv.png","sex_marker_concordance.png"]
        st.write("### QC Figures")
        for f in qc_figs:
            p = os.path.join(fig_dir, f)
            if os.path.exists(p): st.image(p, caption=f, use_column_width=True)

# ---- PCA & Embeddings
with tabs[2]:
    if not out_curr: st.info("No run loaded yet.")
    else:
        fig_dir = out_curr["figdir"]
        pcs = ["pca_clean_groups.png","enhanced_pca_analysis.png","pca_loadings_pc1.png",
               "pca_loadings_pc2.png","umap_by_group.png","tsne_by_group.png"]
        st.write("### PCA / UMAP / t-SNE")
        for f in pcs:
            p = os.path.join(fig_dir, f)
            if os.path.exists(p): st.image(p, caption=os.path.basename(p), use_column_width=True)

# ---- DE & GSEA
with tabs[3]:
    if not out_curr: st.info("No run loaded yet.")
    else:
        fig_dir = out_curr["figdir"]
        de_dir = os.path.join(out_curr["outdir"], "de")
        de_files = [f for f in os.listdir(de_dir)] if os.path.isdir(de_dir) else []
        contrasts = sorted([f.replace("DE_","").replace(".tsv","") for f in de_files if f.startswith("DE_")])
        pick = st.selectbox("Select contrast", contrasts) if contrasts else None
        if pick:
            for pth in [os.path.join(fig_dir, f"volcano_{pick}.png"),
                        os.path.join(fig_dir, f"ma_{pick}.png"),
                        os.path.join(fig_dir, f"heatmap_top_50_{pick}.png")]:
                if os.path.exists(pth): st.image(pth, caption=os.path.basename(pth), use_column_width=True)
            tsv = os.path.join(de_dir, f"DE_{pick}.tsv")
            try:
                df = pd.read_csv(tsv, sep="\t", index_col=0).head(50)
                st.dataframe(df, use_container_width=True, key=f"de_table__{run_id}__{pick}")
                with open(tsv, "rb") as fh:
                    safe_download_button("⬇️ Download full DE table", fh.read(),
                                         file_name=f"DE_{pick}.tsv", mime="text/tab-separated-values",
                                         key=f"dl_de__{run_id}__{pick}")
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
    if not out_curr: st.info("No run loaded yet.")
    else:
        outliers_path = os.path.join(out_curr["outdir"], "outliers.tsv")
        meta_path = os.path.join(out_curr["outdir"], "metadata.tsv")
        st.caption(f"Run: **{run_id}**  •  Outdir: `{out_curr['outdir']}`")
        if os.path.exists(outliers_path):
            mtime = int(os.path.getmtime(outliers_path)); cache_buster = f"{run_id}__{mtime}"
            try:
                df = pd.read_csv(outliers_path, sep="\t", index_col=0)
                if os.path.exists(meta_path):
                    meta_df = pd.read_csv(meta_path, sep="\t", index_col=0)
                    grp_col = "group_raw" if "group_raw" in meta_df.columns else "group"
                    display_df = (
                        df.copy().assign(sample=df.index)
                        .join(meta_df[["bare_id", grp_col]].rename(columns={grp_col: "group"}), how="left")
                        .set_index("sample").rename(columns={"IsolationForest":"IsolationForest_flag","LOF":"LOF_flag"})
                        [["bare_id","group","IsolationForest_flag","LOF_flag"]]
                    )
                else:
                    display_df = df.rename(columns={"IsolationForest":"IsolationForest_flag","LOF":"LOF_flag"})
                st.write("### Outlier flags (1 = outlier)")
                st.dataframe(display_df, use_container_width=True, key=f"outliers_df__{cache_buster}")
                with open(outliers_path, "rb") as fh:
                    safe_download_button("⬇️ Download outlier table", fh.read(), file_name="outliers.tsv",
                                         mime="text/tab-separated-values", key=f"dl_outliers__{cache_buster}")
            except Exception as e:
                st.warning(f"Could not load outliers for this run: {e}")
        else:
            st.info("No outlier table found for this run.")

# ---- Files
with tabs[5]:
    if not out_curr: st.info("No run loaded yet.")
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
                        safe_download_button(f"⬇️ {label}", fh.read(),
                                             file_name=os.path.basename(path), mime="text/plain",
                                             use_container_width=True, key=f"dl_core__{run_id}__{label}")
        with colB:
            try:
                with open(out_curr["zip"], "rb") as fh:
                    safe_download_button("⬇️ Download ALL results (ZIP)", fh.read(),
                                         file_name="harmonization_results.zip", mime="application/zip",
                                         use_container_width=True, key=f"dl_zip__{run_id}")
            except Exception as e:
                st.warning(f"Could not open ZIP for download: {e}")

# ---- NEW: Dataset Index (browse every dataset, switch quickly)
with tabs[6]:
    multi = st.session_state.get("multi")
    if not multi or not multi.get("runs"):
        st.info("Dataset Index becomes available after a multi-dataset run.")
    else:
        runs = multi["runs"]
        # Summary table
        rows = []
        for name, r in runs.items():
            rep = {}
            try:
                rep = json.load(open(r["report_json"], "r"))
            except Exception:
                pass
            shp = rep.get("shapes", {})
            qc = rep.get("qc", {})
            rows.append({
                "dataset": name,
                "outdir": r.get("outdir",""),
                "samples": shp.get("samples", None),
                "genes": shp.get("genes", None),
                "zero_fraction": qc.get("zero_fraction", None),
                "platform": qc.get("platform", None),
            })
        st.write("### All datasets in this run")
        st.dataframe(pd.DataFrame(rows).set_index("dataset"), use_container_width=True)

        # Individual previews + switcher
        st.write("### Quick previews")
        for name, r in runs.items():
            with st.expander(name, expanded=False):
                outdir = r["outdir"]; figdir = r["figdir"]
                st.caption(outdir)
                imgs = ["pca_clean_groups.png","enhanced_pca_analysis.png","dist_pre_vs_post_log2.png"]
                cols = st.columns(3)
                for i, f in enumerate(imgs):
                    p = os.path.join(figdir, f)
                    if os.path.exists(p):
                        with cols[i % 3]:
                            st.image(p, caption=f, use_column_width=True)
                # Switch button
                if st.button(f"🔀 Switch to: {name}", key=f"switch_{name}"):
                    st.session_state.selected_dataset = name
                    st.experimental_rerun()

# ---- Multi-dataset Summary (existing + kept)
with tabs[7]:
    multi = st.session_state.get("multi")
    if not multi:
        st.info("No multi-dataset run in this session.")
    else:
        dec = multi.get("combine_decision", {}) or {}
        st.write("### Combination decision")
        st.json(dec)
        if multi.get("combined"):
            st.success("A combined run was created (sufficient gene overlap).")
        else:
            st.warning("Datasets were analyzed individually (insufficient overlap).")

        summary_txt = multi.get("summary_txt_path")
        summary_png = multi.get("summary_png_path")
        outputs_zip = None
        comb = multi.get("combined")
        if comb:
            outputs_zip = comb.get("zip")

        if summary_txt and os.path.exists(summary_txt):
            st.write("### Key Findings")
            with open(summary_txt, "r") as fh:
                st.code(fh.read(), language="markdown")
            with open(summary_txt, "rb") as fh:
                safe_download_button("⬇️ Download summary (TXT)", fh.read(),
                                     file_name="final_analysis_summary.txt",
                                     mime="text/plain", key=f"dl_summary_txt__{run_id}")

        if summary_png and os.path.exists(summary_png):
            st.image(summary_png, caption=os.path.basename(summary_png), use_column_width=True)

        meta_dir = multi.get("meta_dir")
        if meta_dir and os.path.isdir(meta_dir):
            st.write("### Meta-analysis tables")
            files = [
                "deg_results_annotated.csv",
                "meta_analysis_results.csv",
                "upregulated_genes_meta.csv",
                "drug_targets_analysis.csv",
                "final_analysis_summary.csv",
            ]
            cols = st.columns(2)
            for i, f in enumerate(files):
                p = os.path.join(meta_dir, f)
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        safe_download_button(f"⬇️ {f}", fh.read(), file_name=f, mime="text/csv",
                                             use_container_width=True, key=f"dl_meta_{f}_{run_id}")
        if outputs_zip and os.path.exists(outputs_zip):
            with open(outputs_zip, "rb") as fh:
                safe_download_button("⬇️ Download ALL combined results (ZIP)", fh.read(),
                                     file_name="harmonization_results.zip", mime="application/zip",
                                     use_container_width=True, key=f"dl_zip_multi__{run_id}")

# ---- Presenter Mode (kept)
with tabs[8]:
    st.markdown("### 🎤 Presenter Mode")
    st.caption("A concise, visual summary for stakeholders. (Auto-populates after a multi-dataset run.)")

    multi = st.session_state.get("multi")
    if not multi:
        st.info("Run a **multi-dataset** analysis to enable Presenter Mode.")
    else:
        summary_txt = multi.get("summary_txt_path")
        summary_png = multi.get("summary_png_path")
        meta_dir = multi.get("meta_dir")
        dec = multi.get("combine_decision", {}) or {}
        n_datasets = len(multi.get("runs", {}))
        overlap = dec.get("overlap_genes", 0)
        combined = dec.get("combined", False)

        met_csv = os.path.join(meta_dir, "meta_analysis_results.csv") if meta_dir else None
        sig_count = 0
        top_rows = []
        if met_csv and os.path.exists(met_csv):
            try:
                mdf = pd.read_csv(met_csv, index_col=0)
                sig = mdf[mdf["q_meta"] < 0.10]
                sig_count = int(sig.shape[0])
                top_rows = mdf.sort_values("q_meta").head(10)
            except Exception:
                pass

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.metric("Datasets", n_datasets)
        with k2: st.metric("Overlap genes", f"{overlap:,}")
        with k3: st.metric("Combined run", "Yes" if combined else "No")
        with k4: st.metric("Significant genes (FDR<0.1)", f"{sig_count:,}")

        if summary_png and os.path.exists(summary_png):
            st.image(summary_png, caption="Comprehensive Meta-analysis Overview", use_column_width=True)

        if len(top_rows):
            st.write("#### Top biomarker candidates (meta)")
            st.dataframe(top_rows[["z_meta","p_meta","q_meta","consistent_dir","consistency","meta_log2FC_proxy"]],
                         use_container_width=True)

        if summary_txt and os.path.exists(summary_txt):
            with open(summary_txt, "r") as fh:
                st.write("#### Key Findings (ready to copy)")
                st.code(fh.read(), language="markdown")

# ---- NEW: Comparisons (pairwise similarities across datasets)
with tabs[9]:
    multi = st.session_state.get("multi")
    if not multi or not multi.get("runs"):
        st.info("Comparisons are available after a multi-dataset run.")
    else:
        st.write("### Pairwise comparisons between datasets")
        runs = multi["runs"]

        # 1) Basic stats
        basic_rows = []
        for name, r in runs.items():
            try:
                rep = json.load(open(r["report_json"], "r"))
            except Exception:
                rep = {}
            shp = rep.get("shapes", {})
            qc = rep.get("qc", {})
            basic_rows.append({
                "dataset": name,
                "samples": shp.get("samples", None),
                "genes": shp.get("genes", None),
                "zero_fraction": qc.get("zero_fraction", None),
                "silhouette_batch": qc.get("silhouette_batch", None),
                "platform": qc.get("platform", None),
            })
        if basic_rows:
            st.dataframe(pd.DataFrame(basic_rows).set_index("dataset"), use_container_width=True)

        # 2) Gene overlap matrix
        def load_genes(path):
            try:
                X = pd.read_csv(os.path.join(path, "expression_combined.tsv"), sep="\t", index_col=0)
                return set(map(str, X.index))
            except Exception:
                return set()

        gene_sets = {name: load_genes(r["outdir"]) for name, r in runs.items()}
        ds_names = list(gene_sets.keys())
        jacc = pd.DataFrame(index=ds_names, columns=ds_names, dtype=float)
        inter = pd.DataFrame(index=ds_names, columns=ds_names, dtype=float)
        for a, b in itertools.product(ds_names, ds_names):
            A, B = gene_sets[a], gene_sets[b]
            if not A or not B:
                jacc.loc[a,b] = None
                inter.loc[a,b] = None
            else:
                inter_set = len(A & B)
                union_set = len(A | B)
                inter.loc[a,b] = inter_set
                jacc.loc[a,b] = inter_set / union_set if union_set else None
        st.write("#### Gene overlap (intersection counts)")
        st.dataframe(inter, use_container_width=True)
        st.write("#### Jaccard index of gene overlap")
        st.dataframe(jacc, use_container_width=True)

        # 3) DE overlap (optional: Disease vs Control if found)
        st.write("#### Overlap of top DE genes (if 'DE_Disease_vs_Control.tsv' exists)")
        def top_de_genes(outdir, topn=200):
            de_dir = os.path.join(outdir, "de")
            target = os.path.join(de_dir, "DE_Disease_vs_Control.tsv")
            if not os.path.exists(target):
                # fallback: pick any DE file
                cands = [f for f in os.listdir(de_dir)] if os.path.isdir(de_dir) else []
                cands = [f for f in cands if f.startswith("DE_") and f.endswith(".tsv")]
                if not cands:
                    return set()
                target = os.path.join(de_dir, cands[0])
            try:
                df = pd.read_csv(target, sep="\t", index_col=0).sort_values("qval").head(topn)
                return set(map(str, df.index))
            except Exception:
                return set()

        top_sets = {name: top_de_genes(r["outdir"]) for name, r in runs.items()}
        de_overlap = pd.DataFrame(index=ds_names, columns=ds_names, dtype=float)
        for a, b in itertools.product(ds_names, ds_names):
            A, B = top_sets[a], top_sets[b]
            if not A or not B:
                de_overlap.loc[a,b] = None
            else:
                de_overlap.loc[a,b] = len(A & B)
        st.dataframe(de_overlap, use_container_width=True)

        st.caption("Tip: Higher intersections and Jaccard suggest stronger similarity across datasets. "
                   "Meta-analysis above already aggregates signals from all datasets.")
with tabs[10]:
    st.markdown("### 🤖 Harmonization Agent")
    st.caption(
        "Talk to the agent. Examples:\n"
        "- `search diabetes on drive`\n"
        "- `run harmonization`\n"
        "- `show summary`"
    )

    # Inputs the agent needs for Drive search
    sa_json = st.file_uploader("Service Account JSON (for agent)", type=["json"], key="sa_json_agent")
    drive_link_agent = st.text_input("deg_data folder (link or ID) for agent", value="", key="drive_link_agent")

    # Show chat history
    for role, msg in st.session_state.agent_messages:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Agent:** {msg}")

    user_msg = st.text_input("Command to agent", key="agent_input", placeholder="e.g. search diabetes on drive")
    if st.button("Send to agent"):
        if user_msg.strip():
            st.session_state.agent_messages.append(("user", user_msg.strip()))
            agent = st.session_state.agent
            sa_bytes = sa_json.getvalue() if sa_json is not None else None
            reply = agent.handle_command(
                user_msg,
                sa_json_bytes=sa_bytes,
                deg_root_link_or_id=drive_link_agent.strip() or None,
            )
            st.session_state.agent_messages.append(("agent", reply))
            st.experimental_rerun()

