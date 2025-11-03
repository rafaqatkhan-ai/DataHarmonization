# app.py (generalized for single- or multi-file)
import os, io, tempfile, shutil, json
import streamlit as st
import pandas as pd
from harmonizer import run_pipeline

st.set_page_config(
    page_title="🧬 Data Harmonization & QC Suite",
    page_icon="🧬",
    layout="wide",
)

# ---- Minimal theming polish
# ---- Custom Title (centered, styled) ----
st.markdown(
    """
    <style>
    .centered-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        color: #1e40af;
        margin-top: -0.5rem;
        background: linear-gradient(90deg, #2563eb, #1e3a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .centered-title span {
        font-size: 2.8rem;
    }
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1rem;
        margin-top: -0.6rem;
        font-style: italic;
    }
    @media (prefers-color-scheme: dark) {
        .centered-title {
            color: #93c5fd;
            background: linear-gradient(90deg, #06b6d4, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #cbd5e1;
        }
    }
    </style>

    <h1 class="centered-title">
        <span>🧬</span> Data Harmonization & QC Suite <span>🧬</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# ---- Expression Upload Mode ----
mode = st.radio(
    "Expression upload mode",
    ["Single expression matrix", "Multiple files (one per group)"],
    horizontal=True,
)

# ---- Caption under mode selector ----
st.caption("Upload expression data (single matrix OR one file per group) + metadata, then click **Run Harmonization**.")

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

    # Preview detected metadata columns to help users type exact names
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

    # Build inputs for pipeline
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

    # Load report for KPIs
    report = {}
    try:
        with open(out["report_json"], "r") as fh:
            report = json.load(fh)
    except Exception:
        pass

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
        # Show a few key figures if present
        fig_dir = out["figdir"]
        previews = [
            "dist_pre_vs_post_log2.png",
            "pca_clean_groups.png",
            "enhanced_pca_analysis.png",
        ]
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
                st.image(p, caption=f, use_column_width=True)

    # ---- DE & GSEA
    with tabs[3]:
        fig_dir = out["figdir"]
        de_dir = os.path.join(out["outdir"], "de")
        de_files = [f for f in os.listdir(de_dir)] if os.path.isdir(de_dir) else []
        # pick contrast
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
                    st.download_button("⬇️ Download full DE table", fh.read(), file_name=f"DE_{pick}.tsv", mime="text/tab-separated-values")
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
        try:
            df = pd.read_csv(os.path.join(out["outdir"], "outliers.tsv"), sep="\t", index_col=0)
            st.write("### Outlier flags (1 = outlier)")
            st.dataframe(df, use_container_width=True)
            with open(os.path.join(out["outdir"], "outliers.tsv"), "rb") as fh:
                st.download_button("⬇️ Download outlier table", fh.read(), file_name="outliers.tsv", mime="text/tab-separated-values")
        except Exception:
            st.info("No outlier table found.")

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
                        st.download_button(f"⬇️ {label}", fh.read(), file_name=os.path.basename(path), mime="text/plain", use_container_width=True)
        with colB:
            try:
                with open(out["zip"], "rb") as fh:
                    st.download_button(
                        label="⬇️ Download ALL results (ZIP)",
                        data=fh.read(),
                        file_name="harmonization_results.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Could not open ZIP for download: {e}")

    # Cleanup temp GMT if used
    if gmt_file:
        shutil.rmtree(os.path.dirname(gmt_path), ignore_errors=True)







