# app.py (generalized for single- or multi-file)
import os, io, tempfile, shutil
import streamlit as st
from harmonizer import run_pipeline

st.set_page_config(page_title="Multi-Modal Data Harmonization", layout="wide")

st.title("🧬 Data Harmonization & QC Suite")
st.caption("Upload expression data (single matrix OR one file per group) + metadata, then click **Run Harmonization**.")

mode = st.radio("Expression upload mode", ["Single expression matrix", "Multiple files (one per group)"], horizontal=True)

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

with st.expander("2) Upload Metadata (TSV/CSV/XLSX)"):
    metadata_file = st.file_uploader("Metadata file", type=["tsv","csv","txt","xlsx"], key="meta")
    id_cols = st.text_input("Candidate ID columns (comma-separated)", "Id,ID,id,CleanID,sample,Sample")
    grp_cols = st.text_input("Candidate GROUP columns (comma-separated)", "group,Group,condition,Condition,phenotype,Phenotype")
    batch_col = st.text_input("Batch column name (optional; leave blank to auto-detect)", "")

with st.expander("3) Optional: GSEA gene set (.gmt)"):
    gmt_file = st.file_uploader("Gene set GMT (optional)", type=["gmt"])    

advanced = st.expander("Advanced settings")
with advanced:
    out_dir = st.text_input("Output directory", "out")
    pca_topk = st.number_input("Top variable genes for PCA", min_value=500, max_value=50000, value=5000, step=500)
    do_nonlinear = st.checkbox("Make UMAP/t-SNE (if available)", value=True)

run = st.button("🚀 Run Harmonization", type="primary", use_container_width=True)

if run:
    if not metadata_file:
        st.error("Please upload a metadata file."); st.stop()

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
        with open(gmt_path, "wb") as fh: fh.write(gmt_file.getvalue())
        kwargs["gsea_gmt"] = gmt_path

    if mode == "Multiple files (one per group)":
        groups = {}
        if normal_file: groups["Normal"] = io.BytesIO(normal_file.getvalue())
        if atypia_file: groups["Atypia"] = io.BytesIO(atypia_file.getvalue())
        if hpv_pos_file: groups["HPV_Pos"] = io.BytesIO(hpv_pos_file.getvalue())
        if hpv_neg_file: groups["HPV_Neg"] = io.BytesIO(hpv_neg_file.getvalue())
        if len(groups)==0:
            st.error("Please upload at least one expression file."); st.stop()
        kwargs["group_to_file"] = groups
    else:
        if not single_expr_file:
            st.error("Please upload the expression matrix."); st.stop()
        kwargs["single_expression_file"] = io.BytesIO(single_expr_file.getvalue())
        kwargs["single_expression_name_hint"] = single_expr_file.name

    with st.spinner("Running harmonization..."):
        out = run_pipeline(**kwargs)

    st.success("Done!")

    # Show report + download
    st.subheader("Results")
    with open(out["report_json"], "r") as fh:
        st.json(fh.read())

    # Figures preview (if any)
    fig_dir = out["figdir"]
    figs = [f for f in os.listdir(fig_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if figs:
        st.write("### Figures")
        for f in sorted(figs):
            st.image(os.path.join(fig_dir, f), caption=f, use_column_width=True)

    # Provide ZIP download
    with open(out["zip"], "rb") as fh:
        st.download_button(
            label="⬇️ Download all results (ZIP)",
            data=fh.read(),
            file_name="harmonization_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if gmt_file:
        shutil.rmtree(os.path.dirname(gmt_path), ignore_errors=True)
