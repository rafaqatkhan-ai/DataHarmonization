# app.py
import os, io, tempfile, shutil
import streamlit as st
from harmonizer import run_pipeline

st.set_page_config(page_title="Multi-Modal Data Harmonization", layout="wide")

st.title("🧬 Data Harmonization & QC Suite")
st.caption("Upload expression spreadsheets (per group) + metadata, then click **Run Harmonization**.")

with st.expander("1) Upload Expression XLSX (one file per group)"):
    col1, col2 = st.columns(2)
    with col1:
        normal_file = st.file_uploader("Normal (XLSX)", type=["xlsx"], key="normal")
        atypia_file = st.file_uploader("Atypia (XLSX)", type=["xlsx"], key="atypia")
    with col2:
        hpv_pos_file = st.file_uploader("HPV Positive (XLSX)", type=["xlsx"], key="hpvp")
        hpv_neg_file = st.file_uploader("HPV Negative (XLSX)", type=["xlsx"], key="hpvn")

with st.expander("2) Upload Metadata (TSV/CSV/XLSX)"):
    metadata_file = st.file_uploader("Metadata file", type=["tsv","csv","txt","xlsx"], key="meta")
    id_cols = st.text_input("Candidate ID columns (comma-separated)", "Id,ID,id,CleanID,sample,Sample")
    batch_col = st.text_input("Batch column name (optional; leave blank to auto-detect)", "")

with st.expander("3) Optional: GSEA gene set (.gmt)"):
    gmt_file = st.file_uploader("Gene set GMT (optional)", type=["gmt"])

run = st.button("🚀 Run Harmonization", type="primary", use_container_width=True)

if run:
    if not metadata_file:
        st.error("Please upload a metadata file."); st.stop()

    groups = {}
    if normal_file: groups["Normal"] = io.BytesIO(normal_file.getvalue())
    if atypia_file: groups["Atypia"] = io.BytesIO(atypia_file.getvalue())
    if hpv_pos_file: groups["HPV_Pos"] = io.BytesIO(hpv_pos_file.getvalue())
    if hpv_neg_file: groups["HPV_Neg"] = io.BytesIO(hpv_neg_file.getvalue())

    if len(groups)==0:
        st.error("Please upload at least one expression XLSX file."); st.stop()

    meta_bytes = io.BytesIO(metadata_file.getvalue())
    gmt_path = None
    if gmt_file:
        tmpdir = tempfile.mkdtemp()
        gmt_path = os.path.join(tmpdir, gmt_file.name)
        with open(gmt_path, "wb") as fh: fh.write(gmt_file.getvalue())

    with st.spinner("Running harmonization..."):
        out = run_pipeline(
            group_to_file=groups,
            metadata_file=meta_bytes,
            metadata_name_hint=metadata_file.name,   # <-- crucial for robust parsing
            metadata_id_cols=[c.strip() for c in id_cols.split(",") if c.strip()],
            metadata_batch_col=(batch_col.strip() or None),
            out_root="out",
            pca_topk_features=5000,
            make_nonlinear=True,
            gsea_gmt=gmt_path,
        )

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
