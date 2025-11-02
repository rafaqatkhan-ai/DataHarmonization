# -*- coding: utf-8 -*-
# harmonizer.py
# End-to-end pipeline: batch harmonization + QC + DE + optional GSEA
import os, re, io, json, warnings, zipfile
from typing import Dict, Tuple, List, Iterable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

# ---------- Optional imports ----------
_HAVE_SCANPY = _HAVE_GSEAPY = _HAVE_UMAP = _HAVE_TSNE = False
try:
    import scanpy as sc, anndata as ad
    _HAVE_SCANPY = True
except Exception:
    pass

try:
    import gseapy as gp
    _HAVE_GSEAPY = True
except Exception:
    pass

try:
    from umap import UMAP
    _HAVE_UMAP = True
except Exception:
    pass

try:
    from sklearn.manifold import TSNE
    _HAVE_TSNE = True
except Exception:
    pass

# ---------------- Config defaults ----------------
ZERO_INFLATION_THRESH = 0.4
MICROARRAY_RANGE_MAX  = 20.0
VAR_EPS = 1e-12

HOUSEKEEPING_GENES = ["ACTB","GAPDH","RPLP0","B2M","HPRT1","PGK1","TBP","GUSB"]
SEX_MARKERS = {"female":["XIST"], "male":["RPS4Y1","KDM5D","UTY"]}

# ---------------- Utilities ----------------
def roman_to_int(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    s = s.upper().strip()
    if not s or not re.fullmatch(r"[MDCLXVI]+", s):
        return None
    values = dict(M=1000,D=500,C=100,L=50,X=10,V=5,I=1)
    total, prev = 0, 0
    for ch in reversed(s):
        v = values[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total

def normalize_batch_token(x: str) -> str:
    if pd.isna(x): return np.nan
    t = str(x).strip()
    if re.fullmatch(r"\d+", t): return t
    r = roman_to_int(t)
    if r is not None: return f"RN_{r}"
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^A-Za-z0-9_]+", "", t)
    return t if t else np.nan

def read_expression_xlsx(bytes_or_path, group_name: str) -> pd.DataFrame:
    # bytes_or_path: str path or BytesIO
    df = pd.read_excel(bytes_or_path, sheet_name=0, engine="openpyxl")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    lower = [str(c).strip().lower() for c in df.columns]
    for key in ["biomarkers","biomarker","marker","gene","feature","id","name"]:
        if key in lower: biomarker_col = df.columns[lower.index(key)]; break
    else:
        biomarker_col = df.columns[0]
    df = df.rename(columns={biomarker_col: "Biomarker"}).set_index("Biomarker")
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    df.columns = [f"{group_name}__{str(c).strip()}" for c in df.columns]
    return df.groupby(level=0).median(numeric_only=True)

def build_metadata(loaded_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return (pd.DataFrame(
        [{"sample": s, "group": g} for g, df in loaded_dict.items() for s in df.columns]
    ).set_index("sample").sort_index())

def infer_batches(metadata: pd.DataFrame) -> pd.Series:
    md = metadata.copy()
    for col in ["batch","Batch","run","Run","lane","Lane","plate","Plate","sequencer","flowcell","Flowcell"]:
        if col in md.columns:
            s = md[col].astype(str); s.index = md.index; return s
    def guess_token(s):
        m = re.search(r"(FC\w+|L\d{3}|P\d+|\d{4}[-_]\d{2}[-_]\d{2}|\d{8})", s)
        return m and m.group(0) or "B0"
    return pd.Series([guess_token(x) for x in md.index.astype(str)], index=md.index, name="batch")

def zscore_rows(M: pd.DataFrame) -> pd.DataFrame:
    mu = M.mean(axis=1)
    sd = M.std(axis=1, ddof=1).replace(0, np.nan)
    return M.sub(mu, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)

def row_mean_impute(X):  # per-gene
    X = X.apply(lambda r: r.fillna(r.mean()), axis=1)
    return X.fillna(0)

def drop_zero_variance(X):
    var = X.var(axis=1, ddof=1).astype(float).fillna(0.0)
    nz = var > VAR_EPS
    return X.loc[nz]

def safe_matrix_for_pca(matrix, topk: int = 5000):
    X = matrix.copy().apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    var = X.var(axis=1, ddof=1)
    nz = var > VAR_EPS
    if not nz.any():
        raise RuntimeError("No non-zero-variance features for PCA.")
    if topk and nz.sum() > topk:
        X = X.loc[var.loc[nz].nlargest(topk).index]
    else:
        X = X.loc[nz]
    X = X.T
    X.columns = X.columns.map(str)
    return X

def detect_data_type_and_platform(X: pd.DataFrame) -> Tuple[str, str, Dict]:
    vals = X.values.ravel()
    vals = vals[np.isfinite(vals)]
    zero_frac = float((X==0).sum().sum()) / float(X.size) if X.size else 0.0
    rng = (np.nanpercentile(vals, 99.5) - np.nanpercentile(vals, 0.5)) if vals.size else 0
    idx_str = X.index.astype(str)
    has_ilumn = any(s.startswith("ILMN_") for s in idx_str)
    has_affy  = any(re.match(r"^\d+_at$", s) for s in idx_str)
    has_ensembl = any(s.startswith("ENSG") for s in idx_str)
    data_type = "bulk"
    if zero_frac >= ZERO_INFLATION_THRESH:
        data_type = "scRNA-seq"
    platform = "Unknown"
    if has_ilumn or has_affy:
        platform = "Microarray (Illumina/Affy)"; data_type = "microarray"
    elif rng > MICROARRAY_RANGE_MAX and not has_ilumn and not has_affy:
        platform = "Long-read/Counts-like (Illumina/PacBio)"
    elif has_ensembl:
        platform = "Short-read RNA-seq (Illumina)"
    diags = {"zero_fraction": zero_frac, "value_range_approx": rng}
    return data_type, platform, diags

# ---------- Robust metadata reader (BytesIO/paths; Excel/CSV/TSV) ----------
def _read_metadata_any(metadata_obj, name_hint: str | None = None) -> pd.DataFrame:
    """
    Robustly read metadata from path/BytesIO/bytes.
    Uses name_hint (filename) when available; otherwise attempts Excel, then CSV/TSV.
    """
    is_pathlike = isinstance(metadata_obj, (str, os.PathLike))
    suffix = (os.path.splitext(name_hint)[1].lower() if name_hint else
              (os.path.splitext(str(metadata_obj))[1].lower() if is_pathlike else ""))

    def _as_bytesio(x):
        if isinstance(x, io.BytesIO):
            x.seek(0); return x
        if isinstance(x, bytes):
            return io.BytesIO(x)
        return None

    def _try_excel_then_csv(buf_or_path):
        try:
            if isinstance(buf_or_path, (str, os.PathLike)):
                return pd.read_excel(buf_or_path, engine="openpyxl")
            else:
                buf_or_path.seek(0)
                return pd.read_excel(buf_or_path, engine="openpyxl")
        except Exception:
            try:
                if isinstance(buf_or_path, (str, os.PathLike)):
                    if str(buf_or_path).lower().endswith((".tsv", ".txt")):
                        return pd.read_csv(buf_or_path, sep="\t")
                    return pd.read_csv(buf_or_path, sep=None, engine="python")
                else:
                    buf_or_path.seek(0)
                    return pd.read_csv(buf_or_path, sep=None, engine="python")
            except Exception as e2:
                raise ValueError(f"Could not parse metadata as Excel or text table: {e2}")

    if suffix in (".xlsx", ".xls"):
        if is_pathlike:
            return pd.read_excel(metadata_obj, engine="openpyxl")
        else:
            bio = _as_bytesio(metadata_obj)
            return pd.read_excel(bio, engine="openpyxl")
    elif suffix in (".tsv", ".txt"):
        if is_pathlike:
            return pd.read_csv(metadata_obj, sep="\t")
        else:
            bio = _as_bytesio(metadata_obj); bio.seek(0)
            return pd.read_csv(bio, sep="\t")
    elif suffix == ".csv":
        if is_pathlike:
            return pd.read_csv(metadata_obj)
        else:
            bio = _as_bytesio(metadata_obj); bio.seek(0)
            return pd.read_csv(bio)

    if is_pathlike:
        return _try_excel_then_csv(metadata_obj)
    else:
        bio = _as_bytesio(metadata_obj)
        return _try_excel_then_csv(bio)

# ---------------- Batch harmonization ----------------
def _combat(expr_imputed: pd.DataFrame, meta_batch: pd.Series) -> Optional[pd.DataFrame]:
    if not _HAVE_SCANPY: return None
    try:
        adata = ad.AnnData(expr_imputed.T.copy())
        adata.obs["batch"] = meta_batch.loc[adata.obs_names].astype(str).values
        sc.pp.combat(adata, key="batch")
        Xc = pd.DataFrame(adata.X.T, index=expr_imputed.index, columns=expr_imputed.columns, dtype=float)
        Xc = Xc.replace([np.inf, -np.inf], np.nan).apply(lambda r: r.fillna(r.mean()), axis=1).fillna(0.0)
        return Xc
    except Exception:
        return None

def _fallback_center(expr_imputed: pd.DataFrame, meta_batch: pd.Series) -> pd.DataFrame:
    X = expr_imputed.copy()
    grand_mean = X.values.mean()
    grand_std  = X.values.std()
    b = meta_batch.astype(str)
    for batch in pd.unique(b):
        cols = b.index[b == batch]
        gmean = X[cols].values.mean()
        gstd  = X[cols].values.std()
        X[cols] = (X[cols] - gmean) * (grand_std / (gstd if gstd>0 else 1)) + grand_mean
    return X

def smart_batch_collapse(meta: pd.DataFrame, min_size: int) -> pd.Series:
    b = meta['batch'].astype(str)
    counts = b.value_counts()
    large = counts[counts >= min_size].index
    small = counts[counts < min_size].index
    mapping = {k:k for k in large}
    for batch in small:
        grp_series = meta.loc[b.index[b==batch], 'group']
        main_group = grp_series.value_counts().idxmax() if len(grp_series) else "mixed"
        mapping[batch] = f"small_{main_group}"
    return b.map(mapping)

# ---------------- QC plots ----------------
def _savefig(path: str):
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

def create_basic_qc_figures(expr_log2, expr_z, expr_harmonized, meta, figdir: str) -> List[str]:
    os.makedirs(figdir, exist_ok=True)
    paths = []
    # pre vs post hist
    plt.figure(figsize=(12,6))
    for arr,label,a in [(expr_log2.values.ravel(),"Pre-harmonization",0.5),
                        (expr_harmonized.values.ravel(),"Post-harmonization",0.5)]:
        arr = arr[np.isfinite(arr)]
        plt.hist(arr, bins=120, density=True, alpha=a, label=label)
    plt.title("Expression Distribution: Pre vs Post (log2)"); plt.xlabel("log2(Expression + 1)"); plt.ylabel("Density"); plt.legend()
    p = os.path.join(figdir, "dist_pre_vs_post_log2.png"); _savefig(p); paths.append(p)

    # z-score dist
    plt.figure(figsize=(12,6))
    arr = expr_z.values.ravel(); arr = arr[np.isfinite(arr)]
    plt.hist(arr, bins=120, density=True)
    plt.title("Distribution of Z-scored Expression (All Samples)"); plt.xlabel("Z-score"); plt.ylabel("Density")
    p = os.path.join(figdir, "dist_zscore.png"); _savefig(p); paths.append(p)

    # per-group densities
    plt.figure(figsize=(12,6))
    for grp in ["Normal","Atypia","HPV_Pos","HPV_Neg"]:
        cols = meta.index[meta["group"]==grp]
        vals = expr_harmonized[cols].values.ravel() if len(cols) else np.array([])
        vals = vals[np.isfinite(vals)]
        plt.hist(vals, bins=100, density=True, alpha=0.35, label=grp)
    plt.title("Per-group Expression Distributions (log2, post-harmonization)")
    plt.xlabel("log2(Expression + 1)"); plt.ylabel("Density"); plt.legend()
    p = os.path.join(figdir, "group_density_post_log2.png"); _savefig(p); paths.append(p)

    # boxplot
    group_vals, labels = [], []
    for grp in ["Atypia","HPV_Neg","HPV_Pos","Normal"]:
        cols = meta.index[meta["group"]==grp]
        vals = expr_harmonized[cols].values.ravel() if len(cols) else np.array([])
        vals = vals[np.isfinite(vals)]
        group_vals.append(vals); labels.append(grp)
    plt.figure(figsize=(12,6)); plt.boxplot(group_vals, tick_labels=labels, showfliers=True)
    plt.title("Expression Distribution After Harmonization (log2)"); plt.ylabel("log2(Expression + 1)")
    p = os.path.join(figdir, "boxplot_groups_harmonized_log2.png"); _savefig(p); paths.append(p)

    # sample correlation heatmap (guard huge matrices)
    if expr_harmonized.shape[1] <= 600:
        C = np.corrcoef(expr_harmonized.fillna(0).T)
        plt.figure(figsize=(10,8)); plt.imshow(C, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation"); plt.title("Sample Correlation Heatmap (post-harmonization)")
        plt.xlabel("Samples"); plt.ylabel("Samples")
        p = os.path.join(figdir, "sample_correlation_heatmap.png"); _savefig(p); paths.append(p)
    return paths

def create_enhanced_pca_plots(pca_df, pca_model, meta, output_dir, harmonization_mode):
    groups = ['Normal','Atypia','HPV_Pos','HPV_Neg']
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # PC1 vs PC2 by group
    ax1 = axes[0,0]
    for i,g in enumerate(groups):
        sub = pca_df[pca_df["group"]==g]
        if sub.empty: continue
        ax1.scatter(sub["PC1"], sub["PC2"], c=colors[i], s=50, alpha=0.7, label=f"{g} (n={len(sub)})", edgecolors='white', linewidth=0.5)
        if len(sub) > 5:
            try:
                cov = np.cov(sub[["PC1","PC2"]].T)
                if np.linalg.det(cov) > 1e-10:
                    vals, vecs = np.linalg.eig(cov)
                    vals = np.sqrt(vals)*2
                    ell = Ellipse((sub["PC1"].mean(), sub["PC2"].mean()), vals[0], vals[1], angle=np.degrees(np.arctan2(vecs[1,0], vecs[0,0])), alpha=0.2, color=colors[i])
                    ax1.add_patch(ell)
            except Exception:
                pass
    ax1.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title(f"PCA: Biological Groups\n({harmonization_mode} harmonization)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    # PC1 vs PC2 by batch
    ax2 = axes[0,1]
    batch_series = meta.loc[pca_df.index, "batch_collapsed"].astype(str)
    top_batches = batch_series.value_counts().head(8).index
    lab = batch_series.where(batch_series.isin(top_batches), other="other_small")
    for b in lab.unique():
        sub = pca_df[lab == b]
        ax2.scatter(sub["PC1"], sub["PC2"], s=30, label=f"{b} (n={len(sub)})", alpha=0.7)
    ax2.set_xlabel(ax1.get_xlabel()); ax2.set_ylabel(ax1.get_ylabel())
    ax2.set_title("PCA: Batch Effects (Top 8)"); ax2.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    # PC3 vs PC4
    ax3 = axes[1,0]
    for i,g in enumerate(groups):
        sub = pca_df[pca_df["group"]==g]
        if not sub.empty:
            ax3.scatter(sub["PC3"], sub["PC4"], c=colors[i], s=40, alpha=0.7, label=g)
    ax3.set_xlabel(f"PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)")
    ax3.set_ylabel(f"PC4 ({pca_model.explained_variance_ratio_[3]*100:.1f}%)")
    ax3.set_title("Higher Components: PC3 vs PC4")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    # Scree
    ax4 = axes[1,1]
    n = min(10, len(pca_model.explained_variance_ratio_))
    xs = np.arange(1,n+1); vals = pca_model.explained_variance_ratio_[:n]; cum = np.cumsum(vals)
    ax4.bar(xs, vals, alpha=0.7, label='Individual')
    ax4.plot(xs, cum, 'ro-', linewidth=2, markersize=6, label='Cumulative')
    for i,v in enumerate(vals):
        ax4.text(i+1, v+0.01, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=9)
    ax4.set_xlabel("Principal Components"); ax4.set_ylabel("Explained Variance Ratio")
    ax4.set_title("Variance Explained by Components"); ax4.legend(); ax4.grid(True, alpha=0.3); ax4.set_xticks(xs)
    plt.tight_layout()
    _savefig(os.path.join(output_dir, "enhanced_pca_analysis.png"))

    # Clean PC1/PC2 by group
    plt.figure(figsize=(10,8))
    for i,g in enumerate(groups):
        sub = pca_df[pca_df["group"]==g]
        if not sub.empty:
            plt.scatter(sub["PC1"], sub["PC2"], c=colors[i], s=60, alpha=0.85,
                        edgecolors='black', linewidth=0.5, label=f"{g} (n={len(sub)})")
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"PCA: Cervical Cancer Sample Groups\n({harmonization_mode} harmonization)")
    plt.legend(fontsize=11); plt.grid(True, alpha=0.3); plt.tight_layout()
    _savefig(os.path.join(output_dir, "pca_clean_groups.png"))

def nonlinear_embedding_plots(Xc, meta, figdir, harmonization_mode, make=True):
    if not make: return
    Emb, name = None, None
    n_embed = max(2, min(30, Xc.shape[1]-1, Xc.shape[0]-1))
    Xp_embed = PCA(n_components=n_embed, random_state=42).fit_transform(Xc)
    if _HAVE_UMAP:
        try:
            reducer = UMAP(n_neighbors=15, min_dist=0.15, metric="euclidean", random_state=42)
            Emb = reducer.fit_transform(Xp_embed); name = "umap"
        except Exception: pass
    if Emb is None and _HAVE_TSNE:
        try:
            perplexity = max(5, min(30, (Xp_embed.shape[0]-1)//3))
            tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                        perplexity=perplexity, n_iter=1000, random_state=42)
            Emb = tsne.fit_transform(Xp_embed); name = "tsne"
        except Exception: pass
    if Emb is None: return
    emb_df = pd.DataFrame(Emb, index=Xc.index, columns=["E1","E2"]).join(meta)
    plt.figure(figsize=(9,7))
    for grp in ["Normal","Atypia","HPV_Pos","HPV_Neg"]:
        sub = emb_df[emb_df["group"]==grp]
        if not sub.empty: plt.scatter(sub["E1"], sub["E2"], s=25, label=f"{grp} (n={len(sub)})")
    plt.title(f"{name.upper()} on ~{n_embed} PCs ({harmonization_mode})"); plt.legend(frameon=False)
    _savefig(os.path.join(figdir, f"{name}_by_group.png"))

# ---------------- Outliers / Checks ----------------
def detect_outliers(expr_log2: pd.DataFrame) -> pd.DataFrame:
    X = expr_log2.T.fillna(0)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(contamination="auto", random_state=42)
    iso_flag = iso.fit_predict(Xs)
    try:
        n_samples = len(Xs)
        n_neighbors = max(2, min(20, n_samples - 1))
        lof_flag = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(Xs) if n_samples>=3 else np.ones(n_samples)
    except Exception:
        lof_flag = np.ones(len(Xs))
    return pd.DataFrame({"IsolationForest": (iso_flag==-1).astype(int),
                         "LOF": (lof_flag==-1).astype(int)}, index=expr_log2.columns)

# ---------------- Differential Expression ----------------
def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty(n); ranked[order] = np.arange(1, n+1)
    q = p * n / ranked
    q = np.minimum.accumulate(q[order[::-1]])[::-1]  # monotone
    q = np.clip(q, 0, 1)
    out = np.empty_like(q); out[order] = q
    return out

def differential_expression(expr_log2: pd.DataFrame, meta: pd.DataFrame,
                            contrasts: Iterable[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
    """
    Simple DE: Welch's t-test on log2 values; returns dict of dataframes per contrast
    with columns: meanA, meanB, log2FC(A-B), t, pval, qval (BH)
    """
    from scipy.stats import ttest_ind
    res = {}
    for A, B in contrasts:
        A_cols = meta.index[meta["group"]==A]
        B_cols = meta.index[meta["group"]==B]
        if len(A_cols) < 2 or len(B_cols) < 2:
            # need at least 2 samples each for Welch's t-test
            continue
        Xa = expr_log2[A_cols]; Xb = expr_log2[B_cols]
        ma = Xa.mean(axis=1); mb = Xb.mean(axis=1)
        log2fc = ma - mb
        t, p = ttest_ind(Xa.T, Xb.T, equal_var=False, nan_policy="omit")
        q = _bh_fdr(p)
        df = pd.DataFrame({"mean_"+A: ma, "mean_"+B: mb,
                           "log2FC": log2fc, "t": t, "pval": p, "qval": q}).sort_values("qval")
        res[f"{A}_vs_{B}"] = df
    return res

# ---------------- GSEA (optional) ----------------
def gsea_from_ranks(rank_series: pd.Series, gmt_path: str, outdir: str) -> Optional[pd.DataFrame]:
    """
    rank_series: pd.Series where index=gene, values=ranking score (e.g., t or log2FC)
    """
    if not (_HAVE_GSEAPY and os.path.exists(gmt_path)):
        return None
    os.makedirs(outdir, exist_ok=True)
    rnk_df = rank_series.dropna().sort_values(ascending=False)
    rnk_file = os.path.join(outdir, "ranks.rnk")
    rnk_df.to_csv(rnk_file, sep="\t", header=False)
    prer = gp.prerank(rnk=rnk_file, gene_sets=gmt_path, min_size=5, max_size=3000,
                      outdir=outdir, seed=42, processes=4, verbose=False)
    res = getattr(prer, "res2d", None)
    return res if isinstance(res, pd.DataFrame) else None

# ---------------- The main orchestrator ----------------
def run_pipeline(
    group_to_file: Dict[str, io.BytesIO | str],
    metadata_file: io.BytesIO | str,
    metadata_name_hint: str | None = None,   # <— NEW: pass filename to infer format
    metadata_id_cols: List[str] = ["Id","ID","id","CleanID","sample","Sample"],
    metadata_batch_col: Optional[str] = None,
    out_root: str = "out",
    fig_subdir: str = "figs",
    min_batch_size_for_combat: int = 5,
    pca_topk_features: int = 5000,
    make_nonlinear: bool = True,
    gsea_gmt: Optional[str] = None,  # optional path to .gmt
) -> Dict[str, str]:
    """
    Returns dict with paths to key outputs and a ZIP file consolidating everything.
    """
    OUTDIR = os.path.join(out_root)
    FIGDIR = os.path.join(OUTDIR, fig_subdir)
    REPORT_DIR = os.path.join(OUTDIR, "report")
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)

    # 1) Load expression
    loaded = {g: read_expression_xlsx(f, g) for g, f in group_to_file.items()}
    combined_expr = pd.concat([loaded[g] for g in loaded.keys()], axis=1, join="outer")
    combined_expr = combined_expr.replace([np.inf,-np.inf], np.nan)

    # 2) Build & align metadata
    meta_base = build_metadata(loaded)
    meta_base["bare_id"] = meta_base.index.str.split("__", n=1).str[-1]

    # 2a) read metadata robustly (xlsx/tsv/csv; paths or bytes)
    m = _read_metadata_any(metadata_file, name_hint=metadata_name_hint)

    id_col = next((c for c in metadata_id_cols if c in m.columns), None)
    if id_col is None:
        raise ValueError(f"Could not find an ID column among {metadata_id_cols} in metadata: {list(m.columns)}")

    if metadata_batch_col is None:
        if "Batch" in m.columns: metadata_batch_col = "Batch"
        elif "batch" in m.columns: metadata_batch_col = "batch"
        else:
            batch_like = [c for c in m.columns if str(c).lower().startswith("batch")]
            metadata_batch_col = batch_like[0] if batch_like else None

    m_align = m.set_index(id_col)
    meta = meta_base.join(m_align, on="bare_id", how="left", rsuffix="_ext")
    if metadata_batch_col is not None and metadata_batch_col in meta.columns:
        meta["batch_external_raw"] = meta[metadata_batch_col]
        meta["batch_external"] = meta[metadata_batch_col].map(normalize_batch_token)
        meta["batch"] = meta["batch_external"].where(meta["batch_external"].notna(), infer_batches(meta_base))
    else:
        meta["batch"] = infer_batches(meta_base)
    keep = ["group","batch"]
    for extra in ("Age","age","Sex","sex","Gender","gender"):
        if extra in meta.columns: keep.append(extra)
    meta = meta[keep]

    # Save inputs
    combined_expr.to_csv(os.path.join(OUTDIR, "expression_combined.tsv"), sep="\t")
    meta.to_csv(os.path.join(OUTDIR, "metadata.tsv"), sep="\t")

    # 3) Detect type/platform
    dtype, platform, diags = detect_data_type_and_platform(combined_expr)

    # 4) log2 & z
    expr_log2 = np.log2(combined_expr + 1).replace([np.inf,-np.inf], np.nan)
    row_mean = expr_log2.mean(axis=1); row_std = expr_log2.std(axis=1, ddof=1).replace(0, np.nan)
    expr_z = expr_log2.sub(row_mean, axis=0).div(row_std, axis=0).replace([np.inf,-np.inf], np.nan).fillna(0)

    # 5) Impute + variance filter
    expr_imputed = drop_zero_variance(row_mean_impute(expr_log2))
    gene_vars = expr_imputed.var(axis=1)
    pos = gene_vars[gene_vars > 0]
    topk = min(pca_topk_features, len(pos))
    expr_filtered = expr_imputed.loc[pos.nlargest(topk).index]

    # 6) Batch collapsing + harmonization
    meta["batch_collapsed"] = smart_batch_collapse(meta, min_batch_size_for_combat)
    x_combat = _combat(expr_filtered, meta["batch_collapsed"])
    expr_harmonized = x_combat or _fallback_center(expr_filtered, meta["batch_collapsed"])
    mode = "ComBat" if x_combat is not None else "fallback_center"

    # 7) PCA
    Xc = safe_matrix_for_pca(zscore_rows(expr_harmonized), topk=topk)
    pca = PCA(n_components=6, random_state=42).fit(Xc)
    Xp = pca.transform(Xc)
    pca_df = pd.DataFrame(Xp[:, :6], columns=[f"PC{i+1}" for i in range(6)], index=Xc.index).join(meta)

    # 8) Figures
    figs = create_basic_qc_figures(expr_log2, expr_z, expr_harmonized, meta, FIGDIR)
    create_enhanced_pca_plots(pca_df, pca, meta, FIGDIR, mode)
    nonlinear_embedding_plots(Xc, meta, FIGDIR, mode, make=make_nonlinear)

    # 9) Outliers
    outliers = detect_outliers(expr_log2); outliers.to_csv(os.path.join(OUTDIR, "outliers.tsv"), sep="\t")

    # 10) DE (group contrasts)
    groups = meta["group"].unique().tolist()
    default_contrasts = [("Atypia","Normal"),("HPV_Pos","Normal"),("HPV_Neg","Normal"),("HPV_Pos","HPV_Neg")]
    contrasts = [(a,b) for (a,b) in default_contrasts if a in groups and b in groups]
    de = differential_expression(expr_log2, meta, contrasts)
    de_dir = os.path.join(OUTDIR, "de"); os.makedirs(de_dir, exist_ok=True)
    for k, df in de.items():
        df.to_csv(os.path.join(de_dir, f"DE_{k}.tsv"), sep="\t")

    # 11) GSEA (optional; uses t-stat as ranking; falls back gracefully)
    gsea_dir = os.path.join(OUTDIR, "gsea"); os.makedirs(gsea_dir, exist_ok=True)
    if gsea_gmt and os.path.exists(gsea_gmt):
        for k, df in de.items():
            ranks = df["t"]; ranks.index = df.index
            res = gsea_from_ranks(ranks, gsea_gmt, os.path.join(gsea_dir, k))
            if isinstance(res, pd.DataFrame):
                res.to_csv(os.path.join(gsea_dir, f"GSEA_{k}.tsv"), sep="\t")

    # 12) Save harmonized + PCA tables
    expr_harmonized.to_csv(os.path.join(OUTDIR, "expression_harmonized.tsv"), sep="\t")
    pca_df.to_csv(os.path.join(OUTDIR, "pca_scores.tsv"), sep="\t")

    # 13) Simple report
    rep = {
        "qc": {"data_type": dtype, "platform": platform,
               "zero_fraction": float(diags.get("zero_fraction", np.nan)),
               "harmonization_mode": mode},
        "shapes": {"genes": int(combined_expr.shape[0]), "samples": int(combined_expr.shape[1])}
    }
    with open(os.path.join(OUTDIR, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # 14) Zip everything
    zip_path = os.path.join(OUTDIR, "results_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTDIR):
            for name in files:
                if name.endswith(".zip"): continue
                p = os.path.join(root, name)
                zf.write(p, arcname=os.path.relpath(p, OUTDIR))

    return {
        "outdir": OUTDIR,
        "figdir": FIGDIR,
        "report_json": os.path.join(OUTDIR, "report.json"),
        "zip": zip_path
    }
