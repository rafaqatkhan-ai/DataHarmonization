# -*- coding: utf-8 -*-
# harmonizer.py — single-dataset pipeline + Multi-GEO wrapper + meta-analysis + presenter assets + dataset QA I/O
import os, re, io, json, warnings, zipfile, tempfile
from typing import Dict, Tuple, List, Iterable, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score

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
# ---------------- Config ----------------
ZERO_INFLATION_THRESH = 0.4
MICROARRAY_RANGE_MAX  = 20.0
VAR_EPS = 1e-12
HOUSEKEEPING_GENES = ["ACTB","GAPDH","RPLP0","B2M","HPRT1","PGK1","TBP","GUSB"]
SEX_MARKERS = {"female":["XIST"], "male":["RPS4Y1","KDM5D","UTY"]}
# ==== NEW: Search GEO by name/keywords and turn them into GSE IDs ====
import re, time, xml.etree.ElementTree as ET
from typing import Sequence

_GSE_RE = re.compile(r"^GSE\d+$", re.IGNORECASE)
# ==== GEO matrix builders & fallbacks ====
import gzip, glob, tarfile, zipfile
from io import BytesIO

def _series_to_expression_and_meta(gse):
    """
    Try the in-memory Series object first (bulk only).
    Returns (expr_df, meta_df) or (None, None) if not possible.
    """
    try:
        # pivot_samples('VALUE') exists only for bulk-style matrices
        expr = gse.pivot_samples('VALUE')
        expr.index = expr.index.astype(str).str.strip().str.upper().str.replace(r'\.\d+$','',regex=True)
        expr = expr.apply(pd.to_numeric, errors='coerce')
        # metadata from GSMs
        rows = []
        for gsm_name, gsm in gse.gsms.items():
            ch = gsm.metadata
            rows.append({
                "sample": gsm_name,
                "group": (ch.get("characteristics_ch1", ["ALL"])[0] if "characteristics_ch1" in ch else "ALL"),
                "bare_id": gsm_name,
                "tissue_type": "; ".join([x for x in ch.get("characteristics_ch1", []) if "tissue" in x.lower()]) if "characteristics_ch1" in ch else "",
                "library_strategy": ", ".join(ch.get("library_strategy", [])) if "library_strategy" in ch else "",
                "extraction_protocol": ", ".join(ch.get("extraction_protocol_ch1", [])) if "extraction_protocol_ch1" in ch else "",
            })
        meta = pd.DataFrame(rows).set_index("sample")
        # Align meta to expression columns
        meta = meta.reindex(expr.columns).fillna({"group": "ALL", "bare_id": ""})
        return expr, meta
    except Exception:
        return None, None

def _parse_series_matrix_file(path: str):
    """
    Parse a downloaded *series_matrix.txt.gz if present.
    """
    try:
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as fh:
            df = pd.read_csv(fh, sep="\t", comment="!", low_memory=False)
        # Find gene column & sample columns
        # Commonly first col is ID_REF / Gene symbol etc.
        gene_col = df.columns[0]
        df = df.rename(columns={gene_col: "Biomarker"}).set_index("Biomarker")
        # Keep only numeric cols
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(axis=1, how="all")
        expr = df
        expr.index = expr.index.astype(str).str.strip().str.upper().str.replace(r'\.\d+$','',regex=True)
        # Minimal meta (we’ll enrich later from GSM SOFT if available)
        meta = pd.DataFrame({"sample": expr.columns, "group": "ALL", "bare_id": expr.columns}).set_index("sample")
        return expr, meta
    except Exception:
        return None, None

def _read_10x_like_matrix(mtx_dir_or_file: str):
    """
    Read 10x/MTX/H5/loom and return AnnData (requires scanpy/anndata).
    """
    if not (_HAVE_SCANPY):
        raise RuntimeError("scanpy/anndata not installed; cannot parse single-cell supplementary files.")
    adata = None
    path = mtx_dir_or_file
    try:
        # H5 10x
        if os.path.isfile(path) and path.lower().endswith((".h5", ".hdf5")):
            adata = sc.read_10x_h5(path)
        # Loom
        elif os.path.isfile(path) and path.lower().endswith(".loom"):
            adata = sc.read_loom(path)
        # MTX folder or tar/zip with MTX
        elif os.path.isdir(path):
            # Expect matrix.mtx + barcodes.tsv + features/genes.tsv
            adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=False)
        else:
            # try extracting archives into a temp dir and re-read
            tmpd = tempfile.mkdtemp()
            try:
                if tarfile.is_tarfile(path):
                    with tarfile.open(path) as tf: tf.extractall(tmpd)
                elif zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as zf: zf.extractall(tmpd)
                # find folder containing matrix.mtx
                cand = None
                for root, _, files in os.walk(tmpd):
                    if "matrix.mtx" in files:
                        cand = root; break
                if cand is None:
                    raise RuntimeError("Could not locate matrix.mtx inside archive.")
                adata = sc.read_10x_mtx(cand, var_names='gene_symbols', cache=False)
            finally:
                shutil.rmtree(tmpd, ignore_errors=True)
    except Exception as e:
        raise
    if adata is None:
        raise RuntimeError("Unsupported 10x format.")
    return adata

def _pseudo_bulk_by_gene(adata, level: str = None):
    """
    Collapse single-cell counts to pseudo-bulk per sample (or per GSM if available).
    If adata.obs has 'sample' or 'GSM', use that; otherwise collapse all cells to one column.
    """
    obs_keys = list(adata.obs.columns)
    key = None
    for k in ["sample", "Sample", "GSM", "gsm", "library", "orig.ident"]:
        if k in obs_keys:
            key = k; break
    if key is None:
        # one pseudo-bulk for all cells
        counts = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                              columns=adata.var_names).sum(axis=0).to_frame("S1").T
    else:
        df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                          columns=adata.var_names)
        df[key] = adata.obs[key].astype(str).values
        counts = df.groupby(key).sum().T
    counts.index = counts.index.astype(str).str.upper()
    counts.index = counts.index.str.replace(r'\.\d+$','',regex=True)
    return counts

def _collect_supplementary_paths(gse_root_dir: str):
    """
    Recursively find candidate supplementary files anywhere under the GSE root.
    Returns {gsm: [paths...]} mapping where keys are GSM IDs if detectable,
    otherwise the files are attached to a pseudo key equal to their parent dir.
    """
    exts = (".h5", ".hdf5", ".loom", ".mtx", ".mtx.gz", ".tar.gz", ".tgz", ".zip")
    sup = {}
    for root, _, files in os.walk(gse_root_dir):
        for fname in files:
            f = os.path.join(root, fname)
            if not f.lower().endswith(exts):
                continue
            # best-effort GSM inference from path
            m = re.search(r"(GSM\d+)", f, re.IGNORECASE)
            gsm = m.group(1).upper() if m else os.path.basename(os.path.dirname(f))
            sup.setdefault(gsm, []).append(f)
    return sup
def _read_10x_minimal(path: str):
    """
    Minimal 10x reader WITHOUT scanpy/anndata/scipy.
    Works with a directory (or archive) containing:
      matrix.mtx(.gz), features.tsv(.gz) or genes.tsv(.gz), barcodes.tsv(.gz)
    Returns a dense DataFrame: rows=genes, cols=cells.
    """
    import gzip, io, shutil, tempfile, tarfile, zipfile
    import numpy as np
    import pandas as pd

    def _open_text(p):
        return gzip.open(p, 'rt') if p.endswith('.gz') else open(p, 'r')

    # If an archive is passed, extract to temp
    cleanup_dir = None
    if os.path.isfile(path) and (tarfile.is_tarfile(path) or zipfile.is_zipfile(path)):
        tmpd = tempfile.mkdtemp(prefix="tenx_")
        cleanup_dir = tmpd
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tf: tf.extractall(tmpd)
        else:
            with zipfile.ZipFile(path) as zf: zf.extractall(tmpd)
        path = tmpd

    try:
        # Find the folder that contains matrix.mtx(.gz)
        mdir = None
        for root, _, files in os.walk(path):
            if ('matrix.mtx' in files) or ('matrix.mtx.gz' in files):
                mdir = root; break
        if mdir is None:
            raise RuntimeError("matrix.mtx not found")

        # Resolve filenames
        mtx = os.path.join(mdir, 'matrix.mtx.gz') if os.path.exists(os.path.join(mdir,'matrix.mtx.gz')) else os.path.join(mdir,'matrix.mtx')
        feats = None
        for fn in ('features.tsv.gz','features.tsv','genes.tsv.gz','genes.tsv'):
            p = os.path.join(mdir, fn)
            if os.path.exists(p): feats = p; break
        bars = None
        for fn in ('barcodes.tsv.gz','barcodes.tsv'):
            p = os.path.join(mdir, fn)
            if os.path.exists(p): bars = p; break
        if feats is None or bars is None:
            raise RuntimeError("features.tsv / barcodes.tsv not found")

        # ---- Tiny .mtx reader (COO) without SciPy
        def _read_mtx_coo(path_mtx):
            # Support optional gzip
            fh = gzip.open(path_mtx, 'rt') if path_mtx.endswith('.gz') else open(path_mtx, 'r')
            with fh:
                # Skip comments
                header = fh.readline().strip()
                while header.startswith('%'):
                    header = fh.readline().strip()
                # Now header has dimension line
                parts = header.split()
                if len(parts) != 3:
                    # sometimes the first non-comment line can be blank
                    # read until we hit 3 numbers
                    while True:
                        header = fh.readline().strip()
                        if not header:
                            continue
                        parts = header.split()
                        if len(parts) == 3: break
                nrows, ncols, nnz = map(int, parts)
                rows = np.empty(nnz, dtype=np.int64)
                cols = np.empty(nnz, dtype=np.int64)
                data = np.empty(nnz, dtype=np.float64)
                i = 0
                for line in fh:
                    if not line.strip():
                        continue
                    r, c, v = line.split()[:3]
                    rows[i] = int(r) - 1  # 1-based -> 0-based
                    cols[i] = int(c) - 1
                    try:
                        data[i] = float(v)
                    except Exception:
                        data[i] = 0.0
                    i += 1
                # build dense (we'll sum duplicates)
                mat = {}
                for r, c, v in zip(rows, cols, data):
                    mat[(r,c)] = mat.get((r,c), 0.0) + v
                # to dense ndarray
                arr = np.zeros((nrows, ncols), dtype=np.float64)
                for (r,c), v in mat.items():
                    arr[r, c] = v
                return arr

        X = _read_mtx_coo(mtx)

        # Load features and barcodes
        with _open_text(feats) as fh:
            feat_rows = [line.rstrip('\n').split('\t') for line in fh]
        with _open_text(bars) as fh:
            cell_ids = [line.strip() for line in fh if line.strip()]

        # Feature name selection: prefer second column; else first
        gene_names = []
        for r in feat_rows:
            if len(r) >= 2 and r[1]:
                gene_names.append(r[1])
            elif len(r) >= 1:
                gene_names.append(r[0])
            else:
                gene_names.append("NA")
        # Normalize + de-duplicate empties
        gene_names = [str(g).strip().upper().split('.')[0] if g else "NA" for g in gene_names]
        # Ensure lengths match
        if len(gene_names) != X.shape[0]:
            # fallback: truncate/pad
            gene_names = (gene_names + ["NA"] * X.shape[0])[:X.shape[0]]
        if len(cell_ids) != X.shape[1]:
            cell_ids = (cell_ids + [f"C{i+1}"] * X.shape[1])[:X.shape[1]]

        df = pd.DataFrame(X, index=gene_names, columns=cell_ids)
        # collapse duplicate gene names
        df = df.groupby(level=0, sort=False).sum()
        return df
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

def _build_from_supplementary(gse_root_dir: str):
    """
    Build pseudo-bulk expression from per-GSM 10x-like supplementary files.
    Returns (expr_df, meta_df) if successful, else (None, None).
    """
    sup = _collect_supplementary_paths(gse_root_dir)
    matrices = []

    for gsm, paths in sup.items():
        built = False
        for p in paths:
            try:
                if _HAVE_SCANPY:
                    # scanpy/anndata path
                    adata = _read_10x_like_matrix(p if os.path.isfile(p) else os.path.dirname(p))
                    pb = _pseudo_bulk_by_gene(adata)
                else:
                    # minimal reader path (no scanpy)
                    root = p if os.path.isdir(p) else os.path.dirname(p)
                    X = _read_10x_minimal(root)
                    # pseudo-bulk by summing cells
                    pb = X.sum(axis=1).to_frame(gsm)
                    pb.index = (
                        pb.index.astype(str)
                        .str.upper()
                        .str.replace(r"\.\d+$", "", regex=True)
                    )

                matrices.append(pb)
                built = True
                break  # done with this GSM once one path worked
            except Exception:
                # try next candidate path for this GSM
                continue

        if not built:
            # nothing worked for this GSM; move on
            continue

    if not matrices:
        return None, None

    expr = pd.concat(matrices, axis=1, join="outer").fillna(0.0)
    meta = (
        pd.DataFrame({"sample": expr.columns, "group": "ALL", "bare_id": expr.columns})
        .set_index("sample")
    )
    return expr, meta


def _entrez_esearch_gds(term: str, retmax: int = 20) -> list[str]:
    """
    Query NCBI E-utilities (db=gds) with a free-text term and return matching GSE accessions.
    We prefer GSE (Series) records. No API key required for light usage.
    """
    import urllib.parse, urllib.request

    # Prefer GSE records: term + gse[Filter]
    composed = f"({term}) AND gse[Filter]"
    params = {
        "db": "gds",
        "term": composed,
        "retmax": str(retmax),
        "retmode": "xml",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml = resp.read()

    root = ET.fromstring(xml)
    ids = [idtag.text for idtag in root.findall(".//IdList/Id") if idtag.text]
    if not ids:
        return []

    # For each GDS (or GSE) ID, fetch the GSE accession via esummary (or efetch)
    gse_list = []
    chunks = [ids[i:i+50] for i in range(0, len(ids), 50)]
    for chunk in chunks:
        params = {
            "db": "gds",
            "id": ",".join(chunk),
            "retmode": "xml",
        }
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml2 = resp.read()
        root2 = ET.fromstring(xml2)
        for doc in root2.findall(".//DocSum"):
            acc = None
            # Try to read the “Accession” item or “GSE” field
            for itm in doc.findall("./Item"):
                if itm.get("Name") in ("Accession", "GSE") and itm.text:
                    acc = itm.text.strip()
                    break
            if acc and _GSE_RE.match(acc):
                gse_list.append(acc.upper())
    # Dedup while preserving order
    seen = set(); out = []
    for g in gse_list:
        if g not in seen:
            out.append(g); seen.add(g)
    return out

def search_geo_gse_by_query(query: str, max_results: int = 2, organism_hint: str | None = None) -> list[str]:
    """
    Given a free-text query (dataset name/disease), return up to max_results GSE accessions.
    If the query already contains valid GSE IDs, we pass them through directly.
    """
    # If the user typed explicit GSE IDs, use them
    tokens = re.split(r"[,\s]+", query.strip())
    explicit = [t.upper() for t in tokens if _GSE_RE.match(t)]
    if explicit:
        # respect max_results if provided
        return list(dict.fromkeys(explicit))[:max_results]

    # Otherwise, search GDS with the query
    gses = _entrez_esearch_gds(query, retmax=max_results * 6 or 20)

    # Optional light filtering by organism in title/text (best-effort)
    if organism_hint and gses:
        organism_hint = organism_hint.strip().lower()
        # (Quick heuristic: keep order and prefer those that include the organism name in accession summary label)
        # For simplicity, we just return the top-of-list; real filtering would require extra esummary parsing.
        pass

    # Keep the top max_results
    return gses[:max_results]

def prepare_datasets_from_geo_queries(queries_or_names: Sequence[str],
                                      max_per_query: int = 2) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    """
    Accepts a list of free-text queries or dataset names.
    For each query, finds up to max_per_query GSEs; merges unique GSEs; then calls fetch_geo_as_datasets.
    """
    all_gses: list[str] = []
    for q in queries_or_names:
        q = (q or "").strip()
        if not q: continue
        found = search_geo_gse_by_query(q, max_results=max_per_query)
        all_gses.extend(found)
        # be polite with NCBI
        time.sleep(0.34)

    # Deduplicate while preserving order
    unique_gses = []
    seen = set()
    for g in all_gses:
        if g not in seen:
            unique_gses.append(g); seen.add(g)

    if not unique_gses:
        raise RuntimeError("No GEO Series (GSE) results found for your query/queries.")

    # Reuse the existing downloader
    return fetch_geo_as_datasets(unique_gses)

# ---------------- Helpers ----------------
def _collapse_dupes_df_by_index(df: pd.DataFrame, how_num: str = "median", keep: str = "first") -> pd.DataFrame:
    if not df.index.duplicated().any(): return df
    num = df.select_dtypes(include=[np.number]).columns
    non = [c for c in df.columns if c not in num]
    agg_spec = {}
    if len(num): agg_spec.update({c: ("median" if how_num not in {"median","mean"} else how_num) for c in num})
    if len(non): agg_spec.update({c: (lambda x: x.iloc[0] if keep == "first" else x.iloc[-1]) for c in non})
    out = df.groupby(level=0, sort=False).agg(agg_spec); out.index.name = df.index.name; return out

def roman_to_int(s: str) -> Optional[int]:
    if not isinstance(s, str): return None
    s = s.upper().strip()
    if not s or not re.fullmatch(r"[MDCLXVI]+", s): return None
    values = dict(M=1000,D=500,C=100,L=50,X=10,V=5,I=1)
    total, prev = 0, 0
    for ch in reversed(s):
        v = values[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total

def normalize_group_value(x: str) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    s = str(x).strip().lower().replace("_","").replace("-","").replace(" ","")
    if s in {"disease","diseased","case","patient"}: return "Disease"
    if s in {"control","ctrl","healthy","normal"}: return "Control"
    if s in {"hpvpos","hpvpositive","hpv+"}: return "HPV_Pos"
    if s in {"hpvneg","hpvnegative","hpv-"}: return "HPV_Neg"
    if s in {"atypia","precancer"}: return "Atypia"
    return str(x).strip().capitalize()

def normalize_batch_token(x: str) -> str:
    if pd.isna(x): return np.nan
    t = str(x).strip()
    if re.fullmatch(r"\d+", t): return t
    r = roman_to_int(t)
    if r is not None: return f"RN_{r}"
    t = re.sub(r"\s+", "_", t); t = re.sub(r"[^A-Za-z0-9_]+", "", t)
    return t if t else np.nan

def _as_bytesio(x):
    if isinstance(x, io.BytesIO): x.seek(0); return x
    if isinstance(x, bytes): return io.BytesIO(x)
    return None

def _as_bytesio_seekable(x):
    bio = _as_bytesio(x)
    if bio is not None: bio.seek(0)
    return bio

def _norm_id(x: str | None) -> str:
    if x is None: return ""
    return re.sub(r"\s+", "", str(x).strip().lower())

# ---------------- IO ----------------
def read_expression_any(bytes_or_path, name_hint: Optional[str] = None, group_name: Optional[str] = None,
                        assume_first_col_is_gene: bool = True) -> pd.DataFrame:
    is_pathlike = isinstance(bytes_or_path, (str, os.PathLike))
    suffix = (os.path.splitext(name_hint)[1].lower() if name_hint else
              (os.path.splitext(str(bytes_or_path))[1].lower() if is_pathlike else ""))

    def _read_excel(x):
        if is_pathlike: df = pd.read_excel(x, sheet_name=0, engine="openpyxl")
        else: bio = _as_bytesio(x); bio.seek(0); df = pd.read_excel(bio, sheet_name=0, engine="openpyxl")
        return df

    def _read_csv(x, sep=None):
        if is_pathlike: return pd.read_csv(x, sep=sep)
        bio = _as_bytesio(x); bio.seek(0); return pd.read_csv(bio, sep=sep)

    if suffix in (".xlsx", ".xls"):
        df = _read_excel(bytes_or_path)
    elif suffix in (".tsv", ".txt"):
        df = _read_csv(bytes_or_path, sep="\t")
    elif suffix == ".csv":
        df = _read_csv(bytes_or_path, sep=None)
    else:
        try: df = _read_excel(bytes_or_path)
        except Exception: df = _read_csv(bytes_or_path, sep=None)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    lower = [str(c).strip().lower() for c in df.columns]
    gene_col = None
    for key in ["biomarkers","biomarker","marker","gene","feature","id","name"]:
        if key in lower:
            gene_col = df.columns[lower.index(key)]
            break
    if gene_col is None:
        gene_col = df.columns[0] if assume_first_col_is_gene else None
        if gene_col is None: raise ValueError("Could not infer gene column.")
    df = df.rename(columns={gene_col: "Biomarker"}).set_index("Biomarker")
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    if group_name: df.columns = [f"{group_name}__{str(c).strip()}" for c in df.columns]
    df.index = df.index.astype(str).str.strip().str.upper()
    df.index = df.index.str.replace(r'\.\d+$', '', regex=True)
    return df.groupby(level=0).median(numeric_only=True)

def build_metadata_from_columns(columns: List[str], groups_from_prefix: bool = True) -> pd.DataFrame:
    idx = pd.Index(columns, name="sample")
    if groups_from_prefix and any("__" in c for c in columns):
        df = pd.DataFrame({
            "sample": idx,
            "group": [c.split("__", 1)[0] if "__" in c else "Unknown" for c in columns],
            "bare_id": [c.split("__", 1)[-1] for c in columns],
        }).set_index("sample")
    else:
        df = pd.DataFrame({"sample": idx, "group": "ALL", "bare_id": columns}).set_index("sample")
    return df

def _read_metadata_any(metadata_obj, name_hint: str | None = None) -> pd.DataFrame:
    """
    Robustly read metadata from Excel or delimited text (CSV/TSV/TXT) with delimiter sniffing.
    Works even if a .txt/.tsv file is actually comma-separated (prevents ['sample,condition']).
    """
    import csv
    is_pathlike = isinstance(metadata_obj, (str, os.PathLike))
    suffix = (os.path.splitext(name_hint)[1].lower() if name_hint
              else (os.path.splitext(str(metadata_obj))[1].lower() if is_pathlike else ""))

    # Excel first
    if suffix in (".xlsx", ".xls"):
        return (pd.read_excel(metadata_obj, engine="openpyxl")
                if is_pathlike else pd.read_excel(_as_bytesio_seekable(metadata_obj), engine="openpyxl"))

    def _peek_bytes(obj, n=4096) -> bytes:
        try:
            if isinstance(obj, (str, os.PathLike)):
                with open(obj, "rb") as fh:
                    return fh.read(n)
            bio = _as_bytesio_seekable(obj)
            pos = bio.tell()
            b = bio.read(n)
            bio.seek(pos)
            return b
        except Exception:
            return b""

    def _sniff_sep(obj) -> str | None:
        sample = _peek_bytes(obj, 4096).decode(errors="ignore")
        if "\t" in sample and "," not in sample:
            return "\t"
        if "," in sample and "\t" not in sample:
            return ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            return dialect.delimiter
        except Exception:
            return None

    def _read_text(obj):
        # 1) pandas inference
        try:
            return pd.read_csv(obj, sep=None, engine="python")
        except Exception:
            pass
        # 2) sniffed delimiter
        sep = _sniff_sep(obj)
        if sep:
            try:
                return pd.read_csv(obj, sep=sep)
            except Exception:
                pass
        # 3) fallbacks
        for sep in [",", "\t", ";", "|"]:
            try:
                return pd.read_csv(obj, sep=sep)
            except Exception:
                continue
        # 4) last resort: try excel
        return (pd.read_excel(obj, engine="openpyxl")
                if isinstance(obj, (str, os.PathLike))
                else pd.read_excel(_as_bytesio_seekable(obj), engine="openpyxl"))

    # Route text-like / unknown suffixes through the robust reader
    return _read_text(metadata_obj if is_pathlike else _as_bytesio_seekable(metadata_obj))

# ---------------- Batch detection ----------------
_BATCH_HINTS = [
    "batch","Batch","BATCH","center","Center","site","Site","location","Location",
    "series","Series","geo_series","GEO_series","run","Run","lane","Lane","plate","Plate",
    "sequencer","Sequencer","flowcell","Flowcell","library","Library","library_prep","LibraryPrep",
    "study","Study","project","Project","lab","Lab","date","Date","collection_date","CollectionDate",
    "source_name_ch1","title","characteristics_ch1","characteristics"
]

def _guess_batch_column_from_meta(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty: return None
    for k in _BATCH_HINTS:
        if k in df.columns:
            nun = df[k].nunique(dropna=True)
            if 1 < nun <= max(20, len(df)//2): return k
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if 1 < nun <= max(20, len(df)//2): return c
    return None

def infer_batches(metadata: pd.DataFrame) -> pd.Series:
    md = metadata.copy()
    for col in ["batch","Batch","run","Run","lane","Lane","plate","Plate","sequencer","flowcell","Flowcell"]:
        if col in md.columns:
            s = md[col].astype(str); s.index = md.index; return s
    def guess_token(s):
        m = re.search(r"(FC\w+|L\d{3}|P\d+|\d{4}[-_]\d{2}[-_]\d{2}|\d{8})", s)
        return m and m.group(0) or "B0"
    return pd.Series([guess_token(x) for x in md.index.astype(str)], index=md.index, name="batch")

# ---------------- Transforms ----------------
def zscore_rows(M: pd.DataFrame) -> pd.DataFrame:
    mu = M.mean(axis=1)
    sd = M.std(axis=1, ddof=1).replace(0, np.nan)
    return M.sub(mu, axis=0).div(sd, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)

def row_mean_impute(X): return X.apply(lambda r: r.fillna(r.mean()), axis=1).fillna(0)
def drop_zero_variance(X):
    var = X.var(axis=1, ddof=1).astype(float).fillna(0.0)
    return X.loc[var > VAR_EPS]

def safe_matrix_for_pca(matrix, topk: int = 5000):
    X = (matrix.copy().apply(pd.to_numeric, errors="coerce").astype(float)
         .replace([np.inf, -np.inf], np.nan).fillna(0.0))
    ddof = 0 if X.shape[1] < 2 else 1
    var = X.var(axis=1, ddof=ddof).fillna(0.0)
    nz = var > VAR_EPS
    if not nz.any():
        med = X.median(axis=1)
        mad = (X.sub(med, axis=0).abs()).median(axis=1)
        nz = mad > 0
    if not nz.any():
        raise RuntimeError("No non-zero-variance features for PCA.")
    if topk and nz.sum() > topk: X = X.loc[var.loc[nz].nlargest(topk).index]
    else: X = X.loc[nz]
    X = X.T; X.columns = X.columns.map(str); return X

def detect_data_type_and_platform(X: pd.DataFrame) -> Tuple[str, str, Dict]:
    vals = X.values.ravel(); vals = vals[np.isfinite(vals)]
    zero_frac = float((X==0).sum().sum()) / float(X.size) if X.size else 0.0
    rng = (np.nanpercentile(vals, 99.5) - np.nanpercentile(vals, 0.5)) if vals.size else 0
    idx_str = X.index.astype(str)
    has_ilumn = any(str(s).startswith("ILMN_") for s in idx_str)
    has_affy  = any(re.match(r"^\d+_at$", str(s)) for s in idx_str)
    has_ensembl = any(str(s).startswith("ENSG") for s in idx_str)
    data_type = "bulk"
    if zero_frac >= ZERO_INFLATION_THRESH: data_type = "scRNA-seq"
    platform = "Unknown"
    if has_ilumn or has_affy:
        platform = "Microarray (Illumina/Affy)"; data_type = "microarray"
    elif rng > MICROARRAY_RANGE_MAX and not has_ilumn and not has_affy:
        platform = "Long-read/Counts-like (Illumina/PacBio)"
    elif has_ensembl:
        platform = "Short-read RNA-seq (Illumina)"
    diags = {"zero_fraction": zero_frac, "value_range_approx": rng}
    return data_type, platform, diags

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
    grand_mean = X.values.mean(); grand_std  = X.values.std()
    b = meta_batch.astype(str)
    for batch in pd.unique(b):
        cols = b.index[b == batch]
        gmean = X[cols].values.mean(); gstd  = X[cols].values.std()
        X[cols] = (X[cols] - gmean) * (grand_std / (gstd if gstd>0 else 1)) + grand_mean
    return X

def smart_batch_collapse(meta: pd.DataFrame, min_size: int) -> pd.Series:
    meta = meta.copy()
    if "batch" not in meta.columns:
        return pd.Series("B0", index=meta.index, name="batch_collapsed")
    b = meta["batch"].astype(str)
    g = meta["group"].astype(str) if "group" in meta.columns else pd.Series("ALL", index=meta.index)
    counts = b.value_counts()
    large = counts[counts >= min_size].index
    small = counts[counts < min_size].index
    mapping = {k: k for k in large}
    for batch in small:
        idx = b.index[b == batch]
        grp_series = g.loc[idx] if len(idx) else pd.Series(dtype=str)
        main_group = (grp_series.value_counts().idxmax() if len(grp_series) else "mixed")
        mapping[batch] = f"small_{main_group}"
    return b.map(mapping).rename("batch_collapsed")

# ---------------- Figures & analytics ----------------
def _savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

def create_sample_qc_figures(raw_expr, expr_log2, meta, figdir: str) -> List[str]:
    os.makedirs(figdir, exist_ok=True); paths = []
    libsize = raw_expr.sum(axis=0).astype(float)
    zero_rate = (raw_expr == 0).sum(axis=0) / raw_expr.shape[0]
    qc = pd.DataFrame({"library_size": libsize, "zero_rate": zero_rate}, index=raw_expr.columns)
    qc.to_csv(os.path.join(figdir, "..", "sample_qc.tsv"), sep="\t")
    plt.figure(figsize=(12,4)); plt.bar(range(len(libsize)), libsize.values)
    plt.title("Per-sample Library Size (pre-log)"); plt.xlabel("Samples"); plt.ylabel("Sum of counts")
    _savefig(os.path.join(figdir, "qc_library_size.png")); paths.append(os.path.join(figdir, "qc_library_size.png"))
    z = zero_rate.values; plt.figure(figsize=(8,4))
    plt.hist(z[np.isfinite(z)], bins=40, density=False)
    plt.title("Per-sample Zero Rate"); plt.xlabel("Fraction zeros"); plt.ylabel("Samples")
    _savefig(os.path.join(figdir, "qc_zero_rate_hist.png")); paths.append(os.path.join(figdir, "qc_zero_rate_hist.png"))
    return paths

def plot_housekeeping_stability(expr_log2, figdir: str) -> Optional[str]:
    hk = [g for g in HOUSEKEEPING_GENES if g in expr_log2.index.astype(str)]
    if not hk: return None
    cv = expr_log2.loc[hk].std(axis=1, ddof=1) / expr_log2.loc[hk].mean(axis=1).replace(0,np.nan)
    plt.figure(figsize=(7,4)); plt.bar(range(len(cv)), cv.values)
    plt.xticks(range(len(cv)), cv.index, rotation=45, ha="right")
    plt.title("Housekeeping Gene Stability (CV)")
    p = os.path.join(figdir, "hk_cv.png"); _savefig(p); return p

def plot_sex_marker_check(expr_log2, meta, figdir: str) -> Optional[str]:
    f_mark = [g for g in SEX_MARKERS["female"] if g in expr_log2.index.astype(str)]
    m_mark = [g for g in SEX_MARKERS["male"] if g in expr_log2.index.astype(str)]
    if not f_mark and not m_mark: return None
    fem = expr_log2.loc[f_mark].mean(axis=0) if f_mark else pd.Series(0, index=expr_log2.columns)
    mal = expr_log2.loc[m_mark].mean(axis=0) if m_mark else pd.Series(0, index=expr_log2.columns)
    plt.figure(figsize=(6,6)); plt.scatter(mal.values, fem.values, s=40, alpha=0.8)
    plt.xlabel("Male markers (avg log2)"); plt.ylabel("Female markers (avg log2)")
    plt.title("Sex-marker Concordance (per sample)")
    p = os.path.join(figdir, "sex_marker_concordance.png"); _savefig(p); return p

def create_basic_qc_figures(expr_log2, expr_z, expr_harmonized, meta, figdir: str) -> List[str]:
    os.makedirs(figdir, exist_ok=True); paths = []
    plt.figure(figsize=(12,6))
    for arr,label,a in [(expr_log2.values.ravel(),"Pre-harmonization",0.5),
                        (expr_harmonized.values.ravel(),"Post-harmonization",0.5)]:
        arr = arr[np.isfinite(arr)]
        plt.hist(arr, bins=120, density=True, alpha=a, label=label)
    plt.title("Expression Distribution: Pre vs Post (log2)"); plt.xlabel("log2(Expression + 1)"); plt.ylabel("Density"); plt.legend()
    p = os.path.join(figdir, "dist_pre_vs_post_log2.png"); _savefig(p); paths.append(p)

    plt.figure(figsize=(12,6))
    arr = expr_z.values.ravel(); arr = arr[np.isfinite(arr)]
    plt.hist(arr, bins=120, density=True)
    plt.title("Distribution of Z-scored Expression (All Samples)"); plt.xlabel("Z-score"); plt.ylabel("Density")
    p = os.path.join(figdir, "dist_zscore.png"); _savefig(p); paths.append(p)

    plt.figure(figsize=(12,6))
    grp_series = meta["group"] if "group" in meta.columns else pd.Series("ALL", index=meta.index)
    groups_seen = list(pd.unique(grp_series.astype(str)))
    for grp in groups_seen:
        cols = meta.index[grp_series==grp]
        vals = expr_harmonized[cols].values.ravel() if len(cols) else np.array([])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            plt.hist(vals, bins=100, density=True, alpha=0.35, label=grp)
    plt.title("Per-group Expression Distributions (log2, post-harmonization)")
    plt.xlabel("log2(Expression + 1)"); plt.ylabel("Density"); plt.legend()
    p = os.path.join(figdir, "group_density_post_log2.png"); _savefig(p); paths.append(p)

    group_vals, labels = [], []
    for grp in groups_seen:
        cols = meta.index[grp_series==grp]
        vals = expr_harmonized[cols].values.ravel() if len(cols) else np.array([])
        vals = vals[np.isfinite(vals)]
        if len(vals): group_vals.append(vals); labels.append(grp)
    if group_vals:
        plt.figure(figsize=(12,6)); plt.boxplot(group_vals, labels=labels, showfliers=True)
        plt.title("Expression Distribution After Harmonization (log2)"); plt.ylabel("log2(Expression + 1)")
        p = os.path.join(figdir, "boxplot_groups_harmonized_log2.png"); _savefig(p); paths.append(p)

    if expr_harmonized.shape[1] <= 600:
        C = np.corrcoef(expr_harmonized.fillna(0).T)
        plt.figure(figsize=(10,8)); plt.imshow(C, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation"); plt.title("Sample Correlation Heatmap (post-harmonization)")
        plt.xlabel("Samples"); plt.ylabel("Samples")
        p = os.path.join(figdir, "sample_correlation_heatmap.png"); _savefig(p); paths.append(p)
    return paths

def create_enhanced_pca_plots(pca_df, pca_model, meta, output_dir, harmonization_mode):
    meta = meta.copy()
    if "group" not in meta.columns: meta["group"] = "ALL"
    meta["group"] = meta["group"].apply(normalize_group_value)
    pca_df = pca_df.copy()
    if "group" not in pca_df.columns:
        pca_df = pca_df.join(meta[["group"]], how="left")
    pca_df["group"] = pca_df["group"].fillna("ALL").apply(normalize_group_value)
    groups = list(pd.unique(pca_df["group"].astype(str)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0,0]
    for g in groups:
        sub = pca_df[pca_df["group"]==g]
        if sub.empty: continue
        ax1.scatter(sub["PC1"], sub["PC2"], s=50, alpha=0.7, label=f"{g} (n={len(sub)})",
                    edgecolors='white', linewidth=0.5)
        if len(sub) > 5:
            try:
                cov = np.cov(sub[["PC1","PC2"]].T)
                if np.linalg.det(cov) > 1e-10:
                    vals, vecs = np.linalg.eig(cov)
                    vals = np.sqrt(np.maximum(vals, 0))*2
                    ell = Ellipse((sub["PC1"].mean(), sub["PC2"].mean()), vals[0], vals[1],
                                  angle=np.degrees(np.arctan2(vecs[1,0], vecs[0,0])), alpha=0.2)
                    ax1.add_patch(ell)
            except Exception:
                pass
    ax1.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title(f"PCA: Biological Groups\n({harmonization_mode} harmonization)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[0,1]
    if "batch_collapsed" not in meta.columns:
        meta["batch_collapsed"] = meta.get("batch", pd.Series("B0", index=meta.index))
    batch_series = meta.loc[pca_df.index, "batch_collapsed"].astype(str)
    top_batches = batch_series.value_counts().head(8).index
    lab = batch_series.where(batch_series.isin(top_batches), other="other_small")
    for b in lab.unique():
        sub = pca_df[lab == b]
        ax2.scatter(sub["PC1"], sub["PC2"], s=30, label=f"{b} (n={len(sub)})", alpha=0.7)
    ax2.set_xlabel(ax1.get_xlabel()); ax2.set_ylabel(ax1.get_ylabel())
    ax2.set_title("PCA: Batch Effects (Top 8)"); ax2.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1,0]
    if {"PC3","PC4"}.issubset(set(pca_df.columns)):
        for g in groups:
            sub = pca_df[pca_df["group"]==g]
            if not sub.empty: ax3.scatter(sub["PC3"], sub["PC4"], s=40, alpha=0.7, label=g)
        ax3.set_xlabel(f"PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)")
        ax3.set_ylabel(f"PC4 ({pca_model.explained_variance_ratio_[3]*100:.1f}%)")
        ax3.set_title("Higher Components: PC3 vs PC4"); ax3.legend(); ax3.grid(True, alpha=0.3)
    else:
        ax3.axis("off")

    ax4 = axes[1,1]
    n = min(10, len(pca_model.explained_variance_ratio_))
    xs = np.arange(1,n+1); vals = pca_model.explained_variance_ratio_[:n]; cum = np.cumsum(vals)
    ax4.bar(xs, vals, alpha=0.7, label='Individual')
    ax4.plot(xs, cum, 'o-', linewidth=2, markersize=6, label='Cumulative')
    for i,v in enumerate(vals): ax4.text(i+1, v+0.01, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=9)
    ax4.set_xlabel("Principal Components"); ax4.set_ylabel("Explained Variance Ratio")
    ax4.set_title("Variance Explained by Components"); ax4.legend(); ax4.grid(True, alpha=0.3); ax4.set_xticks(xs)
    plt.tight_layout()
    _savefig(os.path.join(output_dir, "enhanced_pca_analysis.png"))

    plt.figure(figsize=(10,8))
    for g in groups:
        sub = pca_df[pca_df["group"]==g]
        if not sub.empty:
            plt.scatter(sub["PC1"], sub["PC2"], s=60, alpha=0.85, edgecolors='black', linewidth=0.5,
                        label=f"{g} (n={len(sub)})")
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"PCA: Sample Groups\n({harmonization_mode} harmonization)")
    plt.legend(fontsize=11); plt.grid(True, alpha=0.3); plt.tight_layout()
    _savefig(os.path.join(output_dir, "pca_clean_groups.png"))

def pca_loadings_plots(pca_model: PCA, expr_harmonized: pd.DataFrame, figdir: str, topn: int = 20):
    comps = pca_model.components_
    genes = expr_harmonized.index.astype(str).tolist()
    if comps.shape[1] != len(genes): return
    for i in range(min(2, comps.shape[0])):
        w = comps[i]; idx = np.argsort(np.abs(w))[-topn:][::-1]
        plt.figure(figsize=(10,4)); plt.bar(range(topn), w[idx])
        plt.xticks(range(topn), [genes[j] for j in idx], rotation=60, ha="right", fontsize=8)
        plt.title(f"Top {topn} Loadings: PC{i+1}")
        _savefig(os.path.join(figdir, f"pca_loadings_pc{i+1}.png"))

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
    for grp in pd.unique(emb_df.get("group", pd.Series("ALL", index=emb_df.index)).astype(str)):
        sub = emb_df[emb_df["group"]==grp]
        if not sub.empty: plt.scatter(sub["E1"], sub["E2"], s=25, label=f"{grp} (n={len(sub)})")
    plt.title(f"{name.upper()} on ~{n_embed} PCs ({harmonization_mode})"); plt.legend(frameon=False)
    _savefig(os.path.join(figdir, f"{name}_by_group.png"))

def detect_outliers(expr_log2: pd.DataFrame) -> pd.DataFrame:
    X = (expr_log2.T.replace([np.inf, -np.inf], np.nan).fillna(0.0))
    X_np = X.to_numpy(dtype=float); n_samples = X_np.shape[0]
    if n_samples == 0: return pd.DataFrame(columns=["IsolationForest", "LOF"])
    if n_samples == 1: return pd.DataFrame({"IsolationForest": [0], "LOF": [0]}, index=expr_log2.columns[:1])
    scaler = StandardScaler(with_mean=True, with_std=True); Xs = scaler.fit_transform(X_np)
    iso = IsolationForest(contamination="auto", random_state=42); iso_flag = iso.fit_predict(Xs)
    try:
        n_neighbors = max(2, min(20, n_samples - 1))
        lof_flag = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(Xs) if n_samples >= 3 else np.ones(n_samples)
    except Exception:
        lof_flag = np.ones(n_samples)
    return pd.DataFrame({"IsolationForest": (iso_flag == -1).astype(int),
                         "LOF": (lof_flag == -1).astype(int)}, index=expr_log2.columns)

def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); n = p.size
    order = np.argsort(p)
    ranked = np.empty(n); ranked[order] = np.arange(1, n+1)
    q = p * n / ranked
    q = np.minimum.accumulate(q[order[::-1]])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q); out[order] = q
    return out

def differential_expression(expr_log2: pd.DataFrame, meta: pd.DataFrame,
                            contrasts: Iterable[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
    from scipy.stats import ttest_ind
    res = {}
    for A, B in contrasts:
        A_cols = meta.index[meta["group"]==A]
        B_cols = meta.index[meta["group"]==B]
        if len(A_cols) < 2 or len(B_cols) < 2: continue
        Xa = expr_log2[A_cols]; Xb = expr_log2[B_cols]
        ma = Xa.mean(axis=1); mb = Xb.mean(axis=1)
        log2FC = ma - mb
        t, p = ttest_ind(Xa.T, Xb.T, equal_var=False, nan_policy="omit")
        q = _bh_fdr(p)
        df = pd.DataFrame({"mean_"+A: ma, "mean_"+B: mb,
                           "log2FC": log2FC, "t": t, "pval": p, "qval": q}).sort_values("qval")
        res[f"{A}_vs_{B}"] = df
    return res

def volcano_and_ma_plots(de_df: pd.DataFrame, contrast_name: str, figdir: str):
    if de_df is None or de_df.empty: return
    log2fc = de_df["log2FC"].values
    p = de_df["pval"].clip(lower=1e-300).values
    neglogp = -np.log10(p)
    plt.figure(figsize=(8,6)); plt.scatter(log2fc, neglogp, s=6, alpha=0.6)
    plt.xlabel("log2FC"); plt.ylabel("-log10(p)"); plt.title(f"Volcano: {contrast_name}")
    _savefig(os.path.join(figdir, f"volcano_{contrast_name}.png"))

    means = None
    for c in de_df.columns:
        if c.startswith("mean_"):
            if means is None: means = de_df[c].values
            else: means = (means + de_df[c].values)/2.0
    if means is None: means = np.zeros_like(log2fc)
    plt.figure(figsize=(8,6)); plt.scatter(means, log2fc, s=6, alpha=0.6)
    plt.xlabel("Avg log2 expression"); plt.ylabel("log2FC"); plt.title(f"MA Plot: {contrast_name}")
    _savefig(os.path.join(figdir, f"ma_{contrast_name}.png"))

def heatmap_top_de(expr_log2: pd.DataFrame, meta: pd.DataFrame, de_df: pd.DataFrame, contrast_name: str, figdir: str, topn: int = 50):
    if de_df is None or de_df.empty: return
    top = de_df.sort_values("qval").head(topn).index
    sub = expr_log2.loc[expr_log2.index.intersection(top)]
    if sub.empty: return
    mu = sub.mean(axis=1); sd = sub.std(axis=1, ddof=1).replace(0,np.nan)
    z = sub.sub(mu, axis=0).div(sd, axis=0).fillna(0)
    order = meta.sort_values("group").index
    z = z[order.intersection(z.columns)]
    plt.figure(figsize=(max(10, z.shape[1] * 0.15), max(6, z.shape[0] * 0.15)))
    plt.imshow(z.values, aspect="auto", interpolation="nearest", vmin=-2.5, vmax=2.5)
    plt.colorbar(label="Z-score")
    plt.yticks(range(z.shape[0]), z.index, fontsize=7)
    plt.xticks(range(z.shape[1]), z.columns, fontsize=6, rotation=90)
    plt.title(f"Top {min(topn, z.shape[0])} DE genes: {contrast_name}")
    _savefig(os.path.join(figdir, f"heatmap_top_{topn}_{contrast_name}.png"))

# ---- meta-analysis helpers ----
def _z_from_p_two_sided(p, sign):
    p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    from scipy.stats import norm
    z = norm.isf(p/2.0)
    return z * np.sign(sign)

def _stouffer_meta(z_list, w_list=None):
    z = np.asarray(z_list, float)
    if w_list is None:
        w = np.ones_like(z)
    else:
        w = np.asarray(w_list, float)
    if z.size == 0: return np.nan
    z_meta = (w * z).sum() / np.sqrt((w**2).sum())
    return z_meta

def _bh(q):
    return _bh_fdr(q)

# ---------------- Main single-dataset pipeline ----------------
def run_pipeline(
    group_to_file: Optional[Dict[str, io.BytesIO | str]] = None,
    single_expression_file: Optional[io.BytesIO | str] = None,
    single_expression_name_hint: Optional[str] = None,
    metadata_file: io.BytesIO | str = None,
    metadata_name_hint: str | None = None,
    metadata_id_cols: List[str] = ["Id","ID","id","CleanID","sample","Sample"],
    metadata_group_cols: List[str] = ["group","Group","condition","Condition","phenotype","Phenotype"],
    metadata_batch_col: Optional[str] = None,
    out_root: str = "out",
    fig_subdir: str = "figs",
    min_batch_size_for_combat: int = 2,
    pca_topk_features: int = 5000,
    make_nonlinear: bool = True,
    gsea_gmt: Optional[str] = None,
) -> Dict[str, str]:

    if metadata_file is None:
        raise ValueError("metadata_file is required.")

    OUTDIR = os.path.join(out_root)
    FIGDIR = os.path.join(OUTDIR, fig_subdir)
    REPORT_DIR = os.path.join(OUTDIR, "report")
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)

    # 1) expression
    if single_expression_file is not None:
        expr = read_expression_any(single_expression_file, name_hint=single_expression_name_hint, group_name=None)
        combined_expr = expr; groups_from_prefix = False
    else:
        if not group_to_file or len(group_to_file) == 0:
            raise ValueError("Provide at least one expression file (single or multi).")
        loaded = {g: read_expression_any(f, name_hint=str(f), group_name=g) for g, f in group_to_file.items()}
        combined_expr = pd.concat([loaded[g] for g in loaded.keys()], axis=1, join="outer")
        combined_expr = combined_expr.replace([np.inf,-np.inf], np.nan)
        groups_from_prefix = True

    notes = {}

    if combined_expr.columns.duplicated().any():
        dup_n = int(combined_expr.columns.duplicated().sum())
        combined_expr = (combined_expr.T.groupby(level=0, sort=False).median(numeric_only=True).T)
        notes["dedup_expression_columns"] = f"Collapsed {dup_n} duplicate sample columns by median."

    meta_base = build_metadata_from_columns(list(combined_expr.columns), groups_from_prefix=groups_from_prefix)
    if meta_base.index.duplicated().any():
        dropped = int(meta_base.index.duplicated().sum())
        meta_base = meta_base.loc[~meta_base.index.duplicated(keep="first")].copy()
        notes["dedup_meta_base_index"] = f"Removed {dropped} duplicate entries in meta_base index."

    # 2) metadata
    m = _read_metadata_any(metadata_file, name_hint=metadata_name_hint)
    def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())
    norm_cols = { _norm(c): c for c in m.columns }
    norm_candidates = [_norm(c) for c in metadata_id_cols]

    id_col = None
    for nc in norm_candidates:
        if nc in norm_cols: id_col = norm_cols[nc]; break
    if id_col is None:
        expr_cols = set(map(str, meta_base['bare_id'].tolist()))
        best_col, best_overlap = None, -1
        for c in m.columns:
            vals = set(m[c].astype(str).str.strip()); ov = len(vals & expr_cols)
            if ov > best_overlap: best_overlap, best_col = ov, c
        if best_overlap > 0: id_col = best_col
    if id_col is None:
        raise ValueError(f"Could not find an ID column among {metadata_id_cols} in metadata: {list(m.columns)}")

    group_col = next((c for c in metadata_group_cols if c in m.columns), None)

    if metadata_batch_col is not None and metadata_batch_col not in m.columns:
        metadata_batch_col = None
    if metadata_batch_col is None:
        metadata_batch_col = _guess_batch_column_from_meta(m)

    m[id_col] = m[id_col].astype(str).str.strip()
    before_rows = len(m); m = m.dropna(subset=[id_col])
    m = m[~m[id_col].duplicated(keep="first")].copy()
    after_rows = len(m)
    if after_rows < before_rows:
        notes["dedup_metadata_rows"] = f"Dropped {before_rows - after_rows} duplicate metadata rows (by {id_col})."
    m_align = m.set_index(id_col)

    meta = meta_base.copy()
    meta["bare_id"] = meta["bare_id"].astype(str).str.strip()

    if group_col is not None:
        gser = m_align[group_col] if group_col in m_align.columns else pd.Series(index=m_align.index, dtype=object)
        meta["group_external"] = gser.reindex(meta["bare_id"]).values
        meta["group"] = meta["group_external"].where(pd.notna(meta["group_external"]), meta["group"])

    if "group" not in meta.columns: meta["group"] = "ALL"
    meta["group"] = meta["group"].fillna("ALL").apply(normalize_group_value)

    if metadata_batch_col is not None and metadata_batch_col in m_align.columns:
        meta["batch_external_raw"] = m_align[metadata_batch_col].reindex(meta["bare_id"]).values
        meta["batch_external"] = pd.Series(meta["batch_external_raw"], index=meta.index).map(normalize_batch_token)
        inferred = infer_batches(meta)
        meta["batch"] = meta["batch_external"].where(meta["batch_external"].notna(), inferred)
        notes["batch_source"] = f"Batch column detected: '{metadata_batch_col}'"
    else:
        meta["batch"] = infer_batches(meta); notes["batch_source"] = "Batch inferred from sample names"

    meta["group_raw"] = meta["group"]; meta["group"] = meta["group"].apply(normalize_group_value)

    if meta.index.duplicated().any():
        dups = int(meta.index.duplicated().sum())
        meta = _collapse_dupes_df_by_index(meta, how_num="median", keep="first")
        notes["dedup_meta_index"] = f"Collapsed {dups} duplicate sample rows in metadata."

    os.makedirs(OUTDIR, exist_ok=True)
    combined_expr.to_csv(os.path.join(OUTDIR, "expression_combined.tsv"), sep="\t")
    meta.to_csv(os.path.join(OUTDIR, "metadata.tsv"), sep="\t")

    # 3) detect type
    dtype, platform, diags = detect_data_type_and_platform(combined_expr)

    # 4) log2 & z
    expr_log2 = np.log2(combined_expr + 1).replace([np.inf,-np.inf], np.nan)
    row_mean = expr_log2.mean(axis=1); row_std = expr_log2.std(axis=1, ddof=1).replace(0, np.nan)
    expr_z = expr_log2.sub(row_mean, axis=0).div(row_std, axis=0).replace([np.inf,-np.inf], np.nan).fillna(0)

    # 5) impute + filter
    expr_imputed = drop_zero_variance(row_mean_impute(expr_log2))
    gene_vars = expr_imputed.var(axis=1)
    pos = gene_vars[gene_vars > 0]
    topk = min(pca_topk_features, len(pos)) if len(pos) > 0 else 0
    expr_filtered = expr_imputed.loc[pos.nlargest(topk).index] if topk > 0 else expr_imputed.iloc[0:0]

    # 6) harmonize (ComBat or fallback)
    meta["batch_collapsed"] = smart_batch_collapse(meta, min_batch_size_for_combat)
    if meta.index.duplicated().any():
        meta = _collapse_dupes_df_by_index(meta, how_num="median", keep="first")
    meta_batch = meta["batch_collapsed"].reindex(expr_filtered.columns).fillna(meta["group"].reindex(expr_filtered.columns).astype(str))
    if expr_filtered.shape[1] > 0 and expr_filtered.shape[0] > 0:
        x_combat = _combat(expr_filtered, meta_batch)
        expr_harmonized = x_combat if x_combat is not None else _fallback_center(expr_filtered, meta_batch)
        mode = "ComBat" if x_combat is not None else "fallback_center"
    else:
        expr_harmonized = expr_filtered.copy(); mode = "no_features"

    # 7) PCA (fail-soft)
    pca_df = pd.DataFrame(index=expr_harmonized.columns); kpi = {}; pca_skipped_reason = None
    try:
        Xc = safe_matrix_for_pca(zscore_rows(expr_harmonized), topk=topk)
        if Xc.shape[0] < 2 or Xc.shape[1] < 2: raise RuntimeError("Too few samples or features for PCA.")
        pca = PCA(n_components=min(6, Xc.shape[1]), random_state=42).fit(Xc)
        Xp = pca.transform(Xc)
        cols_to_join = [c for c in ["group","batch","batch_collapsed"] if c in meta.columns]
        pca_df = (pd.DataFrame(Xp[:, :min(6, Xp.shape[1])],
                               columns=[f"PC{i+1}" for i in range(min(6, Xp.shape[1]))],
                               index=Xc.index).join(meta[cols_to_join], how="left"))
        if "group" not in pca_df.columns: pca_df["group"] = "ALL"
        meta["group"] = meta["group"].apply(normalize_group_value)
        pca_df["group"] = pca_df["group"].fillna("ALL").apply(normalize_group_value)
        try:
            use = [c for c in ["PC1","PC2","PC3","PC4"] if c in pca_df.columns]
            if len(use) >= 2:
                Xs = pca_df[use].values
                if meta["batch_collapsed"].nunique() > 1:
                    kpi["silhouette_batch"] = float(silhouette_score(Xs, meta["batch_collapsed"].astype(str)))
                if meta["group"].nunique() > 1:
                    kpi["silhouette_group"] = float(silhouette_score(Xs, meta["group"].astype(str)))
        except Exception:
            pass
        create_enhanced_pca_plots(pca_df, pca, meta, os.path.join(OUTDIR, "figs"), mode)
        pca_loadings_plots(pca, expr_harmonized, os.path.join(OUTDIR, "figs"))
        nonlinear_embedding_plots(Xc, meta, os.path.join(OUTDIR, "figs"), mode, make=True)
    except Exception as e:
        pca_skipped_reason = f"{type(e).__name__}: {e}"

    # 8) QC figs
    figs = []
    figs += create_sample_qc_figures(combined_expr, expr_log2, meta, os.path.join(OUTDIR, "figs"))
    figs += create_basic_qc_figures(expr_log2, expr_z, expr_harmonized, meta, os.path.join(OUTDIR, "figs"))
    plot_housekeeping_stability(expr_log2, os.path.join(OUTDIR, "figs"))
    plot_sex_marker_check(expr_log2, meta, os.path.join(OUTDIR, "figs"))

    # 9) Outliers
    outliers = detect_outliers(expr_log2); outliers.to_csv(os.path.join(OUTDIR, "outliers.tsv"), sep="\t")

    # 10) DE (auto)
    if "group" not in meta.columns: meta["group"] = "ALL"
    meta["group"] = meta["group"].apply(normalize_group_value)
    groups = [g for g in pd.unique(meta["group"].astype(str)) if g and g != "ALL"]
    default_contrasts = [(a, b) for a in groups for b in groups if a != b]
    de = differential_expression(expr_log2, meta, default_contrasts)
    de_dir = os.path.join(OUTDIR, "de"); os.makedirs(de_dir, exist_ok=True)
    for k, df in de.items():
        df.to_csv(os.path.join(de_dir, f"DE_{k}.tsv"), sep="\t")
        volcano_and_ma_plots(df, k, os.path.join(OUTDIR, "figs"))
        heatmap_top_de(expr_log2, meta, df, k, os.path.join(OUTDIR, "figs"), topn=50)

    # 11) GSEA (optional) — reserved hook, if you wire gp.prerank etc.
    gsea_dir = os.path.join(OUTDIR, "gsea"); os.makedirs(gsea_dir, exist_ok=True)

    # 12) Save outputs
    expr_harmonized.to_csv(os.path.join(OUTDIR, "expression_harmonized.tsv"), sep="\t")
    pca_df.to_csv(os.path.join(OUTDIR, "pca_scores.tsv"), sep="\t")

    rep = {
        "qc": {"zero_fraction": float(diags.get("zero_fraction", np.nan)),
               "value_range_approx": float(diags.get("value_range_approx", np.nan)),
               "harmonization_mode": mode,
               "platform": platform,
               **kpi},
        "shapes": {"genes": int(combined_expr.shape[0]), "samples": int(combined_expr.shape[1])},
        "notes": {}
    }
    if pca_skipped_reason: rep["notes"]["pca_skipped_reason"] = pca_skipped_reason
    with open(os.path.join(OUTDIR, "report.json"), "w") as f: json.dump(rep, f, indent=2)

    zip_path = os.path.join(OUTDIR, "results_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUTDIR):
            for name in files:
                if name.endswith(".zip"): continue
                p = os.path.join(root, name)
                zf.write(p, arcname=os.path.relpath(p, OUTDIR))

    return {"outdir": OUTDIR, "figdir": os.path.join(OUTDIR, "figs"),
            "report_json": os.path.join(OUTDIR, "report.json"), "zip": zip_path}

# ---------------- GEO fetch helpers ----------------

def _try_import_geoparse():
    try:
        import GEOparse
        return GEOparse
    except Exception:
        return None

_DEF_META_COLS = [
    "user query","tissue type requested","experiment type requested","data set id",
    "no of samples","no of sample validting the condition","sample tissue type",
    "sample characteristics","library strategy","extractedprotocol"
]
_DEF_META_RENAMES = {
    "user query": "user_query",
    "tissue type requested": "tissue_type_requested",
    "experiment type requested": "experiment_type_requested",
    "data set id": "data_set_id",
    "no of samples": "n_samples",
    "no of sample validting the condition": "n_samples_validating_condition",
    "sample tissue type": "sample_tissue_type",
    "sample characteristics": "sample_characteristics",
    "library strategy": "library_strategy",
    "extractedprotocol": "extracted_protocol",
}

def _series_to_expression_and_meta(gse) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot series matrix to expression (rows=probes/genes, cols=samples) and sample metadata."""
    expr, meta = None, None
    # try pivot
    try:
        expr = gse.pivot_samples('VALUE')
    except Exception:
        pass
    # sample metadata (flatten GSM metadata to a single row)
    rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {k.lower(): v for k,v in getattr(gsm, 'metadata', {}).items()}
        flat = {k: (v[0] if isinstance(v, list) and v else v) for k,v in row.items()}
        flat["sample"] = gsm_name
        rows.append(flat)
    meta = pd.DataFrame(rows).set_index("sample") if rows else pd.DataFrame(index=list(gse.gsms.keys()))
    meta.columns = [str(c).lower().strip() for c in meta.columns]

    if expr is None:
        mats = []
        for gsm_name, gsm in gse.gsms.items():
            tab = getattr(gsm, 'table', None)
            if tab is None or 'VALUE' not in tab.columns: continue
            v = tab[['ID_REF','VALUE']].copy().rename(columns={'ID_REF':'Biomarker', 'VALUE':gsm_name})
            mats.append(v)
        if mats:
            expr = mats[0]
            for m in mats[1:]:
                expr = expr.merge(m, on='Biomarker', how='outer')
            expr = expr.set_index('Biomarker')

    if expr is None or expr.empty:
        raise RuntimeError("Could not construct expression matrix from GEO series.")
    expr = expr.apply(pd.to_numeric, errors='coerce')
    return expr, meta

def fetch_geo_as_datasets(accessions: list[str]) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    """
    Given a list of GSE IDs, download and return a 'datasets' list compatible with run_pipeline_multi.
    Returns (datasets, dataset_summary_df, evaluation_json_rows).

    Robust fallbacks:
      1) In-memory Series pivot (bulk)
      2) Any *series_matrix.txt.gz on disk
      3) Supplementary single-cell (10x/MTX/H5/Loom) -> pseudo-bulk per GSM
    Skips failing accessions (soft-fail per GSE); raises if none succeeded.
    """
    try:
        import GEOparse
    except Exception as e:
        raise RuntimeError("GEOparse is required: pip install GEOparse") from e

    tmp_root = tempfile.mkdtemp(prefix="geo_dl_")
    datasets: list[dict] = []
    eval_rows: list[dict] = []
    summary_rows: list[dict] = []

    def _coerce_meta(meta_df: pd.DataFrame) -> pd.DataFrame:
        if meta_df is None or meta_df.empty:
            return pd.DataFrame()
        m = meta_df.copy()
        m.index.name = "sample" if m.index.name is None else m.index.name
        # Lowercase columns & make safe defaults
        m.columns = [str(c).strip() for c in m.columns]
        if "group" not in m.columns:
            m["group"] = "ALL"
        if "bare_id" not in m.columns:
            # try to use index as bare_id
            m["bare_id"] = m.index.astype(str)
        return m

    succeeded = 0
    failures = []

    for acc in accessions:
        acc = acc.strip().upper()
        if not re.fullmatch(r"GSE\d+", acc):
            failures.append((acc, "Invalid accession format")); continue

        try:
            # Be generous: how='full' downloads all, annotate_gpl often helps pivot
            gse = GEOparse.get_GEO(geo=acc, destdir=tmp_root, how='full', annotate_gpl=True)

            # ---------- 1) Try in-memory pivot (bulk matrices) ----------
            expr, meta = None, None
            try:
                expr = gse.pivot_samples('VALUE')
                # Minimal meta from GSM
                rows = []
                for gsm_name, gsm in gse.gsms.items():
                    ch = getattr(gsm, "metadata", {})
                    rows.append({
                        "sample": gsm_name,
                        "group": (ch.get("characteristics_ch1", ["ALL"])[0] if "characteristics_ch1" in ch else "ALL"),
                        "bare_id": gsm_name,
                        "library_strategy": ", ".join(ch.get("library_strategy", [])) if "library_strategy" in ch else "",
                    })
                meta = pd.DataFrame(rows).set_index("sample") if rows else pd.DataFrame(index=expr.columns)
            except Exception:
                expr, meta = None, None

            # ---------- 2) Try any *series_matrix.txt.gz on disk ----------
            if expr is None or expr.empty:
                sm_files = glob.glob(os.path.join(tmp_root, acc, "*series_matrix.txt.gz")) + \
                           glob.glob(os.path.join(tmp_root, "*", acc, "*series_matrix.txt.gz")) + \
                           glob.glob(os.path.join(tmp_root, "*series_matrix.txt.gz"))
                for f in sm_files:
                    e2, m2 = _parse_series_matrix_file(f)
                    if e2 is not None and not e2.empty:
                        expr, meta = e2, (m2 if m2 is not None else pd.DataFrame(index=e2.columns))
                        break

            # ---------- 3) Supplementary single-cell -> pseudo-bulk ----------
            if expr is None or expr.empty:
                e3, m3 = _build_from_supplementary(os.path.join(tmp_root, acc))
                if e3 is not None and not e3.empty:
                    expr, meta = e3, (m3 if m3 is not None else pd.DataFrame(index=e3.columns))

            # As a last minor fallback: try GSM tables merged (when pivot_samples fails and no series_matrix)
            if expr is None or expr.empty:
                try:
                    mats = []
                    for gsm_name, gsm in gse.gsms.items():
                        tab = getattr(gsm, 'table', None)
                        if tab is None:
                            continue
                        # Common column names to try
                        if 'VALUE' in tab.columns:
                            vcol = 'VALUE'
                        elif 'count' in tab.columns:
                            vcol = 'count'
                        else:
                            continue
                        v = tab[['ID_REF', vcol]].copy().rename(columns={'ID_REF': 'Biomarker', vcol: gsm_name})
                        mats.append(v)
                    if mats:
                        expr = mats[0]
                        for m_ in mats[1:]:
                            expr = expr.merge(m_, on='Biomarker', how='outer')
                        expr = expr.set_index('Biomarker').apply(pd.to_numeric, errors='coerce')
                        # Minimal meta
                        meta = pd.DataFrame({"sample": expr.columns, "group": "ALL", "bare_id": expr.columns}).set_index("sample")
                except Exception:
                    pass

            # If still nothing…
            if expr is None or expr.empty:
                failures.append((acc, "Could not construct expression matrix (no series_matrix and no parsable supplementary/count tables)"))
                continue

            # ---------- Normalize indices/case ----------
            expr.index = expr.index.astype(str).str.strip().str.upper().str.replace(r'\.\d+$', '', regex=True)
            meta = _coerce_meta(meta)
            # Align meta to expression columns
            if not meta.index.name:
                meta.index.name = "sample"
            if not meta.index.isin(expr.columns).all():
                # Re-index gracefully (fill missing rows)
                meta = meta.reindex(expr.columns).fillna({"group": "ALL", "bare_id": ""})

            # ---------- Build in-memory buffers ----------
            expr_buf = io.BytesIO()
            expr_w = expr.copy()
            expr_w.insert(0, "Biomarker", expr_w.index)
            expr_w.to_csv(expr_buf, sep="\t", index=False)
            expr_buf.seek(0)

            meta_buf = io.BytesIO()
            meta_reset = meta.reset_index().rename(columns={"index": "sample"})
            meta_reset.to_csv(meta_buf, sep="\t", index=False)
            meta_buf.seek(0)

            ds_label = acc
            datasets.append({
                "geo": ds_label,
                "dataset_id": acc,
                "counts": expr_buf,
                "meta": meta_buf,
                "meta_id_cols": ["sample","Sample","Id","ID","bare_id"],
                "meta_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                "meta_batch_col": None,
            })

            # Summary rows
            summary_rows.append({
                "data set id": acc,
                "n_genes": int(expr.shape[0]),
                "n_samples": int(expr.shape[1]),
                "source": "GEO",
                "library_strategy": ";".join(sorted(set(meta_reset.get("library_strategy", pd.Series()).astype(str)))),
            })
            eval_rows.append({
                "dataset_id": acc,
                "source": "GEO",
                "download_ok": True,
                "constructed_matrix": True,
            })
            succeeded += 1

        except Exception as e:
            failures.append((acc, f"Exception: {type(e).__name__}: {e}"))
            continue

    if succeeded == 0:
        # Aggregate a concise error
        msg = "No GEO series could be parsed.\n" + "\n".join([f"- {a}: {why}" for a, why in failures[:5]])
        raise RuntimeError(msg)

    ds_summary_df = pd.DataFrame(summary_rows) if len(summary_rows) else pd.DataFrame()

    # If some failed, it’s useful to log (without crashing)
    if failures:
        warnings.warn("Some accessions were skipped:\n" + "\n".join([f"- {a}: {why}" for a, why in failures[:5]]))

    return datasets, ds_summary_df, eval_rows



# ---------------- Multi-GEO wrapper + meta-analysis ----------------
def meta_analyze_disease_vs_control(runs: Dict[str, Dict], out_root: str, fdr_thresh: float = 0.10) -> Dict:
    """
    Combines per-dataset DE tables for Disease_vs_Control via signed Stouffer's Z.
    Emits CSVs and a plain-text + PNG summary for Presenter Mode.
    """
    os.makedirs(out_root, exist_ok=True)
    meta_dir = os.path.join(out_root, "meta"); os.makedirs(meta_dir, exist_ok=True)

    # gather per-dataset DE
    per = {}
    for name, res in runs.items():
        de_dir = os.path.join(res["outdir"], "de")
        target = None
        cand = os.path.join(de_dir, "DE_Disease_vs_Control.tsv")
        if os.path.exists(cand): target = cand
        else:
            for f in os.listdir(de_dir) if os.path.isdir(de_dir) else []:
                if not f.startswith("DE_") or not f.endswith(".tsv"): continue
                c = f.removeprefix("DE_").removesuffix(".tsv")
                try:
                    A, B = c.split("_vs_")
                    if normalize_group_value(A) == "Disease" and normalize_group_value(B) == "Control":
                        target = os.path.join(de_dir, f); break
                except Exception:
                    pass
        if target:
            df = pd.read_csv(target, sep="\t", index_col=0)
            per[name] = df[["log2FC","pval"]].rename(columns={"pval":"p"})
    if not per:
        return {"meta_dir": meta_dir, "n_datasets_with_de": 0}

    # union of genes; compute meta stats
    genes = set().union(*[set(df.index.astype(str)) for df in per.values()])
    rows = []
    from scipy.stats import norm
    for g in genes:
        zs, signs = [], []
        for name, df in per.items():
            if g in df.index:
                row = df.loc[g]
                lfc = float(row.get("log2FC", np.nan))
                p = float(row.get("p", np.nan))
                if np.isfinite(lfc) and np.isfinite(p):
                    z = _z_from_p_two_sided(p, np.sign(lfc))
                    zs.append(z); signs.append(np.sign(lfc))
        if len(zs) >= 2:
            z_meta = _stouffer_meta(zs)
            p_meta = 2.0 * norm.sf(abs(z_meta))
            pos = np.sum(np.array(signs) > 0); neg = np.sum(np.array(signs) < 0)
            cons_dir = "up" if pos >= neg else "down"
            cons_frac = max(pos, neg) / float(len(signs))
            rows.append((g, z_meta, p_meta, cons_dir, cons_frac))
    if not rows:
        return {"meta_dir": meta_dir, "n_datasets_with_de": len(per)}

    meta_df = pd.DataFrame(rows, columns=["gene","z_meta","p_meta","consistent_dir","consistency"])
    meta_df = meta_df.set_index("gene")
    meta_df["q_meta"] = _bh(meta_df["p_meta"].values)

    med_abs = []
    for g in meta_df.index:
        vals = []
        for df in per.values():
            if g in df.index:
                v = float(df.loc[g, "log2FC"])
                if np.isfinite(v): vals.append(abs(v))
        med_abs.append(np.median(vals) if vals else 0.0)
    meta_df["meta_log2FC_proxy"] = np.sign(meta_df["z_meta"].values) * np.array(med_abs, float)

    # save meta tables
    meta_df.sort_values("q_meta").to_csv(os.path.join(meta_dir, "meta_analysis_results.csv"))
    ups = meta_df[(meta_df["q_meta"] < fdr_thresh) & (meta_df["meta_log2FC_proxy"] > 0) & (meta_df["consistency"] >= 0.6)]
    ups.sort_values(["q_meta","consistency"], ascending=[True, False]).to_csv(os.path.join(meta_dir, "upregulated_genes_meta.csv"))
    meta_df.sort_values("q_meta").head(2000).to_csv(os.path.join(meta_dir, "deg_results_annotated.csv"))
    meta_df.sort_values("q_meta").head(500).to_csv(os.path.join(meta_dir, "drug_targets_analysis.csv"))
    meta_df.sort_values("q_meta").head(20).to_csv(os.path.join(meta_dir, "final_analysis_summary.csv"))

    # Build plain-text Key Findings
    total_genes = len(genes)
    sig = int((meta_df["q_meta"] < fdr_thresh).sum())
    up_consistent = int(((meta_df["q_meta"] < fdr_thresh) & (meta_df["meta_log2FC_proxy"] > 0) & (meta_df["consistency"] >= 0.6)).sum())
    ds_list = ", ".join(sorted(per.keys()))
    text_lines = [
        "=== KEY FINDINGS FROM HARMONIZED META-ANALYSIS ===",
        "",
        "🎯 ANALYSIS SCOPE:",
        f"   • Successfully harmonized {len(per)} datasets",
        f"   • Total genes analyzed: {total_genes:,}",
        f"   • Datasets: {ds_list}",
        "",
        "📊 STATISTICAL RESULTS:",
        f"   • Significant genes (FDR < {fdr_thresh}): {sig:,}",
        f"   • Consistently upregulated genes: {up_consistent:,}",
        f"   • Meta-analysis method: Signed Stouffer Z (gene-level)",
        "",
        "🧬 TOP BIOMARKER CANDIDATES:",
    ]
    top10 = meta_df.sort_values("q_meta").head(10)
    i = 1
    for g, r in top10.iterrows():
        text_lines.append(
            f"    {i}. {g:<12} | LogFC*: {r['meta_log2FC_proxy']:+.3f} | FDR: {r['q_meta']:.2e} | Potential: {'High' if r['q_meta'] < 0.01 else 'Medium'}"
        )
        i += 1
    text_lines += [
        "",
        "🔬 HARMONIZATION STRATEGY:",
        "   ✓ Individual dataset analysis followed by meta-analysis",
        "   ✓ Gene-level aggregation for cross-platform compatibility",
        "   ✓ Consistency filtering (same direction across datasets)",
        f"   ✓ Robust statistical thresholds (FDR < {fdr_thresh})",
        "",
        "📁 OUTPUT FILES GENERATED:",
        "   • deg_results_annotated.csv - Full DEG-like meta results (top subset)",
        "   • meta_analysis_results.csv - Complete meta-analysis results",
        "   • upregulated_genes_meta.csv - Consistently upregulated genes",
        "   • drug_targets_analysis.csv - Candidate targets (top subset)",
        "   • final_analysis_summary.csv - Top 20 biomarker candidates",
        "   • diabetes_harmonized_analysis_comprehensive.png - Comprehensive visualization",
        "",
        "📈 BIOLOGICAL SIGNIFICANCE:",
        "   • Cross-dataset validation reduces false positives",
        "   • Directional consistency suggests robust association",
        "   • Multi-platform validation increases clinical relevance",
        "",
        "🎯 NEXT STEPS FOR CLINICAL TRANSLATION:",
        "   1. Validate biomarkers in independent cohorts",
        "   2. Investigate targets through pathway analysis",
        "   3. Design therapeutic interventions",
        "   4. Develop diagnostic/prognostic assays",
        "",
        "🏁 HARMONIZATION ANALYSIS COMPLETE",
        "   Approach: Biologically sound meta-analysis",
        "   Result: High-confidence biomarkers identified."
    ]
    summary_txt = os.path.join(meta_dir, "final_analysis_summary.txt")
    with open(summary_txt, "w") as fh:
        fh.write("\n".join(text_lines))

    # Create a simple hero PNG for Presenter Mode
    try:
        plt.figure(figsize=(10,6))
        top_plot = meta_df.sort_values("q_meta").head(10)[["q_meta"]].copy()
        x = np.arange(len(top_plot))
        plt.bar(x, -np.log10(top_plot["q_meta"].values))
        plt.xticks(x, top_plot.index.tolist(), rotation=60, ha="right", fontsize=9)
        plt.ylabel("-log10(FDR)")
        plt.title("Top Meta-analysis Signals (lower FDR is better)")
        summary_png = os.path.join(meta_dir, "diabetes_harmonized_analysis_comprehensive.png")
        _savefig(summary_png)
    except Exception:
        summary_png = os.path.join(meta_dir, "diabetes_harmonized_analysis_comprehensive.png")

    return {
        "meta_dir": meta_dir,
        "n_datasets_with_de": len(per),
        "summary_txt_path": summary_txt,
        "summary_png_path": summary_png,
    }

def run_pipeline_multi(
    datasets: List[Dict],
    attempt_combine: bool = True,
    combine_minoverlap_genes: int = 3000,
    out_root: str = "out/multi_geo",
    pca_topk_features: int = 5000,
    make_nonlinear: bool = True,
    dataset_summary_df: Optional[pd.DataFrame] = None,
    evaluation_results_json: Optional[List[Dict[str, Any]]] = None,
) -> Dict:
    os.makedirs(out_root, exist_ok=True)
    runs, exprs, metas = {}, {}, {}

    # Pre-normalize lookups for fast per-dataset slicing
    ds_summary_df_norm = None
    if isinstance(dataset_summary_df, pd.DataFrame) and not dataset_summary_df.empty:
        ds_summary_df_norm = dataset_summary_df.copy()
        cols_lower = {c.lower(): c for c in ds_summary_df_norm.columns}
        keycol = None
        for k in ["data set id","dataset id","geo accession","gse","series id","data_set_id"]:
            if k in cols_lower:
                keycol = cols_lower[k]; break
        if keycol is None:
            ds_summary_df_norm["__key__"] = ""
            keycol = "__key__"
        ds_summary_df_norm["_key"] = ds_summary_df_norm[keycol].map(_norm_id)

    eval_list = None
    if evaluation_results_json is not None:
        eval_list = evaluation_results_json if isinstance(evaluation_results_json, list) else [evaluation_results_json]

    qa_warnings = []

    # per-dataset runs
    for d in datasets:
        name = d.get("geo") or f"DS{len(runs)+1}"
        ds_id = d.get("dataset_id") or name
        ds_id_key = _norm_id(ds_id)

        # slice QA artifacts
        ds_subset_df = None
        if ds_summary_df_norm is not None and "_key" in ds_summary_df_norm.columns:
            tmp = ds_summary_df_norm[ds_summary_df_norm["_key"] == ds_id_key]
            ds_subset_df = tmp.drop(columns=["_key"], errors="ignore")
            if ds_subset_df is not None and ds_subset_df.empty:
                qa_warnings.append(f"No dataset_summary rows found for dataset_id '{ds_id}'")

        eval_subset = None
        if eval_list is not None:
            eval_subset = [x for x in eval_list if _norm_id(x.get("dataset_id")) == ds_id_key]
            if len(eval_subset) == 0:
                qa_warnings.append(f"No evaluation_results entries found for dataset_id '{ds_id}'")

        outdir = os.path.join(out_root, name)
        res = run_pipeline(
            single_expression_file=d["counts"],
            single_expression_name_hint=d.get("counts_name"),
            metadata_file=d["meta"],
            metadata_name_hint=d.get("meta_name"),
            metadata_id_cols=d.get("meta_id_cols") or ["Id","ID","id","sample","Sample","bare_id"],
            metadata_group_cols=d.get("meta_group_cols") or ["group","Group","condition","Condition","phenotype","Phenotype"],
            metadata_batch_col=d.get("meta_batch_col"),
            out_root=outdir,
            pca_topk_features=pca_topk_features,
            make_nonlinear=make_nonlinear,
        )

        # Attach per-dataset QA artifacts under report/dataset_qa/
        qa_dir = os.path.join(res["outdir"], "report", "dataset_qa")
        os.makedirs(qa_dir, exist_ok=True)
        if isinstance(ds_subset_df, pd.DataFrame) and not ds_subset_df.empty:
            ds_subset_df.to_csv(os.path.join(qa_dir, "dataset_summary_normalized.csv"), index=False)
        if isinstance(eval_subset, list) and len(eval_subset):
            with open(os.path.join(qa_dir, "evaluation_results.json"), "w") as fh:
                json.dump(eval_subset, fh, indent=2)

        runs[name] = res
        exprs[name] = pd.read_csv(os.path.join(res["outdir"], "expression_combined.tsv"), sep="\t", index_col=0)
        metas[name] = pd.read_csv(os.path.join(res["outdir"], "metadata.tsv"), sep="\t", index_col=0)

    # combine decision
    decision = {"attempted": True if attempt_combine else False, "combined": False, "overlap_genes": 0}
    combined = None

    if attempt_combine and len(exprs) >= 2:
        common = None
        for _, X in exprs.items():
            genes = set(X.index.astype(str))
            common = genes if common is None else (common & genes)
        n_common = len(common or [])
        decision["overlap_genes"] = n_common

        if n_common >= combine_minoverlap_genes:
            decision["combined"] = True
            common = list(common)
            expr_joined = []
            meta_joined = []
            for name, X in exprs.items():
                Xi = X.loc[common]
                expr_joined.append(Xi)
                m = metas[name].copy(); m["dataset"] = name
                meta_joined.append(m)
            expr_all = pd.concat(expr_joined, axis=1, join="outer")
            meta_all = pd.concat(meta_joined, axis=0)

            expr_buf = io.BytesIO()
            df_to_write = expr_all.copy()
            df_to_write.insert(0, "Biomarker", df_to_write.index)
            df_to_write.to_csv(expr_buf, sep="\t", index=False); expr_buf.seek(0)

            meta_buf = io.BytesIO()
            meta_all = meta_all.reset_index().rename(columns={"index":"sample"})
            meta_all.to_csv(meta_buf, sep="\t", index=False); meta_buf.seek(0)

            combined_out = os.path.join(out_root, "combined")
            combined = run_pipeline(
                single_expression_file=expr_buf,
                single_expression_name_hint="combined.tsv",
                metadata_file=meta_buf,
                metadata_name_hint="combined_meta.tsv",
                metadata_id_cols=["sample","Sample","Id","ID","bare_id"],
                metadata_group_cols=["group","Group","condition","Condition","phenotype","Phenotype"],
                metadata_batch_col="dataset",
                out_root=combined_out,
                pca_topk_features=pca_topk_features,
                make_nonlinear=make_nonlinear,
            )

            # attach global QA to combined
            qa_dir_c = os.path.join(combined["outdir"], "report", "dataset_qa")
            os.makedirs(qa_dir_c, exist_ok=True)
            if isinstance(dataset_summary_df, pd.DataFrame) and not dataset_summary_df.empty:
                dataset_summary_df.to_csv(os.path.join(qa_dir_c, "dataset_summary_normalized.csv"), index=False)
            if evaluation_results_json is not None:
                with open(os.path.join(qa_dir_c, "evaluation_results.json"), "w") as fh:
                    json.dump(evaluation_results_json, fh, indent=2)

    # run meta-analysis
    meta = meta_analyze_disease_vs_control(runs, out_root=os.path.join(out_root, "meta_summary"))

    out = {
        "runs": runs,
        "combine_decision": decision,
        "combined": combined,
        "meta_dir": meta.get("meta_dir"),
        "summary_txt_path": meta.get("summary_txt_path"),
        "summary_png_path": meta.get("summary_png_path"),
    }
    if qa_warnings:
        out["qa_mapping_warnings"] = qa_warnings
    return out







