# drive_ingest.py
# Robust recursive Drive crawler that finds *all* prep folders and pairs counts+meta in each.
# Provides: DriveClient.from_service_account_bytes(...) and make_ingest_plan(...)

import io
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ---------- Configuration ----------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# File name heuristics (case-insensitive regex)
COUNT_HINTS = [
    r"(^|[_\-\s])counts?($|[_\-\s])",
    r"expr(ession)?",
    r"\bmatrix\b",
    r"\btable\b",
    r"prep[_\-]?count",
    r"genes?[_\-]?(counts?|expr)",
]
META_HINTS = [
    r"\bmeta(data)?\b",
    r"pheno(type)?",
    r"sample(s)?",
    r"annot(ation)?",
    r"group(s)?",
    r"prep[_\-]?meta",
]

# Acceptable tabular extensions
TXT_EXT = (".tsv", ".csv", ".txt", ".xlsx", ".xls")


@dataclass
class DriveFile:
    id: str
    name: str
    mime: str
    parent: Optional[str]


@dataclass
class DatasetHit:
    label: str            # human readable, unique
    disease: str          # top-level disease folder name
    prep_path: str        # relative path from disease folder to prep folder
    counts_id: str
    counts_name: str
    meta_id: str
    meta_name: str


class DriveClient:
    """Thin wrapper over Drive v3 with helpers for recursive listing and download."""
    def __init__(self, credentials: Credentials):
        self.creds = credentials
        self.svc = build("drive", "v3", credentials=self.creds, cache_discovery=False)

    @classmethod
    def from_service_account_file(cls, path: str) -> "DriveClient":
        creds = Credentials.from_service_account_file(path, scopes=SCOPES)
        return cls(creds)

    @classmethod
    def from_service_account_bytes(cls, raw_bytes: bytes) -> "DriveClient":
        # Streamlit uploader gives bytes; parse as JSON dict
        import json
        info = json.loads(raw_bytes.decode("utf-8"))
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return cls(creds)

    # ------------ basic helpers ------------
    def _list_children(self, folder_id: str) -> List[DriveFile]:
        q = f"'{folder_id}' in parents and trashed=false"
        files, page_token = [], None
        while True:
            resp = self.svc.files().list(
                q=q,
                fields="nextPageToken, files(id, name, mimeType, parents)",
                pageToken=page_token
            ).execute()
            for f in resp.get("files", []):
                files.append(DriveFile(
                    id=f["id"],
                    name=f["name"],
                    mime=f["mimeType"],
                    parent=(f.get("parents") or [None])[0]
                ))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return files

    @staticmethod
    def _is_folder(mime: str) -> bool:
        return mime == "application/vnd.google-apps.folder"

    def _download_raw(self, file_id: str) -> bytes:
        req = self.svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        return buf.read()

    # ------------ search utilities ------------
    @staticmethod
    def _match_any(name: str, patterns: List[str]) -> bool:
        s = name.lower()
        for pat in patterns:
            if re.search(pat, s, flags=re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _uniqify(base: str, seen: Dict[str, int]) -> str:
        if base not in seen:
            seen[base] = 1
            return base
        seen[base] += 1
        return f"{base} (#{seen[base]})"

    # ------------ core: recursive harvest ------------
    def harvest_datasets(
        self,
        deg_root_id: str,
        disease_query: Optional[str] = None,
    ) -> List[DatasetHit]:
        """
        deg_root_id: ID of the 'deg_data' folder
        disease_query: tokens like 'acute, myeloid' (matches ANY token, substring, case-insensitive)
        """
        hits: List[DatasetHit] = []
        label_seen: Dict[str, int] = {}

        # Tokenize disease query
        tokens: List[str] = []
        if disease_query:
            tokens = [t.strip().lower() for t in re.split(r"[,\s]+", disease_query) if t.strip()]

        # 1) list disease folders under root
        disease_folders = [f for f in self._list_children(deg_root_id) if self._is_folder(f.mime)]
        if tokens:
            def _match_disease(name: str) -> bool:
                n = name.lower()
                return any(tok in n for tok in tokens)
            disease_folders = [f for f in disease_folders if _match_disease(f.name)]

        for disease in disease_folders:
            # 2) recursively find all 'prep' folders under this disease
            stack: List[Tuple[DriveFile, str]] = [(disease, "")]  # (folder, relpath)
            prep_folders: List[Tuple[DriveFile, str]] = []

            while stack:
                folder, rel = stack.pop()
                children = self._list_children(folder.id)
                for ch in children:
                    if self._is_folder(ch.mime):
                        rel2 = os.path.join(rel, ch.name)
                        # treat any folder whose name *contains* 'prep' as a prep folder
                        if "prep" in ch.name.lower():
                            prep_folders.append((ch, rel2))
                        # keep traversing
                        stack.append((ch, rel2))

            # 3) inside each prep folder, find counts + meta files by heuristic
            for prep, relpath in prep_folders:
                children = self._list_children(prep.id)
                files_only = [f for f in children if not self._is_folder(f.mime) and f.name.lower().endswith(TXT_EXT)]

                count_candidates = [f for f in files_only if self._match_any(f.name, COUNT_HINTS)]
                meta_candidates  = [f for f in files_only if self._match_any(f.name, META_HINTS)]

                # broaden if nothing matched strictly
                if not count_candidates:
                    count_candidates = [f for f in files_only if re.search(r"(count|expr|matrix|table|gene)", f.name, re.I)]
                if not meta_candidates:
                    meta_candidates = [f for f in files_only if re.search(r"(meta|pheno|sample|annot|group)", f.name, re.I)]

                # Also check one-level deeper (common layout: prep/<subdir>/*.tsv)
                if not count_candidates or not meta_candidates:
                    for ch in children:
                        if self._is_folder(ch.mime):
                            grand = self._list_children(ch.id)
                            grand_files = [g for g in grand if not self._is_folder(g.mime) and g.name.lower().endswith(TXT_EXT)]
                            if not count_candidates:
                                count_candidates.extend([g for g in grand_files if self._match_any(g.name, COUNT_HINTS)])
                            if not meta_candidates:
                                meta_candidates.extend([g for g in grand_files if self._match_any(g.name, META_HINTS)])

                # Pairing strategy
                pairs: List[Tuple[DriveFile, DriveFile]] = []
                if count_candidates and meta_candidates:
                    # Try to pair by common normalized prefix
                    def base_key(s: str) -> str:
                        s = os.path.splitext(s)[0]
                        s = re.sub(r"(?i)(counts?|meta(data)?)", "", s)
                        return re.sub(r"[^a-z0-9]+", "", s.lower())

                    metas_by_key: Dict[str, List[DriveFile]] = {}
                    for m in meta_candidates:
                        metas_by_key.setdefault(base_key(m.name), []).append(m)

                    used_meta: Dict[str, int] = {}
                    for c in count_candidates:
                        k = base_key(c.name)
                        if k in metas_by_key and metas_by_key[k]:
                            m = metas_by_key[k].pop(0)
                        else:
                            # fallback: first remaining meta
                            m = meta_candidates[used_meta.get("fallback", 0) % len(meta_candidates)]
                            used_meta["fallback"] = used_meta.get("fallback", 0) + 1
                        pairs.append((c, m))

                if not pairs:
                    # Need both to proceed
                    continue

                for cfile, mfile in pairs:
                    # label like: allergic_disease :: df62cf82/.../prep  [counts: X | meta: Y]
                    short_rel = relpath.replace("\\", "/").strip("/")
                    disp_rel = short_rel or prep.name
                    base_label = f"{disease.name} :: {disp_rel}  [counts: {cfile.name} | meta: {mfile.name}]"
                    label = self._uniqify(base_label, label_seen)

                    hits.append(DatasetHit(
                        label=label,
                        disease=disease.name,
                        prep_path=disp_rel,
                        counts_id=cfile.id, counts_name=cfile.name,
                        meta_id=mfile.id,   meta_name=mfile.name
                    ))

        return hits

    # ----------------- materialization helpers -----------------
    def download_tsv_like(self, file_id: str) -> bytes:
        # Returns raw bytes (xlsx/csv/tsv/txt). The app will parse by extension.
        return self._download_raw(file_id)


# ----------------- Public plan builder -----------------
_DRIVE_FOLDER_RE = re.compile(
    r"(?:/folders/|id=)([a-zA-Z0-9\-_]{10,})"
)

def _extract_id(link_or_id: str) -> str:
    s = link_or_id.strip()
    # If already looks like an ID (no slashes, ~25+ chars), accept it
    if re.fullmatch(r"[a-zA-Z0-9\-_]{10,}", s):
        return s
    m = _DRIVE_FOLDER_RE.search(s)
    if not m:
        raise ValueError("Could not parse Google Drive folder ID from the provided link/ID.")
    return m.group(1)


def make_ingest_plan(
    drv: DriveClient,
    deg_root_link_or_id: str,
    disease_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns a dict with keys:
      - mode: "none" | "single" | "multi_dataset"
      - reason: (if mode == "none")
      - single: {counts (BytesIO), counts_name, meta (BytesIO), meta_name, label}  (if mode == "single")
      - datasets: [ {label, counts(BytesIO), counts_name, meta(BytesIO), meta_name}, ... ] (if mode == "multi_dataset")
      - disease: (best-effort: if only one disease folder matched)
      - prep_path: (best-effort: if only one prep)
    """
    root_id = _extract_id(deg_root_link_or_id)

    hits = drv.harvest_datasets(root_id, disease_query=disease_query)
    if not hits:
        return {"mode": "none", "reason": "No (counts, meta) pairs found under deg_data with the given query."}

    # Download into memory
    datasets = []
    for h in hits:
        try:
            cbytes = drv.download_tsv_like(h.counts_id)
            mbytes = drv.download_tsv_like(h.meta_id)
        except Exception as e:
            # Skip broken files but continue others
            continue
        datasets.append({
            "label": h.label,
            "disease": h.disease,
            "prep_path": h.prep_path,
            "counts": io.BytesIO(cbytes),
            "counts_name": h.counts_name,
            "meta": io.BytesIO(mbytes),
            "meta_name": h.meta_name,
        })

    if not datasets:
        return {"mode": "none", "reason": "Found candidate pairs but failed to download any."}

    # If exactly one dataset -> single mode; else multi
    if len(datasets) == 1:
        d = datasets[0]
        return {
            "mode": "single",
            "single": {
                "counts": d["counts"],
                "counts_name": d["counts_name"],
                "meta": d["meta"],
                "meta_name": d["meta_name"],
                "label": d["label"],
            },
            "disease": d.get("disease"),
            "prep_path": d.get("prep_path"),
        }

    # Multi-dataset
    # Keep all; app will attempt to combine based on overlap, otherwise compare separately
    # Also expose a stable disease if all the same; otherwise omit.
    uniq_dis = sorted(set(d["disease"] for d in datasets if d.get("disease")))
    meta = {}
    if len(uniq_dis) == 1:
        meta["disease"] = uniq_dis[0]

    return {
        "mode": "multi_dataset",
        "datasets": datasets,
        **meta
    }
