# drive_ingest.py
# Robust recursive Drive crawler that finds *all* prep folders and pairs counts+meta in each.

import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ---------- Configuration ----------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# File name heuristics
COUNT_HINTS = [
    r"^counts?\b", r"\bcount(s)?\b", r"\bexpr(ession)?\b", r"\bmatrix\b", r"\bprep[_-]?count\b"
]
META_HINTS = [
    r"\bmeta(data)?\b", r"\bpheno(type)?\b", r"\bsample(s)?\b", r"\bannotation\b", r"\bprep[_-]?meta\b"
]

# Extensions we accept
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
    def __init__(self, service_account_json_path: str):
        creds = Credentials.from_service_account_file(service_account_json_path, scopes=SCOPES)
        self.svc = build("drive", "v3", credentials=creds, cache_discovery=False)

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
                files.append(DriveFile(id=f["id"], name=f["name"], mime=f["mimeType"], parent=f.get("parents", [None])[0]))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return files

    def _is_folder(self, mime: str) -> bool:
        return mime == "application/vnd.google-apps.folder"

    def _download(self, file_id: str) -> bytes:
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
            if re.search(pat, s):
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
        disease_query: user input like 'acute' or 'myeloid' (matches substrings, case-insensitive)
        """
        hits: List[DatasetHit] = []
        label_seen: Dict[str, int] = {}

        # 1) list disease folders under root
        disease_folders = [f for f in self._list_children(deg_root_id) if self._is_folder(f.mime)]
        if disease_query:
            q = disease_query.lower().strip()
            disease_folders = [f for f in disease_folders if q in f.name.lower()]

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
                        # keep traversing (so we can find lower-level prep as well)
                        stack.append((ch, rel2))

            # 3) inside each prep folder, find counts + meta files by heuristic
            for prep, relpath in prep_folders:
                children = self._list_children(prep.id)
                # candidate tables
                count_candidates = [f for f in children if not self._is_folder(f.mime) and f.name.lower().endswith(TXT_EXT) and self._match_any(f.name, COUNT_HINTS)]
                meta_candidates  = [f for f in children if not self._is_folder(f.mime) and f.name.lower().endswith(TXT_EXT) and self._match_any(f.name, META_HINTS)]

                # also accept generic names when folder is clearly 'prep'
                if not count_candidates:
                    count_candidates = [f for f in children if not self._is_folder(f.mime) and f.name.lower().endswith(TXT_EXT) and re.search(r"(count|expr|matrix|table|gene)", f.name.lower())]
                if not meta_candidates:
                    meta_candidates = [f for f in children if not self._is_folder(f.mime) and f.name.lower().endswith(TXT_EXT) and re.search(r"(meta|pheno|sample|annot|group)", f.name.lower())]

                if not count_candidates:
                    # Try one level deeper (some people put files in a nested folder under 'prep')
                    for ch in children:
                        if self._is_folder(ch.mime):
                            grand = self._list_children(ch.id)
                            count_candidates.extend([f for f in grand if not self._is_folder(f.mime) and f.name.lower().endswith(TXT_EXT) and self._match_any(f.name, COUNT_HINTS)])
                            meta_candidates.extend([f for f in grand if not self._is_folder(f.mime) and f.name.lower().endswith(TXT_EXT) and self._match_any(f.name, META_HINTS)])

                # Pair up files. If multiple, we try to pair by simple same prefix; else produce all combos.
                pairs: List[Tuple[DriveFile, DriveFile]] = []
                if count_candidates and meta_candidates:
                    # prefix map
                    def base(s: str) -> str:
                        s = os.path.splitext(s)[0]
                        s = re.sub(r"(?i)(counts?|meta(data)?)", "", s)
                        return re.sub(r"[^a-z0-9]+", "", s.lower())

                    m_by_prefix: Dict[str, List[DriveFile]] = {}
                    for m in meta_candidates:
                        m_by_prefix.setdefault(base(m.name), []).append(m)

                    for c in count_candidates:
                        pref = base(c.name)
                        if pref in m_by_prefix and m_by_prefix[pref]:
                            pairs.append((c, m_by_prefix[pref].pop(0)))
                        else:
                            # fallback: pair with first meta
                            pairs.append((c, meta_candidates[0]))
                # if only counts OR only meta, skip (we need both)
                if not pairs:
                    continue

                for cfile, mfile in pairs:
                    # build unique, descriptive label like:
                    # allergic_disease :: df62cf82/.../prep  [counts: prep_count.tsv | meta: prep_meta.tsv]
                    short_rel = relpath.replace("\\", "/").strip("/")
                    base_label = f"{disease.name} :: {short_rel or prep.name}  [counts: {cfile.name} | meta: {mfile.name}]"
                    label = self._uniqify(base_label, label_seen)

                    

                    hits.append(DatasetHit(
                        label=label,
                        disease=disease.name,
                        prep_path=short_rel or prep.name,
                        counts_id=cfile.id, counts_name=cfile.name,
                        meta_id=mfile.id,   meta_name=mfile.name
                    ))

        return hits

    # ----------------- materialization helpers -----------------
    def download_tsv_like(self, file_id: str) -> bytes:
        # Just returns raw bytes (xlsx/csv/tsv/txt). The app will parse by extension.
        return self._download(file_id)
