# drive_ingest.py
# Discover nested `prep/` folders in Google Drive and download counts + metadata.
import io
import re
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

FOLDER_MIME = "application/vnd.google-apps.folder"
VALID_DATA_EXT = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
COUNT_HINTS = ("count", "counts", "expr", "expression", "matrix")
META_HINTS = ("meta", "metadata", "pheno", "clinical", "sample", "phenotype")


def parse_folder_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    # Raw ID (Drive ids are long-ish base64-like)
    if re.fullmatch(r"[A-Za-z0-9_\-]{20,}", s):
        return s
    # /folders/<id>
    m = re.search(r"/folders/([A-Za-z0-9_\-]+)", s)
    if m:
        return m.group(1)
    # ?id=<id>
    m = re.search(r"[?&]id=([A-Za-z0-9_\-]+)", s)
    if m:
        return m.group(1)
    raise ValueError("Could not parse a Drive folder ID from the provided link.")


def _is_data_file(name: str) -> bool:
    nm = name.lower()
    return nm.endswith(VALID_DATA_EXT)


def _looks_like_counts(name: str) -> bool:
    nm = name.lower()
    return any(h in nm for h in COUNT_HINTS) and _is_data_file(name)


def _looks_like_meta(name: str) -> bool:
    nm = name.lower()
    return any(h in nm for h in META_HINTS) and _is_data_file(name)


@dataclass
class DriveFile:
    id: str
    name: str
    mimeType: str
    parents: List[str]
    path_hint: str = ""


@dataclass
class PrepBundle:
    disease: str
    path: str
    prep_folder_id: str
    counts: List[DriveFile]
    metas: List[DriveFile]


class DriveClient:
    """Thin wrapper over Google Drive API v3."""

    def __init__(self, creds: Credentials):
        self.creds = creds
        # cache_discovery=False avoids a write in restricted envs
        self.svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    @classmethod
    def from_service_account_bytes(cls, json_bytes: bytes, scopes: Optional[List[str]] = None):
        """Build a DriveClient from the uploaded Service Account JSON (bytes)."""
        scopes = scopes or ["https://www.googleapis.com/auth/drive.readonly"]
        info = json.loads(json_bytes.decode("utf-8"))
        creds = Credentials.from_service_account_info(info=info, scopes=scopes)
        return cls(creds)

    # ------------------ Drive queries ------------------

    def get_children(self, folder_id: str) -> List[DriveFile]:
        q = f"'{folder_id}' in parents and trashed=false"
        fields = "nextPageToken, files(id, name, mimeType, parents)"
        out: List[DriveFile] = []
        page_token = None
        while True:
            resp = self.svc.files().list(q=q, fields=fields, pageToken=page_token).execute()
            for f in resp.get("files", []):
                out.append(DriveFile(
                    id=f["id"],
                    name=f["name"],
                    mimeType=f["mimeType"],
                    parents=f.get("parents", []),
                ))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return out

    def get_file_path_chain(self, file_or_folder_id: str, stop_id: Optional[str] = None) -> List[DriveFile]:
        """Walk parents up to root (or stop_id), materialize names."""
        fields = "id, name, mimeType, parents"
        chain: List[DriveFile] = []
        cur = file_or_folder_id
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            f = self.svc.files().get(fileId=cur, fields=fields).execute()
            df = DriveFile(
                id=f["id"],
                name=f["name"],
                mimeType=f["mimeType"],
                parents=f.get("parents", []),
            )
            chain.append(df)
            if stop_id and df.id == stop_id:
                break
            if df.parents:
                cur = df.parents[0]
            else:
                break
        chain.reverse()
        return chain

    def download_file(self, file_id: str) -> Tuple[bytes, str]:
        """Return (bytes, suggested_name). Retries on transient errors."""
        meta = self.svc.files().get(fileId=file_id, fields="id,name").execute()
        name = meta.get("name", "file.bin")
        req = self.svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req, chunksize=1 << 20)
        done = False
        backoff = 1.0
        while not done:
            try:
                _, done = downloader.next_chunk()
            except Exception:
                time.sleep(backoff)
                backoff = min(10.0, backoff * 1.8)
        buf.seek(0)
        return buf.read(), name

    # ------------------ Recursive discovery ------------------

    def find_all_prep_folders(self, root_folder_id: str) -> List[PrepBundle]:
        """
        BFS through nested folders:
        deg_data / <disease> / ... / prep / {counts/meta files}
        Returns a list of PrepBundle with discovered files.
        """
        results: List[PrepBundle] = []

        # 1) diseases directly under root
        root_children = self.get_children(root_folder_id)
        disease_folders = [c for c in root_children if c.mimeType == FOLDER_MIME]

        for disease in disease_folders:
            disease_name = disease.name
            # BFS under each disease to find leaf folders named "prep"
            queue = [disease]
            visited = set()
            while queue:
                cur = queue.pop(0)
                if cur.id in visited:
                    continue
                visited.add(cur.id)
                kids = self.get_children(cur.id)
                for k in kids:
                    if k.mimeType == FOLDER_MIME:
                        if k.name.strip().lower() == "prep":
                            # collect files *inside* this prep folder
                            prep_files = self.get_children(k.id)
                            counts = [f for f in prep_files if _looks_like_counts(f.name)]
                            metas = [f for f in prep_files if _looks_like_meta(f.name)]
                            chain = self.get_file_path_chain(k.id, stop_id=root_folder_id)
                            path_hint = "/".join([x.name for x in chain])
                            for f in counts + metas:
                                f.path_hint = path_hint
                            results.append(PrepBundle(
                                disease=disease_name,
                                path=path_hint,
                                prep_folder_id=k.id,
                                counts=counts,
                                metas=metas
                            ))
                        else:
                            queue.append(k)
        return results


# ------------------ Planning logic ------------------

def make_ingest_plan(drive: DriveClient, root_url_or_id: str) -> Dict:
    """
    Build a plan describing how to run:
      - 'single'                    → one prep with 1 counts & 1 meta
      - 'multi_files_one_meta'     → one prep with >=2 counts and exactly 1 meta
      - 'multi_dataset'            → >=2 preps, each with their own counts+meta
    """
    root_id = parse_folder_id(root_url_or_id)
    preps = drive.find_all_prep_folders(root_id)

    if not preps:
        return {"mode": "none", "reason": "No prep folders found.", "preps": []}

    usable = [p for p in preps if len(p.counts) > 0]
    if not usable:
        return {"mode": "none", "reason": "No counts files detected in any prep folder.", "preps": []}

    # If multiple prep folders -> treat as multi-dataset (require each to have a meta)
    if len(usable) >= 2:
        datasets = []
        for p in usable:
            # choose best meta
            meta_file = None
            if p.metas:
                def _score_meta(df: DriveFile):
                    nm = df.name.lower()
                    hits = sum(int(h in nm) for h in META_HINTS)
                    return (-hits, len(nm))
                meta_file = sorted(p.metas, key=_score_meta)[0]
            if not meta_file:
                # skip preps without metadata in multi-dataset mode
                continue

            # choose best counts
            def _score_counts(df: DriveFile):
                nm = df.name.lower()
                hits = sum(int(h in nm) for h in COUNT_HINTS)
                ext_rank = 0 if nm.endswith((".tsv", ".txt")) else (1 if nm.endswith(".csv") else 2)
                return (-hits, ext_rank, len(nm))
            best_counts = sorted(p.counts, key=_score_counts)[0]

            c_bytes, c_name = drive.download_file(best_counts.id)
            m_bytes, m_name = drive.download_file(meta_file.id)
            label = f"{p.disease} :: {p.path.split('/')[-2]} / prep" if "/" in p.path else p.path

            datasets.append({
                "label": label,
                "counts": io.BytesIO(c_bytes),
                "counts_name": c_name,
                "meta": io.BytesIO(m_bytes),
                "meta_name": m_name,
            })

        if len(datasets) >= 2:
            return {"mode": "multi_dataset", "datasets": datasets, "preps_found": len(usable)}

        # Fallback: if exactly one valid dataset, treat as single
        if len(datasets) == 1:
            single = datasets[0]
            return {
                "mode": "single",
                "single": {
                    "counts": single["counts"],
                    "counts_name": single["counts_name"],
                    "meta": single["meta"],
                    "meta_name": single["meta_name"],
                },
                "preps_found": len(usable),
            }

    # Exactly one usable prep
    p = usable[0]

    # One meta + multiple counts → multi-files-one-meta
    if len(p.metas) >= 1 and len(p.counts) >= 2:
        def _score_meta(df: DriveFile):
            nm = df.name.lower()
            hits = sum(int(h in nm) for h in META_HINTS)
            return (-hits, len(nm))
        meta_file = sorted(p.metas, key=_score_meta)[0]
        m_bytes, m_name = drive.download_file(meta_file.id)

        groups: Dict[str, Tuple[io.BytesIO, str]] = {}
        for i, cf in enumerate(p.counts, 1):
            c_bytes, c_name = drive.download_file(cf.id)
            base = re.sub(r"\.[^.]+$", "", cf.name)
            label = base if len(base) <= 40 else f"Group_{i}"
            groups[label] = (io.BytesIO(c_bytes), c_name)

        return {
            "mode": "multi_files_one_meta",
            "meta": io.BytesIO(m_bytes),
            "meta_name": m_name,
            "groups": groups,
            "prep_path": p.path,
        }

    # One counts + one meta → single
    if len(p.counts) == 1 and len(p.metas) >= 1:
        def _score_meta(df: DriveFile):
            nm = df.name.lower()
            hits = sum(int(h in nm) for h in META_HINTS)
            return (-hits, len(nm))
        meta_file = sorted(p.metas, key=_score_meta)[0]
        best_counts = p.counts[0]
        c_bytes, c_name = drive.download_file(best_counts.id)
        m_bytes, m_name = drive.download_file(meta_file.id)
        return {
            "mode": "single",
            "single": {
                "counts": io.BytesIO(c_bytes),
                "counts_name": c_name,
                "meta": io.BytesIO(m_bytes),
                "meta_name": m_name,
            },
            "prep_path": p.path,
        }

    # Ambiguous contents
    return {
        "mode": "none",
        "reason": f"Ambiguous contents in prep folder: counts={len(p.counts)}, metas={len(p.metas)}",
        "prep_path": p.path,
    }
