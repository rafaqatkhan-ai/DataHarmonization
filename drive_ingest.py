# drive_ingest.py
# Discover nested `prep/` folders in Google Drive and download counts + metadata.
import io
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

FOLDER_MIME = "application/vnd.google-apps.folder"
VALID_DATA_EXT = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
COUNT_HINTS = ("count", "counts", "expr", "expression", "matrix")
META_HINTS = ("meta", "metadata", "pheno", "clinical", "sample", "phenotype")

# ---------- Helpers ----------

def parse_folder_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    # Accept raw ID
    if re.fullmatch(r"[A-Za-z0-9_\-]{20,}", s):
        return s
    # /folders/<id>
    m = re.search(r"/folders/([A-Za-z0-9_\-]+)", s)
    if m:
        return m.group(1)
    # Fallback: query param id=...
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
    def __init__(self, creds: Credentials):
        self.creds = creds
        self.svc = build("drive", "v3", credentials=creds, cache_discovery=False)

    @classmethod
    def from_service_account_bytes(cls, json_bytes: bytes, scopes: Optional[List[str]] = None):
        scopes = scopes or ["https://www.googleapis.com/auth/drive.readonly"]
        creds = Credentials.from_service_account_info(
            info=__class__._bytes_to_dict(json_bytes),
            scopes=scopes
        )
        return cls(creds)

    @staticmethod
    def _bytes_to_dict(b: bytes) -> dict:
        import json
        return json.loads(b.decode("utf-8"))

    # ---- Drive queries ----
    def get_children(self, folder_id: str) -> List[DriveFile]:
        q = f"'{folder_id}' in parents and trashed=false"
        fields = "nextPageToken, files(id, name, mimeType, parents)"
        out = []
        page_token = None
        while True:
            resp = self.svc.files().list(
                q=q, fields=fields, pageToken=page_token
            ).execute()
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
        """Walk parents up to root (or stop_id), materialize names. (Best-effort; uses additional API calls.)"""
        # Minimal path resolution to build "disease/sub/..." hints
        fields = "id, name, mimeType, parents"
        chain = []
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
        """Return (bytes, suggested_name)"""
        meta = self.svc.files().get(fileId=file_id, fields="id,name").execute()
        name = meta.get("name", "file.bin")
        req = self.svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req, chunksize=1 << 20)  # 1MB chunks
        done = False
        backoff = 1.0
        while not done:
            try:
                status, done = downloader.next_chunk()
            except Exception as e:
                # Simple retry
                time.sleep(backoff)
                backoff = min(10.0, backoff * 1.8)
                continue
        buf.seek(0)
        return buf.read(), name

    # ---- Recursive discovery ----
    def find_all_prep_folders(self, root_folder_id: str) -> List[PrepBundle]:
        """
        BFS through nested folders:
        deg_data / <disease> / ... / prep / {counts/meta files}
        Returns a list of PrepBundle with discovered files.
        """
        results: List[PrepBundle] = []

        # Collect immediate diseases: children under root that are folders
        root_children = self.get_children(root_folder_id)
        disease_folders = [c for c in root_children if c.mimeType == FOLDER_MIME]

        for dis in disease_folders:
            disease_name = dis.name
            # BFS under each disease folder until we find folders named 'prep' (case-insensitive)
            queue = [dis]
            visited = set()
            while queue:
                cur = queue.pop(0)
                if cur.id in visited:
                    continue
                visited.add(cur.id)
                kids = self.get_children(cur.id)
                # any folder named prep?
                for k in kids:
                    if k.mimeType == FOLDER_MIME:
                        if k.name.strip().lower() == "prep":
                            # gather files in this prep
                            prep_files = self.get_children(k.id)
                            counts = [f for f in prep_files if _looks_like_counts(f.name)]
                            metas  = [f for f in prep_files if _looks_like_meta(f.name)]
                            # Build a path hint
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
            # End BFS disease
        return results

# ---------- Planning logic ----------

def make_ingest_plan(
    drive: DriveClient,
    root_url_or_id: str
) -> Dict:
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

    # Separate usable preps (must have at least one counts; meta optional for multi-files-one-meta and single)
    usable = []
    for p in preps:
        if len(p.counts) == 0:
            continue
        usable.append(p)

    if not usable:
        return {"mode": "none", "reason": "No counts files detected in any prep folder.", "preps": []}

    # If multiple prep folders → multi-dataset (each prep expected to include its *own* meta)
    if len(usable) >= 2:
        datasets = []
        for p in usable:
            # pick best meta (if multiple, prefer file with 'meta' in name, then the first)
            meta_file = None
            if p.metas:
                # Rank: number of META_HINT matches (more is better), then shorter name
                def _score_meta(df: DriveFile):
                    nm = df.name.lower()
                    hits = sum(int(h in nm) for h in META_HINTS)
                    return (-hits, len(nm))
                meta_file = sorted(p.metas, key=_score_meta)[0]

            # If no meta found for this prep, skip dataset (we require meta per dataset in this branch)
            if not meta_file:
                continue

            # Choose best counts file (or take the first if only one)
            def _score_counts(df: DriveFile):
                nm = df.name.lower()
                hits = sum(int(h in nm) for h in COUNT_HINTS)
                # prefer tabular text over excel slightly
                ext_rank = 0 if nm.endswith((".tsv", ".txt")) else (1 if nm.endswith(".csv") else 2)
                return (-hits, ext_rank, len(nm))
            best_counts = sorted(p.counts, key=_score_counts)[0]

            # Download the pair
            c_bytes, c_name = drive.download_file(best_counts.id)
            m_bytes, m_name = drive.download_file(meta_file.id)

            datasets.append({
                "label": f"{p.disease} :: {p.path.split('/')[-2]} / prep",  # e.g., disease :: run-id / prep
                "counts": io.BytesIO(c_bytes),
                "counts_name": c_name,
                "meta": io.BytesIO(m_bytes),
                "meta_name": m_name,
            })

        if len(datasets) >= 2:
            return {"mode": "multi_dataset", "datasets": datasets, "preps_found": len(usable)}

        # If multiple preps but only one produced a valid pair, treat it as single
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

    # Exactly one usable prep folder
    p = usable[0]

    # Case A: One meta + multiple counts in the same prep → multi-files-one-meta
    if len(p.metas) >= 1 and len(p.counts) >= 2:
        # choose one meta (best-scored)
        def _score_meta(df: DriveFile):
            nm = df.name.lower()
            hits = sum(int(h in nm) for h in META_HINTS)
            return (-hits, len(nm))
        meta_file = sorted(p.metas, key=_score_meta)[0]
        m_bytes, m_name = drive.download_file(meta_file.id)

        groups: Dict[str, Tuple[io.BytesIO, str]] = {}
        # create group labels from filename (before first dot)
        for i, cf in enumerate(p.counts, 1):
            c_bytes, c_name = drive.download_file(cf.id)
            base = re.sub(r"\.[^.]+$", "", cf.name)  # drop extension
            label = base
            # Fallback to generic label if too long
            if len(label) > 40:
                label = f"Group_{i}"
            groups[label] = (io.BytesIO(c_bytes), c_name)

        return {
            "mode": "multi_files_one_meta",
            "meta": io.BytesIO(m_bytes),
            "meta_name": m_name,
            "groups": groups,
            "prep_path": p.path,
        }

    # Case B: One counts + one meta → single
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

    # If we get here, data is ambiguous (no meta, or weird layout)
    return {
        "mode": "none",
        "reason": f"Ambiguous contents in prep folder: counts={len(p.counts)}, metas={len(p.metas)}",
        "prep_path": p.path,
    }
