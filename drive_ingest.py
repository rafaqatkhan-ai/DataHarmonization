# drive_ingest.py
import io
import re
import json
from typing import Dict, List, Optional, Tuple

# ---- Safe imports with helpful error messages ----
_MISSING_GOOGLE_MSG = (
    "Google API client libraries are not installed. "
    "Please add to requirements.txt: "
    "'google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib httplib2'"
)

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from googleapiclient.errors import HttpError  # re-exported for app
except Exception as _e:
    Credentials = None
    build = None
    MediaIoBaseDownload = None
    HttpError = Exception  # fallback type
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# -------------------- ID parsing --------------------
_ID_PATTERNS = [
    r"/folders/([a-zA-Z0-9_-]{10,})",
    r"id=([a-zA-Z0-9_-]{10,})",
    r"/file/d/([a-zA-Z0-9_-]{10,})",
]
def extract_drive_id(url: str) -> Optional[str]:
    for pat in _ID_PATTERNS:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

# -------------------- Drive client --------------------
def build_drive_client_from_sa_bytes(sa_bytes: bytes):
    if _IMPORT_ERROR is not None or Credentials is None or build is None:
        raise ImportError(_MISSING_GOOGLE_MSG) from _IMPORT_ERROR
    info = json.loads(sa_bytes.decode("utf-8"))
    creds = Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# -------------------- List / download --------------------
def list_children(drive, folder_id: str) -> List[dict]:
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id, name, mimeType)"
    out, token = [], None
    while True:
        resp = drive.files().list(
            q=q,
            fields=fields,
            pageToken=token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return out

def download_file_to_bytes(drive, file_id: str) -> io.BytesIO:
    if MediaIoBaseDownload is None:
        raise ImportError(_MISSING_GOOGLE_MSG)  # safety
    bio = io.BytesIO()
    req = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    downloader = MediaIoBaseDownload(bio, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    bio.seek(0)
    return bio

# -------------------- Traverse to deepest 'prep' --------------------
def find_prep_leaves(drive, root_folder_id: str) -> List[Tuple[str, str]]:
    """
    Return [(prep_folder_id, full_path_str), ...]
    """
    results = []

    def dfs(folder_id: str, path: List[str]):
        children = list_children(drive, folder_id)
        folders = [c for c in children if c["mimeType"] == "application/vnd.google-apps.folder"]
        for f in folders:
            name_norm = f["name"].strip().lower()
            new_path = path + [f["name"]]
            if name_norm == "prep":
                results.append((f["id"], "/".join(new_path)))
            dfs(f["id"], new_path)

    try:
        meta = drive.files().get(fileId=root_folder_id, fields="id,name", supportsAllDrives=True).execute()
        root_name = meta.get("name", "root")
    except Exception:
        root_name = "root"

    dfs(root_folder_id, [root_name])

    # de-duplicate by folder id
    uniq = {}
    for fid, p in results:
        uniq[fid] = p
    return [(fid, uniq[fid]) for fid in uniq]

# -------------------- Heuristics --------------------
_TEXT_EXT = (".tsv", ".csv", ".txt", ".xlsx", ".xls")

def is_table_like(name: str) -> bool:
    n = name.lower()
    return n.endswith(_TEXT_EXT)

def is_counts(name: str) -> bool:
    n = name.lower()
    return is_table_like(n) and any(k in n for k in ["count", "counts", "expr", "expression", "matrix"])

def is_meta(name: str) -> bool:
    n = name.lower()
    return is_table_like(n) and any(k in n for k in ["meta", "metadata", "phenotype", "pheno", "sample", "clinic"])

# -------------------- Collect plan --------------------
def collect_from_root(drive, root_folder_url: str):
    """
    Returns one of:
      {"mode":"single", "single": (counts_bio, counts_name), "meta": (meta_bio, meta_name)}
      {"mode":"multi_files", "groups": {group: (bio, name)}, "meta": (bio, name)}
      {"mode":"multi_dataset", "datasets":[{"geo":label,"counts":(bio,name),"meta":(bio,name)}]}
    """
    root_id = extract_drive_id(root_folder_url)
    if not root_id:
        raise ValueError("Could not parse Drive folder ID from the provided URL.")

    prep_leaves = find_prep_leaves(drive, root_id)
    if not prep_leaves:
        raise RuntimeError("No 'prep' folders found under the provided root Drive folder.")

    datasets = []
    for prep_id, path in prep_leaves:
        items = list_children(drive, prep_id)

        # Resolve shortcuts
        resolved = []
        for f in items:
            if f.get("mimeType") == "application/vnd.google-apps.shortcut":
                try:
                    sc = drive.files().get(
                        fileId=f["id"],
                        fields="shortcutDetails",
                        supportsAllDrives=True
                    ).execute()
                    target_id = sc["shortcutDetails"]["targetId"]
                    target = drive.files().get(
                        fileId=target_id,
                        fields="id,name,mimeType",
                        supportsAllDrives=True
                    ).execute()
                    target["name"] = f["name"]
                    resolved.append(target)
                except Exception:
                    continue
            else:
                resolved.append(f)

        count_files = [f for f in resolved
                       if f["mimeType"] != "application/vnd.google-apps.folder" and is_counts(f["name"])]
        meta_files = [f for f in resolved
                      if f["mimeType"] != "application/vnd.google-apps.folder" and is_meta(f["name"])]

        counts_payload = [(download_file_to_bytes(drive, f["id"]), f["name"]) for f in count_files]
        meta_payload = [(download_file_to_bytes(drive, f["id"]), f["name"]) for f in meta_files]

        if counts_payload:
            datasets.append({
                "label": path,
                "counts": counts_payload,
                "meta": meta_payload
            })

    if not datasets:
        raise RuntimeError("Found 'prep' folders, but no usable count/meta files inside them.")

    if len(datasets) == 1 and len(datasets[0]["counts"]) == 1 and len(datasets[0]["meta"]) >= 1:
        c_bio, c_name = datasets[0]["counts"][0]
        m_bio, m_name = datasets[0]["meta"][0]
        return {"mode": "single", "single": (c_bio, c_name), "meta": (m_bio, m_name)}

    if len(datasets) == 1 and len(datasets[0]["counts"]) >= 2 and len(datasets[0]["meta"]) >= 1:
        groups = {}
        for (bio, name) in datasets[0]["counts"]:
            g = re.sub(r"\.[^.]+$", "", name).strip()
            groups[g or f"group_{len(groups)+1}"] = (bio, name)
        return {"mode": "multi_files", "groups": groups, "meta": datasets[0]["meta"][0]}

    md = []
    def pick_best(lst, is_fn):
        lst_sorted = sorted(lst, key=lambda x: (0 if is_fn(x[1]) else 1, len(x[1])))
        return lst_sorted[0]

    for d in datasets:
        if len(d["counts"]) >= 1 and len(d["meta"]) >= 1:
            c_bio, c_name = pick_best(d["counts"], is_counts)
            m_bio, m_name = pick_best(d["meta"], is_meta)
            md.append({"geo": d["label"], "counts": (c_bio, c_name), "meta": (m_bio, m_name)})

    if len(md) >= 2:
        return {"mode": "multi_dataset", "datasets": md}

    raise RuntimeError(
        f"Drive discovery did not fit expected patterns. "
        f"Found datasets: {[(x['label'], len(x['counts']), len(x['meta'])) for x in datasets]}"
    )
