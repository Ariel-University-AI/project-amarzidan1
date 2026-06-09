"""
download_traffic_counts.py
==========================
Downloads Israeli traffic count data from data.gov.il and saves to CSV.

Datasets found:
1. ספירות תנועה (Traffic Count Surveys Index) — datastore API, 8,219 records
   Resource ID: cb930bf3-388f-48da-b501-a69038ea959a
   Dataset ID:  aa7747c4-fd59-4563-805a-d30e2fd431c8

2. נתוני ספירות תנועה מזדמנות (Ad-hoc Traffic Counts, ZIP archives by year range)
   Resource IDs (ZIP files, not datastore):
     - up to 2010:   b9fc10f1-d753-44be-8aae-5f925d669571
     - 2010-2015:    817599ba-1330-4b28-9650-74e512967c7e
     - 2015-2020:    ca453c88-d320-49de-b4ea-0c71878024b0
     - 2020+:        e32d4418-f2fc-42dc-8abb-9ef352c2b756

3. ספירות תנועה 2019 (Ayalon Counts) — ZIP only, no datastore
4. ספירות תנועה 2018 / CBS PUF — ZIP only, no datastore

The script:
  - Part A: Downloads the full datastore CSV (8,219 survey records) via API.
  - Part B: Downloads ad-hoc count ZIP archives, extracts inner CSV/XLSX files,
            and concatenates them into one big CSV.

Usage:
    pip install requests pandas openpyxl tqdm
    python download_traffic_counts.py
"""

import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://data.gov.il/api/3/action/datastore_search"
DOWNLOAD_BASE = "https://e.data.gov.il/dataset/{dataset_id}/resource/{resource_id}/download/{filename}"

# ─────────────────────────────────────────────────────────────────────────────
# PART A — Download the traffic count survey INDEX from the datastore API
# ─────────────────────────────────────────────────────────────────────────────
SFIROT_RESOURCE_ID = "cb930bf3-388f-48da-b501-a69038ea959a"

def download_datastore_all(resource_id: str, page_size: int = 1000) -> pd.DataFrame:
    """Pages through the CKAN datastore and returns a full DataFrame."""
    records = []
    offset = 0
    total = None

    print(f"[Part A] Downloading datastore resource: {resource_id}")
    while True:
        params = {
            "resource_id": resource_id,
            "limit": page_size,
            "offset": offset,
        }
        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            raise RuntimeError(f"API error: {data.get('error')}")

        result = data["result"]
        if total is None:
            total = result["total"]
            print(f"  Total records: {total}")

        batch = result["records"]
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        print(f"  Fetched {offset}/{total} records...", end="\r")

        if offset >= total:
            break

    print(f"\n  Done — {len(records)} records loaded.")
    df = pd.DataFrame(records)
    # Drop internal CKAN _id column if present
    df = df.drop(columns=["_id"], errors="ignore")
    return df


def save_part_a():
    df = download_datastore_all(SFIROT_RESOURCE_ID)
    out_path = OUTPUT_DIR / "traffic_counts_index.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Part A] Saved {len(df)} rows → {out_path}")
    print(f"[Part A] Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART B — Download ad-hoc count ZIP archives and extract data tables
# ─────────────────────────────────────────────────────────────────────────────
AD_HOC_DATASET_ID = "944e53ea-5fb0-40c6-9427-d9d7ba1407a2"

AD_HOC_RESOURCES = [
    {
        "resource_id": "b9fc10f1-d753-44be-8aae-5f925d669571",
        "filename": "countsvol1_upto2010_ver1.4.zip",
        "label": "up_to_2010",
    },
    {
        "resource_id": "817599ba-1330-4b28-9650-74e512967c7e",
        "filename": "countsvol2_2010-2015_ver1.4.zip",
        "label": "2010_2015",
    },
    {
        "resource_id": "ca453c88-d320-49de-b4ea-0c71878024b0",
        "filename": "countsvol3_2015-2020_ver1.4.zip",
        "label": "2015_2020",
    },
    {
        "resource_id": "e32d4418-f2fc-42dc-8abb-9ef352c2b756",
        "filename": "countsvol4_2020_ver1.4.zip",
        "label": "2020_plus",
    },
]


def download_zip_resource(resource: dict) -> bytes:
    url = DOWNLOAD_BASE.format(
        dataset_id=AD_HOC_DATASET_ID,
        resource_id=resource["resource_id"],
        filename=resource["filename"],
    )
    print(f"  Downloading {resource['filename']} ...")
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    chunks = []
    total_bytes = 0
    for chunk in resp.iter_content(chunk_size=1024 * 256):
        chunks.append(chunk)
        total_bytes += len(chunk)
        print(f"    {total_bytes / 1_048_576:.1f} MB downloaded...", end="\r")
    print()
    return b"".join(chunks)


def read_tables_from_zip(zip_bytes: bytes, label: str) -> list[pd.DataFrame]:
    """Extract CSV or XLSX files from a ZIP and return list of DataFrames."""
    dfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        print(f"    ZIP contains {len(names)} files. Looking for CSV/XLSX ...")
        data_files = [n for n in names if n.lower().endswith((".csv", ".xlsx", ".xls"))]
        if not data_files:
            print(f"    WARNING: no CSV/XLSX found in ZIP. Files: {names[:20]}")
            return dfs
        for fname in data_files:
            print(f"    Reading: {fname}")
            raw = zf.read(fname)
            try:
                if fname.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", low_memory=False)
                    except UnicodeDecodeError:
                        df = pd.read_csv(io.BytesIO(raw), encoding="windows-1255", low_memory=False)
                else:
                    df = pd.read_excel(io.BytesIO(raw))
                df["_source_file"] = fname
                df["_period_label"] = label
                dfs.append(df)
                print(f"      → {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"      ERROR reading {fname}: {e}")
    return dfs


def save_part_b():
    all_dfs = []
    for resource in AD_HOC_RESOURCES:
        label = resource["label"]
        local_zip = OUTPUT_DIR / resource["filename"]

        # Cache ZIP locally to avoid re-downloading
        if local_zip.exists():
            print(f"[Part B] Using cached ZIP: {local_zip}")
            zip_bytes = local_zip.read_bytes()
        else:
            print(f"[Part B] Fetching ZIP: {resource['filename']}")
            try:
                zip_bytes = download_zip_resource(resource)
                local_zip.write_bytes(zip_bytes)
                print(f"  Saved to {local_zip}")
            except Exception as e:
                print(f"  SKIPPING {resource['filename']} — download failed: {e}")
                continue

        dfs = read_tables_from_zip(zip_bytes, label)
        all_dfs.extend(dfs)

    if not all_dfs:
        print("[Part B] No data tables extracted from ZIP files.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = OUTPUT_DIR / "traffic_counts_adhoc.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Part B] Saved {len(combined)} total rows → {out_path}")
    print(f"[Part B] Columns: {list(combined.columns)}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Quick summary of what was downloaded
# ─────────────────────────────────────────────────────────────────────────────
def summarize():
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    idx_path = OUTPUT_DIR / "traffic_counts_index.csv"
    adhoc_path = OUTPUT_DIR / "traffic_counts_adhoc.csv"

    if idx_path.exists():
        df = pd.read_csv(idx_path, encoding="utf-8-sig")
        print(f"\ntraffic_counts_index.csv")
        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {list(df.columns)}")
        if "EXEC_YEAR" in df.columns:
            print(f"  Years   : {sorted(df['EXEC_YEAR'].dropna().unique().tolist())}")
        if "COUNT_NAME" in df.columns:
            print(f"  Sample locations:")
            for loc in df["COUNT_NAME"].dropna().head(5):
                print(f"    - {loc}")

    if adhoc_path.exists():
        df2 = pd.read_csv(adhoc_path, encoding="utf-8-sig", low_memory=False)
        print(f"\ntraffic_counts_adhoc.csv")
        print(f"  Rows    : {len(df2):,}")
        print(f"  Columns : {list(df2.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Israeli Traffic Counts Downloader — data.gov.il")
    print("=" * 60)

    # Part A: Survey index (metadata table, 8,219 records, fully in datastore)
    save_part_a()

    print()

    # Part B: Ad-hoc count archives (ZIP → CSV/XLSX inside)
    # NOTE: These ZIPs can be 10-100 MB each. Comment out if bandwidth is limited.
    save_part_b()

    # Summary
    summarize()

    print("\nDone.")
