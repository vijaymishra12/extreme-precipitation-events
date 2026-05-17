"""
CMIP6 Data Downloader for IITM-ESM Historical Data
====================================================
Downloads pr, ts, hus from ESGF mirrors (CEDA UK / DKRZ Germany)
for the IITM-ESM model historical experiment.

Only downloads files covering 2000-2014 to combine with your
existing 2015-2025 data = 25 years total.
"""

import requests
import os
import re

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")

# Use DKRZ search endpoint (more reliable than LLNL)
SEARCH_URL = "https://esgf-data.dkrz.de/esg-search/search"

# Preferred download mirrors (try in order)
PREFERRED_NODES = ["esgf.ceda.ac.uk", "esgf3.dkrz.de"]

VARIABLES = {
    "pr":  {"table": "Amon", "desc": "Precipitation"},
    "ts":  {"table": "Amon", "desc": "Surface Temperature"},
    "hus": {"table": "Amon", "desc": "Specific Humidity"},
}

NEED_START = 2000
NEED_END   = 2014


def file_year_range(filename):
    """Extract start/end year from filename like pr_Amon_..._200001-200912.nc"""
    match = re.search(r"_(\d{6})-(\d{6})\.nc", filename)
    if match:
        return int(match.group(1)[:4]), int(match.group(2)[:4])
    return None, None


def get_best_url(url_list):
    """Pick the best download URL from preferred mirrors."""
    # First try preferred nodes
    for pref in PREFERRED_NODES:
        for entry in url_list:
            parts = entry.split("|")
            if len(parts) >= 2 and "HTTPServer" in parts[-1] and pref in parts[0]:
                return parts[0]
    # Fallback: any HTTP URL
    for entry in url_list:
        parts = entry.split("|")
        if len(parts) >= 2 and "HTTPServer" in parts[-1]:
            return parts[0]
    return None


def search_files(variable_id, table_id):
    """Search ESGF for IITM-ESM historical files via DKRZ."""
    params = {
        "project": "CMIP6",
        "source_id": "IITM-ESM",
        "experiment_id": "historical",
        "variable_id": variable_id,
        "frequency": "mon",
        "variant_label": "r1i1p1f1",
        "table_id": table_id,
        "grid_label": "gn",
        "type": "File",
        "format": "application/solr+json",
        "limit": "100",
        "latest": "true",
        "distrib": "true",
    }

    print(f"  Searching ESGF (via DKRZ) for {variable_id}...")
    resp = requests.get(SEARCH_URL, params=params, timeout=120)
    resp.raise_for_status()
    docs = resp.json()["response"]["docs"]
    print(f"  Found {len(docs)} total file(s)")

    files = []
    for doc in docs:
        title = doc.get("title", "unknown")
        size = doc.get("size", 0)
        urls = doc.get("url", [])

        start_yr, end_yr = file_year_range(title)
        if start_yr is None or end_yr < NEED_START or start_yr > NEED_END:
            continue

        url = get_best_url(urls)
        if url:
            node = url.split("/")[2]
            files.append({
                "title": title,
                "url": url,
                "size_mb": round(size / 1024 / 1024, 1),
                "years": f"{start_yr}-{end_yr}",
                "node": node,
            })

    files.sort(key=lambda f: f["years"])
    return files


def download_file(url, filename):
    """Download a file with progress."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"  Already exists: {filename}, skipping.")
        return filepath

    print(f"  Downloading from {url.split('/')[2]}...")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  [{filename}] {pct:.0f}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end="", flush=True)

    print(f"\n  Saved: {filename}")
    return filepath


def main():
    print("=" * 60)
    print("CMIP6 IITM-ESM Historical Data Downloader")
    print(f"Period: {NEED_START}-{NEED_END}")
    print(f"Mirrors: CEDA (UK), DKRZ (Germany)")
    print("=" * 60)

    all_files = {}
    total_size = 0

    for var_id, info in VARIABLES.items():
        print(f"\n--- {info['desc']} ({var_id}) ---")
        try:
            files = search_files(var_id, info["table"])
            if not files:
                print(f"  No files found for {NEED_START}-{NEED_END}")
                continue

            var_size = sum(f["size_mb"] for f in files)
            total_size += var_size
            print(f"  Need {len(files)} files ({var_size:.1f} MB):")
            for f in files:
                print(f"    {f['title']} ({f['size_mb']}MB) [{f['years']}] from {f['node']}")

            all_files[var_id] = files
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_files:
        print("\nNo files found!")
        return

    n = sum(len(v) for v in all_files.values())
    print(f"\n{'=' * 60}")
    print(f"Total: {n} files, {total_size:.1f} MB")
    print(f"Save to: {SAVE_DIR}")

    confirm = input("Download? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    for var_id, files in all_files.items():
        print(f"\n>>> {VARIABLES[var_id]['desc']}...")
        for f in files:
            try:
                download_file(f["url"], f["title"])
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\n{'=' * 60}")
    print("Done! Now run: python merge_data.py")


if __name__ == "__main__":
    main()
