import requests
import json

# Try multiple ESGF search endpoints to find mirrors
urls_to_try = [
    "https://esgf-data.dkrz.de/esg-search/search",
    "https://esgf-index1.ceda.ac.uk/esg-search/search",
]

for search_url in urls_to_try:
    params = {
        "project": "CMIP6",
        "source_id": "IITM-ESM",
        "experiment_id": "historical",
        "variable_id": "pr",
        "frequency": "mon",
        "variant_label": "r1i1p1f1",
        "type": "File",
        "format": "application/solr+json",
        "limit": "5",
        "latest": "true",
        "distrib": "true",
    }
    try:
        r = requests.get(search_url, params=params, timeout=30)
        if r.status_code == 200:
            docs = r.json()["response"]["docs"]
            print(f"\n{search_url}: {len(docs)} files")
            nodes = set()
            for d in docs:
                for u in d.get("url", []):
                    if "HTTPServer" in u:
                        host = u.split("/")[2]
                        nodes.add(host)
                        print(f"  -> {u.split('|')[0][:120]}")
            print(f"  Nodes: {nodes}")
        else:
            print(f"\n{search_url}: HTTP {r.status_code}")
    except Exception as e:
        print(f"\n{search_url}: ERROR - {e}")

# Also try direct HTTPS on the data node (sometimes HTTP fails but HTTPS works)
print("\n\nTrying direct HTTPS connection to diasjp.net...")
try:
    r = requests.head("https://esgf-data04.diasjp.net/thredds/fileServer/esg_dataroot/CMIP6/CMIP/CCCR-IITM/IITM-ESM/historical/r1i1p1f1/Amon/pr/gn/v20191226/pr_Amon_IITM-ESM_historical_r1i1p1f1_gn_200001-200912.nc", timeout=15)
    print(f"HTTPS status: {r.status_code}")
except Exception as e:
    print(f"HTTPS also failed: {str(e)[:100]}")

# Try IITM's own portal
print("\n\nTrying IITM's own data portal...")
try:
    r = requests.get("https://cccr.tropmet.res.in/", timeout=10)
    print(f"IITM CCCR portal: HTTP {r.status_code}")
except Exception as e:
    print(f"IITM portal: {str(e)[:100]}")
