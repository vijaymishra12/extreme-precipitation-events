import json
import requests

with open("district_names.json") as f:
    target_districts = json.load(f)

# States bounding boxes from app.py
STATES = {
    "Jammu & Kashmir": {"lat_min": 32.5, "lat_max": 36.0, "lon_min": 73.0, "lon_max": 79.0},
    "Himachal Pradesh": {"lat_min": 30.5, "lat_max": 33.5, "lon_min": 75.0, "lon_max": 79.0},
    "Punjab":           {"lat_min": 29.0, "lat_max": 32.5, "lon_min": 73.0, "lon_max": 77.0},
    "Uttarakhand":      {"lat_min": 29.0, "lat_max": 31.5, "lon_min": 78.0, "lon_max": 81.0},
    "Haryana":          {"lat_min": 27.5, "lat_max": 30.5, "lon_min": 74.5, "lon_max": 77.5},
    "Delhi":            {"lat_min": 27.5, "lat_max": 29.5, "lon_min": 76.0, "lon_max": 78.0},
    "Rajasthan":        {"lat_min": 23.0, "lat_max": 30.0, "lon_min": 69.0, "lon_max": 77.0},
    "Uttar Pradesh":    {"lat_min": 23.5, "lat_max": 30.5, "lon_min": 77.5, "lon_max": 85.0},
    "Bihar":            {"lat_min": 24.0, "lat_max": 28.0, "lon_min": 83.0, "lon_max": 88.5},
    "Madhya Pradesh":   {"lat_min": 21.0, "lat_max": 26.5, "lon_min": 74.0, "lon_max": 83.0},
    "West Bengal":      {"lat_min": 21.0, "lat_max": 27.5, "lon_min": 86.0, "lon_max": 90.5},
}

url = "https://raw.githubusercontent.com/nshntarora/Indian-Cities-JSON/master/cities.json"
cities = requests.get(url).json()

city_coords = {}
for c in cities:
    name = c.get("name")
    if name and c.get("lat") and c.get("lng"):
        city_coords[name.lower()] = (float(c.get("lat")), float(c.get("lng")))

final_districts = {}

for state, dlist in target_districts.items():
    if state not in STATES: continue
    final_districts[state] = {}
    bounds = STATES[state]
    c_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
    c_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2
    
    for i, dname in enumerate(dlist):
        dname_clean = dname.split("(")[0].strip()
        if dname_clean.lower() in city_coords:
            final_districts[state][dname] = city_coords[dname_clean.lower()]
        else:
            # Fallback: distribute points in a spiral or grid around center
            offset = (i - len(dlist)/2) * 0.1
            final_districts[state][dname] = (c_lat + offset, c_lon + offset)

with open("final_districts_py.json", "w") as f:
    json.dump(final_districts, f, indent=4)
print("Saved final_districts_py.json")
