"""
Merge Historical + Scenario Data
==================================
Combines the downloaded historical (1850–2014) IITM-ESM data
with your existing ScenarioMIP (2015–2025) data.

Only keeps 2000–2025 (25 years) and crops to the same
lat/lon region as your existing files.

Output: Overwrites your existing .nc files with the merged 25-year data.
"""

import xarray as xr
import numpy as np
import os
import glob
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(BASE_DIR, "historical_data")

# Your existing scenario files
SCENARIO_FILES = {
    "pr":  os.path.join(BASE_DIR, "precipitation.nc"),
    "ts":  os.path.join(BASE_DIR, "surface temperature.nc"),
    "hus": os.path.join(BASE_DIR, "specific humidity.nc"),
}

# Output files (same names — we back up originals first)
OUTPUT_FILES = {
    "pr":  os.path.join(BASE_DIR, "precipitation.nc"),
    "ts":  os.path.join(BASE_DIR, "surface temperature.nc"),
    "hus": os.path.join(BASE_DIR, "specific humidity.nc"),
}

# Date range we want to keep
START_YEAR = 2000
END_YEAR   = 2025


def find_historical_files(variable):
    """Find downloaded historical NetCDF files for a variable."""
    pattern = os.path.join(HIST_DIR, f"*_{variable}_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try without underscore pattern
        pattern = os.path.join(HIST_DIR, f"*{variable}*.nc")
        files = sorted(glob.glob(pattern))
    return files


def merge_variable(var_name):
    """Merge historical + scenario data for one variable."""
    print(f"\n--- Processing: {var_name} ---")
    
    # 1. Load existing scenario data
    scenario_path = SCENARIO_FILES[var_name]
    if not os.path.exists(scenario_path):
        print(f"  ERROR: Scenario file not found: {scenario_path}")
        return False
    
    scenario = xr.open_dataset(scenario_path)
    print(f"  Scenario data: {scenario.time.values[0]} to {scenario.time.values[-1]}")
    print(f"  Lat range: {float(scenario.lat.min()):.2f} to {float(scenario.lat.max()):.2f}")
    print(f"  Lon range: {float(scenario.lon.min()):.2f} to {float(scenario.lon.max()):.2f}")
    
    # 2. Find and load historical files
    hist_files = find_historical_files(var_name)
    if not hist_files:
        print(f"  ERROR: No historical files found for {var_name} in {HIST_DIR}")
        print(f"  Looking for files matching: *{var_name}*.nc")
        print(f"  Files in directory: {os.listdir(HIST_DIR) if os.path.exists(HIST_DIR) else 'DIR NOT FOUND'}")
        return False
    
    print(f"  Found {len(hist_files)} historical file(s)")
    
    # Open all historical files and merge
    historical = xr.open_mfdataset(hist_files, combine="by_coords")
    print(f"  Historical data: {historical.time.values[0]} to {historical.time.values[-1]}")
    
    # 3. Crop historical to our lat/lon region
    lat_min, lat_max = float(scenario.lat.min()), float(scenario.lat.max())
    lon_min, lon_max = float(scenario.lon.min()), float(scenario.lon.max())
    
    historical_cropped = historical.sel(
        lat=slice(lat_min - 1, lat_max + 1),
        lon=slice(lon_min - 1, lon_max + 1),
    )
    
    # Match exact lat/lon from scenario
    historical_cropped = historical_cropped.sel(
        lat=scenario.lat,
        lon=scenario.lon,
        method="nearest"
    )
    
    # 4. Filter to 2000 onwards
    historical_cropped = historical_cropped.sel(
        time=slice(f"{START_YEAR}-01-01", f"2014-12-31")
    )
    print(f"  Historical cropped: {historical_cropped.time.values[0]} to {historical_cropped.time.values[-1]}")
    print(f"  Historical records: {len(historical_cropped.time)}")
    
    # 5. Concatenate historical + scenario along time
    # Only keep the main variable + coordinates (drop bounds etc.)
    hist_var = historical_cropped[[var_name]]
    scen_var = scenario[[var_name]]
    
    # Handle plev dimension for humidity
    if "plev" in hist_var.dims and "plev" not in scen_var.dims:
        hist_var = hist_var.mean(dim="plev")
    elif "plev" in scen_var.dims and "plev" not in hist_var.dims:
        scen_var = scen_var.mean(dim="plev")
    
    merged = xr.concat([hist_var, scen_var], dim="time")
    
    # Filter to our desired date range
    merged = merged.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    
    total_records = len(merged.time)
    print(f"  Merged: {merged.time.values[0]} to {merged.time.values[-1]}")
    print(f"  Total records: {total_records} months ({total_records // 12} years)")
    
    # 6. Back up original file
    backup_path = scenario_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(scenario_path, backup_path)
        print(f"  Backed up original to: {os.path.basename(backup_path)}")
    
    # 7. Save merged file
    output_path = OUTPUT_FILES[var_name]
    merged.to_netcdf(output_path)
    print(f"  Saved merged file: {os.path.basename(output_path)}")
    
    # Cleanup
    scenario.close()
    historical.close()
    
    return True


def main():
    print("=" * 60)
    print("Merging Historical + Scenario Data")
    print(f"Target: {START_YEAR}–{END_YEAR} ({END_YEAR - START_YEAR + 1} years)")
    print("=" * 60)
    
    if not os.path.exists(HIST_DIR):
        print(f"\nERROR: Historical data directory not found!")
        print(f"Expected: {HIST_DIR}")
        print("Run 'python download_data.py' first to download the data.")
        return
    
    success = 0
    for var_name in ["pr", "ts", "hus"]:
        if merge_variable(var_name):
            success += 1
    
    print(f"\n{'='*60}")
    if success == 3:
        print("SUCCESS! All 3 variables merged.")
        print(f"Your data now covers {START_YEAR}–{END_YEAR} ({END_YEAR - START_YEAR + 1} years).")
        print()
        print("IMPORTANT: Update DATA_START_DATE in app.py:")
        print(f'  DATA_START_DATE = "{START_YEAR}-01-01"')
        print()
        print("Original files backed up as *.nc.backup")
    else:
        print(f"Partially done: {success}/3 variables merged.")
        print("Check errors above and try again.")


if __name__ == "__main__":
    main()
