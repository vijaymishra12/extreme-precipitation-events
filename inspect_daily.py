"""Inspect the daily NetCDF files to understand their structure."""
import xarray as xr

files = {
    "pr":  "precipitation daily.nc",
    "ts":  "surface temperature daily.nc",
    "hus": "specific humidity daily.nc",
}

for var, fname in files.items():
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"{'='*60}")
    ds = xr.open_dataset(fname)
    print(ds)
    print(f"\nTime range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"Time steps: {len(ds.time)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")
    ds.close()
