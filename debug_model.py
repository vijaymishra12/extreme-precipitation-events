"""Debug script to diagnose why predictions are flat."""
import xarray as xr, pandas as pd, numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Load merged data exactly like app.py
pr  = xr.open_dataset("precipitation_merged.nc")
ts  = xr.open_dataset("surface temperature_merged.nc")
hus = xr.open_dataset("specific humidity_merged.nc")

pr_df  = pr[["pr"]].mean(dim=["lat", "lon"]).to_dataframe().reset_index()
pr_df["pr"] = pr_df["pr"] * 86400  # Convert to mm/day

ts_df  = ts[["ts"]].mean(dim=["lat", "lon"]).to_dataframe().reset_index()
hus_dims = [d for d in hus["hus"].dims if d != "time"]
hus_df = hus[["hus"]].mean(dim=hus_dims).to_dataframe().reset_index()

n = len(pr_df)
time_index = pd.date_range(start="2000-01-01", periods=n, freq="MS")
pr_df["time"]  = time_index
ts_df["time"]  = time_index
hus_df["time"] = time_index[:len(hus_df)]

df = pr_df[["time", "pr"]].merge(
    ts_df[["time", "ts"]], on="time"
).merge(
    hus_df[["time", "hus"]], on="time"
)
df["month"] = df["time"].dt.month

print("=== DATA OVERVIEW ===")
print(f"Records: {len(df)}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print()

print("=== FEATURE STATS ===")
print(df[["pr", "ts", "hus", "month"]].describe().to_string())
print()

print("=== MONTHLY AVERAGES ===")
monthly = df.groupby("month")[["pr", "ts", "hus"]].mean()
print(monthly.to_string())
print()

print("=== VARIANCE CHECK ===")
print(f"pr  std across months: {monthly['pr'].std():.4f}")
print(f"ts  std across months: {monthly['ts'].std():.4f}")
print(f"hus std across months: {monthly['hus'].std():.6f}")
print()

# Train XGBoost
FEATURE_COLS = ["month", "ts", "hus"]
X = df[FEATURE_COLS]
y = df["pr"]

xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb.fit(X, y)

# Train Random Forest for comparison
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X, y)

# Predict for future
future_dates = pd.date_range(start="2026-01-01", end="2027-12-01", freq="MS")
future = pd.DataFrame({"time": future_dates})
future["month"] = future["time"].dt.month
monthly_means = df.groupby("month")[["ts", "hus"]].mean()
future = future.merge(monthly_means, on="month", how="left")

future["xgb_pred"] = xgb.predict(future[FEATURE_COLS])
future["rf_pred"]  = rf.predict(future[FEATURE_COLS])

print("=== PREDICTIONS (XGBoost vs RandomForest) ===")
print(future[["time", "month", "ts", "hus", "xgb_pred", "rf_pred"]].to_string())
print()

print("=== PREDICTION UNIQUENESS ===")
print(f"XGBoost unique values: {len(future['xgb_pred'].unique())}")
print(f"RF unique values:      {len(future['rf_pred'].unique())}")
print(f"XGBoost range: {future['xgb_pred'].min():.4f} to {future['xgb_pred'].max():.4f}")
print(f"RF range:      {future['rf_pred'].min():.4f} to {future['rf_pred'].max():.4f}")

# Feature importance
print()
print("=== XGB FEATURE IMPORTANCE ===")
for name, imp in zip(FEATURE_COLS, xgb.feature_importances_):
    print(f"  {name}: {imp:.4f}")
