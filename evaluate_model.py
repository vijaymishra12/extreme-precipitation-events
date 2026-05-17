"""Evaluate the daily XGBoost model accuracy."""
import xarray as xr, pandas as pd, numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Load data (same as app.py) ──
pr  = xr.open_dataset("precipitation daily.nc")
ts  = xr.open_dataset("surface temperature daily.nc")
hus = xr.open_dataset("specific humidity daily.nc")

pr_df  = pr[["pr"]].mean(dim=["lat", "lon"]).to_dataframe().reset_index()
pr_df["pr"] = pr_df["pr"] * 86400
ts_df  = ts[["tas"]].mean(dim=["lat", "lon"]).to_dataframe().reset_index()
hus_df = hus[["huss"]].mean(dim=["lat", "lon"]).to_dataframe().reset_index()

n = len(pr_df)
start_date = pd.Timestamp(str(pr.time.values[0])[:10])
time_index = pd.date_range(start=start_date, periods=n, freq="D")
pr_df["time"] = time_index
ts_df["time"] = time_index[:len(ts_df)]
hus_df["time"] = time_index[:len(hus_df)]

df = pr_df[["time", "pr"]].merge(ts_df[["time", "tas"]], on="time").merge(hus_df[["time", "huss"]], on="time")
df["day_of_year"] = df["time"].dt.dayofyear
df["month"] = df["time"].dt.month
df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
df["pr_lag1"]  = df["pr"].shift(1)
df["pr_lag3"]  = df["pr"].rolling(3).mean().shift(1)
df["pr_lag7"]  = df["pr"].rolling(7).mean().shift(1)
df["ts_lag1"]  = df["tas"].shift(1)
df["hus_lag1"] = df["huss"].shift(1)
df = df.dropna().reset_index(drop=True)

FEATURES = ["month", "day_sin", "day_cos", "tas", "huss", "pr_lag1", "pr_lag3", "pr_lag7", "ts_lag1", "hus_lag1"]
X = df[FEATURES]
y = df["pr"]

print(f"Total samples: {len(X)}")
print(f"Date range: {df['time'].min().date()} to {df['time'].max().date()}")
print()

# ── 1. Train/Test Split (last 20% = ~2 years as test) ──
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

print("=" * 50)
print("TRAIN/TEST SPLIT (80/20)")
print("=" * 50)
print(f"Train R²:  {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred_test):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.3f} mm/day")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_test):.3f} mm/day")
print(f"Train RMSE:{np.sqrt(mean_squared_error(y_train, y_pred_train)):.3f} mm/day")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f} mm/day")
print()

# ── 2. Time-Series Cross-Validation (5 folds) ──
print("=" * 50)
print("TIME-SERIES CROSS-VALIDATION (5 folds)")
print("=" * 50)
tscv = TimeSeriesSplit(n_splits=5)
cv_r2  = cross_val_score(model, X, y, cv=tscv, scoring="r2")
cv_mae = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")

print(f"CV R² scores:  {[f'{s:.4f}' for s in cv_r2]}")
print(f"CV R² mean:    {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"CV MAE scores: {[f'{s:.3f}' for s in cv_mae]}")
print(f"CV MAE mean:   {cv_mae.mean():.3f} ± {cv_mae.std():.3f} mm/day")
print()

# ── 3. Feature Importance ──
print("=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)
importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
for name, imp in importances:
    bar = "█" * int(imp * 50)
    print(f"  {name:12s} {imp:.4f} {bar}")
