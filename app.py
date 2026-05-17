"""
Precipitation Predictor – Flask App  (Daily + State-wise Edition)
==================================================================
A fully offline Flask application that:
  1. Loads DAILY precipitation, temperature, and humidity from NetCDF.
  2. Maps grid cells to North Indian states using lat/lon bounding boxes.
  3. Trains XGBoost with lag features + state encoding.
  4. Predicts daily precipitation per state or for all states combined.
  5. Shows interactive Plotly charts with state comparison.

HOW TO CHANGE DATA IN THE FUTURE:
  • Replace the .nc files and update file paths below.
  • Update variable names if they differ.
  • To add/change states, edit the STATES dictionary.
"""

from flask import Flask, render_template, request, jsonify
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from xgboost import XGBRegressor
import json
import os
import traceback
import gc

# ──────────────────────────────────────────────
# ▸  CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PR_FILE   = os.path.join(BASE_DIR, "precipitation daily.nc")
TEMP_FILE = os.path.join(BASE_DIR, "surface temperature daily.nc")
HUS_FILE  = os.path.join(BASE_DIR, "specific humidity daily.nc")

VAR_PR  = "pr"
VAR_TS  = "tas"
VAR_HUS = "huss"

# ── North Indian States → approximate lat/lon bounding boxes ──
# Grid: lat 21.9–35.2,  lon 69.4–90.0  (spacing ~1.9°)
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

import json

# ── Major Districts for Extreme Event Analysis ──
dist_file = os.path.join(BASE_DIR, "final_districts_py.json")
try:
    with open(dist_file) as f:
        DISTRICTS = json.load(f)
except Exception:
    DISTRICTS = {}
# ──────────────────────────────────────────────

app = Flask(__name__)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
_cache = {}

def _extract_state_series(ds, var, bounds):
    """Extract spatial average for a state's bounding box."""
    subset = ds[var].sel(
        lat=slice(bounds["lat_min"], bounds["lat_max"]),
        lon=slice(bounds["lon_min"], bounds["lon_max"]),
    )
    if subset.size == 0:
        # Fall back to nearest point
        center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
        center_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2
        subset = ds[var].sel(lat=center_lat, lon=center_lon, method="nearest")
    spatial_dims = [d for d in subset.dims if d != "time"]
    return subset.mean(dim=spatial_dims).values


def load_data():
    """Load daily data and build per-state DataFrames with features."""
    if "all_data" in _cache:
        return _cache["all_data"], _cache["district_data"], _cache["state_list"]

    pr  = xr.open_dataset(PR_FILE)
    ts  = xr.open_dataset(TEMP_FILE)
    hus = xr.open_dataset(HUS_FILE)

    n = len(pr.time)
    start_date = pd.Timestamp(str(pr.time.values[0])[:10])
    time_index = pd.date_range(start=start_date, periods=n, freq="D")

    all_frames = []
    state_names = []

    for state_id, (state_name, bounds) in enumerate(STATES.items()):
        pr_vals  = _extract_state_series(pr, VAR_PR, bounds) * 86400.0  # → mm/day
        ts_vals  = _extract_state_series(ts, VAR_TS, bounds)
        hus_vals = _extract_state_series(hus, VAR_HUS, bounds)

        min_len = min(len(pr_vals), len(ts_vals), len(hus_vals), n)

        sdf = pd.DataFrame({
            "time":    time_index[:min_len],
            VAR_PR:    pr_vals[:min_len],
            VAR_TS:    ts_vals[:min_len],
            VAR_HUS:   hus_vals[:min_len],
            "state":   state_name,
            "state_id": state_id,
        })

        # Time features
        sdf["day_of_year"] = sdf["time"].dt.dayofyear
        sdf["month"]       = sdf["time"].dt.month
        sdf["day_sin"]     = np.sin(2 * np.pi * sdf["day_of_year"] / 365.25)
        sdf["day_cos"]     = np.cos(2 * np.pi * sdf["day_of_year"] / 365.25)

        # Lag features (within each state)
        sdf["pr_lag1"]  = sdf[VAR_PR].shift(1)
        sdf["pr_lag3"]  = sdf[VAR_PR].rolling(3).mean().shift(1)
        sdf["pr_lag7"]  = sdf[VAR_PR].rolling(7).mean().shift(1)
        sdf["ts_lag1"]  = sdf[VAR_TS].shift(1)
        sdf["hus_lag1"] = sdf[VAR_HUS].shift(1)

        sdf = sdf.dropna().reset_index(drop=True)
        all_frames.append(sdf)
        state_names.append(state_name)

    # ── Extract District Data (Vectorized for Speed) ──
    district_frames = []
    lat_list, lon_list, dist_names, dist_states, state_ids = [], [], [], [], []
    
    for state_id, (state_name, _) in enumerate(STATES.items()):
        if state_name not in DISTRICTS: continue
        for dist_name, (d_lat, d_lon) in DISTRICTS[state_name].items():
            lat_list.append(d_lat)
            lon_list.append(d_lon)
            dist_names.append(dist_name)
            dist_states.append(state_name)
            state_ids.append(state_id)

    lats_da = xr.DataArray(lat_list, dims="district")
    lons_da = xr.DataArray(lon_list, dims="district")

    d_pr_all  = pr[VAR_PR].sel(lat=lats_da, lon=lons_da, method="nearest").values * 86400.0
    d_ts_all  = ts[VAR_TS].sel(lat=lats_da, lon=lons_da, method="nearest").values
    d_hus_all = hus[VAR_HUS].sel(lat=lats_da, lon=lons_da, method="nearest").values

    for i in range(len(dist_names)):
        min_len = min(d_pr_all.shape[0], d_ts_all.shape[0], d_hus_all.shape[0], n)
        ddf = pd.DataFrame({
            "time": time_index[:min_len], VAR_PR: d_pr_all[:min_len, i], 
            VAR_TS: d_ts_all[:min_len, i], VAR_HUS: d_hus_all[:min_len, i],
            "state": dist_states[i], "state_id": state_ids[i], "district": dist_names[i]
        })
        ddf["day_of_year"] = ddf["time"].dt.dayofyear
        ddf["month"]       = ddf["time"].dt.month
        ddf["day_sin"]     = np.sin(2 * np.pi * ddf["day_of_year"] / 365.25)
        ddf["day_cos"]     = np.cos(2 * np.pi * ddf["day_of_year"] / 365.25)
        ddf["pr_lag1"]  = ddf[VAR_PR].shift(1)
        ddf["pr_lag3"]  = ddf[VAR_PR].rolling(3).mean().shift(1)
        ddf["pr_lag7"]  = ddf[VAR_PR].rolling(7).mean().shift(1)
        ddf["ts_lag1"]  = ddf[VAR_TS].shift(1)
        ddf["hus_lag1"] = ddf[VAR_HUS].shift(1)
        district_frames.append(ddf.dropna().reset_index(drop=True))

    pr.close(); ts.close(); hus.close()
    del pr, ts, hus
    gc.collect()

    combined = pd.concat(all_frames, ignore_index=True)
    combined_districts = pd.concat(district_frames, ignore_index=True)

    # Downcast to float32 to save memory
    for col in combined.select_dtypes(include=['float64']).columns:
        combined[col] = combined[col].astype(np.float32)
    for col in combined_districts.select_dtypes(include=['float64']).columns:
        combined_districts[col] = combined_districts[col].astype(np.float32)

    _cache["all_data"]   = combined
    _cache["district_data"] = combined_districts
    _cache["state_list"] = state_names
    return combined, combined_districts, state_names


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "state_id", "month", "day_sin", "day_cos",
    VAR_TS, VAR_HUS,
    "pr_lag1", "pr_lag3", "pr_lag7",
    "ts_lag1", "hus_lag1",
]

def train_model(df):
    """Train one XGBoost model across all states using numpy for faster inference."""
    if "model" in _cache:
        return _cache["model"]

    X = df[FEATURE_COLS].values
    y = df[VAR_PR].values

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X, y)
    _cache["model"] = model
    gc.collect()
    return model


def predict_batch(model, df, targets, start_date, end_date):
    """
    Highly optimized vectorized prediction for multiple targets (states or districts).
    'targets' is a list of tuples: (name, state_id, filter_df)
    """
    future_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(future_dates) == 0 or not targets:
        return pd.DataFrame()

    num_targets = len(targets)
    num_days = len(future_dates)
    
    # Initialize state for each target
    states_data = []
    for name, sid, t_df in targets:
        daily_mean = t_df.groupby("day_of_year")[[VAR_TS, VAR_HUS]].mean()
        daily_std  = t_df.groupby("day_of_year")[[VAR_TS, VAR_HUS]].std()
        states_data.append({
            "name": name, "sid": sid,
            "mean": daily_mean, "std": daily_std,
            "last_pr": list(t_df[VAR_PR].tail(7).values),
            "last_ts": float(t_df[VAR_TS].iloc[-1]),
            "last_hus": float(t_df[VAR_HUS].iloc[-1]),
            "results": []
        })

    np.random.seed(42)
    doys = future_dates.dayofyear
    months = future_dates.month
    day_sins = np.sin(2 * np.pi * doys / 365.25)
    day_coss = np.cos(2 * np.pi * doys / 365.25)

    # Loop day by day, but predict all targets at once!
    for d_idx, date in enumerate(future_dates):
        doy = doys[d_idx]
        month = months[d_idx]
        
        batch_input = []
        batch_ts_hus = [] # Store for updates
        
        for t_idx in range(num_targets):
            sd = states_data[t_idx]
            if doy in sd["mean"].index:
                ts_val  = sd["mean"].loc[doy, VAR_TS] + np.random.normal(0, sd["std"].loc[doy, VAR_TS] or 0.5)
                hus_val = sd["mean"].loc[doy, VAR_HUS] + np.random.normal(0, sd["std"].loc[doy, VAR_HUS] or 0.0001)
            else:
                ts_val, hus_val = sd["last_ts"], sd["last_hus"]
            
            batch_ts_hus.append((ts_val, hus_val))
            
            batch_input.append([
                sd["sid"], month, day_sins[d_idx], day_coss[d_idx],
                ts_val, hus_val,
                sd["last_pr"][-1], np.mean(sd["last_pr"][-3:]), np.mean(sd["last_pr"][-7:]),
                sd["last_ts"], sd["last_hus"]
            ])

        # Batch predict!
        preds = np.maximum(model.predict(np.array(batch_input, dtype=np.float32)), 0.0)
        
        # Update states
        for t_idx in range(num_targets):
            sd = states_data[t_idx]
            pr_val = float(preds[t_idx])
            ts_val, hus_val = batch_ts_hus[t_idx]
            
            sd["results"].append({
                "time": date, "predicted_pr": pr_val,
                "month": month, "day_of_year": doy,
                "year": date.year, "state": sd["name"],
            })
            
            sd["last_pr"].append(pr_val)
            if len(sd["last_pr"]) > 7: sd["last_pr"].pop(0)
            sd["last_ts"], sd["last_hus"] = ts_val, hus_val

    # Combine all results
    all_rows = []
    for sd in states_data:
        all_rows.extend(sd["results"])
    
    return pd.DataFrame(all_rows)


def find_extreme_events(model, dist_df, state_name, state_id, start_date, end_date):
    """Predict for districts and identify extreme precipitation events (> 64.5 mm/day)."""
    state_districts = dist_df[dist_df["state"] == state_name]
    if state_districts.empty: return []
    
    dnames = state_districts["district"].unique()
    targets = []
    for dname in dnames:
        tmp_df = state_districts[state_districts["district"] == dname].copy()
        targets.append((dname, state_id, tmp_df))
    
    # Use batch prediction for speed!
    future_all = predict_batch(model, dist_df, targets, start_date, end_date)
    if future_all.empty: return []

    events = []
    for dname in dnames:
        pred_df = future_all[future_all["state"] == dname]
        # Filter directly by > 64.5 mm/day
        extremes = pred_df[pred_df["predicted_pr"] >= 64.5]
        for _, row in extremes.iterrows():
            pr_val = round(row["predicted_pr"], 2)
            if pr_val >= 204.5:
                cat, lvl = "Extremely Heavy Rain", "Emergency"
            elif pr_val >= 115.6:
                cat, lvl = "Very Heavy Rain", "Alert"
            else:
                cat, lvl = "Heavy Rain", "Warning"
                
            events.append({
                "date": str(row["time"].date()),
                "district": dname,
                "predicted_pr": pr_val,
                "category": cat,
                "level": lvl
            })
            
    return sorted(events, key=lambda x: x["date"])


# ──────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)

def _to_html(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return pio.to_html(fig, full_html=False, config={"displayModeBar": True})


def build_pred_line(future_df):
    """Daily trend line with original pink color and extreme event markers."""
    states = future_df["state"].unique()
    
    if len(states) == 1:
        fig = px.line(future_df, x="time", y="predicted_pr",
                      labels={"time": "Date", "predicted_pr": "Precipitation (mm/day)"},
                      title=f"Predicted Daily Precipitation – {states[0]}")
        fig.update_traces(line=dict(color="#f472b6", width=2))
    else:
        fig = px.line(future_df, x="time", y="predicted_pr", color="state",
                      labels={"time": "Date", "predicted_pr": "Precipitation (mm/day)"},
                      title="Predicted Daily Precipitation – State Comparison")

    # Overlay colored markers for extreme events as requested previously
    extremes = future_df[future_df["predicted_pr"] >= 64.5]
    if not extremes.empty:
        color_map = {
            "Heavy Rain": "#3b82f6",      # Blue
            "Very Heavy Rain": "#22c55e", # Green
            "Extremely Heavy Rain": "#991b1b" # Dark Red
        }
        for cat, color in color_map.items():
            cat_df = extremes[extremes["Category"] == cat]
            if not cat_df.empty:
                fig.add_trace(go.Scatter(
                    x=cat_df["time"], y=cat_df["predicted_pr"],
                    mode='markers', name=cat,
                    marker=dict(color=color, size=7, line=dict(width=1, color='white')),
                    hovertemplate="<b>%{x}</b><br>Rain: %{y} mm/day<br>Category: " + cat + "<extra></extra>"
                ))
    
    fig.update_layout(**PLOTLY_LAYOUT)
    return _to_html(fig)


def build_pred_bar(future_df):
    """Monthly average bar chart."""
    tmp = future_df.copy()
    tmp["month_label"] = tmp["time"].dt.to_period("M").astype(str)
    monthly = tmp.groupby(["month_label", "state"])["predicted_pr"].mean().reset_index()
    states = future_df["state"].unique()
    if len(states) == 1:
        fig = px.bar(monthly, x="month_label", y="predicted_pr",
                     labels={"month_label": "Month", "predicted_pr": "Avg (mm/day)"},
                     title=f"Monthly Avg – {states[0]}")
        fig.update_traces(marker_color="#818cf8")
    else:
        fig = px.bar(monthly, x="month_label", y="predicted_pr", color="state",
                     barmode="group",
                     labels={"month_label": "Month", "predicted_pr": "Avg (mm/day)"},
                     title="Monthly Avg – State Comparison")
    return _to_html(fig)


def build_pred_seasonal(future_df):
    """Seasonal box plot."""
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    tmp = future_df.copy()
    tmp["season"] = tmp["month"].map(season_map)
    states = future_df["state"].unique()
    if len(states) == 1:
        fig = px.box(tmp, x="season", y="predicted_pr",
                     category_orders={"season": ["Winter", "Spring", "Summer", "Autumn"]},
                     labels={"season": "Season", "predicted_pr": "mm/day"},
                     title=f"Seasonal Distribution – {states[0]}",
                     color="season",
                     color_discrete_sequence=["#60a5fa", "#34d399", "#fbbf24", "#f87171"])
    else:
        fig = px.box(tmp, x="season", y="predicted_pr", color="state",
                     category_orders={"season": ["Winter", "Spring", "Summer", "Autumn"]},
                     labels={"season": "Season", "predicted_pr": "mm/day"},
                     title="Seasonal Distribution – State Comparison")
    return _to_html(fig)


def build_pred_heatmap(future_df):
    """Heatmap (Month × Year) for single state, or State × Month for multi."""
    states = future_df["state"].unique()
    if len(states) == 1:
        pivot = future_df.pivot_table(index="month", columns="year",
                                       values="predicted_pr", aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto",
                        labels=dict(x="Year", y="Month", color="mm/day"),
                        title=f"Heatmap – {states[0]}",
                        color_continuous_scale="YlGnBu",
                        y=[f"{m:02d}" for m in pivot.index])
    else:
        pivot = future_df.pivot_table(index="state", columns="month",
                                       values="predicted_pr", aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto",
                        labels=dict(x="Month", y="State", color="mm/day"),
                        title="State × Month Precipitation Heatmap",
                        color_continuous_scale="YlGnBu",
                        x=[f"{m:02d}" for m in pivot.columns])
    return _to_html(fig)


def build_combined_chart(future_df, df, selected_states):
    """Historical + predicted comparison."""
    fig = go.Figure()
    colors_hist = ["#6ee7b7", "#60a5fa", "#fbbf24", "#a78bfa", "#fb923c",
                   "#f472b6", "#34d399", "#818cf8", "#f87171", "#38bdf8", "#e879f9"]
    colors_pred = ["#f472b6", "#818cf8", "#f87171", "#e879f9", "#fb923c",
                   "#34d399", "#60a5fa", "#fbbf24", "#a78bfa", "#6ee7b7", "#38bdf8"]

    for i, state in enumerate(selected_states):
        hist = df[df["state"] == state].set_index("time")[VAR_PR].resample("W").mean().reset_index()
        fig.add_trace(go.Scatter(
            x=hist["time"], y=hist[VAR_PR], mode="lines",
            name=f"{state} (Hist)", line=dict(color=colors_hist[i % len(colors_hist)], width=1),
            opacity=0.6,
        ))
        pred = future_df[future_df["state"] == state]
        fig.add_trace(go.Scatter(
            x=pred["time"], y=pred["predicted_pr"], mode="lines",
            name=f"{state} (Pred)", line=dict(color=colors_pred[i % len(colors_pred)], width=2, dash="dot"),
        ))

    fig.update_layout(title="Historical + Predicted", xaxis_title="Date",
                      yaxis_title="Precipitation (mm/day)")
    fig.update_layout(**PLOTLY_LAYOUT)
    return pio.to_html(fig, full_html=False, config={"displayModeBar": True})


def build_state_comparison_bar(future_df):
    """Overall average precipitation per state – bar chart."""
    avg = future_df.groupby("state")["predicted_pr"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg, x="state", y="predicted_pr",
                 labels={"state": "State", "predicted_pr": "Avg Precipitation (mm/day)"},
                 title="Average Predicted Precipitation by State",
                 color="predicted_pr", color_continuous_scale="YlGnBu")
    return _to_html(fig)


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def home():
    df, dist_df, state_names = load_data()

    data_info = {
        "start":   df["time"].min().strftime("%Y-%m-%d"),
        "end":     df["time"].max().strftime("%Y-%m-%d"),
        "records": len(df),
        "states":  state_names,
        "districts": {s: list(d.keys()) for s, d in DISTRICTS.items()}
    }

    return render_template("index.html", data_info=data_info)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_date = request.form.get("start_date")
        end_date   = request.form.get("end_date")
        selected   = request.form.getlist("states")  # list of state names

        if not start_date or not end_date:
            return jsonify({"error": "Please provide both start and end dates."}), 400

        try:
            pd.Timestamp(start_date)
            pd.Timestamp(end_date)
        except Exception:
            return jsonify({"error": "Invalid date format."}), 400

        pred_mode = request.form.get("pred_mode", "state")
        selected_states = request.form.getlist("states")
        selected_districts = request.form.getlist("districts")

        df, dist_df, state_names = load_data()
        model = train_model(df)

        all_preds_df = pd.DataFrame()
        selected_names = []

        if pred_mode == "state":
            if not selected_states or "All States" in selected_states:
                selected_states = state_names
            selected_names = selected_states
            
            targets = []
            for sname in selected_states:
                sid = list(STATES.keys()).index(sname)
                targets.append((sname, sid, df[df["state"] == sname]))
            
            all_preds_df = predict_batch(model, df, targets, start_date, end_date)
                    
        else:
            # District mode
            if not selected_districts:
                return jsonify({"error": "Please select at least one district."}), 400
                
            targets = []
            for dist_val in selected_districts:
                sname, dname = dist_val.split("|")
                selected_names.append(dname)
                sid = list(STATES.keys()).index(sname)
                
                tmp_df = dist_df[dist_df["district"] == dname].copy()
                targets.append((dname, sid, tmp_df))
                
            all_preds_df = predict_batch(model, dist_df, targets, start_date, end_date)

        if all_preds_df.empty:
            return jsonify({"error": "Prediction failed for selection."}), 400

        future = all_preds_df
        
        # Add severity category for coloring in graphs
        def get_category(pr):
            if pr >= 204.5: return "Extremely Heavy Rain"
            elif pr >= 115.6: return "Very Heavy Rain"
            elif pr >= 64.5: return "Heavy Rain"
            else: return "Normal/Moderate"
            
        future["Category"] = future["predicted_pr"].apply(get_category)

        # Build charts
        pred_charts = {
            "pred_line":     build_pred_line(future),
            "pred_bar":      build_pred_bar(future),
            "pred_seasonal": build_pred_seasonal(future),
            "pred_heatmap":  build_pred_heatmap(future),
        }
        
        # Use selected_names for combined chart
        if pred_mode == "state":
            pred_charts["combined"] = build_combined_chart(future, df, selected_names)
        else:
            tmp_dist = dist_df.copy()
            tmp_dist["state"] = tmp_dist["district"]
            pred_charts["combined"] = build_combined_chart(future, tmp_dist, selected_names)

        # Add comparison chart if multiple items
        if len(selected_names) > 1:
            pred_charts["state_compare"] = build_state_comparison_bar(future)

        # Monthly summary table
        table_rows = []
        for name in selected_names:
            state_pred = future[future["state"] == name]
            monthly = state_pred.groupby(
                state_pred["time"].dt.to_period("M")
            )["predicted_pr"].agg(["mean", "min", "max", "sum"]).reset_index()

            for _, row in monthly.iterrows():
                table_rows.append({
                    "state": name,
                    "date":  str(row["time"]),
                    "avg":   f"{row['mean']:.2f}",
                    "min":   f"{row['min']:.2f}",
                    "max":   f"{row['max']:.2f}",
                    "total": f"{row['sum']:.1f}",
                })

        # Extreme events calculation
        extreme_events = []
        if pred_mode == "state" and len(selected_names) == 1:
            sid = list(STATES.keys()).index(selected_names[0])
            extreme_events = find_extreme_events(model, dist_df, selected_names[0], sid, start_date, end_date)
        elif pred_mode == "district":
            # Already calculated in future_all if we want to filter them
            for name in selected_names:
                p_df = future[future["state"] == name]
                ext = p_df[p_df["predicted_pr"] >= 64.5]
                for _, row in ext.iterrows():
                    pr_val = round(row["predicted_pr"], 2)
                    if pr_val >= 204.5: cat, lvl = "Extremely Heavy Rain", "Emergency"
                    elif pr_val >= 115.6: cat, lvl = "Very Heavy Rain", "Alert"
                    else: cat, lvl = "Heavy Rain", "Warning"
                    extreme_events.append({
                        "date": str(row["time"].date()), "district": name,
                        "predicted_pr": pr_val, "category": cat, "level": lvl
                    })
            extreme_events.sort(key=lambda x: x["date"])

        return jsonify({
            "charts": pred_charts,
            "table": table_rows,
            "multi_state": len(selected_names) > 1,
            "extreme_events": extreme_events,
            "pred_mode": pred_mode
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
