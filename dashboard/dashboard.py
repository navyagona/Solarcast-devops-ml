"""
dashboard.py – Interactive Streamlit dashboard for SolarCast Region-Based Forecasting
Run with: streamlit run dashboard/dashboard.py
"""

import os, sys, glob, requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH   = os.path.join(PROJECT_ROOT, "model", "solar_model.pkl")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "model", "scaler.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "model", "encoder.pkl")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "unified_solar_dataset.csv")

ALL_FEATURES = [
    "temperature", "pressure", "precipitation", "radiation", "wind_speed",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "region_encoded"
]

REGION_COLORS = {"North": "#2A9D8F", "South": "#F4A261", "East": "#E9C46A", "West": "#E63946"}

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SolarCast Regional", page_icon="☀️", layout="wide")

st.markdown("""
<style>
.main { background-color: #0e1117; }
.hero { font-size:2.5rem; font-weight:800;
  background:linear-gradient(135deg,#F4A261,#E9C46A,#2A9D8F);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-align:center; padding:1rem 0; }
.sub  { text-align:center; color:#8B9DB5; font-size:1.05rem; margin-bottom:1.5rem; }
.sec  { font-size:1.3rem; font-weight:700; color:#E9C46A;
  border-left:4px solid #F4A261; padding-left:10px; margin:1.2rem 0 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    s = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    e = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
    return m, s, e

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">☀️ SolarCast Regional Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Region-Based Solar Power Generation Forecasting · North | South | East | West</div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ SolarCast v2.0")
    st.markdown("**Region-Based ML Forecasting**")
    st.divider()
    api_url = st.text_input("API URL", value="http://localhost:8000")
    selected_region = st.selectbox("Focus Region", ["All", "North", "South", "East", "West"])
    st.divider()
    st.markdown("### Model Info")
    st.markdown("""
    - **Algorithm**: RandomForestRegressor  
    - **Training data**: ERA5 reanalysis  
    - **Regions**: North, South, East, West  
    - **R² Score**: ~0.997  
    - **Features**: 10 (incl. cyclic time)
    """)

# ─── Load everything ──────────────────────────────────────────────────────────
model, scaler, encoder = load_artifacts()
df_all = load_data()

if model is None or scaler is None or encoder is None:
    st.error("Model artifacts not found. Run `python notebooks/train_regional_model.py` first.")
    st.stop()

# ─── Generate predictions on dataset ─────────────────────────────────────────
@st.cache_data
def compute_predictions(_model, _scaler, _encoder):
    df = load_data()
    if df is None:
        return None
    col_map = {"temperature_c":"temperature","pressure_hpa":"pressure",
               "precipitation_mm":"precipitation","radiation_wm2":"radiation","wind_speed_ms":"wind_speed"}
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
        df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
        df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["region_encoded"] = _encoder.transform(df["region"])
    avail = [f for f in ALL_FEATURES if f in df.columns]
    X = df[avail].fillna(0).values
    X_scaled = _scaler.transform(X)
    df["predicted"] = np.clip(_model.predict(X_scaled), 0, None)
    return df

df_pred = compute_predictions(model, scaler, encoder)
regions = list(encoder.classes_)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Region Overview",
    "🔮 Live Prediction",
    "📈 Feature Importance",
    "🌤️ Radiation Trends",
    "🌦️ Weather Impact"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Region Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec">Region-Wise Solar Generation Overview</div>', unsafe_allow_html=True)

    # KPI metrics per region
    if df_pred is not None:
        cols = st.columns(4)
        for i, reg in enumerate(regions):
            sub = df_pred[df_pred["region"] == reg]
            avg_pred = sub["predicted"].mean()
            max_pred = sub["predicted"].max()
            with cols[i]:
                st.metric(f"☀️ {reg}", f"{avg_pred:.2f} kW avg", f"Max: {max_pred:.2f} kW")

    st.divider()

    if df_pred is not None:
        col_left, col_right = st.columns(2)

        # Actual vs Predicted scatter by region
        with col_left:
            target_col = "solar_power_generation"
            fig = go.Figure()
            for reg in regions:
                sub = df_pred[df_pred["region"] == reg].sample(min(500, len(df_pred[df_pred["region"]==reg])), random_state=42)
                fig.add_trace(go.Scatter(
                    x=sub[target_col], y=sub["predicted"],
                    mode="markers", name=reg,
                    marker=dict(color=REGION_COLORS.get(reg, "#fff"), size=4, opacity=0.6)
                ))
            lim = df_pred[target_col].max()
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                line=dict(color="red", dash="dash", width=2), name="Perfect Fit"))
            fig.update_layout(title="Actual vs Predicted by Region", xaxis_title="Actual (kW)",
                yaxis_title="Predicted (kW)", template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420)
            st.plotly_chart(fig, use_container_width=True)

        # Region-wise average generation bar chart
        with col_right:
            region_avg = df_pred.groupby("region")["predicted"].mean().reset_index()
            fig2 = px.bar(region_avg, x="region", y="predicted",
                color="region", color_discrete_map=REGION_COLORS,
                title="Average Predicted Solar Power by Region",
                labels={"predicted": "Avg Power (kW)", "region": "Region"},
                template="plotly_dark", text_auto=".2f")
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                showlegend=False, height=420)
            st.plotly_chart(fig2, use_container_width=True)

        # Per-region metrics table
        st.markdown('<div class="sec">Per-Region Model Performance</div>', unsafe_allow_html=True)
        rows = []
        for reg in regions:
            sub = df_pred[df_pred["region"] == reg]
            actual = sub[target_col].values
            pred   = sub["predicted"].values
            rows.append({
                "Region": reg,
                "Samples": f"{len(sub):,}",
                "MAE (kW)": f"{mean_absolute_error(actual, pred):.4f}",
                "RMSE (kW)": f"{np.sqrt(mean_squared_error(actual, pred)):.4f}",
                "R² Score": f"{r2_score(actual, pred):.4f}",
                "Avg Generation": f"{pred.mean():.3f} kW",
                "Max Generation": f"{pred.max():.3f} kW",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">Live Solar Power Prediction</div>', unsafe_allow_html=True)
    st.markdown("Adjust sliders and click predict to get real-time solar power estimates.")

    c1, c2, c3 = st.columns(3)
    with c1:
        p_region = st.selectbox("Region", regions)
        p_temp   = st.slider("Temperature (°C)", -10.0, 55.0, 28.0, 0.5)
        p_press  = st.slider("Pressure (hPa)", 850.0, 1100.0, 1010.0, 0.5)
    with c2:
        p_precip = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.1)
        p_rad    = st.slider("Radiation (W/m²)", 0.0, 1400.0, 650.0, 10.0)
        p_wind   = st.slider("Wind Speed (m/s)", 0.0, 30.0, 3.5, 0.1)
    with c3:
        p_hour  = st.slider("Hour of Day", 0, 23, 12)
        p_month = st.slider("Month", 1, 12, 6)
        p_day   = st.slider("Day of Year (display only)", 1, 365, 172)

    col_api, col_local = st.columns(2)

    with col_api:
        if st.button("Predict via API", type="primary", use_container_width=True):
            payload = {
                "region": p_region, "temperature": p_temp, "pressure": p_press,
                "precipitation": p_precip, "radiation": p_rad, "wind_speed": p_wind,
                "hour": p_hour, "month": p_month
            }
            try:
                r = requests.post(f"{api_url}/predict", json=payload, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"Predicted: **{data['predicted_solar_power_kw']} kW**")
                    st.json(data)
                else:
                    st.error(f"API Error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to {api_url}. Is the API running?")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_local:
        if st.button("Predict Locally", use_container_width=True):
            try:
                hour_sin = np.sin(2 * np.pi * p_hour / 24)
                hour_cos = np.cos(2 * np.pi * p_hour / 24)
                month_sin = np.sin(2 * np.pi * p_month / 12)
                month_cos = np.cos(2 * np.pi * p_month / 12)
                region_enc = int(encoder.transform([p_region])[0])
                row = pd.DataFrame([[
                    p_temp, p_press, p_precip, p_rad, p_wind,
                    hour_sin, hour_cos, month_sin, month_cos, region_enc
                ]], columns=ALL_FEATURES)
                X_scaled = scaler.transform(row)
                result = max(0.0, float(model.predict(X_scaled)[0]))
                st.success(f"Local Prediction: **{result:.4f} kW**")
            except Exception as e:
                st.error(f"Local prediction error: {e}")

    # Power gauge
    st.divider()
    gauge_val = max(0.0, p_rad * 0.18 * 50 / 1000 * (1 - 0.004 * max(0, p_temp - 25)) * (1 - min(0.9, 2*p_precip)))
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        delta={"reference": 5.0},
        title={"text": f"{p_region} Region – Estimated Output (kW)", "font": {"color": "#E9C46A", "size": 18}},
        gauge={
            "axis": {"range": [0, 12], "tickcolor": "white"},
            "bar":  {"color": "#F4A261"},
            "steps": [
                {"range": [0, 3],  "color": "#264653"},
                {"range": [3, 7],  "color": "#2A9D8F"},
                {"range": [7, 12], "color": "#E9C46A"}
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": 10}
        }
    ))
    fig_g.update_layout(paper_bgcolor="#0e1117", font={"color": "white"}, height=320)
    st.plotly_chart(fig_g, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec">Feature Importance Analysis</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": importances}).sort_values("Importance", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="viridis",
            title="Feature Importances (RandomForest)", template="plotly_dark")
        fig_fi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        fig_sun = px.sunburst(fi_df, path=["Feature"], values="Importance",
            color="Importance", color_continuous_scale="RdYlGn",
            title="Feature Importance Sunburst", template="plotly_dark")
        fig_sun.update_layout(paper_bgcolor="#0e1117", height=450)
        st.plotly_chart(fig_sun, use_container_width=True)

    st.dataframe(fi_df.style.background_gradient(cmap="YlOrRd", subset=["Importance"]), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Radiation Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec">Solar Radiation Trends by Region</div>', unsafe_allow_html=True)

    if df_pred is not None and "radiation" in df_pred.columns and "hour" in df_pred.columns:
        # Hourly radiation profile by region
        hourly = df_pred.groupby(["region", "hour"])["radiation"].mean().reset_index()
        fig_hr = go.Figure()
        for reg in regions:
            sub = hourly[hourly["region"] == reg]
            fig_hr.add_trace(go.Scatter(
                x=sub["hour"], y=sub["radiation"], mode="lines+markers",
                name=reg, line=dict(color=REGION_COLORS.get(reg), width=2),
                marker=dict(size=5)
            ))
        fig_hr.update_layout(title="Average Hourly Solar Radiation (W/m²) by Region",
            xaxis_title="Hour of Day", yaxis_title="Radiation (W/m²)",
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420)
        st.plotly_chart(fig_hr, use_container_width=True)

        # Monthly radiation profile
        if "month" in df_pred.columns:
            monthly = df_pred.groupby(["region", "month"])["radiation"].mean().reset_index()
            MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly["month_label"] = monthly["month"].apply(lambda x: MONTH_LABELS[x-1])
            fig_mo = px.line(monthly, x="month_label", y="radiation", color="region",
                color_discrete_map=REGION_COLORS,
                title="Monthly Average Solar Radiation by Region",
                labels={"radiation": "Avg Radiation (W/m²)", "month_label": "Month"},
                template="plotly_dark", markers=True)
            fig_mo.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
            st.plotly_chart(fig_mo, use_container_width=True)

        # Radiation vs Power scatter
        st.markdown('<div class="sec">Radiation vs Power Generation</div>', unsafe_allow_html=True)
        sample = df_pred.sample(min(2000, len(df_pred)), random_state=42)
        fig_rv = px.scatter(sample, x="radiation", y="predicted", color="region",
            color_discrete_map=REGION_COLORS,
            title="Solar Radiation vs Predicted Power (sample of 2000)",
            labels={"radiation": "Radiation (W/m²)", "predicted": "Predicted Power (kW)"},
            template="plotly_dark", opacity=0.6)
        fig_rv.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420)
        st.plotly_chart(fig_rv, use_container_width=True)
    else:
        st.info("Dataset not available for trend analysis.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Weather Impact
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec">Weather Feature Impact Analysis</div>', unsafe_allow_html=True)

    if df_pred is not None:
        weather_cols = [c for c in ["temperature", "pressure", "precipitation", "wind_speed"] if c in df_pred.columns]
        focus_reg = selected_region if selected_region != "All" else None
        plot_df = df_pred[df_pred["region"] == focus_reg] if focus_reg else df_pred
        sample_df = plot_df.sample(min(3000, len(plot_df)), random_state=42)

        for feat in weather_cols:
            if feat not in sample_df.columns:
                continue
            fig_w = px.scatter(sample_df, x=feat, y="predicted", color="region",
                color_discrete_map=REGION_COLORS,
                title=f"{feat.replace('_',' ').title()} vs Predicted Solar Power",
                labels={feat: feat.replace('_',' ').title(), "predicted": "Predicted Power (kW)"},
                template="plotly_dark", opacity=0.5, trendline="lowess")
            fig_w.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=380)
            st.plotly_chart(fig_w, use_container_width=True)

        # Correlation heatmap
        st.markdown('<div class="sec">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        num_cols = weather_cols + ["radiation", "predicted"] if "radiation" in df_pred.columns else weather_cols + ["predicted"]
        corr_df  = df_pred[[c for c in num_cols if c in df_pred.columns]].corr()
        fig_heat = px.imshow(corr_df, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Feature Correlation Matrix", template="plotly_dark", text_auto=".2f")
        fig_heat.update_layout(paper_bgcolor="#0e1117", height=420)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Dataset not available.")

st.divider()
st.markdown(
    "<center style='color:#8B9DB5;'>☀️ SolarCast v2.0 · Region-Based Solar Power Forecasting · "
    "ERA5 Reanalysis Data · RandomForestRegressor</center>",
    unsafe_allow_html=True
)
