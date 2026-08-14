import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="10Pearls Shine – Kohat AQI Forecaster",
    page_icon="🍃",
    layout="wide"
)

st.title("🍃 Kohat Air Quality Index (AQI) 3-Day Forecaster")
st.caption("Automated Machine Learning pipeline predicting Air Quality Index using local weather and air quality datasets.")

# ---------------------------------------------------------
# 2. LOAD, MERGE & ENGINEER FEATURES
# ---------------------------------------------------------
@st.cache_data
def fetch_live_features():
    # ⚠️ UPDATE THESE FILENAMES IF YOUR FILES ARE NAMED DIFFERENTLY ON GITHUB
    aqi_file = "air quality.csv"   
    weather_file = "weather.csv"   
    
    if not os.path.exists(aqi_file) or not os.path.exists(weather_file):
        st.error(f"⚠️ Files missing in repository! Could not find `{aqi_file}` or `{weather_file}`.")
        st.stop()
        
    # Read CSV files, skipping top metadata if present
    try:
        df_aqi = pd.read_csv(aqi_file, skiprows=3)
    except Exception:
        df_aqi = pd.read_csv(aqi_file)

    try:
        df_weather = pd.read_csv(weather_file, skiprows=3)
    except Exception:
        df_weather = pd.read_csv(weather_file)
    
    # Clean column headers
    df_aqi.columns = [c.split(' ')[0] for c in df_aqi.columns]
    df_weather.columns = [c.split(' ')[0] for c in df_weather.columns]

    # Parse timestamps and merge
    df_aqi['time'] = pd.to_datetime(df_aqi['time'])
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    df = pd.merge(df_aqi, df_weather, on='time').sort_values('time').reset_index(drop=True)
    
    if 'european_aqi' in df.columns:
        df.rename(columns={'european_aqi': 'aqi'}, inplace=True)

    # --- FEATURE ENGINEERING (Matches your training steps) ---
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month

    if 'aqi' in df.columns:
        df['aqi_lag_1h'] = df['aqi'].shift(1)
        df['aqi_lag_24h'] = df['aqi'].shift(24)
        df['aqi_change_rate'] = df['aqi'].diff()
        df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24).mean()

    # Handle missing values created by shifts
    df = df.bfill().ffill()

    return df

df_live = fetch_live_features()
latest_row = df_live.iloc[-1]

# ---------------------------------------------------------
# 3. METRICS DISPLAY
# ---------------------------------------------------------
st.subheader("📍 Latest Recorded Environmental Metrics (Kohat)")
col1, col2, col3, col4 = st.columns(4)

col1.metric("PM2.5", f"{latest_row.get('pm2_5', 0):.1f} µg/m³")
col2.metric("PM10", f"{latest_row.get('pm10', 0):.1f} µg/m³")
col3.metric("Temperature", f"{latest_row.get('temperature_2m', 0):.1f} °C")
col4.metric("Humidity", f"{latest_row.get('relative_humidity_2m', 0):.1f} %")

# ---------------------------------------------------------
# 4. LOAD MODELS & PREDICT
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 3-Day Air Quality Index Forecast")

@st.cache_resource
def load_models():
    m1 = joblib.load("best_aqi_model_day1.pkl")
    m2 = joblib.load("best_aqi_model_day2.pkl")
    m3 = joblib.load("best_aqi_model_day3.pkl")
    return m1, m2, m3

try:
    model1, model2, model3 = load_models()
    
    # Automatically get exact features model expects
    expected_features = getattr(model1, "feature_names_in_", None)
    
    if expected_features is not None:
        X_input = df_live[expected_features].iloc[[-1]]
    else:
        feature_cols = [c for c in df_live.columns if c not in ['time', 'date', 'aqi']]
        X_input = df_live[feature_cols].iloc[[-1]]

    pred_day1 = model1.predict(X_input)[0]
    pred_day2 = model2.predict(X_input)[0]
    pred_day3 = model3.predict(X_input)[0]

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Day 1 Forecast", f"{pred_day1:.1f} AQI")
    fc2.metric("Day 2 Forecast", f"{pred_day2:.1f} AQI")
    fc3.metric("Day 3 Forecast", f"{pred_day3:.1f} AQI")

    forecast_df = pd.DataFrame({
        "Day": ["Day 1", "Day 2", "Day 3"],
        "Predicted AQI": [pred_day1, pred_day2, pred_day3]
    }).set_index("Day")

    st.line_chart(forecast_df)

except Exception as e:
    st.error(f"❌ Forecast Pipeline Error: `{type(e).__name__}: {e}`")

# ---------------------------------------------------------
# 5. SIDEBAR & SHAP
# ---------------------------------------------------------
st.sidebar.header("📊 Model Metrics")
st.sidebar.metric("Day 1 MAE", "±5.82 AQI")
st.sidebar.metric("Day 2 MAE", "±7.59 AQI")
st.sidebar.metric("Day 3 MAE", "±8.10 AQI")

if os.path.exists("shap_summary.png"):
    st.markdown("---")
    st.subheader("🔍 Model Interpretability (SHAP Analysis)")
    st.image("shap_summary.png", caption="Feature Importance & SHAP Values", use_container_width=True)



   
        



   
    
   
   
    









  

