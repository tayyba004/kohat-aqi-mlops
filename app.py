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
# 2. LOAD & MERGE CSV DATASETS
# ---------------------------------------------------------

        
@st.cache_data
def fetch_live_features():
    aqi_file = "air quality.csv"   # Match your exact GitHub filename
    weather_file = "weather.csv"   # Match your exact GitHub filename
    
    if not os.path.exists(aqi_file) or not os.path.exists(weather_file):
        st.error(f"⚠️ Missing files! Could not find `{aqi_file}` or `{weather_file}` in repo root.")
        st.stop()
        
    try:
        # 1. Skip the top 3 lines of metadata for the Air Quality CSV
        df_aqi = pd.read_csv(aqi_file, skiprows=3)
    except Exception as e:
        st.error(f"❌ Error reading `{aqi_file}`: {e}")
        st.stop()

    try:
        # 2. Check if weather CSV also has metadata; if so, add skiprows=3 here too
        df_weather = pd.read_csv(weather_file, skiprows=3)
    except Exception:
        # Fallback if weather file doesn't have metadata lines at top
        df_weather = pd.read_csv(weather_file)
    
    # 3. Clean up column names (removes units like ' (μg/m³)' if present)
    df_aqi.columns = [c.split(' ')[0] for c in df_aqi.columns]
    df_weather.columns = [c.split(' ')[0] for c in df_weather.columns]

    # 4. Parse timestamps and merge
    df_aqi['time'] = pd.to_datetime(df_aqi['time'])
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    df = pd.merge(df_aqi, df_weather, on='time').sort_values('time').reset_index(drop=True)
    
    if 'european_aqi' in df.columns:
        df.rename(columns={'european_aqi': 'aqi'}, inplace=True)
        
    return df

df_live = fetch_live_features()
latest_row = df_live.iloc[-1]

# ---------------------------------------------------------
# 3. CURRENT METRICS DISPLAY
# ---------------------------------------------------------
st.subheader("📍 Latest Recorded Environmental Metrics (Kohat)")
col1, col2, col3, col4 = st.columns(4)

col1.metric("PM2.5", f"{latest_row.get('pm2_5', 0):.1f} µg/m³")
col2.metric("PM10", f"{latest_row.get('pm10', 0):.1f} µg/m³")
col3.metric("Temperature", f"{latest_row.get('temperature_2m', 0):.1f} °C")
col4.metric("Humidity", f"{latest_row.get('relative_humidity_2m', 0):.1f} %")

# ---------------------------------------------------------
# 4. LOAD MODELS & GENERATE 3-DAY AQI FORECAST
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
    
    # Drop timestamp or non-feature columns if needed before feeding to model
    feature_cols = [c for c in df_live.columns if c not in ['time', 'date']]
    X_input = df_live[feature_cols].iloc[[-1]]

    pred_day1 = model1.predict(X_input)[0]
    pred_day2 = model2.predict(X_input)[0]
    pred_day3 = model3.predict(X_input)[0]

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Day 1 Forecast", f"{pred_day1:.1f} AQI")
    fc2.metric("Day 2 Forecast", f"{pred_day2:.1f} AQI")
    fc3.metric("Day 3 Forecast", f"{pred_day3:.1f} AQI")

    # Forecast Trend Line Chart
    forecast_df = pd.DataFrame({
        "Day": ["Day 1", "Day 2", "Day 3"],
        "Predicted AQI": [pred_day1, pred_day2, pred_day3]
    }).set_index("Day")

    st.line_chart(forecast_df)

except Exception as e:
    st.info("💡 Make sure model `.pkl` files are present in the repo to display live model predictions.")

# ---------------------------------------------------------
# 5. MODEL EVALUATION & EXPLAINABILITY
# ---------------------------------------------------------
st.sidebar.header("📊 Model Metrics")
st.sidebar.metric("Day 1 MAE", "±5.82 AQI")
st.sidebar.metric("Day 2 MAE", "±7.59 AQI")
st.sidebar.metric("Day 3 MAE", "±8.10 AQI")

if os.path.exists("shap_summary.png"):
    st.markdown("---")
    st.subheader("🔍 Model Interpretability (SHAP Analysis)")
    st.image("shap_summary.png", caption="Feature Importance & SHAP Values", use_container_width=True)




   
        



   
    
   
   
    









  

