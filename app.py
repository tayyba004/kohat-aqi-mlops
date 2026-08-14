import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Kohat AQI 3-Day Forecast",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌤️ 10Pearls Shine – Kohat (AQI) 3-Day Serverless Forecaster")
st.markdown("Automated Machine Learning pipeline predicting Air Quality Index using live Open-Meteo weather data.")

# Coordinates for Kohat
LAT = 33.5889
LON = 71.4429

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
@st.cache_data
def fetch_live_features():
    """Fetch recent 24-hour weather & air quality data to build live model input features."""
    aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi&past_days=2"
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m&past_days=2"
    
    # 1. Fetch & Validate Air Quality Data
    res_aqi = requests.get(aqi_url)
    data_aqi = res_aqi.json()
    
    if 'hourly' not in data_aqi:
        st.error(f"⚠️ Open-Meteo AQI API Error: {data_aqi.get('reason', 'Failed to retrieve AQI data.')}")
        st.write("Raw API Response:", data_aqi)
        st.stop()
        
    df_aqi = pd.DataFrame(data_aqi['hourly'])
    
    # 2. Fetch & Validate Weather Data
    res_weather = requests.get(weather_url)
    data_weather = res_weather.json()
    
    if 'hourly' not in data_weather:
        st.error(f"⚠️ Open-Meteo Weather API Error: {data_weather.get('reason', 'Failed to retrieve weather data.')}")
        st.write("Raw API Response:", data_weather)
        st.stop()
        
    df_weather = pd.DataFrame(data_weather['hourly'])
    
    # 3. Merge Datasets on Timestamp
    df_aqi['time'] = pd.to_datetime(df_aqi['time'])
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df = pd.merge(df_aqi, df_weather, on='time')
    
    df.rename(columns={'european_aqi': 'aqi'}, inplace=True)

    
    
    # Feature engineering matching training data
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    df['aqi_lag_1h'] = df['aqi'].shift(1)
    df['aqi_lag_24h'] = df['aqi'].shift(24)
    df['aqi_change_rate'] = df['aqi'] - df['aqi_lag_1h']
    
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def get_aqi_status(aqi_val):
    """Categorize European AQI values into health advisories."""
    if aqi_val <= 20:
        return "Good 🟢", "Air quality is considered satisfactory."
    elif aqi_val <= 40:
        return "Fair 🟡", "Air quality is acceptable for most people."
    elif aqi_val <= 60:
        return "Moderate 🟧", "Sensitive groups may experience mild discomfort."
    elif aqi_val <= 80:
        return "Poor 🔴", "Unhealthy air quality; limit outdoor activity."
    else:
        return "Very Poor / Hazardous 🟣", "Hazardous air quality alert! Stay indoors."

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    m1 = joblib.load('best_aqi_model_day1.pkl')
    m2 = joblib.load('best_aqi_model_day2.pkl')
    m3 = joblib.load('best_aqi_model_day3.pkl')
    return m1, m2, m3

model_day1, model_day2, model_day3 = load_models()

# ---------------------------------------------------------
# Sidebar & Data Ingestion
# ---------------------------------------------------------
st.sidebar.header("📍 Location & Controls")
st.sidebar.write("**City:** Kohat, Khyber Pakhtunkhwa, Pakistan")
st.sidebar.write(f"**Coordinates:** {LAT}, {LON}")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Fetching latest satellite & weather data..."):
    df_live = fetch_live_features()

latest_record = df_live.iloc[-1:]

feature_cols = [
    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone',
    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m',
    'hour', 'day_of_week', 'month', 'aqi_lag_1h', 'aqi_lag_24h', 'aqi_change_rate'
]

X_live = latest_record[feature_cols]

# Compute Predictions
pred_day1 = round(float(model_day1.predict(X_live)[0]), 1)
pred_day2 = round(float(model_day2.predict(X_live)[0]), 1)
pred_day3 = round(float(model_day3.predict(X_live)[0]), 1)

current_aqi = round(float(latest_record['aqi'].values[0]), 1)

# ---------------------------------------------------------
# Dashboard Metrics View
# ---------------------------------------------------------
st.subheader("📊 3-Day Forecast Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    status, desc = get_aqi_status(current_aqi)
    st.metric(label="Current AQI (Live)", value=current_aqi)
    st.caption(f"Status: **{status}**")

with col2:
    status1, _ = get_aqi_status(pred_day1)
    st.metric(label="Tomorrow (+24h)", value=pred_day1, delta=round(pred_day1 - current_aqi, 1))
    st.caption(f"Forecast: **{status1}**")

with col3:
    status2, _ = get_aqi_status(pred_day2)
    st.metric(label="Day 2 (+48h)", value=pred_day2, delta=round(pred_day2 - pred_day1, 1))
    st.caption(f"Forecast: **{status2}**")

with col4:
    status3, _ = get_aqi_status(pred_day3)
    st.metric(label="Day 3 (+72h)", value=pred_day3, delta=round(pred_day3 - pred_day2, 1))
    st.caption(f"Forecast: **{status3}**")

st.markdown("---")

# ---------------------------------------------------------
# Interactive Chart & Advanced Analytics
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Forecast Trend Chart", "🔬 Feature Importance (SHAP)", "🌡️ Current Live Metrics"])

with tab1:
    forecast_df = pd.DataFrame({
        'Timeline': ['Current', 'Day 1 (+24h)', 'Day 2 (+48h)', 'Day 3 (+72h)'],
        'AQI Forecast': [current_aqi, pred_day1, pred_day2, pred_day3]
    })
    
    fig = px.line(forecast_df, x='Timeline', y='AQI Forecast', markers=True, 
                  title="3-Day Predicted Air Quality Trend for Kohat",
                  labels={'AQI Forecast': 'European AQI Index'})
    fig.update_traces(line_color='#0083B0', line_width=4, marker_size=10)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Model Interpretability via SHAP")
    st.markdown("This chart reveals which environmental variables have the strongest impact on AQI forecasts.")
    try:
        image = Image.open('shap_summary.png')
        st.image(image, caption='SHAP Feature Importance Summary Plot', use_container_width=True)
    except FileNotFoundError:
        st.warning("SHAP plot image 'shap_summary.png' not found.")

with tab3:
    st.subheader("Latest Environmental Parameters in Kohat")
    st.dataframe(X_live.T.rename(columns={X_live.index[0]: 'Value'}))

