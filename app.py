# crop_recommender.py - Complete All-in-One Solution
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(layout="wide", page_title="🌱 Crop Advisor Pro")

# 1. Data Loading and Model Training (Runs only once)
@st.cache_resource
def load_data_and_train():
    # Load data with correct column names
    df = pd.read_csv("Crop_Recommendation.csv")
    
    # Rename columns to simpler names for easier reference
    df = df.rename(columns={
        'Nitrogen': 'N',
        'Phosphorus': 'P', 
        'Potassium': 'K',
        'pH_Value': 'pH'
    })
    
    # Train model
    le = LabelEncoder()
    df['Crop'] = le.fit_transform(df['Crop'])
    model = RandomForestClassifier()
    model.fit(df.drop('Crop', axis=1), df['Crop'])
    
    # Calculate crop requirements
    crop_ranges = {}
    for crop in le.classes_:
        crop_data = df[df['Crop'] == le.transform([crop])[0]]
        ranges = {
            'N': (crop_data['N'].min(), crop_data['N'].max()),
            'P': (crop_data['P'].min(), crop_data['P'].max()),
            'K': (crop_data['K'].min(), crop_data['K'].max()),
            'Temperature': (crop_data['Temperature'].min(), crop_data['Temperature'].max()),
            'Humidity': (crop_data['Humidity'].min(), crop_data['Humidity'].max()),
            'pH': (crop_data['pH'].min(), crop_data['pH'].max()),
            'Rainfall': (crop_data['Rainfall'].min(), crop_data['Rainfall'].max())
        }
        crop_ranges[crop] = ranges
    
    return model, le, crop_ranges

# 2. Load everything
model, le, crop_ranges = load_data_and_train()
all_crops = sorted(le.classes_)

# 3. Streamlit App
st.title("Crop Suitability Analyzer")
st.write(f"Available crops: {', '.join(all_crops)}")

# Input Section
with st.sidebar:
    st.header("🌦️ Environmental Inputs")
    selected_crop = st.selectbox("Your desired crop", all_crops)
    n = st.slider("Nitrogen (N)", 0, 140, 50)
    p = st.slider("Phosphorus (P)", 5, 145, 50)
    k = st.slider("Potassium (K)", 5, 205, 50)
    temp = st.slider("Temperature (°C)", 8.8, 43.7, 25.0)
    humidity = st.slider("Humidity (%)", 14.3, 99.9, 50.0)
    ph = st.slider("pH Level", 3.5, 9.9, 6.5)
    rainfall = st.slider("Rainfall (mm)", 20.2, 298.6, 100.0)

# Analysis Logic
if st.button("Analyze Conditions"):
    user_inputs = {
        'N': n, 'P': p, 'K': k,
        'Temperature': temp, 'Humidity': humidity,
        'pH': ph, 'Rainfall': rainfall
    }
    
    # Primary Suitability Check
    st.header("🔍 Analysis Results")
    requirements = crop_ranges[selected_crop]
    violations = []
    
    for param, (min_val, max_val) in requirements.items():
        if not (min_val <= user_inputs[param] <= max_val):
            violations.append(
                f"{param}: {user_inputs[param]} (needs {min_val}-{max_val})"
            )
    
    if not violations:
        st.success(f"✅ Perfect for {selected_crop}!")
    else:
        st.error(f"❌ Unsuitable for {selected_crop}:")
        for issue in violations:
            st.write(f"- {issue}")
    
    # All Suitable Crops
    st.subheader("🌾 All Suitable Crops")
    suitable = []
    for crop in all_crops:
        reqs = crop_ranges[crop]
        if all(
            reqs[param][0] <= user_inputs[param] <= reqs[param][1]
            for param in user_inputs
        ):
            suitable.append(crop)
    
    if suitable:
        cols = st.columns(3)
        for i, crop in enumerate(suitable):
            cols[i%3].write(f"- {crop}")
    else:
        st.warning("No crops match these extreme conditions")

# Additional Features
with st.expander("ℹ️ About this app"):
    st.write("""
    **Crop Advisor Pro** helps farmers and agricultural professionals determine:
    - The suitability of their land for specific crops
    - Optimal growing conditions for maximum yield
    - Alternative crops that might thrive in given conditions
    
    The recommendations are based on analysis of soil nutrients (N, P, K), 
    weather conditions (temperature, humidity, rainfall), and soil pH levels.
    """)

with st.expander("📊 View Crop Requirements Table"):
    # Create a summary table of all crop requirements
    summary_data = []
    for crop in all_crops:
        ranges = crop_ranges[crop]
        summary_data.append({
            'Crop': crop,
            'N Range': f"{ranges['N'][0]} - {ranges['N'][1]}",
            'P Range': f"{ranges['P'][0]} - {ranges['P'][1]}",
            'K Range': f"{ranges['K'][0]} - {ranges['K'][1]}",
            'Temp (°C)': f"{ranges['Temperature'][0]} - {ranges['Temperature'][1]}",
            'Humidity (%)': f"{ranges['Humidity'][0]} - {ranges['Humidity'][1]}",
            'pH': f"{ranges['pH'][0]} - {ranges['pH'][1]}",
            'Rainfall (mm)': f"{ranges['Rainfall'][0]} - {ranges['Rainfall'][1]}"
        })
    
    st.dataframe(pd.DataFrame(summary_data), height=500)