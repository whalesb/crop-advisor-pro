# crop_recommender.py - Optimized Combined Version with Soil Moisture Mapping
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Page config ---
st.set_page_config(layout="wide", page_title="🌱 Smart Crop Advisor", page_icon="🌾")

# --- Helper function to map Rainfall to Soil Moisture ---
def calculate_soil_moisture(rainfall_mm):
    # Linear mapping from Rainfall to Soil Moisture percentage
    min_rainfall = 20.2
    max_rainfall = 298.6
    min_moisture = 10
    max_moisture = 90
    moisture = min_moisture + (rainfall_mm - min_rainfall) * (max_moisture - min_moisture) / (max_rainfall - min_rainfall)
    return round(moisture, 1)

# --- Data Loading and Model Training ---
@st.cache_resource
def load_data_and_train():
    # Load dataset
    df = pd.read_csv("Crop_Recommendation.csv")

    # Rename columns
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

    # Calculate comprehensive crop statistics
    crop_stats = {}
    for crop in le.classes_:
        crop_data = df[df['Crop'] == le.transform([crop])[0]]
        stats = {
            'N_avg': crop_data['N'].mean(),
            'P_avg': crop_data['P'].mean(),
            'K_avg': crop_data['K'].mean(),
            'Temperature': (crop_data['Temperature'].min(), crop_data['Temperature'].max()),
            'Humidity': (crop_data['Humidity'].min(), crop_data['Humidity'].max()),
            'pH': (crop_data['pH'].min(), crop_data['pH'].max()),
            'Rainfall': (crop_data['Rainfall'].min(), crop_data['Rainfall'].max())
        }
        crop_stats[crop] = stats

    return model, le, crop_stats

# --- Load model and data ---
model, le, crop_stats = load_data_and_train()
all_crops = sorted(le.classes_)

# --- App Layout ---
st.title("🌾 Smart Crop Advisor")
st.markdown("""
*Using average soil nutrient values (N-P-K) with customizable environmental factors*  
*Perfect for areas without soil testing capabilities*
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🌿 Crop Selection")
    selected_crop = st.selectbox("Choose your crop", all_crops, 
                                 help="Select your desired crop to see its requirements")
    
    # Auto-calculated NPK values
    with st.expander("🧪 Auto-calculated Soil Nutrients", expanded=True):
        avg_n = round(crop_stats[selected_crop]['N_avg'], 1)
        avg_p = round(crop_stats[selected_crop]['P_avg'], 1)
        avg_k = round(crop_stats[selected_crop]['K_avg'], 1)

        cols = st.columns(3)
        cols[0].metric("Nitrogen (N)", f"{avg_n} ppm")
        cols[1].metric("Phosphorus (P)", f"{avg_p} ppm")
        cols[2].metric("Potassium (K)", f"{avg_k} ppm")
        st.caption("Values calculated from historical growing data")

    st.header("🌦️ Environmental Factors")

    temp = st.slider("Temperature (°C)", 8.8, 43.7, 25.0,
                     help="Average daytime temperature during growing season")
    humidity = st.slider("Humidity (%)", 14.3, 99.9, 50.0,
                         help="Average relative humidity")
    ph = st.slider("Soil pH Level", 3.5, 9.9, 6.5,
                   help="Soil acidity/alkalinity (7 is neutral)")

    # Rainfall slider
    rainfall = st.slider("Rainfall (mm)", 20.2, 298.6, 100.0,
                         help="Annual or seasonal rainfall amount")

    # --- Dynamic Moisture Display ---
    soil_moisture = calculate_soil_moisture(rainfall)
    st.caption(f"🌧️ Rainfall: **{rainfall:.1f} mm**  |  🌿 Estimated Soil Moisture: **{soil_moisture}%**")

# --- Analysis Logic ---
if st.button("🧑‍🌾 Analyze Growing Conditions", type="primary"):
    st.header("🔍 Analysis Results")

    # Prepare input for analysis
    analysis_data = {
        'N': avg_n,
        'P': avg_p,
        'K': avg_k,
        'Temperature': temp,
        'Humidity': humidity,
        'pH': ph,
        'Rainfall': rainfall
    }

    # Suitability check
    requirements = crop_stats[selected_crop]
    violations = []
    for param in ['Temperature', 'Humidity', 'pH', 'Rainfall']:
        min_val, max_val = requirements[param]
        if not (min_val <= analysis_data[param] <= max_val):
            violations.append(f"{param}: {analysis_data[param]} (requires {min_val}-{max_val})")

    if not violations:
        st.success(f"✅ Excellent conditions for {selected_crop}!")
        st.balloons()
    else:
        st.error(f"⚠️ Potential challenges for {selected_crop}:")
        for issue in violations:
            st.write(f"- {issue}")
        st.warning("Consider adjusting environmental factors or choosing a different crop")

    # Alternative crops
    st.subheader("🌱 Alternative Suitable Crops")
    suitable_crops = []
    for crop in all_crops:
        reqs = crop_stats[crop]
        if all(
            reqs[param][0] <= analysis_data[param] <= reqs[param][1]
            for param in ['Temperature', 'Humidity', 'pH', 'Rainfall']
        ):
            suitable_crops.append((crop, reqs['N_avg'], reqs['P_avg'], reqs['K_avg']))

    if suitable_crops:
        cols = st.columns(3)
        for i, (crop, n, p, k) in enumerate(suitable_crops):
            with cols[i % 3]:
                with st.expander(f"**{crop}**", expanded=True):
                    st.metric("N-P-K", f"{n:.1f}-{p:.1f}-{k:.1f}")
                    st.caption(f"Temp: {crop_stats[crop]['Temperature'][0]}–{crop_stats[crop]['Temperature'][1]}°C")
                    st.caption(f"pH: {crop_stats[crop]['pH'][0]}–{crop_stats[crop]['pH'][1]}")
    else:
        st.warning("No crops perfectly match these conditions. Consider modifying your environment.")

# --- Additional Information Sections ---
with st.expander("📚 About This Tool", expanded=False):
    st.markdown("""
    **Smart Crop Advisor** helps farmers optimize crop selection by:
    - Using scientifically calculated average soil nutrient values
    - Analyzing environmental compatibility
    - Suggesting alternative crops when needed

    *Why fixed NPK values?*  
    Without soil sensors, we use historical averages that represent typical successful growing conditions.
    """)

with st.expander("📊 Complete Crop Requirements", expanded=False):
    st.write("Detailed growing requirements for all available crops:")
    summary_data = []
    for crop in all_crops:
        stats = crop_stats[crop]
        summary_data.append({
            'Crop': crop,
            'Avg NPK': f"{stats['N_avg']:.1f}-{stats['P_avg']:.1f}-{stats['K_avg']:.1f}",
            'Temp (°C)': f"{stats['Temperature'][0]}–{stats['Temperature'][1]}",
            'Humidity (%)': f"{stats['Humidity'][0]}–{stats['Humidity'][1]}",
            'pH Range': f"{stats['pH'][0]}–{stats['pH'][1]}",
            'Rainfall (mm)': f"{stats['Rainfall'][0]}–{stats['Rainfall'][1]}"
        })

    st.dataframe(
        pd.DataFrame(summary_data),
        height=500,
        use_container_width=True,
        column_config={
            "Avg NPK": st.column_config.TextColumn(
                "NPK (Avg)",
                help="Average Nitrogen-Phosphorus-Potassium requirements"
            )
        }
    )
