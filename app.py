import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from streamlit_folium import folium_static
import folium

# ======================
# Load Data and Trained Model
# ======================
# Load the dataset for visualization
data = pd.read_csv("processed_noise_pollution_dataset.csv")

# Load the trained Gradient Boosting model
model = joblib.load("best_gb_model.pkl")  # Ensure you save your trained model using joblib

# ======================
# Streamlit App Layout
# ======================
st.set_page_config(page_title="Stop Noise Pollution from Honking", layout="wide")

st.title("Stop Noise Pollution from Honking Dashboard")

# Sidebar for Prediction Inputs
st.sidebar.header("Predict Honking Incidents")

# Input fields
zone = st.sidebar.selectbox("Select Zone", data["Zone"].unique())
weather = st.sidebar.selectbox("Select Weather Condition", data["Weather_Condition"].unique())
traffic_density = st.sidebar.slider("Traffic Density", 10, 100, 50)
noise_level = st.sidebar.slider("Noise Level (dB)", 50, 120, 80)

# ======================
# Prediction Logic
# ======================
st.header("Honking Prediction")

# Prepare input data for prediction
input_data = pd.DataFrame({
    "Noise_Level_dB": [noise_level],
    "Traffic_Density": [traffic_density],
    "Zone": [zone],
    "Weather_Condition": [weather]
})

# One-hot encode input data
input_data = pd.get_dummies(input_data, columns=["Zone", "Weather_Condition"], drop_first=True)

# Align input data with the model's feature names
required_columns = model.feature_names_in_  # Feature names used during model training
for col in required_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns with a value of 0

# Reorder columns to match the model's input format
input_data = input_data[required_columns]

# Predict using the trained model
prediction = model.predict(input_data)
result = "Honking Incident Likely" if prediction[0] == 1 else "No Honking Incident"

st.write(f"**Prediction Result:** {result}")

# ======================
# Visualizations
# ======================
st.header("Visualizations")

# Noise Level Distribution
st.subheader("Noise Level Distribution")
fig1 = px.histogram(data, x="Noise_Level_dB", nbins=30, title="Noise Level Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Incident Type Distribution
st.subheader("Incident Type Distribution")
fig2 = px.pie(data, names="Incident_Type", title="Incident Type Distribution")
st.plotly_chart(fig2, use_container_width=True)

# Zone-wise Noise Levels
st.subheader("Noise Levels by Zone")
fig3 = px.box(data, x="Zone", y="Noise_Level_dB", title="Noise Levels by Zone", color="Zone")
st.plotly_chart(fig3, use_container_width=True)

# ======================
# Noise Clustering Map with Dynamic Noise Levels
# ======================
st.subheader("Noise Clustering Map")

# Center the map on the overall data
map_center = [data["Latitude"].mean(), data["Longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=12)

# Add points to the map for all noise levels
for _, row in data.iterrows():
    # Set color dynamically based on noise level
    color = 'red' if row['Noise_Level_dB'] > 90 else 'green'
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"Noise Level: {row['Noise_Level_dB']} dB"
    ).add_to(m)

# Show loading spinner and render map
with st.spinner("Loading Noise Clustering Map..."):
    folium_static(m)
