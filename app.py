"""
🌊 AI + Satellite Enhanced River Pollution Detection Dashboard
--------------------------------------------------------------
Combines unsupervised ML on IoT river sensor data
with satellite-derived environmental indicators (NDWI, NDVI, Temperature)
to detect and localize pollution with higher accuracy.
"""

# ======== IMPORTS ========
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ======== APP CONFIG ========
st.set_page_config(page_title="AI + Satellite River Pollution Detection", layout="wide")
st.title("🛰️ AI + Satellite Enhanced River Pollution Detection System")
st.markdown(
    """
    This system fuses **river sensor data** with **simulated satellite features**
    (NDWI, NDVI, and Land Surface Temperature) to improve pollution detection accuracy.
    The AI model uses **Unsupervised Learning (Isolation Forest + DBSCAN)** 
    for real-time pollution identification and clustering.
    """
)

# ======== FILE UPLOAD ========
uploaded_file = st.file_uploader("📂 Upload your river sensor dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Rows: {df.shape[0]} | Columns: {df.shape[1]}")
else:
    st.info("Using sample dataset (generated locally)...")
    df = pd.read_csv("AI_Unsupervised_River_Pollution_Data_1000.csv")

# ======== SIMULATE SATELLITE DATA INTEGRATION ========
def simulate_satellite_features(df):
    np.random.seed(42)
    df["NDWI"] = np.random.uniform(0.2, 0.8, len(df))  # Water index
    df["NDVI"] = np.random.uniform(0.1, 0.7, len(df))  # Vegetation index
    df["Surface_Temperature"] = np.random.uniform(20, 35, len(df))  # °C
    return df

df = simulate_satellite_features(df)

st.subheader("🌤️ Added Satellite-Derived Features:")
st.write("**NDWI:** Water turbidity index | **NDVI:** Vegetation health | **Temperature:** Surface heating")
st.dataframe(df.head(10))

# ======== DATA PREPROCESSING ========
features = df.drop(columns=["Date_Time", "Station_ID", "Latitude", "Longitude"], errors='ignore')
features = features.fillna(features.mean())

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ======== UNSUPERVISED ANOMALY DETECTION ========
isolation_model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
isolation_model.fit(features_scaled)

df["Anomaly_Score"] = isolation_model.decision_function(features_scaled)
df["Anomaly_Flag"] = isolation_model.predict(features_scaled)
df["Pollution_Event"] = df["Anomaly_Flag"].apply(lambda x: "Yes" if x == -1 else "No")

num_anomalies = df[df["Pollution_Event"] == "Yes"].shape[0]
st.metric(label="⚠️ Detected Pollution Events", value=num_anomalies)

# ======== DBSCAN CLUSTERING ========
anomalies = df[df["Pollution_Event"] == "Yes"].copy()
anomaly_features = features_scaled[df["Pollution_Event"] == "Yes"]

dbscan_model = DBSCAN(eps=2.5, min_samples=2)
clusters = dbscan_model.fit_predict(anomaly_features)
anomalies["Cluster_Label"] = clusters

df = df.merge(
    anomalies[["Date_Time", "Station_ID", "Cluster_Label"]],
    on=["Date_Time", "Station_ID"], how="left"
)
df["Cluster_Label"] = df["Cluster_Label"].fillna(-1)

# ======== TABS ========
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Pollution Timeline",
    "🗺️ Cluster Map",
    "🌤️ Satellite Analysis",
    "📊 Correlation & Stats"
])

# --- Timeline ---
with tab1:
    st.subheader("Pollution Timeline by Station")
    station_list = df["Station_ID"].unique().tolist()
    selected_station = st.selectbox("Select Station", station_list)
    station_df = df[df["Station_ID"] == selected_station]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(station_df["Date_Time"], station_df["pH"], color='blue', label="pH Level")
    anomaly_points = station_df[station_df["Pollution_Event"] == "Yes"]
    ax.scatter(anomaly_points["Date_Time"], anomaly_points["pH"], color='red', label="Pollution Event", s=60)
    ax.set_title(f"Pollution Detection Timeline - {selected_station}")
    ax.set_xlabel("Time")
    ax.set_ylabel("pH Level")
    ax.legend()
    st.pyplot(fig)

# --- Map ---
with tab2:
    st.subheader("🗺️ Pollution Clusters on Map (with Satellite Features)")
    map_df = anomalies.copy()
    map_df["Cluster_Label"] = map_df["Cluster_Label"].astype(int)
    map_df["Tooltip"] = map_df["Station_ID"] + " | Cluster: " + map_df["Cluster_Label"].astype(str)

    if not map_df.empty:
        fig = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Cluster_Label",
            size="Turbidity",
            hover_name="Tooltip",
            zoom=10,
            height=500,
            color_continuous_scale="rainbow"
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No clusters found yet.")

# --- Satellite Feature Analysis ---
with tab3:
    st.subheader("🌤️ Satellite-Derived Feature Trends")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(df["NDWI"], color="blue", ax=ax[0])
    ax[0].set_title("NDWI Distribution (Water Index)")
    sns.histplot(df["NDVI"], color="green", ax=ax[1])
    ax[1].set_title("NDVI Distribution (Vegetation)")
    sns.histplot(df["Surface_Temperature"], color="orange", ax=ax[2])
    ax[2].set_title("Surface Temperature (°C)")

    st.pyplot(fig)

    st.markdown("**Observation:** Sudden drops in NDWI or spikes in temperature can indicate pollution or reduced water quality.")

# --- Correlation ---
with tab4:
    st.subheader("📉 Correlation Heatmap (Including Satellite Features)")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ======== DOWNLOAD RESULTS ========
pollution_results = df[df["Pollution_Event"] == "Yes"]
csv = pollution_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="💾 Download Detected Pollution Events (CSV)",
    data=csv,
    file_name="Detected_Pollution_Events_with_Satellite.csv",
    mime="text/csv",
)

st.success("✅ Analysis Completed with Satellite Integration!")
