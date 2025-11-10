"""
AI-Based Unsupervised River Pollution Detection Dashboard 🌊
------------------------------------------------------------
Streamlit app that detects and visualizes river pollution sources
using upstream–downstream sensor data and unsupervised learning (Isolation Forest + DBSCAN).
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
st.set_page_config(page_title="AI River Pollution Detection", layout="wide")
st.title("🌊 AI-Based Unsupervised River Pollution Detection System")
st.markdown(
    "This system uses **unsupervised machine learning** (Isolation Forest + DBSCAN)** "
    "to detect and group pollution events in river networks based on multi-sensor data."
)

# ======== FILE UPLOAD ========
uploaded_file = st.file_uploader("📂 Upload your river sensor dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded successfully! Rows: {df.shape[0]} | Columns: {df.shape[1]}")
else:
    st.info("Using sample dataset...")
    df = pd.read_csv("AI_Unsupervised_River_Pollution_Data_1000.csv")

# ======== DATA PREVIEW ========
with st.expander("🔍 View Dataset Sample"):
    st.dataframe(df.head())

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

# Merge cluster info
df = df.merge(
    anomalies[["Date_Time", "Station_ID", "Cluster_Label"]],
    on=["Date_Time", "Station_ID"], how="left"
)
df["Cluster_Label"] = df["Cluster_Label"].fillna(-1)

# ======== VISUALIZATIONS ========
tab1, tab2, tab3 = st.tabs(["📈 Pollution Timeline", "📊 Cluster Map", "📉 Correlation & Stats"])

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

with tab2:
    st.subheader("🗺️ Pollution Clusters on River Map")
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

with tab3:
    st.subheader("📉 Correlation Heatmap of Water Parameters")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Pollution Events by Station")
    event_count = df.groupby("Station_ID")["Pollution_Event"].value_counts().unstack().fillna(0)
    st.bar_chart(event_count)

# ======== DOWNLOAD RESULTS ========
pollution_results = df[df["Pollution_Event"] == "Yes"]
csv = pollution_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="💾 Download Detected Pollution Events (CSV)",
    data=csv,
    file_name="Detected_Pollution_Events.csv",
    mime="text/csv",
)

st.success("✅ Analysis Completed Successfully!")
