# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from brain import predict_cluster, kmeans_model, scaler_model

st.set_page_config(page_title="Chicago Crime Clustering", layout="wide")
st.title("🔍 Chicago Crime Cluster Explorer")

# Load clustered data with memory-safe strategy
@st.cache_data
def load_data():
    df = pd.read_csv("crime_clustered_output.csv")
    df = df.sample(n=50000, random_state=42)
    df['Latitude'] = pd.to_numeric(df['Latitude'], downcast='float')
    df['Longitude'] = pd.to_numeric(df['Longitude'], downcast='float')
    df['Cluster'] = pd.to_numeric(df['Cluster'], downcast='unsigned')
    return df

# Load raw crime records for individual events
@st.cache_data
def load_raw_crimes():
    files = [
        "Chicago_Crimes_2001_to_2004.csv",
        "Chicago_Crimes_2005_to_2007.csv",
        "Chicago_Crimes_2008_to_2011.csv",
        "Chicago_Crimes_2012_to_2017.csv"
    ]
    dfs = [pd.read_csv(f, usecols=['Primary Type', 'Latitude', 'Longitude', 'Date'], low_memory=False) for f in files]
    df = pd.concat(dfs)
    df = df.dropna(subset=['Latitude', 'Longitude', 'Primary Type'])
    df = df[df['Date'].str.contains("2012|2013|2014|2015|2016|2017")]
    return df.sample(n=50000, random_state=42)

df = load_data()
raw_df = load_raw_crimes()

# Tabs: Cluster Map vs Raw Crime Map
tabs = st.tabs(["📍 Cluster Map", "📌 Raw Crime Map"])

with tabs[0]:
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Filters")
        selected_clusters = st.multiselect("Select Clusters to View", sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique()))
        show_data = st.checkbox("Show Sampled Raw Data", value=False)

    # Filter by cluster
    filtered_df = df[df['Cluster'].isin(selected_clusters)]

    # Plot cluster map
    st.subheader("🗺️ Cluster Map of Crime Locations")
    fig = px.scatter_map(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        color_continuous_scale="Turbo",
        height=600,
        zoom=10
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster summary
    st.subheader("📊 Cluster Summary")
    summary = filtered_df.groupby("Cluster").size().reset_index(name="Locations in Cluster")
    st.dataframe(summary, use_container_width=True)

    if show_data:
        st.subheader("🧾 Raw Sampled Data (First 1000 Rows)")
        st.dataframe(filtered_df.head(1000), use_container_width=True)

    # Optional: Predict cluster from user input
    st.subheader("🧮 Predict Cluster for a New Location")

    # Load crime type columns from full dataset (once)
    @st.cache_data
    def load_crime_types():
        full_df = pd.read_csv("crime_clustered_output.csv")
        return full_df.drop(['Latitude', 'Longitude', 'Cluster'], axis=1).columns

    crime_types = load_crime_types()
    user_input = {}

    cols = st.columns(3)
    for idx, crime in enumerate(crime_types):
        user_input[crime] = cols[idx % 3].number_input(f"{crime}", min_value=0, value=0)

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([{col: user_input.get(col, 0) for col in crime_types}])
        cluster = predict_cluster(input_df, kmeans_model, scaler_model)
        st.success(f"📌 This location belongs to **Cluster {cluster[0]}**")

with tabs[1]:
    st.subheader("📌 Raw Crime Map (2012–2017 Sample)")
    fig2 = px.scatter_map(
        raw_df,
        lat="Latitude",
        lon="Longitude",
        color="Primary Type",
        title="Individual Crimes by Type",
        height=600,
        zoom=10
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(raw_df.head(1000), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ by <b>Ayushman Kar</b><br>"
    "&copy; 2025 • All Rights Reserved"
    "</div>",
    unsafe_allow_html=True
)
