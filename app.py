import streamlit as st
import pandas as pd
import plotly.express as px
import os
import gdown
from brain import predict_cluster, load_model_and_scaler

st.set_page_config(page_title="Chicago Crime Clustering", layout="wide")
st.title("🔍 Chicago Crime Cluster Explorer")

# ----------- 📥 Download files from Google Drive if not present -----------------
def download_if_missing(file_id, output_path):
    if not os.path.exists(output_path):
        st.info(f"📥 Downloading {output_path} ...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Required files
download_if_missing("1_aqWE9NJqa2GRj9DNj2YfEohdlMf6p9k", "crime_clustered_output.csv")
download_if_missing("1uNpuLhzMaqFkJsWNzWf1yucx9RQVfK5L", "crime_kmeans_model.pkl")
download_if_missing("1DjN_WStn7aUVxErBKuhzg53RJYmzd-NS", "crime_scaler.pkl")
download_if_missing("1v8ui1H1zwG9SPztGR3HUAUT7iL-Z4kWX", "Chicago_Crimes_2001_to_2004.csv")
download_if_missing("12isIDKoEaCSIm0VbCOdZ6c6JOToWjFRh", "Chicago_Crimes_2005_to_2007.csv")
download_if_missing("1EZqAMiO89IKCYlqry57kLlC6aNeQMSZV", "Chicago_Crimes_2008_to_2011.csv")
download_if_missing("16HjoQqK0Aop63QVuV5APThCS47MJnB5g", "Chicago_Crimes_2012_to_2017.csv")

# ----------- ✅ Load Models AFTER Download -----------------
kmeans_model, scaler_model = load_model_and_scaler()

# ----------- 📦 Load Data -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("crime_clustered_output.csv")
    df = df.sample(n=50000, random_state=42)
    df['Latitude'] = pd.to_numeric(df['Latitude'], downcast='float')
    df['Longitude'] = pd.to_numeric(df['Longitude'], downcast='float')
    df['Cluster'] = pd.to_numeric(df['Cluster'], downcast='unsigned')
    return df

@st.cache_data
def load_raw_crimes():
    files = [
        "Chicago_Crimes_2001_to_2004.csv",
        "Chicago_Crimes_2005_to_2007.csv",
        "Chicago_Crimes_2008_to_2011.csv",
        "Chicago_Crimes_2012_to_2017.csv"
    ]
    dfs = []
    for f in files:
        if os.path.exists(f):
            df_part = pd.read_csv(f, usecols=['Primary Type', 'Latitude', 'Longitude', 'Date'], low_memory=False)
            dfs.append(df_part)
    if not dfs:
        return pd.DataFrame(columns=['Primary Type', 'Latitude', 'Longitude', 'Date'])
    df = pd.concat(dfs)
    df = df.dropna(subset=['Latitude', 'Longitude', 'Primary Type'])
    df = df[df['Date'].str.contains("2012|2013|2014|2015|2016|2017")]
    return df.sample(n=50000, random_state=42)

df = load_data()
raw_df = load_raw_crimes()

# ----------- 📦 Load Static Cluster Summary -----------------
# 🔧 FIXED: Removed external file dependency
training_summary_df = df.drop(['Latitude', 'Longitude', 'Cluster'], axis=1).sum().reset_index()
training_summary_df.columns = ['Crime Type', 'Count']
training_summary_df = training_summary_df.sort_values(by='Count', ascending=False)

# ----------- 🗺️ Tabs -----------------
tabs = st.tabs(["📍 Cluster Map", "📌 Raw Crime Map"])

with tabs[0]:
    with st.sidebar:
        st.header("🔧 Filters")
        selected_clusters = st.multiselect("Select Clusters to View", sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique()))
        show_data = st.checkbox("Show Sampled Raw Data", value=False)

    filtered_df = df[df['Cluster'].isin(selected_clusters)]

    st.subheader("🗺️ Cluster Map of Crime Locations")
    fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        color_continuous_scale="Turbo",
        height=600,
        zoom=10,
        mapbox_style="open-street-map"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Cluster Summary")
    summary = filtered_df.groupby("Cluster").size().reset_index(name="Locations in Cluster")
    st.dataframe(summary, use_container_width=True)

    st.subheader("📦 Training Cluster Distribution (Dynamic)")
    st.dataframe(training_summary_df, use_container_width=True)

    if show_data:
        st.subheader("🧾 Raw Sampled Data (First 1000 Rows)")
        st.dataframe(filtered_df.head(1000), use_container_width=True)

    st.subheader("🧮 Predict Cluster for a New Location")

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

        # Show user input breakdown
        st.subheader("🧾 Your Input Crime Distribution")
        transposed = input_df.T.reset_index()
        transposed.columns = ["Crime Type", "Count"]
        transposed = transposed[transposed["Count"] > 0].sort_values(by="Count", ascending=False)

        if transposed.empty:
            st.info("You entered 0 for all crime types.")
        else:
            st.dataframe(transposed, use_container_width=True)

with tabs[1]:
    st.subheader("📌 Raw Crime Map (2012–2017 Sample)")
    if not raw_df.empty:
        fig2 = px.scatter_mapbox(
            raw_df,
            lat="Latitude",
            lon="Longitude",
            color="Primary Type",
            title="Individual Crimes by Type",
            height=600,
            zoom=10,
            mapbox_style="open-street-map"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(raw_df.head(1000), use_container_width=True)
    else:
        st.warning("❌ Raw crime CSV files are not available locally. Upload them or host online.")

# ----------- Footer -----------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ by <b>Ayushman Kar</b><br>"
    "&copy; 2025 • All Rights Reserved"
    "</div>",
    unsafe_allow_html=True
)
