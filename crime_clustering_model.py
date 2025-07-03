# crime_clustering_model.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os

# STEP 1: Load and concatenate all CSV files
print("Loading all crime data...")

file_paths = [
    "Chicago_Crimes_2001_to_2004.csv",
    "Chicago_Crimes_2005_to_2007.csv",
    "Chicago_Crimes_2008_to_2011.csv",
    "Chicago_Crimes_2012_to_2017.csv"
]

dfs = []
for file in file_paths:
    print(f"Reading: {file}")
    df_part = pd.read_csv(file, low_memory=False, on_bad_lines='skip')
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Combined shape: {df.shape}")

# STEP 2: Basic filtering and cleanup
print("Cleaning data...")
df = df[['Primary Type', 'Latitude', 'Longitude', 'Date']]
df = df.dropna(subset=['Latitude', 'Longitude', 'Primary Type'])

# Optional: Filter only recent years
print("Filtering recent years...")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df = df[df['Date'].dt.year >= 2012]

# STEP 3: Aggregate crime counts by location and type
print("Aggregating by location and crime type...")
grouped = df.groupby(['Latitude', 'Longitude', 'Primary Type']).size().reset_index(name='Count')

# Pivot: rows = location, columns = crime types
pivot_table = grouped.pivot_table(index=['Latitude', 'Longitude'],
                                   columns='Primary Type',
                                   values='Count',
                                   fill_value=0).reset_index()

# STEP 4: Scale the features (excluding lat/lon)
print("Scaling features...")
lat_lon = pivot_table[['Latitude', 'Longitude']]
features = pivot_table.drop(['Latitude', 'Longitude'], axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# STEP 5: Fit KMeans model
print("Training KMeans...")
k = 5  # Choose optimal k with elbow method optionally
kmeans = KMeans(n_clusters=k, random_state=42)
pivot_table['Cluster'] = kmeans.fit_predict(scaled_features)

# STEP 6: Save model and scaler
print("Saving model and scaler...")
with open("crime_kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("crime_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Optional: Export clustered data for dashboard use
pivot_table.to_csv("crime_clustered_output.csv", index=False)

print("✅ Model trained and saved! Clusters assigned.")
