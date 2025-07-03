# 🧠 Chicago Crime Pattern Clustering & Exploration

## 🎯 Project Purpose

This project analyzes and visualizes crime patterns in the city of **Chicago (2001–2017)** using **machine learning** and **interactive maps**.

It serves two key purposes:

---

### 1. 🔗 Unsupervised Crime Pattern Clustering (KMeans)

We use the KMeans clustering algorithm to group together **geographic locations** with similar crime type profiles (e.g., areas with high theft vs. assault-heavy regions).

- Crimes are aggregated by **latitude, longitude, and type**
- Each cluster highlights a different **crime zone pattern**
- Output is visualized on an interactive **map**
- Helps law enforcement, researchers, or city officials **identify hotspots**

---

### 2. 🗺️ Interactive Raw Crime Exploration (2012–2017)

We also display a map of **individual crime incidents** (50,000 sampled records for performance), showing:

- Location (Latitude, Longitude)
- Specific Crime Type (e.g., THEFT, BATTERY, ASSAULT)
- Time Period: 2012–2017 (to balance recency and performance)

This mode provides a real-world, detailed view of what crimes occurred where.

---

## 🛠 Tech Stack

- **Python**
- **Pandas** for data cleaning & aggregation
- **Scikit-learn** for clustering
- **Streamlit** for interactive web UI
- **Plotly** for maps and charts
- **Pickle** for model storage

---

## 📌 Features

- 📍 Cluster map of crime zones (2001–2017)
- 📌 Raw crime event map (2012–2017)
- 🧮 Predict cluster based on crime pattern input
- ✅ Optimized for low RAM (sampling + caching)
- ❤️ Made with love by Ayushman Kar

---

## 📂 Dataset

Public crime datasets from:
[Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

Split into:
- Chicago_Crimes_2001_to_2004.csv
- Chicago_Crimes_2005_to_2007.csv
- Chicago_Crimes_2008_to_2011.csv
- Chicago_Crimes_2012_to_2017.csv

---

## 🚀 How to Run

1. Clone the repo
2. Install requirements  
   ```bash
   pip install -r requirements.txt
