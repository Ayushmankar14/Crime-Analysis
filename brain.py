# brain.py

import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
def load_model_and_scaler(model_path='crime_kmeans_model.pkl', scaler_path='crime_scaler.pkl'):
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return kmeans, scaler

# Predict the cluster of a new input row
def predict_cluster(new_data_df, kmeans, scaler):
    """
    new_data_df: a DataFrame with same structure as training crime features (NOT including lat/lon)
    returns: cluster label
    """
    scaled = scaler.transform(new_data_df)
    return kmeans.predict(scaled)

# Load once for Streamlit/global use
kmeans_model, scaler_model = load_model_and_scaler()

# Example usage
if __name__ == "__main__":
    # Simulate a new location with crime type counts (match the training columns!)
    example_input = {
        'BATTERY': 4,
        'THEFT': 3,
        'CRIMINAL DAMAGE': 2,
        'ROBBERY': 0,
        'NARCOTICS': 0,
        # Add all columns used during training (fill with 0 if missing)
    }

    # Load the training feature names (get from the clustered output)
    train_features = pd.read_csv("crime_clustered_output.csv")
    crime_cols = train_features.drop(['Latitude', 'Longitude', 'Cluster'], axis=1).columns

    # Reformat to match model structure
    example_vector = {col: example_input.get(col, 0) for col in crime_cols}
    input_df = pd.DataFrame([example_vector])

    cluster = predict_cluster(input_df, kmeans_model, scaler_model)
    print(f"Predicted Cluster: {cluster[0]}")
