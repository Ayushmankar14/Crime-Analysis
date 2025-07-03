import pickle
import pandas as pd

# Load the trained model and scaler
def load_model_and_scaler(model_path='crime_kmeans_model.pkl', scaler_path='crime_scaler.pkl'):
    """
    Load KMeans model and Scaler from pickle files.
    """
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return kmeans, scaler

# Predict the cluster of a new input row
def predict_cluster(new_data_df, kmeans, scaler):
    """
    Predicts the cluster label for new input crime data.

    Parameters:
        new_data_df (pd.DataFrame): Input data with same structure as training (crime counts per type).
        kmeans (KMeans): Pre-trained KMeans model.
        scaler (Scaler): Pre-fitted StandardScaler or MinMaxScaler.

    Returns:
        np.ndarray: Predicted cluster label(s).
    """
    scaled = scaler.transform(new_data_df)
    return kmeans.predict(scaled)
