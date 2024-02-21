import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def perform_feature_engineering(data_path, scaler_path='../models/scaler.joblib'):
    # Load the processed data
    df = pd.read_csv(data_path)

    # Implement functions for feature engineering tasks
    # Example: Standard Scaling on numerical features
    numerical_features = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
        'furnishingstatus']  # Replace with your actual feature names
    scaler = StandardScaler()

    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # scaler1 = StandardScaler()
    # df['price'] = scaler1.fit_transform(df['price'])
    # joblib.dump(scaler1, scaler_path)

    # Save the engineered features to data/processed_data.csv
    df.to_csv(data_path, index=False)


if __name__ == "__main__":
    data_path = '../data/processed_data.csv'
    perform_feature_engineering(data_path)
