# app/app.py

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


# from src.utils.constants import embedding_dict


# Load the trained model
def load_model(model_path='models/house_price.joblib', scaler_path='models/scaler.joblib'):
    model = joblib.load(model_path)
    # scaler = joblib.load(scaler_path)
    return model


# Feature engineering function
def feature_engineering(input_features, numerical_features):
    embedding_dict = {
        'mainroad': {'yes': 1, 'no': 0},
        'guestroom': {'yes': 1, 'no': 0},
        'basement': {'yes': 1, 'no': 0},
        'hotwaterheating': {'yes': 1, 'no': 0},
        'airconditioning': {'yes': 1, 'no': 0},
        'prefarea': {'yes': 1, 'no': 0},
        'furnishingstatus': {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0},
        # 'parking': {'yes': 1, 'no': 0}  # Add this line
    }
    input_df = pd.DataFrame([input_features])

    categorical_features = [col for col in numerical_features if col in embedding_dict]
    numerical_features = [col for col in numerical_features if col not in embedding_dict]

    for column in categorical_features:
        if column in embedding_dict:
            input_df[column] = input_df[column].map(embedding_dict[column])

    # scaler = StandardScaler()
    # input_df[numerical_features] = scaler.fit_transform(input_df[numerical_features])
    print(input_df)
    return input_df


# Streamlit app interface
def main():
    st.title("House Price Prediction App")

    # Load trained model
    # Load trained model and scaler
    model = load_model()

    # User input for prediction
    st.sidebar.header("User Input Features")

    # Example input features (customize based on your model's input features)
    input_features = {
        # 'price': st.sidebar.number_input('Price', min_value=0, step=1, value=100000),
        'area': st.sidebar.number_input('Area', min_value=0, step=1, value=1000),
        'bedrooms': st.sidebar.number_input('Bedrooms', min_value=1, step=1, value=3),
        'bathrooms': st.sidebar.number_input('Bathrooms', min_value=1, step=1, value=2),
        'stories': st.sidebar.number_input('Stories', min_value=1, step=1, value=2),
        'mainroad': st.sidebar.selectbox('Main Road', ['yes', 'no']),
        'guestroom': st.sidebar.selectbox('Guest Room', ['yes', 'no']),
        'basement': st.sidebar.selectbox('Basement', ['yes', 'no']),
        'hotwaterheating': st.sidebar.selectbox('Hot Water Heating', ['yes', 'no']),
        'airconditioning': st.sidebar.selectbox('Air Conditioning', ['yes', 'no']),
        'parking': st.sidebar.number_input('Parking', min_value=0, step=1, value=1),
        'prefarea': st.sidebar.selectbox('Preferred Area', ['yes', 'no']),
        'furnishingstatus': st.sidebar.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished']),
    }

    # Convert input features to a DataFrame for prediction
    numerical_features = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
        'furnishingstatus']  # Replace with your actual feature names

    # Convert input features to a DataFrame for prediction

    # Perform feature engineering
    input_df = feature_engineering(input_features, numerical_features)
    #
    # Display user input
    st.subheader("User Input:")
    st.write(input_df)
    # Make predictions
    prediction = model.predict(input_df)[0]



    # Display prediction
    st.subheader("House Price Prediction:")
    st.write(f"${prediction:}")


if __name__ == "__main__":
    main()
