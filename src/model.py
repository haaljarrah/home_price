import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_save_model(data_path='../data/processed_data.csv', model_path='../models/house_price.joblib'):
    # Load the processed data
    df = pd.read_csv(data_path)

    # Define features and target variable
    X = df.drop('price', axis=1)
    y = df['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train your machine learning model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R2 score: {score}')
    # Define and train your machine learning model
    model = RandomForestRegressor()
    model.fit(X, y)
    joblib.dump(model, model_path)


if __name__ == '__main__':
    train_and_save_model()
