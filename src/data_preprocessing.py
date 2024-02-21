# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from src.utils.constants import embedding_dict

def handle_missing_values(data):
    """
    Handle missing values in the dataset.

    Parameters:
    - data: DataFrame, input dataset.

    Returns:
    - DataFrame, dataset after handling missing values.
    """

    return data.dropna()


def encode_categorical_variables(data, categorical_columns):
    """
    Encode categorical variables in the dataset using the provided embedding dictionary.

    Parameters:
    - data: DataFrame, input dataset.
    - categorical_columns: list, names of categorical columns.
    - embedding_dict: dict, embedding dictionary for categorical values.

    Returns:
    - DataFrame, dataset after encoding.
    """
    # Replace categorical values with numerical representations from embedding_dict
    for column in categorical_columns:
        if column in embedding_dict:
            data[column] = data[column].map(embedding_dict[column])
    return data


def split_dataset(data, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - data: DataFrame, input dataset.
    - target_column: str, the name of the target column.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, seed for random number generation.

    Returns:
    - tuple, (X_train, X_test, y_train, y_test).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_processed_data(data, file_path='data/processed_data.csv'):
    """
    Save the processed dataset to a CSV file.

    Parameters:
    - data: DataFrame, processed dataset.
    - file_path: str, path to the output CSV file.

    Returns:
    - None
    """
    data.to_csv(file_path, index=False)

# Example usage:
if __name__ == "__main__":
    raw_data = pd.read_csv('../data/raw_data.csv')
    processed_data = handle_missing_values(raw_data)
    processed_data = encode_categorical_variables(processed_data, categorical_columns=['mainroad', 'guestroom','basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'])
    X_train, X_test, y_train, y_test = split_dataset(processed_data, target_column='price')
    save_processed_data(processed_data, file_path='../data/processed_data.csv')

