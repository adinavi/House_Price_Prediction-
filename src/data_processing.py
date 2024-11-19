
import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Drop missing values
    df.dropna(inplace=True)
    # Add any feature engineering steps here
    return df
