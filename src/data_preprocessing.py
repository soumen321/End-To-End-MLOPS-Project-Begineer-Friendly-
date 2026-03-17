import pandas as pd
import os

RAW_DATA_PATH = "data/raw/house_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_house_data.csv"

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Example preprocessing steps
    df = df.dropna()  # Drop missing values
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"Preprocessed data saved to {PROCESSED_DATA_PATH}")
    
if __name__ == "__main__":
    preprocess()