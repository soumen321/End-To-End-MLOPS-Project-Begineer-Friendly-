import pandas as pd
import os
import joblib
import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

PROCESSED_DATA_PATH = "data/processed/processed_house_data.csv"
MODEL_PATH = "models/model.pkl"
PARMS_PATH = "params.yaml"

def train():
    with open(PARMS_PATH, "r") as f:
        params = yaml.safe_load(f)
        
    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]  
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    X = df[["size_sqft", "bedrooms", "location_score"]]  
    y = df["price"]  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)   
    
    mlflow.set_experiment("house_price_prediction")
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        
        mlflow.log_artifact(MODEL_PATH)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model saved to {MODEL_PATH}")
        print(f"MSE: {mse}")
        print(f"R2 Score: {r2}")
        
if __name__ == "__main__":
    train()        

