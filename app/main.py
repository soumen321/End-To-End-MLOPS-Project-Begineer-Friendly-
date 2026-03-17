from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

class HouseFeatures(BaseModel):
    size_sqft: float
    bedrooms: int
    location_score: float
    
@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API!"}    
    
@app.post("/predict")
def predict(features: HouseFeatures):
    input_data = pd.DataFrame({
        "size_sqft": [features.size_sqft],
        "bedrooms": [features.bedrooms],
        "location_score": [features.location_score]
    })
    predicted_price = model.predict(input_data)[0]
    
    return {"predicted_price": predicted_price}