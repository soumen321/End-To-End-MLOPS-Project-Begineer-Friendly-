import joblib
import pandas as pd

def predict(size_sqft, bedrooms, location_score):
    model = joblib.load("models/model.pkl")
    input_data = pd.DataFrame({
        "size_sqft": [size_sqft],
        "bedrooms": [bedrooms],
        "location_score": [location_score]
    })
    predicted_price = model.predict(input_data)[0]
    return predicted_price

if __name__ == "__main__":
    result = predict(1700, 3, 8)
    print(f"Predicted house price: ${result:.2f}")