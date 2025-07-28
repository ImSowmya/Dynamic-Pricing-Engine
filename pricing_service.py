from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import dill
import numpy as np
import pandas as pd


class PricingRequest(BaseModel):
    Number_of_riders: int
    Number_of_drivers: int
    Vehicle_type: str
    Expected_Ride_Duration: int

    @field_validator('Number_of_riders', 'Number_of_drivers', 'Expected_Ride_Duration')
    @classmethod
    def must_be_positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive and greater than zero")
        return v

    @field_validator('Vehicle_type')
    @classmethod
    def must_be_valid_type(cls, v, info):
        valid_types = {"Premium", "Economy"}
        if v not in valid_types:
            raise ValueError(f"{info.field_name} must be one of {valid_types}")
        return v

app = FastAPI(title="Dynamic Pricing Engine API")

model_path = r"models/dynamic_pricing_model.pkl"
scaler_path = r"models/standard_scaler.pkl"
with open(model_path, "rb") as f:
    model = dill.load(f)
with open(scaler_path, "rb") as f:
    scaler = dill.load(f)

def get_vehicle_type_numeric(vehicle_type):
    vehicle_type_mapping = {"Premium": 1,"Economy": 0}
    return vehicle_type_mapping.get(vehicle_type)

@app.post("/predict-price")
async def predict_price(request: PricingRequest):
    vehicle_type_numeric = get_vehicle_type_numeric(request.Vehicle_type)
    if vehicle_type_numeric is None:
        raise ValueError("Invalid vehicle type")
    
    # Convert lists to numpy arrays for numerical computation
    number_of_riders = np.array(request.Number_of_riders)
    number_of_drivers = np.array(request.Number_of_drivers)
    
    # Fit a polynomial regression model to the data
    coefficients = np.polyfit(number_of_riders.ravel(), number_of_drivers.ravel(), deg=2)
    poly = np.poly1d(coefficients)
    
    # Calculate division feature
    division_feature = poly(number_of_riders / number_of_drivers)
    
    # Create input data array for prediction
    input_data = np.array([number_of_riders, number_of_drivers, request.Expected_Ride_Duration, vehicle_type_numeric, division_feature])
    
    # Reshape input data for compatibility with the model
    input = pd.DataFrame(input_data.reshape(1, -1))
    
    # Scale input data using scaler object
    scaled_input_data = scaler.transform(input)
    
    # Make price prediction using the model
    predicted_price = model.predict(scaled_input_data)

    return {"predicted_price": round(np.expm1(predicted_price[0]), 2)}