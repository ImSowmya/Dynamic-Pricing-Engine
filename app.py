import streamlit as st
import requests

st.set_page_config(page_title="Dynamic Ride Pricing Demo")
st.title("Dynamic Pricing Engine 🚗")
st.markdown("Enter ride details to get a predicted price:")

# Input fields
number_of_riders = st.number_input("Number of Riders", min_value=1, value=60)
number_of_drivers = st.number_input("Number of Drivers", min_value=1, value=25)
vehicle_type = st.selectbox("Vehicle Type", ["Premium", "Economy"])
expected_ride_duration = st.number_input("Expected Ride Duration (minutes)", min_value=1, value=30)

if st.button("Predict Price"):
    payload = {
        "Number_of_riders": number_of_riders,
        "Number_of_drivers": number_of_drivers,
        "Vehicle_type": vehicle_type,
        "Expected_Ride_Duration": expected_ride_duration
    }
    try:
        response = requests.post("http://127.0.0.1:8000/predict-price", json=payload, timeout=10)
        if response.status_code == 200:
            predicted_price = response.json()["predicted_price"]
            st.success(f"Predicted Price: ₹{predicted_price}")
        else:
            st.error(f"API error: {response.status_code}. Details: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"Could not reach backend API: {e}")