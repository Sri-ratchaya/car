import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the dataset from the CSV file
data = pd.read_csv('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/car predict 2025 assum.csv')

# Combine `Brand` and `Model` into `Name`
data['Name'] = data['Brand'] + ' ' + data['Model']
data.drop(['Brand', 'Model'], axis=1, inplace=True)

# Load pre-trained model
with open('car_price_predictor_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title("Car Price Prediction App")

# Dynamic inputs from dataset
selected_name = st.selectbox("Select Car Name", data['Name'].unique())
selected_year = st.selectbox("Select Year", sorted(data['Year'].unique()))
selected_mileage = st.slider("Select Mileage (in kmpl)", min_value=8.0, max_value=18.0, step=0.1)
selected_fuel_type = st.selectbox("Select Fuel Type", data['Fuel_Type'].unique())
selected_transmission = st.selectbox("Select Transmission", data['Transmission'].unique())

# Prepare input for prediction
if st.button("Predict Price"):
    # Create a DataFrame for the input
    query_df = pd.DataFrame([{
        'Name': selected_name,
        'Year': selected_year,
        'Mileage': selected_mileage,
        'Fuel_Type': selected_fuel_type,
        'Transmission': selected_transmission
    }])

    # Predict the price
    predicted_price = model.predict(query_df)[0]

    # Display the result
    st.title(f"The predicted price of the car is: ₹{int(predicted_price):,}")
