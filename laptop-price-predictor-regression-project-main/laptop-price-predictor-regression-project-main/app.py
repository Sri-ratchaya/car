import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle

# Load the pre-trained model and scaler
pipe = pickle.load(open('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/rf_model.pkl', 'rb'))
scaler = pickle.load(open('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/scaler.pkl', 'rb'))

# Load the car dataset for user inputs
car_data = pd.read_csv('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/dataset.csv')  # Adjust path if necessary

# Set up the title and description of the web app
st.title("Car Price Prediction App")
st.write("This app predicts the price of a car based on several features such as brand, year, kilometers driven, and more.")

# Collect user inputs for the car's features
company = st.selectbox('Car Brand', car_data['Name'].unique())
car_type = st.selectbox('Car Type', car_data['Location'].unique())
year = st.number_input('Year of Manufacture', min_value=2000, max_value=2024, step=1)
kilometers = st.number_input('Kilometers Driven', min_value=0, max_value=500000, step=1000)
fuel_type = st.selectbox('Fuel Type', car_data['Fuel_Type'].unique())
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner_type = st.selectbox('Owner Type', car_data['Owner_Type'].unique())
engine = st.number_input('Engine Capacity (in CC)', min_value=800, max_value=5000, step=50)
power = st.number_input('Power (in bhp)', min_value=50, max_value=500, step=1)
seats = st.number_input('Seats', min_value=2, max_value=10, step=1)

# Predict the price when the user clicks the button
if st.button('Predict Price'):
    # Prepare the data for prediction
    query = np.array([company, car_type, year, kilometers, fuel_type, transmission, owner_type, engine, power, seats])
    query = query.reshape(1, 10)  # Ensure the correct shape for the model

    # Apply scaling to the query (this assumes the scaler has been properly trained and is loaded correctly)
    query_scaled = scaler.transform(query)

    # Get the predicted price
    predicted_price = int(np.exp(pipe.predict(query_scaled)[0]))  # Assuming the model expects log-transformed prices

    # Display the predicted price
    st.title(f"The predicted price of this car is ₹{predicted_price}")
