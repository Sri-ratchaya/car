from dotenv import load_dotenv
import openai
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import pickle

# Load environment variables from .env file
load_dotenv()
import os

# Set the environment variable directly
os.environ["OPENAI_API_KEY"] = "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"

# Now you can access the API key using os.getenv or directly pass it to the LangChain model

# Function to interact with OpenAI
def query_openai(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or whichever engine you prefer
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Example for loading and preprocessing the car data
car_data = pd.read_csv('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/car_data_2020_2025.csv')

# Combining 'Brand' and 'Model' into a 'Name' column
car_data['Name'] = car_data['Brand'] + ' ' + car_data['Model']

# Define the features (X) and the target (y)
X = car_data[['Name', 'Year', 'Mileage', 'Fuel_Type', 'Transmission']]
y = car_data['Price']

# Preprocessing steps
numeric_features = ['Year', 'Mileage']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Name', 'Fuel_Type', 'Transmission']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Saving the trained model
with open('car_price_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# LangChain Setup for OpenAI integration
prompt = """
Given a car description with features such as the car's brand/model, fuel type, transmission, mileage, and year, predict the car price.
For instance, for a car with the following details:

Car Name: {car_name}
Fuel Type: {fuel_type}
Transmission: {transmission}
Mileage: {mileage} km
Year: {year}

What would be the expected price of the car in USD?

"""

template = PromptTemplate(input_variables=["car_name", "fuel_type", "transmission", "mileage", "year"], template=prompt)
llm = OpenAI(temperature=0, model="gpt-4")

# Create an LLMChain
llm_chain = LLMChain(prompt=template, llm=llm)

# Streamlit Interface
st.title("Car Price Prediction")
car_name = st.text_input("Car Name (e.g., Ford Mustang)")
fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'Electric', 'Hybrid'])
transmission = st.selectbox("Transmission", options=['Automatic', 'Manual'])
mileage = st.number_input("Mileage (in km)", min_value=0)
year = st.number_input("Year of Manufacture", min_value=1900, max_value=2024)

if st.button("Predict Car Price"):
    # Use LangChain to generate the car price prediction
    car_description = {
        'car_name': car_name,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'mileage': mileage,
        'year': year
    }

    # Use the LangChain LLM to predict the price
    price_prediction = llm_chain.run(car_description)
    
    st.subheader(f"Predicted Car Price (LangChain Model): {price_prediction}")

    # Alternatively, you can use your trained Random Forest model
    input_data = pd.DataFrame({
        'Name': [car_name],
        'Year': [year],
        'Mileage': [mileage],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission]
    })
    predicted_price = model.predict(input_data)[0]
    st.write(f"Predicted Car Price (Trained Random Forest Model): {predicted_price:.2f} USD")

    # Optionally, query OpenAI (for extra info or interaction)
    openai_response = query_openai(f"Based on the car details: {car_description}, what would be the expected price of the car?")
    st.write(f"OpenAI's Take on Car Price: {openai_response}")
