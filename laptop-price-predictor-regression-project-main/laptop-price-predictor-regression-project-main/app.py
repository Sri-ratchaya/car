import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
car_data = pd.read_csv('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/dataset.csv')

# Clean numeric columns
car_data['Engine'] = car_data['Engine'].str.extract('(\d+)').astype(float)  # Extract numeric values
car_data['Power'] = car_data['Power'].str.extract('(\d+)').astype(float)    # Extract numeric values
car_data['Year'] = pd.to_numeric(car_data['Year'], errors='coerce')         # Ensure Year is numeric
car_data['Kilometers_Driven'] = pd.to_numeric(car_data['Kilometers_Driven'], errors='coerce')
car_data['Seats'] = pd.to_numeric(car_data['Seats'], errors='coerce')

# Fill missing values in numeric columns
numeric_features = ['Year', 'Kilometers_Driven', 'Engine', 'Power', 'Seats']
for col in numeric_features:
    car_data[col].fillna(car_data[col].mean(), inplace=True)

# Extract features and target
X = car_data.drop('Price', axis=1)
y = car_data['Price']

# Define numeric and categorical columns
categorical_features = ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),   # Impute missing values with the mean
    ('scaler', StandardScaler())                  # Standardize features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),   # Impute missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))      # OneHotEncode categories
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

model.fit(X, y)

# Save the model
with open('laptop_price_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the pre-trained model
pipe = pickle.load(open('laptop_price_predictor_model.pkl', 'rb'))

# Streamlit app
st.title("Car Price Prediction App")
st.write("This app predicts the price of a car based on several features like brand, year, kilometers driven, and more.")

# User inputs
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

if st.button('Predict Price'):
    # Prepare the data for prediction
    query = pd.DataFrame([[
        company, car_type, year, kilometers, fuel_type, transmission, owner_type, engine, power, seats
    ]], columns=X.columns)  # Ensure query has the same column names as the training data

    # Transform the query using the pre-trained pipeline
    try:
        predicted_price = pipe.predict(query)[0]
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
        st.stop()

    # Display the predicted price
    st.title(f"The predicted price of this car is ₹{predicted_price:,.2f}")

