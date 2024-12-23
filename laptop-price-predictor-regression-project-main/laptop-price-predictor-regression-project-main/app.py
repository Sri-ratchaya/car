import numpy as np
import pandas as pd
import streamlit as st
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your data
car_data = pd.read_csv('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/dataset.csv') 

# Extract features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Define numeric and categorical columns
numeric_features = ['Year', 'Kilometers_Driven', 'Engine', 'Power', 'Seats']
categorical_features = ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']

# Preprocessing for numeric features: Impute missing values and scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),   # Impute missing values with the mean
    ('scaler', StandardScaler())                   # Scale features to standardize them
])

# Preprocessing for categorical features: Impute missing values and apply OneHotEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),   # Impute missing values with the most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore'))      # Apply OneHotEncoding
])

# Combine both numeric and categorical transformations into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the model pipeline which first preprocesses data and then fits the RandomForest model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the model
model.fit(X, y)

# Save the model
with open('laptop_price_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Now, you can use this model to make predictions or deploy it further

# Load the pre-trained model and scaler
pipe = pickle.load(open('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/rf_model.pkl', 'rb'))
scaler = pickle.load(open('laptop-price-predictor-regression-project-main/laptop-price-predictor-regression-project-main/scaler.pkl', 'rb'))

# Load the car dataset for user inputs
  # Adjust path if necessary

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
    query = query.reshape(1, -1)  # Ensure it's a 2D array (1 sample, n features)

    # Apply scaling
    try:
        query_scaled = scaler.transform(query)
    except AttributeError:
        st.error("Scaler object is not the correct type.")
        st.stop()

    # Get the predicted price
    predicted_price = int(np.exp(pipe.predict(query_scaled)[0]))  # Assuming the model expects log-transformed prices

    # Display the predicted price
    st.title(f"The predicted price of this car is ₹{predicted_price}")
