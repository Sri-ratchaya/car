import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Load the dataset for car prediction
data = pd.read_csv('./car_data_2020_2025.csv')  # Ensure correct path
data['Name'] = data['Brand'] + ' ' + data['Model']
data.drop(['Brand', 'Model'], axis=1, inplace=True)

# Define features (X) and target (y)
X = data.drop(['Price'], axis=1)
y = data['Price']

# Define numeric and categorical columns
numeric_features = ['Year', 'Mileage']
categorical_features = ['Name', 'Fuel_Type', 'Transmission']

# Preprocessing for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors into a single column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('car_price_predictor_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# Load predefined answers (could be a CSV or a dictionary)
answers_dict = {
    'mileage': "Mileage refers to the distance a car can travel per unit of fuel. It's usually measured in kilometers per liter (km/l) or miles per gallon (mpg).",
    'fuel': "Fuel type refers to the kind of fuel a car uses. Common types are petrol, diesel, and electric.",
    'price': "The price of a car depends on various factors like the year, mileage, fuel type, and brand.",
    'year': "The year of manufacture indicates how old the car is and can affect its value."
}

# Function to find the answer to a chatbot question
def get_answer(question):
    # Convert the question to lowercase for easy matching
    question_lower = question.lower()
    
    # Search for keywords in the question
    for keyword, answer in answers_dict.items():
        if keyword in question_lower:
            return answer
    
    # Default response if no match found
    return "Sorry, I didn't understand that. Can you please rephrase?"

# Streamlit App Code
def run_streamlit_app():
    # Load the trained model
    with open('car_price_predictor_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Set up the title and description of the web app
    st.title("Car Price Prediction and Chatbot App")
    st.write("This app predicts the price of a car based on several features such as brand, year, mileage, fuel type, transmission, and more. You can also chat with me and ask questions about cars!")

    # Car Price Prediction Section
    st.header("Car Price Prediction")
    company = st.selectbox('Car Brand and Model', ['Toyota Corolla', 'Honda Civic', 'Ford Focus', 'BMW 3 Series', 'Audi A4', 'Mercedes Benz C-Class', 'Volkswagen Golf', 'Hyundai Elantra', 'Kia Seltos', 'Nissan Altima'])
    year = st.number_input('Year of Manufacture', min_value=2000, max_value=2025, step=1)
    mileage = st.number_input('Mileage (in Km)', min_value=0.0, max_value=500000.0, step=0.1)
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])

    # Prepare the input data for prediction
    query = pd.DataFrame({
        'Name': [company],
        'Year': [year],
        'Mileage': [mileage],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission]
    })

    # Predict the price when the user clicks the button
    if st.button('Predict Price'):
        predicted_price = int(model.predict(query)[0])  # Get the prediction from the model
        st.subheader(f"The predicted price of this car is ₹{predicted_price}")

    # Chatbot Section
    st.header("Chat with the Bot")
    user_question = st.text_input("Ask me a question about cars!")

    if user_question:
        answer = get_answer(user_question)
        st.write(answer)

# Running Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
