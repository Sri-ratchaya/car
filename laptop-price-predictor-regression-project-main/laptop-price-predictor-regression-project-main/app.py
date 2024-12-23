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
    'year': "The year of manufacture indicates how old the car is and can affect its value.",
    'engine': "The engine is the heart of your car. It converts fuel into power, and there are various types like 4-cylinder, V6, and V8.",
    'transmission': "Transmission is how the car shifts gears. Common types are manual and automatic, with automatics being easier to drive.",
    'horsepower': "Horsepower is a measure of how powerful the car's engine is. More horsepower means more acceleration and speed.",
    'torque': "Torque refers to the twisting force the engine produces. It helps with acceleration, especially at lower speeds.",
    'safety features': "Modern cars come with a variety of safety features like airbags, ABS brakes, and lane-keeping assistance to keep you safe.",
    'fuel economy': "Fuel economy refers to how much distance the car can travel per unit of fuel. A car with good fuel economy helps you save on gas.",
    'maintenance': "Maintenance is important to keep your car running smoothly. Regular checkups, oil changes, and tire rotations are key.",
    'service': "Car service includes routine checks like engine diagnostics, oil changes, and tire replacements. Always keep up with it!",
    'warranty': "A warranty is a promise from the manufacturer to fix certain car issues within a specific time or mileage limit.",
    'color': "Car colors not only make the car look good but can also affect its resale value. Popular colors like white and black often have higher resale value.",
    'model': "Car models refer to specific versions of a brand's car, like the Toyota Corolla or Honda Civic. Each model comes with different features.",
    'brand': "Car brands are the names of companies that make the cars, like Toyota, Ford, and BMW. Some are known for luxury, while others for reliability.",
    'insurance': "Car insurance protects you financially in case of an accident. It can cover damage to your car, medical bills, and liability.",
    'fuel tank capacity': "The fuel tank capacity tells you how much fuel the car can hold. Larger tanks mean fewer refuels on long trips.",
    'top speed': "Top speed is the maximum speed a car can reach. Sports cars tend to have high top speeds, while economy cars focus on fuel efficiency.",
    'acceleration': "Acceleration is how quickly a car can increase its speed. It’s important for merging into traffic or overtaking other vehicles.",
    'brake system': "The brake system is crucial for safety. Most modern cars have disc brakes, but older ones may still have drum brakes.",
    'air conditioning': "Air conditioning in cars is a lifesaver during hot days. It keeps you cool and comfortable on the road.",
    'seating capacity': "Seating capacity refers to how many people can comfortably sit in the car. A 5-seater is standard, but larger cars may seat 7 or more.",
    'comfort': "Comfort in a car comes from features like seat material, legroom, and ride quality. Luxury cars often provide the most comfort.",
    'drivetrain': "The drivetrain refers to how power is delivered to the wheels. Front-wheel drive (FWD) and all-wheel drive (AWD) are common systems.",
    'crash test rating': "Crash test ratings show how safe a car is in the event of an accident. Cars with higher ratings are generally safer.",
    'headlights': "Headlights are crucial for night driving. Modern cars have LED or HID lights, which are brighter and last longer than traditional halogen lights.",
    'taillights': "Taillights indicate your car's position to other drivers. They also have safety functions like brake lights and turn signals.",
    'navigation system': "Many modern cars come with built-in GPS navigation systems to help you find your way. Some use apps like Apple Maps or Google Maps.",
    'Bluetooth': "Bluetooth in cars lets you connect your phone wirelessly to make calls, listen to music, and more.",
    'backup camera': "A backup camera helps you see behind your car when reversing. It's a great safety feature to avoid accidents.",
    'sunroof': "A sunroof is a panel in the roof of the car that can be opened to let in fresh air or sunlight.",
    'convertible': "A convertible car has a roof that can be lowered or removed, letting you enjoy the open air while driving.",
    'cargo space': "Cargo space refers to how much room there is to store items in the car. Sedans typically have less space than SUVs.",
    'hybrid': "Hybrid cars use both an electric motor and a gasoline engine, providing better fuel efficiency and lower emissions.",
    'electric car': "Electric cars run entirely on electricity, offering an eco-friendly and low-maintenance alternative to gasoline-powered cars.",
    '4x4': "4x4 cars have four-wheel drive, providing better traction for off-road driving or in slippery conditions like snow.",
    'luxury cars': "Luxury cars offer high-end features like premium materials, advanced technology, and superior comfort.",
    'sports cars': "Sports cars are built for speed and handling. They tend to be lower to the ground and have powerful engines.",
    'SUV': "SUVs are sport utility vehicles that offer more space, higher ground clearance, and are great for off-road driving.",
    'sedan': "Sedans are popular family cars with a separate trunk and comfortable seating for 4-5 passengers.",
    'hatchback': "Hatchbacks have a rear door that swings upwards, providing more cargo space and flexibility in the back of the car.",
    'minivan': "Minivans are designed for families, offering spacious seating for seven or more and large cargo space.",
    'pickup truck': "Pickup trucks are known for their large cargo beds, making them ideal for hauling heavy loads and off-road adventures.",
    'car loan': "A car loan helps you finance your car purchase, and you'll repay it over time with interest.",
    'used cars': "Used cars are pre-owned vehicles that are sold at a lower price than new cars. They can be a great value if well-maintained.",
    'new cars': "New cars are freshly manufactured and come with the latest features, a full warranty, and no wear and tear.",
    'resale value': "Resale value is how much money you can get when selling your car. Cars from reliable brands tend to have better resale value.",
    'climate control': "Climate control systems automatically adjust the temperature inside the car, ensuring comfort no matter the outside weather.",
    'road tax': "Road tax is a fee that car owners pay to legally drive on public roads. It often depends on the car's engine size or emissions.",
    'car registration': "Car registration is the process of officially recording your vehicle with the local authorities. It's required for legal driving.",
    'car loan interest': "The interest rate on a car loan determines how much extra you'll pay over the life of the loan. Shop around for the best rates!",
    'car dealership': "Car dealerships sell both new and used cars. Some specialize in specific brands, while others offer a variety of options.",
    'car financing': "Car financing is when you take out a loan to purchase a car, and you pay it off over time with monthly payments.",
    'off-road': "Off-road vehicles are built to handle rough, uneven terrains like mud, snow, or rocks. They're perfect for adventurous drivers.",
    'performance': "Performance refers to how well a car drives, including acceleration, handling, and braking. Sports cars are often the best performers.",
    'hi': "Hi there! How can I help you today?",
    'hello': "Hello! What's on your mind? Feel free to ask me anything about cars.",
    'hey': "Hey! Ready to talk about cars? What would you like to know?",
    'greetings': "Greetings! I'm here to help with anything you need related to cars. Just ask away!",
    'how are you': "I'm doing great, thanks for asking! How about you? Ready to chat about cars?",
    'hi there': "Hi there! What car-related question can I assist you with today?",
    'what’s up': "Not much, just here to help with your car questions. How can I assist you today?",
    'good morning': "Good morning! Ready to learn more about cars? Ask away!",
    'good afternoon': "Good afternoon! How can I assist you with your car-related questions today?",
    'good evening': "Good evening! I'm here to answer all your car questions, feel free to ask anything.",
    'what is your name': "I’m your car assistant! You can call me CarBot. I'm here to help with anything car-related.",
    'tell me more about you': "I'm CarBot, your friendly assistant! I can help you with car prices, features, and all sorts of car-related questions. Just ask!"
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
