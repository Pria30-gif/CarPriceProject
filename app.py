import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title('Car Price Prediction App')

# Load the trained model
model = joblib.load('tuned_decision_tree_model.joblib')

# Define the input features based on the training data
# The order and names of these features must match the training data (X_train)
# X_train columns: 'Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Owner', 'Seller_Type_Individual', 'Transmission_Manual'

st.header("Enter Car Details:")

year = st.number_input("Year", min_value=2000, max_value=2025, value=2020)
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.0)
kms_driven = st.number_input("Kms Driven", min_value=0, value=50000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
owner = st.selectbox("Number of Owners", [0, 1, 3])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

# Map categorical inputs to the same numerical/one-hot encoded format as in training
fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
fuel_type_encoded = fuel_type_mapping[fuel_type]

seller_type_individual = 1 if seller_type == 'Individual' else 0
transmission_manual = 1 if transmission == 'Manual' else 0

# Create a button to trigger prediction
if st.button('Predict Selling Price'):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type_encoded, owner, seller_type_individual, transmission_manual]],
                                columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Owner', 'Seller_Type_Individual', 'Transmission_Manual'])

    # Make prediction
    predicted_price = model.predict(input_data)

    # Display the predicted price
    st.success(f"Predicted Selling Price: {predicted_price[0]:.2f} Lakhs")
