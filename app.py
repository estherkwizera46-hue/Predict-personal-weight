# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 18:54:30 2025

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 18:04:36 2025

@author: user
"""

import pickle
import pandas as pd
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('C:/Users/user/Desktop/My dataset/weight_data.sav', 'rb'))

# Prediction function
def weight_data_prediction(Gender, Workout_Type, Age, Height_m):
    # Create DataFrame from input
    new_data = pd.DataFrame([{
        'Gender': Gender,
        'Workout_Type': Workout_Type,
        'Age': Age,
        'Height (m)': Height_m
    }])
    
    # Predict weight
    predicted_weight = loaded_model.predict(new_data)
    
    # Return the prediction
    return predicted_weight[0]

# Main Streamlit app
def main():
    st.title("Personal Weight Prediction")

    # Input fields for all features
    Gender = st.text_input('Gender (e.g., 1 for Male, 0 for Female)')
    Workout_Type = st.text_input('Workout Type (e.g., 1, 2, or 3)')
    Age = st.text_input('Age (e.g., 25)')
    Height_m = st.text_input('Height (m) (e.g., 1.75)')

    if st.button('Predict Weight'):
        try:
            # Convert inputs to numeric types
            Gender = int(Gender)
            Workout_Type = int(Workout_Type)
            Age = int(Age)
            Height_m = float(Height_m)

            # Call the prediction function
            weight = weight_data_prediction(Gender, Workout_Type, Age, Height_m)

            st.success(f'The predicted weight for the person is: {weight:.2f} kg')
        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")

if __name__ == "__main__":
    main()