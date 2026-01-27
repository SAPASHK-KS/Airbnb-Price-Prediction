import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Airbnb Price Prediction")

guests = st.number_input("Guests", 1, 20, 2)
bedrooms = st.number_input("Bedrooms", 0, 10, 1)
bathrooms = st.number_input("Bathrooms", 0, 10, 1)
beds = st.number_input("Beds", 1, 10, 1)
rating = st.slider("Rating", 1.0, 5.0, 4.5)
reviews = st.number_input("Number of Reviews", 0, 500, 10)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[bathrooms, beds, guests, reviews, rating, bedrooms]],
                              columns=['bathrooms', 'beds', 'guests', 'reviews', 'rating', 'bedrooms'])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹ {int(prediction[0])}")
