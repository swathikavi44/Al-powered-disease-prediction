# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("AI-Powered Disease Prediction")
st.subheader("Input patient data to predict possible disease")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Collect input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
bp = st.selectbox("Blood Pressure Level", ["Low", "Normal", "High"])
cholesterol = st.selectbox("Cholesterol", ["Normal", "High"])
symptom_score = st.slider("Symptom Severity Score (0–10)", 0, 10, 5)

# Encoding categorical inputs
gender = 1 if gender == "Male" else 0
bp = {"Low": 0, "Normal": 1, "High": 2}[bp]
cholesterol = {"Normal": 0, "High": 1}[cholesterol]

# Make prediction
if st.button("Predict Disease"):
    input_data = np.array([[age, gender, bp, cholesterol, symptom_score]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Disease Category: {prediction}")
