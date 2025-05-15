import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("AI-Powered Disease Prediction")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
bp = st.selectbox("Blood Pressure Level", ["Low", "Normal", "High"])
cholesterol = st.selectbox("Cholesterol", ["Normal", "High"])
symptom_score = st.slider("Symptom Severity Score (0–10)", 0, 10, 5)

gender = 1 if gender == "Male" else 0
bp = {"Low": 0, "Normal": 1, "High": 2}[bp]
cholesterol = {"Normal": 0, "High": 1}[cholesterol]

if st.button("Predict Disease"):
    input_data = np.array([[age, gender, bp, cholesterol, symptom_score]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Disease Category: {prediction}")
