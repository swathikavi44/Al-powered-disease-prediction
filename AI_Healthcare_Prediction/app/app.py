import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and preprocessing objects
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # Optional
encoder = pickle.load(open("encoder.pkl", "rb"))  # Optional

st.set_page_config(page_title="AI Disease Predictor", layout="centered")

st.title("🩺 AI Healthcare Disease Predictor")
st.write("Fill in the patient details below to predict the likelihood of disease.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=120, value=80)
    
    with col2:
        glucose = st.number_input("Glucose Level", min_value=50, max_value=500, value=100)
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=180)
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        symptoms = st.multiselect("Symptoms", ["Fatigue", "Fever", "Chest Pain", "Cough", "Dizziness"])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "systolic_bp": [systolic_bp],
        "diastolic_bp": [diastolic_bp],
        "glucose": [glucose],
        "cholesterol": [cholesterol],
        "smoking": [smoking],
        "symptoms_count": [len(symptoms)]
    })

    # Preprocess inputs
    input_data["gender"] = input_data["gender"].map({"Male": 0, "Female": 1})
    input_data["smoking"] = input_data["smoking"].map({"No": 0, "Yes": 1})
    
    # Scale if required
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][int(prediction)] * 100

    # Display result
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Disease Detected with {probability:.2f}% confidence.")
    else:
        st.success(f"✅ No Disease Detected with {probability:.2f}% confidence.")
