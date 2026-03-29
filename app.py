import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load your trained XGBoost model
@st.cache_resource
def load_model():
    return joblib.load(r"Models\xgb_model.joblib")

model = load_model()

st.title("❤️ HeartGuard - XGBoost Predictor")
st.caption("Binary Classification for Heart Disease Risk")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 55)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain_type = st.selectbox("Chest pain type", [0, 1, 2, 3, 4])
    bp = st.slider("BP (Resting Blood Pressure)", 80, 200, 130)
    cholesterol = st.slider("Cholesterol", 100, 600, 250)

with col2:
    max_hr = st.slider("Max HR", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise angina", [0, 1])
    st_depression = st.slider("ST depression", 0.0, 6.0, 1.0, 0.1)
    slope_of_st = st.selectbox("Slope of ST", [0, 1, 2, 3])
    number_of_vessels_fluro = st.slider("Number of vessels fluro", 0, 4, 0)
    thallium = st.selectbox("Thallium", [0, 1, 2, 3, 4, 5, 6, 7])

# Prediction
if st.button("🚀 Predict Heart Disease Risk", type="primary", use_container_width=True):
    
    input_data = pd.DataFrame({
        "Number of vessels fluro": [number_of_vessels_fluro],
        "ST depression": [st_depression],
        "Age": [age],
        "Cholesterol": [cholesterol],
        "BP": [bp],
        "Max HR": [max_hr],
        "Thallium": [thallium],
        "Sex": [1 if sex == "Male" else 0],
        "Chest pain type": [chest_pain_type],
        "Exercise angina": [exercise_angina],
        "Slope of ST": [slope_of_st]
    })

    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Convert numpy.float32 to regular Python float (this fixes the error)
    probability = float(probability)

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK** of heart disease  \nProbability: **{probability:.1%}**")
        st.snow()
    else:
        st.success(f"✅ **LOW RISK** of heart disease  \nProbability of disease: **{probability:.1%}**")
        st.balloons()

    # Fixed progress bar
    st.progress(probability)
    st.caption(f"Risk Score: {probability:.1%}")