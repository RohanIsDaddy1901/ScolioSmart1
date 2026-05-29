# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Scoliosis Trend & Cobb Predictor", layout="centered")

st.title("Scoliosis Trend and Cobb Angle Predictor")
st.write(
    "Prototype tool that uses a classifier to predict trend "
    "(improving / stable / worsening) and a regressor to estimate a future Cobb angle."
)

# Load models
@st.cache_resource
def load_models():
    clf = joblib.load("trend_classifier.pkl")
    reg = joblib.load("cobb_regressor.pkl")
    return clf, reg

clf, reg = load_models()

st.subheader("Input patient / session data")

col1, col2 = st.columns(2)

with col1:
    current_cobb = st.number_input("Current Cobb angle (°)", min_value=0.0, max_value=100.0, value=25.0)
    lumbar_gyro = st.number_input("Lumbar gyro (deg)", value=3.0)
    thoracic_gyro = st.number_input("Thoracic gyro (deg)", value=6.5)
    cervical_gyro = st.number_input("Cervical gyro (deg)", value=1.4)

with col2:
    wear_time = st.number_input("Brace wear time (hours / day)", min_value=0.0, max_value=24.0, value=16.0)
    pressure = st.number_input("Brace pressure (N)", min_value=0.0, value=37.0)
    age = st.number_input("Age (years)", min_value=1, max_value=25, value=13)
    risser = st.number_input("Risser score (0–5)", min_value=0, max_value=5, value=2)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "current_cobb": current_cobb,
        "lumbar_gyro": lumbar_gyro,
        "thoracic_gyro": thoracic_gyro,
        "cervical_gyro": cervical_gyro,
        "wear_time": wear_time,
        "pressure": pressure,
        "age": age,
        "risser": risser,
    }])

    trend_pred = clf.predict(input_df)[0]
    cobb_pred = reg.predict(input_df)[0]

    st.subheader("Prediction")
    st.write(f"**Predicted trend**: {trend_pred}")
    st.write(f"**Predicted future Cobb angle**: {cobb_pred:.2f}°")

    st.caption("This is a student prototype and not a clinical tool.")
