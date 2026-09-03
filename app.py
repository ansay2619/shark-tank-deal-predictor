import streamlit as st
import pandas as pd
import joblib

# Load artifacts saved in cell [34]
model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Shark Tank Predictor", layout="centered")
st.title("🦈 Shark Tank Deal Predictor")
st.write("Enter pitch details to predict whether the startup gets an investment offer.")

# Input fields
ask = st.number_input("Original Ask Amount ($)", min_value=1000, value=100000, step=5000)
equity = st.number_input("Original Offered Equity (%)", min_value=0.5, max_value=100.0, value=10.0, step=0.5)
viewers = st.number_input("US Viewership (Millions)", min_value=0.1, value=5.5, step=0.1)

# Implied valuation calculation
valuation = (ask / (equity / 100.0)) if equity > 0 else 0
st.caption(f"Implied Valuation: **${valuation:,.2f}**")

industry = st.selectbox("Industry", [
    "Food and Beverage", "Fashion / Beauty", "Lifestyle / Home", 
    "Children / Education", "Fitness / Health", "Software / Tech", "Other"
])
gender = st.selectbox("Pitchers Gender", ["Male", "Female", "Mixed Team"])

if st.button("Predict Deal"):
    # Build dataframe with exact training feature names
    input_data = pd.DataFrame([{
        "Original Ask Amount": ask,
        "Original Offered Equity": equity,
        "Valuation Requested": valuation,
        "US Viewership": viewers,
        "Industry": industry,
        "Pitchers Gender": gender
    }])

    # Preprocess & predict
    input_proc = preprocessor.transform(input_data)
    pred = model.predict(input_proc)[0]
    prob = model.predict_proba(input_proc)[0][1]

    st.divider()
    if pred == 1:
        st.success(f"🎉 **Likely to get a Deal!** (Confidence: {prob * 100:.1f}%)")
    else:
        st.error(f"❌ **Unlikely to get a Deal.** (Confidence: {(1 - prob) * 100:.1f}%)")
