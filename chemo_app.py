import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load all models and scaler
chemo_model = joblib.load("chemo.pkl")
hormone_model = joblib.load("hormone.pkl")
radio_model = joblib.load("radio.pkl")
scaler = joblib.load("scaler.pkl")

# Set up page
st.set_page_config(page_title="Breast Cancer Treatment Predictor", page_icon="🧬")
st.title("🎗️ Breast Cancer Treatment Recommender")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Select the treatment type to predict.
    2. Enter clinical data.
    3. Click **Predict** to view the recommendation and explanation.
    """)

# Select treatment
treatment_choice = st.selectbox("Select Treatment to Predict", ["Chemotherapy", "Hormone Therapy", "Radiotherapy"])

# Clinical inputs
age = st.number_input("Age at Diagnosis", min_value=20, max_value=100, value=50)
tumor_size = st.number_input("Tumor Size (mm)", min_value=1.0, max_value=100.0, value=20.0)
tumor_stage = st.selectbox("Tumor Stage", [1, 2, 3, 4], index=0)
grade = st.selectbox("Histologic Grade", [1, 2, 3], index=0)
er = st.radio("ER Status", ["Positive", "Negative"], index=0)
pr = st.radio("PR Status", ["Positive", "Negative"], index=0)
her2 = st.radio("HER2 Status", ["Positive", "Negative"], index=0)
nodes = st.number_input("Lymph Nodes Positive", min_value=0, max_value=50, value=0)
meno = st.radio("Menopausal State", ["Pre", "Post"], index=0)

# Encode categorical
er = 1 if er == "Positive" else 0
pr = 1 if pr == "Positive" else 0
her2 = 1 if her2 == "Positive" else 0
meno = 1 if meno == "Post" else 0

# Prepare input
input_dict = {
    'Age at Diagnosis': [age],
    'Tumor Size': [tumor_size],
    'Tumor Stage': [tumor_stage],
    'Neoplasm Histologic Grade': [grade],
    'ER Status': [er],
    'PR Status': [pr],
    'HER2 Status': [her2],
    'Lymph nodes examined positive': [nodes],
    'Inferred Menopausal State': [meno]
}
input_df = pd.DataFrame(input_dict)
feature_order = list(input_dict.keys())
input_df = input_df[feature_order]
input_scaled = scaler.transform(input_df)

# Select model
if treatment_choice == "Chemotherapy":
    model = chemo_model
elif treatment_choice == "Hormone Therapy":
    model = hormone_model
else:
    model = radio_model

# Predict and show SHAP
if st.button("Predict"):
    try:
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        if prediction == 1:
            st.success(f"✅ {treatment_choice} is likely **recommended**. ({probability:.2f}% confidence)")
        else:
            st.info(f"❌ {treatment_choice} may **not** be needed. ({100 - probability:.2f}% confidence)")

        

    except Exception as e:
        st.error(f"⚠️ Error during prediction or SHAP plotting: {e}")
