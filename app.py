# --- Imports & Model Loading ---
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("model.pkl")  # consistency with notebook save

# --- App Title ---
st.title("GlycoAID – Diabetes Risk Prediction System")
st.write("This app predicts diabetes risk based on patient diagnostic data.")

# --- Sidebar Inputs (Clean UI) ---
st.sidebar.header("Patient Inputs")
st.sidebar.text_input("Patient Name", key="patient_name")  # Added patient name

preg = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=100)
bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=1000, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# --- Initialize session state for patient records ---
if "records" not in st.session_state:
    st.session_state.records = []

# --- Prediction Block ---
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    patient_name = st.session_state.patient_name

    if prediction == 0:
        st.success(f"{patient_name}: Low Risk of Diabetes ({prob:.2f}% probability)")
    else:
        st.error(f"{patient_name}: High Risk of Diabetes ({prob:.2f}% probability)")

    # --- Risk Probability Progress Bar ---
    st.write("Risk Probability")
    st.progress(int(prob))
    st.metric(label="Probability (%)", value=f"{prob:.2f}")

    # Save record
    st.session_state.records.append({
        "Patient Name": patient_name,
        "Pregnancies": preg,
        "Glucose": glucose,
        "Blood Pressure": bp,
        "Skin Thickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DPF": dpf,
        "Age": age,
        "Prediction": "High Risk" if prediction == 1 else "Low Risk",
        "Probability (%)": f"{prob:.2f}"
    })

# --- Feature Importance Visualization ---
if st.button("Show Feature Importance"):
    features = ["Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age"]
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# --- Model Comparison Table ---
st.subheader("Model Comparison")
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Logistic Regression", "Best Model"],
    "Accuracy": [0.78, 0.82, 0.85],
    "F1-score": [0.75, 0.80, 0.83]
})
st.table(comparison)

# --- Download & View Patient Records ---
if st.session_state.records:
    df_records = pd.DataFrame(st.session_state.records)

    # Show records in the app
    st.subheader("Saved Patient Records")
    st.dataframe(df_records)   # interactive table view

    # Allow staff to download as CSV
    st.download_button(
        label="Download Patient Records",
        data=df_records.to_csv(index=False).encode("utf-8"),
        file_name="patient_records.csv",
        mime="text/csv"
    )
