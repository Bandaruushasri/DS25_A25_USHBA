import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="Energy Prediction", layout="centered")
st.title("Energy Efficiency Prediction")

# Load model and scaler
model = tf.keras.models.load_model("nn_energy_model.keras")
scaler = joblib.load("scaler.pkl")

st.subheader("Enter Building Features")

features = [
    st.number_input("Relative Compactness", value=0.9),
    st.number_input("Surface Area", value=550.0),
    st.number_input("Wall Area", value=300.0),
    st.number_input("Roof Area", value=120.0),
    st.number_input("Overall Height", value=7.0),
    st.number_input("Orientation", min_value=2, max_value=5, value=2),
    st.number_input("Glazing Area", value=0.15),
    st.number_input("Glazing Area Distribution", min_value=0, max_value=5, value=1)
]

if st.button("Predict Heating Load"):
    data = np.array(features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    
    st.success(f"Predicted Heating Load: {prediction[0][0]:.2f}")


