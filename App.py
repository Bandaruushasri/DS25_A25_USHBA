from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("nn_energy_model")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Neural Network Regression Model API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    
    return jsonify({
        "Predicted_Heating_Load": float(prediction[0][0])
    })

if __name__ == "__main__":
    app.run(debug=True)
