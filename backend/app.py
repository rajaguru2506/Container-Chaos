from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
model_path = os.path.join(BASE_DIR, "model", "model_v1.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Amazon Rating Prediction API is Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)

    return jsonify({
        "predicted_rating": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
