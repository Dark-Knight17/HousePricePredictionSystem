# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import sys

app = Flask(__name__)

# Load model at startup
model_path = os.path.join('model', 'house_price_model.pkl')

import os
import subprocess

MODEL_PATH = "model/house_price_model.pkl"

def load_or_train_model():
    """
    Ensures a trained model exists and returns the loaded model.
    Trains the model if it is missing.
    """
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training model...")
        try:
            subprocess.run(
                [sys.executable, "train_model.py"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Model training failed") from e

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model was not created at {MODEL_PATH}. Check train_model.py output."
        )

    return joblib.load(MODEL_PATH)


model = load_or_train_model()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template("index.html", error="Model not loaded. Contact administrator.")

    try:
        # 1. Extract form data
        # We grab all inputs. The keys in request.form match the 'name' attributes in HTML.
        form_data = request.form.to_dict()

        # 2. Convert to DataFrame
        # The pipeline expects specific column names and data types.
        # We strictly define the schema here to match training data.
        input_data = pd.DataFrame([form_data])

        # Convert numeric fields from strings (form input) to floats
        numeric_cols = [
            'longitude', 'latitude', 'housing_median_age', 
            'total_rooms', 'total_bedrooms', 'population', 
            'households', 'median_income'
        ]
        
        for col in numeric_cols:
            if col in input_data.columns:
                # 'coerce' turns invalid text into NaN, which the imputer handles
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        # 3. Predict
        # The pipeline handles all scaling and encoding internally
        prediction = model.predict(input_data)[0]

        # 4. Format Result
        formatted_price = f"${prediction:,.2f}"

        return render_template(
            "index.html", 
            prediction=formatted_price,
            original_input=form_data
        )

    except Exception as e:
        # Graceful error handling
        error_message = f"An error occurred during prediction: {str(e)}"
        return render_template("index.html", error=error_message, original_input=request.form)

if __name__ == "__main__":
    app.run(debug=True, port=5000)