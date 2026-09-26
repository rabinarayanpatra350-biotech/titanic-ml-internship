"""
Titanic Survival Prediction API - Flask Application
=====================================================
Week 4 Task (Capstone): a simple prediction API serving the serialized
logistic regression model (model.pkl, created by train_and_serialize.py).

Endpoints
---------
GET  /health    -> service status and model metadata
GET  /          -> API documentation (JSON)
POST /predict   -> single-passenger survival prediction

Run locally:
    pip install -r requirements.txt
    python app.py
    # then, from another terminal:
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"pclass": 3, "sex": "male", "age": 22, "sibsp": 1,
              "parch": 0, "fare": 7.25, "embarked": "S", "title": "Mr"}'
"""

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the serialized model bundle once at startup
BUNDLE = joblib.load("model.pkl")
MODEL = BUNDLE["model"]
FARE_SCALER = BUNDLE["fare_scaler"]
FEATURES = BUNDLE["features"]
METRICS = BUNDLE["metrics"]


def build_feature_vector(payload):
    """Turn one passenger's raw JSON fields into the model's numeric
    feature vector, applying exactly the same transformations used at
    training time (feature parity between training and serving)."""
    row = {
        "Pclass": int(payload["pclass"]),
        "Sex": 0 if str(payload["sex"]).lower() == "female" else 1,
        "Age": float(payload["age"]),
        "SibSp": int(payload.get("sibsp", 0)),
        "Parch": int(payload.get("parch", 0)),
        "Fare": float(payload.get("fare", 14.45)),   # median fare as default
        "Embarked": str(payload.get("embarked", "S")).upper(),
        "Title": str(payload.get("title", "Mr")),
    }
    family_size = row["SibSp"] + row["Parch"] + 1

    # One-hot columns, mirroring the training-time encoding
    vector = {
        "Pclass": row["Pclass"],
        "Sex": row["Sex"],
        "Age": row["Age"],
        "SibSp": row["SibSp"],
        "Parch": row["Parch"],
        "Fare": float(FARE_SCALER.transform([[row["Fare"]]])[0, 0]),
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
    }
    for port in ("C", "Q", "S"):
        vector[f"Embarked_{port}"] = int(row["Embarked"] == port)
    for title in ("Master", "Miss", "Mr", "Mrs", "Rare"):
        vector[f"Title_{title}"] = int(row["Title"] == title)

    # Order columns exactly as the model expects them
    return pd.DataFrame([[vector[f] for f in FEATURES]], columns=FEATURES)


@app.route("/")
def index():
    return jsonify({
        "service": "Titanic Survival Prediction API",
        "task": "Week 4 capstone - AI Project Deployment",
        "author": "Rabi Narayan Patra",
        "endpoints": {
            "GET /health": "service status and model metadata",
            "POST /predict": "single-passenger survival prediction",
        },
        "predict_example": {
            "pclass": 3, "sex": "male", "age": 22, "sibsp": 1,
            "parch": 0, "fare": 7.25, "embarked": "S", "title": "Mr",
        },
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "LogisticRegression (scikit-learn)",
        "trained_on": BUNDLE["trained_on"],
        "test_metrics": METRICS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not payload or "pclass" not in payload or "age" not in payload:
        return jsonify({
            "error": "provide JSON with at least 'pclass', 'sex', 'age'; "
                     "see GET / for a full example",
        }), 400

    X = build_feature_vector(payload)
    probability = float(MODEL.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.5)
    return jsonify({
        "input": payload,
        "survival_prediction": prediction,
        "survival_probability": round(probability, 3),
        "verdict": "survives" if prediction == 1 else "does not survive",
        "model": "LogisticRegression (scikit-learn)",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
