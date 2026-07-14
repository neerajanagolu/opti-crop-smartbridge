from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Create Flask app FIRST
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

ROOT = os.path.dirname(os.path.dirname(__file__))

# Load model files
with open(os.path.join(ROOT, "model", "crop_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(ROOT, "model", "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(ROOT, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/find")
def find():
    return render_template("find.html")


@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["nitrogen"])
    P = float(request.form["phosphorus"])
    K = float(request.form["potassium"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    values = [[N, P, K, temperature, humidity, ph, rainfall]]

    values = scaler.transform(values)

    pred = model.predict(values)
    prob = model.predict_proba(values)

    crop = le.inverse_transform(pred)[0]
    confidence = round(prob.max() * 100, 1)

    top3_idx = prob[0].argsort()[-3:][::-1]
    top3_crops = le.inverse_transform(top3_idx)
    top3_scores = prob[0][top3_idx]

    top3 = [
        {
            "crop": c,
            "score": round(s * 100, 1)
        }
        for c, s in zip(top3_crops, top3_scores)
    ]

    inputs = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    return render_template(
        "result.html",
        crop=crop,
        confidence=confidence,
        inputs=inputs,
        top3=top3
    )


application = app