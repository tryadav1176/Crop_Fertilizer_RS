from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# --------------------------
# Load Crop Model and Scaler
# --------------------------
with open("model_crop.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Crop dictionary
crop_dict = {
    0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas',
    5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate',
    10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon',
    15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton',
    20: 'jute', 21: 'coffee'
}

# -----------------------------
# Load Fertilizer Model
# -----------------------------
with open("model_fertilizer.pkl", "rb") as f:
    fert_model = pickle.load(f)


# -----------------------------
# Prediction Functions
# -----------------------------
def crop_rec(N, P, K, temp, hum, ph, rain):
    features = np.array([[N, P, K, temp, hum, ph, rain]])
    transformed = scaler.transform(features)
    prediction = crop_model.predict(transformed)[0]
    return f"{crop_dict[prediction]} is the best crop to grow."


def fertilizer_rec(temp, hum, mois, soil, crop, N, K, P):
    # Numeric encoding must match training
    features = np.array([[temp, hum, mois, soil, crop, N, K, P]])
    prediction = fert_model.predict(features)
    return f"Recommended Fertilizer: {prediction}"


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    data = request.form
    result = crop_rec(
        float(data["N"]), float(data["P"]), float(data["K"]),
        float(data["temp"]), float(data["hum"]),
        float(data["ph"]), float(data["rain"])
    )
    return render_template("index.html", crop_result=result)


@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    data = request.form
    result = fertilizer_rec(
        float(data["temp"]), float(data["hum"]), float(data["mois"]),
        int(data["soil"]), int(data["crop"]),
        float(data["N"]), float(data["K"]), float(data["P"])
    )
    return render_template("index.html", fert_result=result)


if __name__ == "__main__":
    app.run(debug=True)
