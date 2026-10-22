
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model once when the API starts
saved_data = joblib.load(
    "saved_model/spam_job_model.pkl"
)

model = saved_data["model"]
tfidf = saved_data["tfidf"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Spam Job Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide a valid JSON object"
        }), 400

    # Read job fields and handle missing values
    fields = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    values = []

    for field in fields:
        value = data.get(field, "")

        if not isinstance(value, str):
            return jsonify({
                "error": f"'{field}' must be a string"
            }), 400

        values.append(value)

    # Require some actual job-posting text
    if not any(value.strip() for value in values):
        return jsonify({
            "error": "Please provide job posting text"
        }), 400

    job_text = " ".join(values)

    # Transform and predict
    features = tfidf.transform([job_text])

    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]
    fraud_index = list(model.classes_).index(1)
    fraud_probability = float(probabilities[fraud_index])

    return jsonify({
        "prediction": prediction,
        "label": (
            "potentially_fraudulent"
            if prediction == 1
            else "likely_legitimate"
        ),
        "fraud_probability": round(
            fraud_probability, 4
        )
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
