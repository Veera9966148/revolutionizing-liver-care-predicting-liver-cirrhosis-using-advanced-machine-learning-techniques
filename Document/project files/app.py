from flask import Flask, render_template, request
import joblib
import logging
import os
import numpy as np

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "model.pkl"

# Load model safely
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found")

    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully.")

except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            error="Prediction service is currently unavailable."
        )

    try:
        age = float(request.form["Age"])
        gender = int(request.form["Gender"])
        total_bilirubin = float(request.form["Total_Bilirubin"])
        direct_bilirubin = float(request.form["Direct_Bilirubin"])
        alkaline_phosphotase = float(request.form["Alkaline_Phosphotase"])
        alt = float(request.form["Alamine_Aminotransferase"])
        ast = float(request.form["Aspartate_Aminotransferase"])
        total_proteins = float(request.form["Total_Protiens"])
        albumin = float(request.form["Albumin"])
        agr = float(request.form["Albumin_and_Globulin_Ratio"])

        # Basic validation
        if age <= 0:
            raise ValueError("Age must be greater than zero.")

        if total_bilirubin < 0 or direct_bilirubin < 0:
            raise ValueError("Bilirubin values cannot be negative.")

        features = np.array([[
            age,
            gender,
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            alt,
            ast,
            total_proteins,
            albumin,
            agr
        ]])

        prediction = model.predict(features)[0]

        result = (
            "⚠️ High Risk of Liver Disease"
            if prediction == 1
            else "✅ Low Risk of Liver Disease"
        )

        return render_template(
            "index.html",
            prediction=result,
            form=request.form
        )

    except ValueError as e:
        app.logger.error(f"Validation error: {e}")

        return render_template(
            "index.html",
            error=str(e),
            form=request.form
        )

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")

        return render_template(
            "index.html",
            error="Something went wrong while making the prediction.",
            form=request.form
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)