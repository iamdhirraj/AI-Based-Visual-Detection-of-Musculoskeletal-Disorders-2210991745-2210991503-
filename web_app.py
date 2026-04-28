#!/usr/bin/env python3
"""Web app for musculoskeletal disorder prediction and exercise recommendation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from muscle_disorder_ai import (
    DEFAULT_OUTPUT_DIR,
    ModelBundle,
    fetch_exercise_recommendations,
    normalize_body_part,
    preprocess_single_image,
    train,
)


APP_ROOT = Path(__file__).parent
MODEL_BUNDLE_PATH = Path(os.getenv("MUSCLE_AI_MODEL_BUNDLE", str(APP_ROOT / DEFAULT_OUTPUT_DIR / "model_bundle.joblib")))
API_KEY = os.getenv("EXERCISE_API_KEY")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def ensure_model_bundle() -> ModelBundle:
    if MODEL_BUNDLE_PATH.exists():
        return joblib.load(MODEL_BUNDLE_PATH)

    print("Model bundle not found; building a sample bundle for the web app.")
    args = type("Args", (), {
        "data_dir": None,
        "output_dir": str(MODEL_BUNDLE_PATH.parent),
        "image_size": 64,
        "test_size": 0.2,
        "random_state": 42,
        "sample": True,
    })()
    bundle = train(args)
    return bundle


MODEL_BUNDLE = ensure_model_bundle()


@app.get("/")
def index():
    return render_template("index.html", api_key_loaded=bool(API_KEY), model_path=str(MODEL_BUNDLE_PATH))


@app.post("/api/predict")
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    uploaded_file = request.files["image"]
    if not uploaded_file.filename:
        return jsonify({"error": "Empty file name."}), 400

    body_part = normalize_body_part(request.form.get("body_part", "general"))
    top_k = int(request.form.get("top_k", 3))

    filename = secure_filename(uploaded_file.filename)
    suffix = Path(filename).suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        uploaded_file.save(temp_file.name)
        image_vector = preprocess_single_image(Path(temp_file.name), MODEL_BUNDLE.image_size)

    predicted_index = int(MODEL_BUNDLE.model.predict(image_vector)[0])
    predicted_label = MODEL_BUNDLE.label_encoder.inverse_transform([predicted_index])[0]

    confidence = None
    if hasattr(MODEL_BUNDLE.model.named_steps["model"], "predict_proba") and len(MODEL_BUNDLE.class_names) == 2:
        confidence = float(MODEL_BUNDLE.model.predict_proba(image_vector)[0][predicted_index])

    recommendations = fetch_exercise_recommendations(API_KEY, body_part, top_k)

    try:
        Path(temp_file.name).unlink(missing_ok=True)
    except Exception:
        pass

    return jsonify(
        {
            "prediction": {
                "label": predicted_label,
                "confidence": confidence,
                "body_part": body_part,
            },
            "recommendations": recommendations,
            "model_bundle": str(MODEL_BUNDLE_PATH),
            "api_key_loaded": bool(API_KEY),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)