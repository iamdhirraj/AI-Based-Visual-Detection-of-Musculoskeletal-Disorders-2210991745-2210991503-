#!/usr/bin/env python3
"""Train an image-based musculoskeletal classifier and recommend exercises.

Usage:
  python muscle_disorder_ai.py train --data-dir path/to/dataset --output-dir muscle_ai_output
  python muscle_disorder_ai.py train --sample --output-dir muscle_ai_output
  python muscle_disorder_ai.py predict --image path/to/image.png --model muscle_ai_output/model_bundle.joblib --body-part wrist

Dataset layout:
  dataset/
    normal/
      image1.png
      image2.png
    abnormal/
      image3.png
      image4.png

The prediction script uses ExerciseAPI.dev for exercise recommendations when an
API key is provided through the EXERCISE_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_OUTPUT_DIR = Path("muscle_ai_output")
EXERCISE_API_BASE = "https://api.exerciseapi.dev/v1"


@dataclass
class ModelBundle:
    model: object
    label_encoder: LabelEncoder
    image_size: tuple[int, int]
    feature_dim: int
    class_names: list[str]


BODY_PART_QUERY_MAP = {
    "shoulder": {"q": "shoulder mobility", "muscle": "shoulder", "category": "mobility"},
    "wrist": {"q": "wrist stretch", "muscle": "forearm", "category": "stretching"},
    "hand": {"q": "hand mobility", "muscle": "forearm", "category": "mobility"},
    "finger": {"q": "finger mobility", "muscle": "forearm", "category": "mobility"},
    "elbow": {"q": "elbow mobility", "muscle": "arm", "category": "stretching"},
    "humerus": {"q": "upper arm mobility", "muscle": "shoulder", "category": "mobility"},
    "forearm": {"q": "forearm stretch", "muscle": "forearm", "category": "stretching"},
    "general": {"q": "gentle mobility", "muscle": "full body", "category": "mobility"},
}

LOCAL_RECOMMENDATIONS = {
    "shoulder": [
        {
            "name": "Shoulder Pendulum Swings",
            "overview": "Gentle swinging movement to reduce stiffness and improve circulation.",
            "instructions": [
                "Lean forward with the unaffected hand supported on a table.",
                "Let the affected arm hang loosely and swing in small circles.",
                "Perform for 30 to 60 seconds in each direction.",
            ],
            "safetyInfo": "Stop if pain increases sharply or you feel instability.",
        },
        {
            "name": "Wall Slides",
            "overview": "Improves shoulder range of motion using a controlled upward slide.",
            "instructions": [
                "Stand with your back against a wall.",
                "Slide your arms upward slowly while keeping elbows and wrists comfortable.",
                "Repeat 8 to 12 times.",
            ],
            "safetyInfo": "Keep movements pain-free and avoid shrugging.",
        },
    ],
    "wrist": [
        {
            "name": "Wrist Flexor Stretch",
            "overview": "A gentle stretch for the front of the forearm and wrist.",
            "instructions": [
                "Extend one arm forward with the palm facing up.",
                "Use the other hand to gently pull the fingers backward.",
                "Hold for 20 to 30 seconds and repeat 3 times.",
            ],
            "safetyInfo": "Do not force the wrist into discomfort or numbness.",
        },
        {
            "name": "Tendon Glides",
            "overview": "Helps improve hand and finger mobility with a simple sequence of hand positions.",
            "instructions": [
                "Move from a straight hand to a hook fist, then a full fist.",
                "Pause 2 to 3 seconds in each position.",
                "Repeat 5 to 10 times.",
            ],
            "safetyInfo": "Keep the motion smooth and avoid painful squeezing.",
        },
    ],
    "hand": [
        {
            "name": "Finger Tendon Glides",
            "overview": "Supports finger flexibility and gentle hand rehabilitation.",
            "instructions": [
                "Open the hand fully.",
                "Curl the fingers into a hook fist, then a full fist.",
                "Repeat 5 to 10 times.",
            ],
            "safetyInfo": "Stop if swelling or pain increases.",
        }
    ],
    "elbow": [
        {
            "name": "Elbow Flexion and Extension",
            "overview": "Restores elbow range of motion in a controlled way.",
            "instructions": [
                "Slowly bend and straighten the elbow through a comfortable range.",
                "Keep the shoulder relaxed.",
                "Repeat 10 times.",
            ],
            "safetyInfo": "Avoid locking the elbow forcefully.",
        }
    ],
    "forearm": [
        {
            "name": "Forearm Stretch",
            "overview": "Targets the forearm muscles used for grip and wrist motion.",
            "instructions": [
                "Extend the arm forward with the palm down.",
                "Use the opposite hand to gently bend the wrist downward.",
                "Hold 20 to 30 seconds.",
            ],
            "safetyInfo": "Use a mild stretch only.",
        }
    ],
    "general": [
        {
            "name": "Gentle Range of Motion",
            "overview": "Useful when the disorder is not specific to one joint.",
            "instructions": [
                "Move the affected area slowly through a comfortable range.",
                "Repeat 5 to 10 times without holding your breath.",
                "Stay within a pain-free zone.",
            ],
            "safetyInfo": "If the movement feels unsafe, stop and seek medical guidance.",
        }
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Musculoskeletal disorder classifier + exercise recommender")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an image classifier from a folder dataset")
    train_parser.add_argument("--data-dir", type=str, help="Root folder with class subfolders")
    train_parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Where to save plots and model bundle")
    train_parser.add_argument("--image-size", type=int, default=64, help="Square image size used for training")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing")
    train_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    train_parser.add_argument("--sample", action="store_true", help="Generate a small synthetic dataset for quick testing")

    predict_parser = subparsers.add_parser("predict", help="Predict a single image and recommend exercises")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to the image to classify")
    predict_parser.add_argument("--model", type=str, default=str(DEFAULT_OUTPUT_DIR / "model_bundle.joblib"), help="Saved model bundle path")
    predict_parser.add_argument("--body-part", type=str, default="general", help="Affected body part for exercise lookup")
    predict_parser.add_argument("--api-key", type=str, default=os.getenv("EXERCISE_API_KEY"), help="ExerciseAPI.dev key (or set EXERCISE_API_KEY)")
    predict_parser.add_argument("--top-k", type=int, default=3, help="How many exercise recommendations to show")

    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_image_vector(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        resized = image.convert("L").resize(image_size)
        vector = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
    return vector


def collect_dataset(data_dir: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_dirs = [path for path in sorted(data_dir.iterdir()) if path.is_dir() and not path.name.startswith(".")]
    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {data_dir}. Expected folders like normal/ and abnormal/.")

    features: list[np.ndarray] = []
    labels: list[str] = []

    for class_dir in class_dirs:
        images = [path for path in sorted(class_dir.rglob("*")) if is_image_file(path)]
        if not images:
            continue
        for image_path in images:
            features.append(load_image_vector(image_path, image_size))
            labels.append(class_dir.name)

    if not features:
        raise SystemExit(f"No images were found under {data_dir}.")

    return np.vstack(features), np.array(labels), [path.name for path in class_dirs]


def build_synthetic_dataset(image_size: tuple[int, int], random_state: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(random_state)
    width, height = image_size
    features: list[np.ndarray] = []
    labels: list[str] = []

    for class_name in ["normal", "abnormal"]:
        for _ in range(60):
            canvas = Image.new("L", image_size, color=0)
            draw = ImageDraw.Draw(canvas)
            if class_name == "normal":
                draw.ellipse((10, 10, width - 10, height - 10), outline=220, width=4)
                for _ in range(3):
                    x = int(rng.integers(8, width - 8))
                    y = int(rng.integers(8, height - 8))
                    draw.point((x, y), fill=180)
            else:
                draw.line((10, 12, width - 10, height - 12), fill=240, width=5)
                draw.line((10, height - 12, width - 10, 12), fill=180, width=3)
                draw.rectangle((16, 16, width - 16, height - 16), outline=120, width=2)
            noise = rng.normal(0, 14, size=(height, width)).astype(np.float32)
            pixels = np.clip(np.asarray(canvas, dtype=np.float32) + noise, 0, 255)
            features.append((pixels.reshape(-1) / 255.0).astype(np.float32))
            labels.append(class_name)

    return np.vstack(features), np.array(labels), ["normal", "abnormal"]


def build_models(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "Support Vector Machine": SVC(probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    }


def make_feature_pipeline(model: object) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def save_training_plots(output_dir: Path, y_test: np.ndarray, y_pred: np.ndarray, model_results: pd.DataFrame) -> None:
    class_counts = pd.Series(y_test).value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(class_counts.index.astype(str), class_counts.values, color="#0f766e", edgecolor="black")
    plt.title("Test Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()

    ordered = model_results.sort_values(by="f1_score", ascending=False)
    positions = np.arange(len(ordered))

    plt.figure(figsize=(12, 6))
    plt.bar(positions - 0.2, ordered["accuracy"], width=0.4, label="Accuracy", color="#ad7d3d")
    plt.bar(positions + 0.2, ordered["f1_score"], width=0.4, label="F1-score", color="#0f766e")
    plt.xticks(positions, ordered["model"], rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    labels = sorted(pd.Series(y_test).unique().tolist())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(labels)), [str(label) for label in labels], rotation=45, ha="right")
    plt.yticks(range(len(labels)), [str(label) for label in labels])
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_roc_curve(output_dir: Path, best_model: Pipeline, x_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder) -> None:
    if len(np.unique(y_test)) != 2 or not hasattr(best_model.named_steps["model"], "predict_proba"):
        return

    positive_index = 1
    y_binary = (y_test == positive_index).astype(int)
    probabilities = best_model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_binary, probabilities)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC curve", color="#0f766e", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close()


def train(args: argparse.Namespace) -> ModelBundle:
    image_size = (args.image_size, args.image_size)
    if args.sample:
        features, labels, class_names = build_synthetic_dataset(image_size, args.random_state)
        print("Using built-in synthetic dataset for a quick demo.")
    else:
        if not args.data_dir:
            raise SystemExit("Provide --data-dir or use --sample.")
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise SystemExit(f"Dataset folder not found: {data_dir}")
        features, labels, class_names = collect_dataset(data_dir, image_size)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    if len(label_encoder.classes_) < 2:
        raise SystemExit("Need at least two classes for classification.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = build_models(args.random_state)
    fitted_models: dict[str, Pipeline] = {}
    rows = []

    for model_name, model in models.items():
        pipeline = make_feature_pipeline(model)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "roc_auc": None,
        }
        if len(np.unique(y_test)) == 2 and hasattr(pipeline.named_steps["model"], "predict_proba"):
            row["roc_auc"] = roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])
        rows.append(row)
        fitted_models[model_name] = pipeline

    results = pd.DataFrame(rows).sort_values(by="f1_score", ascending=False)
    best_row = results.iloc[0]
    best_model = fitted_models[best_row["model"]]
    best_predictions = best_model.predict(x_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_training_plots(output_dir, label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(best_predictions), results)
    save_roc_curve(output_dir, best_model, x_test, y_test, label_encoder)

    results.to_csv(output_dir / "model_metrics.csv", index=False)
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, best_predictions, target_names=label_encoder.classes_, zero_division=0),
        encoding="utf-8",
    )
    summary = {
        "rows": int(features.shape[0]),
        "image_size": list(image_size),
        "classes": label_encoder.classes_.tolist(),
        "best_model": best_row["model"],
        "best_metrics": {
            "accuracy": float(best_row["accuracy"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1_score": float(best_row["f1_score"]),
            "roc_auc": None if pd.isna(best_row["roc_auc"]) else float(best_row["roc_auc"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bundle = ModelBundle(
        model=best_model,
        label_encoder=label_encoder,
        image_size=image_size,
        feature_dim=features.shape[1],
        class_names=label_encoder.classes_.tolist(),
    )
    joblib.dump(bundle, output_dir / "model_bundle.joblib")

    print("Training complete.")
    print(f"Saved results to: {output_dir}")
    print(f"Best model: {best_row['model']}")
    print("Top models by F1-score:")
    for _, row in results.head(5).iterrows():
        auc_value = "n/a" if pd.isna(row["roc_auc"]) else f"{row['roc_auc']:.4f}"
        print(f"- {row['model']}: accuracy={row['accuracy']:.4f}, f1={row['f1_score']:.4f}, auc={auc_value}")

    return bundle


def preprocess_single_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    return load_image_vector(image_path, image_size).reshape(1, -1)


def normalize_body_part(body_part: str) -> str:
    key = body_part.strip().lower()
    return key if key in BODY_PART_QUERY_MAP else "general"


def fetch_exercise_recommendations(api_key: str | None, body_part: str, top_k: int) -> list[dict]:
    normalized = normalize_body_part(body_part)
    query_config = BODY_PART_QUERY_MAP[normalized]

    if not api_key:
        return LOCAL_RECOMMENDATIONS[normalized][:top_k]

    headers = {"X-API-Key": api_key}
    params = {
        "q": query_config["q"],
        "muscle": query_config["muscle"],
        "category": query_config["category"],
        "limit": min(top_k, 20),
    }
    response = requests.get(f"{EXERCISE_API_BASE}/exercises", params=params, headers=headers, timeout=20)
    if response.status_code != 200:
        return LOCAL_RECOMMENDATIONS[normalized][:top_k]

    payload = response.json()
    exercises = payload.get("data", [])
    recommendations = []
    for exercise in exercises[:top_k]:
        recommendations.append(
            {
                "name": exercise.get("name", "Unnamed exercise"),
                "overview": exercise.get("overview", "No overview available."),
                "instructions": exercise.get("instructions", []),
                "exerciseTips": exercise.get("exerciseTips", []),
                "commonMistakes": exercise.get("commonMistakes", []),
                "safetyInfo": exercise.get("safetyInfo", "Follow the guidance of a licensed professional."),
            }
        )

    return recommendations or LOCAL_RECOMMENDATIONS[normalized][:top_k]


def predict(args: argparse.Namespace) -> None:
    bundle_path = Path(args.model)
    if not bundle_path.exists():
        raise SystemExit(f"Model bundle not found: {bundle_path}")

    bundle: ModelBundle = joblib.load(bundle_path)
    image = preprocess_single_image(Path(args.image), bundle.image_size)
    predicted_index = int(bundle.model.predict(image)[0])
    predicted_label = bundle.label_encoder.inverse_transform([predicted_index])[0]

    probability_text = "n/a"
    if hasattr(bundle.model.named_steps["model"], "predict_proba") and len(bundle.class_names) == 2:
        probabilities = bundle.model.predict_proba(image)[0]
        probability_text = f"{probabilities[predicted_index]:.4f}"

    recommendations = fetch_exercise_recommendations(args.api_key, args.body_part, args.top_k)

    print("Prediction result")
    print("-" * 80)
    print(f"Image: {args.image}")
    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {probability_text}")
    print(f"Body part used for recommendations: {normalize_body_part(args.body_part)}")
    print()
    print("Exercise recommendations")
    print("-" * 80)

    for index, item in enumerate(recommendations, start=1):
        print(f"{index}. {item.get('name', 'Exercise')}")
        print(f"   Overview: {item.get('overview', 'No overview.')}")
        instructions = item.get("instructions", [])
        if instructions:
            print("   Instructions:")
            for step in instructions[:4]:
                print(f"   - {step}")
        tips = item.get("exerciseTips", [])
        if tips:
            print("   Tips:")
            for tip in tips[:3]:
                print(f"   - {tip}")
        mistakes = item.get("commonMistakes", [])
        if mistakes:
            print("   Common mistakes:")
            for mistake in mistakes[:3]:
                print(f"   - {mistake}")
        if item.get("safetyInfo"):
            print(f"   Safety: {item['safetyInfo']}")
        print()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        raise SystemExit("Unknown command.")


if __name__ == "__main__":
    main()