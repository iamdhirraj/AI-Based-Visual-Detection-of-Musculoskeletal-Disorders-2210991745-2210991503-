#!/usr/bin/env python3
"""Musculoskeletal image prediction service.

This module trains and serves two classifiers from image features:
- body part prediction
- disorder prediction

The web app uses the saved bundle to infer both from a single uploaded image,
then optionally fetches exercise recommendations from ExerciseAPI.dev.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_OUTPUT_DIR = Path("muscle_ai_output")
EXERCISE_API_BASE = "https://api.exerciseapi.dev/v1"


@dataclass
class MuscleAIModelBundle:
    body_part_model: object
    disorder_model: object
    body_part_encoder: LabelEncoder
    disorder_encoder: LabelEncoder
    image_size: tuple[int, int]
    feature_dim: int
    body_part_classes: list[str]
    disorder_classes: list[str]


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
    parser = argparse.ArgumentParser(description="Train or serve musculoskeletal image prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train body-part and disorder classifiers")
    train_parser.add_argument("--data-dir", type=str, help="Dataset root: body_part/disorder/images or normal/abnormal/images")
    train_parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for the saved bundle")
    train_parser.add_argument("--image-size", type=int, default=64, help="Square image size used for training")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    train_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    train_parser.add_argument("--sample", action="store_true", help="Use a synthetic demo dataset")

    predict_parser = subparsers.add_parser("predict", help="Predict a single image")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to an X-ray image")
    predict_parser.add_argument("--model", type=str, default=str(DEFAULT_OUTPUT_DIR / "model_bundle.joblib"), help="Saved bundle path")
    predict_parser.add_argument("--api-key", type=str, default=os.getenv("EXERCISE_API_KEY"), help="ExerciseAPI.dev key")
    predict_parser.add_argument("--top-k", type=int, default=3, help="Number of exercise recommendations")

    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_image_vector(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        resized = image.convert("L").resize(image_size)
        return np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0


def _find_image_files(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.rglob("*")) if is_image_file(path)]


def collect_dataset(data_dir: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    features: list[np.ndarray] = []
    body_parts: list[str] = []
    disorders: list[str] = []

    root_dirs = [path for path in sorted(data_dir.iterdir()) if path.is_dir() and not path.name.startswith(".")]
    if not root_dirs:
        raise SystemExit(f"No subfolders found in {data_dir}.")

    for root_dir in root_dirs:
        child_dirs = [path for path in sorted(root_dir.iterdir()) if path.is_dir() and not path.name.startswith(".")]
        if child_dirs and any(child.name.lower() in {"normal", "abnormal", "positive", "negative"} for child in child_dirs):
            body_part_name = root_dir.name.lower()
            for disorder_dir in child_dirs:
                disorder_name = disorder_dir.name.lower()
                for image_path in _find_image_files(disorder_dir):
                    features.append(load_image_vector(image_path, image_size))
                    body_parts.append(body_part_name)
                    disorders.append(disorder_name)
        else:
            disorder_name = root_dir.name.lower()
            for image_path in _find_image_files(root_dir):
                features.append(load_image_vector(image_path, image_size))
                body_parts.append("general")
                disorders.append(disorder_name)

    if not features:
        raise SystemExit(f"No images found under {data_dir}.")

    return np.vstack(features), np.array(body_parts), np.array(disorders), sorted(set(body_parts)), sorted(set(disorders))


def build_synthetic_dataset(image_size: tuple[int, int], random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(random_state)
    body_part_names = ["shoulder", "wrist", "hand", "finger", "elbow", "forearm", "humerus"]
    disorder_names = ["normal", "abnormal"]
    width, height = image_size

    features: list[np.ndarray] = []
    body_parts: list[str] = []
    disorders: list[str] = []

    for body_part_name in body_part_names:
        for disorder_name in disorder_names:
            for _ in range(36):
                canvas = Image.new("L", image_size, color=0)
                draw = ImageDraw.Draw(canvas)
                cx, cy = width // 2, height // 2

                if body_part_name == "shoulder":
                    draw.ellipse((12, 8, width - 12, height - 12), outline=220, width=3)
                elif body_part_name == "wrist":
                    draw.rectangle((10, 18, width - 10, height - 18), outline=210, width=3)
                elif body_part_name == "hand":
                    draw.line((10, cy, width - 10, cy), fill=220, width=4)
                    draw.line((cx, 8, cx, height - 8), fill=220, width=3)
                elif body_part_name == "finger":
                    for offset in (-12, -4, 4, 12):
                        draw.line((cx + offset, 8, cx + offset, height - 8), fill=220, width=2)
                elif body_part_name == "elbow":
                    draw.arc((10, 8, width - 10, height - 8), start=25, end=320, fill=220, width=4)
                elif body_part_name == "forearm":
                    draw.polygon([(10, 20), (width - 10, 14), (width - 10, height - 14), (10, height - 20)], outline=220)
                else:
                    draw.line((10, 10, width - 10, height - 10), fill=220, width=4)

                if disorder_name == "abnormal":
                    draw.line((8, height - 14, width - 8, 14), fill=255, width=3)
                    draw.rectangle((18, 18, width - 18, height - 18), outline=150, width=2)
                else:
                    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=140, width=1)

                noise = rng.normal(0, 12, size=(height, width)).astype(np.float32)
                pixels = np.clip(np.asarray(canvas, dtype=np.float32) + noise, 0, 255)
                features.append((pixels.reshape(-1) / 255.0).astype(np.float32))
                body_parts.append(body_part_name)
                disorders.append(disorder_name)

    return np.vstack(features), np.array(body_parts), np.array(disorders), body_part_names, disorder_names


def build_candidates(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=4000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=240, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


def make_pipeline(model: object) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def evaluate_task(x_train, x_test, y_train, y_test, candidates, encoder: LabelEncoder) -> tuple[Pipeline, pd.DataFrame]:
    rows = []
    fitted: dict[str, Pipeline] = {}

    for model_name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
            }
        )
        fitted[model_name] = pipeline

    results = pd.DataFrame(rows).sort_values(by="f1_score", ascending=False)
    best_model = fitted[results.iloc[0]["model"]]
    return best_model, results


def _plot_distribution(path: Path, labels: Iterable[str], title: str, filename: str) -> None:
    counts = pd.Series(list(labels)).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values, color="#0f766e", edgecolor="black")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path / filename, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_comparison(path: Path, results: pd.DataFrame, title: str, filename: str) -> None:
    ordered = results.sort_values(by="f1_score", ascending=False)
    positions = np.arange(len(ordered))
    plt.figure(figsize=(12, 6))
    plt.bar(positions - 0.2, ordered["accuracy"], width=0.4, label="Accuracy", color="#ad7d3d")
    plt.bar(positions + 0.2, ordered["f1_score"], width=0.4, label="F1-score", color="#0f766e")
    plt.xticks(positions, ordered["model"], rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / filename, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], filename: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=("white" if matrix[row, col] > threshold else "black"), fontweight="bold")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path / filename, dpi=160, bbox_inches="tight")
    plt.close()


def train(args: argparse.Namespace) -> MuscleAIModelBundle:
    image_size = (args.image_size, args.image_size)
    if args.sample:
        x, body_parts, disorders, body_part_classes, disorder_classes = build_synthetic_dataset(image_size, args.random_state)
        print("Using built-in synthetic dataset for a quick demo.")
    else:
        if not args.data_dir:
            raise SystemExit("Provide --data-dir or use --sample.")
        x, body_parts, disorders, body_part_classes, disorder_classes = collect_dataset(Path(args.data_dir), image_size)

    body_part_encoder = LabelEncoder().fit(body_parts)
    disorder_encoder = LabelEncoder().fit(disorders)

    combined = np.array([f"{bp}::{dz}" for bp, dz in zip(body_parts, disorders)])
    x_train, x_test, body_train, body_test, disorder_train, disorder_test = train_test_split(
        x,
        body_part_encoder.transform(body_parts),
        disorder_encoder.transform(disorders),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=combined,
    )

    body_part_model, body_results = evaluate_task(
        x_train,
        x_test,
        body_train,
        body_test,
        build_candidates(args.random_state),
        body_part_encoder,
    )
    disorder_model, disorder_results = evaluate_task(
        x_train,
        x_test,
        disorder_train,
        disorder_test,
        build_candidates(args.random_state),
        disorder_encoder,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_distribution(output_dir, body_parts, "Body Part Distribution", "body_part_distribution.png")
    _plot_distribution(output_dir, disorders, "Disorder Distribution", "disorder_distribution.png")
    _plot_comparison(output_dir, body_results, "Body Part Model Comparison", "body_part_models.png")
    _plot_comparison(output_dir, disorder_results, "Disorder Model Comparison", "disorder_models.png")

    body_pred = body_part_model.predict(x_test)
    disorder_pred = disorder_model.predict(x_test)
    _plot_confusion(output_dir, body_test, body_pred, body_part_encoder.classes_.tolist(), "body_part_confusion.png")
    _plot_confusion(output_dir, disorder_test, disorder_pred, disorder_encoder.classes_.tolist(), "disorder_confusion.png")

    bundle = MuscleAIModelBundle(
        body_part_model=body_part_model,
        disorder_model=disorder_model,
        body_part_encoder=body_part_encoder,
        disorder_encoder=disorder_encoder,
        image_size=image_size,
        feature_dim=x.shape[1],
        body_part_classes=body_part_encoder.classes_.tolist(),
        disorder_classes=disorder_encoder.classes_.tolist(),
    )
    joblib.dump(bundle, output_dir / "model_bundle.joblib")

    summary = {
        "samples": int(x.shape[0]),
        "image_size": list(image_size),
        "body_part_classes": bundle.body_part_classes,
        "disorder_classes": bundle.disorder_classes,
        "best_body_part_model": body_results.iloc[0]["model"],
        "best_disorder_model": disorder_results.iloc[0]["model"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Saved results to: {output_dir}")
    print(f"Best body-part model: {body_results.iloc[0]['model']}")
    print(f"Best disorder model: {disorder_results.iloc[0]['model']}")
    return bundle


def preprocess_single_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    return load_image_vector(image_path, image_size).reshape(1, -1)


def body_part_to_query(body_part: str) -> dict[str, str]:
    key = body_part.strip().lower()
    return BODY_PART_QUERY_MAP.get(key, BODY_PART_QUERY_MAP["general"])


def fetch_exercise_recommendations(api_key: str | None, body_part: str, top_k: int) -> list[dict]:
    normalized = body_part.strip().lower() if body_part else "general"
    query_config = BODY_PART_QUERY_MAP.get(normalized, BODY_PART_QUERY_MAP["general"])

    if not api_key:
        return LOCAL_RECOMMENDATIONS.get(normalized, LOCAL_RECOMMENDATIONS["general"] )[:top_k]

    headers = {"X-API-Key": api_key}
    params = {
        "q": query_config["q"],
        "muscle": query_config["muscle"],
        "category": query_config["category"],
        "limit": min(top_k, 20),
    }
    response = requests.get(f"{EXERCISE_API_BASE}/exercises", params=params, headers=headers, timeout=20)
    if response.status_code != 200:
        return LOCAL_RECOMMENDATIONS.get(normalized, LOCAL_RECOMMENDATIONS["general"] )[:top_k]

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

    return recommendations or LOCAL_RECOMMENDATIONS.get(normalized, LOCAL_RECOMMENDATIONS["general"] )[:top_k]


def predict_bundle(image_path: Path, bundle: MuscleAIModelBundle, api_key: str | None = None, top_k: int = 3) -> dict:
    image = preprocess_single_image(image_path, bundle.image_size)

    body_part_index = int(bundle.body_part_model.predict(image)[0])
    disorder_index = int(bundle.disorder_model.predict(image)[0])

    body_part = bundle.body_part_encoder.inverse_transform([body_part_index])[0]
    disorder = bundle.disorder_encoder.inverse_transform([disorder_index])[0]

    body_part_confidence = None
    disorder_confidence = None
    if hasattr(bundle.body_part_model.named_steps["model"], "predict_proba"):
        body_part_confidence = float(bundle.body_part_model.predict_proba(image)[0][body_part_index])
    if hasattr(bundle.disorder_model.named_steps["model"], "predict_proba"):
        disorder_confidence = float(bundle.disorder_model.predict_proba(image)[0][disorder_index])

    normalized_disorder = disorder.strip().lower()
    has_disorder = normalized_disorder not in {"normal", "healthy", "negative", "no disorder"}
    recommendations = fetch_exercise_recommendations(api_key, body_part, top_k) if has_disorder else []

    return {
        "body_part": body_part,
        "disorder": disorder,
        "has_disorder": has_disorder,
        "body_part_confidence": body_part_confidence,
        "disorder_confidence": disorder_confidence,
        "recommendations": recommendations,
    }


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
        return

    if args.command == "predict":
        bundle = joblib.load(Path(args.model))
        result = predict_bundle(Path(args.image), bundle, api_key=args.api_key, top_k=args.top_k)
        print(json.dumps(result, indent=2))
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()