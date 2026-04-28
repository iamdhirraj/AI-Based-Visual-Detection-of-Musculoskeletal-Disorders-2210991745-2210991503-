#!/usr/bin/env python3
"""Train classification models on a CSV dataset and generate graphs.

Usage examples:
  python train_dataset.py --data data.csv --target target
  python train_dataset.py --sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classifier on a CSV dataset and generate plots.")
    parser.add_argument("--data", type=str, help="Path to a CSV file containing features and target.")
    parser.add_argument("--target", type=str, default="target", help="Target column name in the CSV.")
    parser.add_argument("--output-dir", type=str, default="training_output", help="Directory for reports and plots.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for repeatable splits.")
    parser.add_argument("--sample", action="store_true", help="Generate a sample dataset and train on it.")
    return parser.parse_args()


def build_sample_dataset(random_state: int) -> pd.DataFrame:
    from sklearn.datasets import make_classification

    features, labels = make_classification(
        n_samples=1200,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.2,
        random_state=random_state,
    )
    feature_names = [f"feature_{index + 1}" for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=feature_names)
    frame["target"] = np.where(labels == 1, "positive", "negative")
    return frame


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.sample:
        print("Using built-in sample dataset.")
        return build_sample_dataset(args.random_state)

    if not args.data:
        raise SystemExit("Provide --data path/to/data.csv or use --sample to run a demo dataset.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    if data_path.suffix.lower() != ".csv":
        raise SystemExit("This script currently expects a CSV file. Convert your dataset to CSV first.")

    return pd.read_csv(data_path)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            )
        )

    if not transformers:
        raise SystemExit("No usable feature columns were found after removing the target column.")

    return ColumnTransformer(transformers=transformers)


def build_models(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Support Vector Machine": SVC(probability=True, random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    }


def compute_auc(model, x_test, y_test, class_labels) -> float | None:
    if len(class_labels) < 2:
        return None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        probabilities = np.asarray(scores)
    else:
        return None

    if len(class_labels) == 2:
        positive_label = class_labels[1]
        y_binary = (y_test == positive_label).astype(int)
        if probabilities.ndim == 1:
            scores = probabilities
        else:
            scores = probabilities[:, 1]
        return roc_auc_score(y_binary, scores)

    if probabilities.ndim == 1:
        return None

    return roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")


def plot_class_distribution(y: pd.Series, output_dir: Path) -> None:
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values, color="#0f766e", edgecolor="black")
    plt.title("Target Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    for index, value in enumerate(counts.values):
        plt.text(index, value, str(int(value)), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results: pd.DataFrame, output_dir: Path) -> None:
    ordered = results.sort_values(by="f1_score", ascending=False)
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


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    labels = sorted(pd.Series(y_true).unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
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


def plot_roc(best_model, x_test, y_test, class_labels, output_dir: Path) -> None:
    if len(class_labels) != 2 or not hasattr(best_model, "predict_proba"):
        return

    positive_label = class_labels[1]
    y_binary = (y_test == positive_label).astype(int)
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(args)
    if args.target not in frame.columns:
        available_columns = ", ".join(frame.columns.astype(str).tolist())
        raise SystemExit(f"Target column '{args.target}' not found. Available columns: {available_columns}")

    features = frame.drop(columns=[args.target])
    target = frame[args.target]

    if target.nunique() < 2:
        raise SystemExit("Target column must contain at least two classes for classification.")

    plot_class_distribution(target, output_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    preprocessor = build_preprocessor(features)
    models = build_models(args.random_state)
    results = []
    fitted_models: dict[str, Pipeline] = {}

    class_labels = sorted(target.unique().tolist())

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "roc_auc": compute_auc(pipeline, x_test, y_test, class_labels),
        }
        results.append(metrics)
        fitted_models[model_name] = pipeline

    results_frame = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    best_row = results_frame.iloc[0]
    best_model = fitted_models[best_row["model"]]
    best_predictions = best_model.predict(x_test)

    plot_model_comparison(results_frame, output_dir)
    plot_confusion(y_test, best_predictions, output_dir)
    plot_roc(best_model, x_test, y_test, class_labels, output_dir)

    results_frame.to_csv(output_dir / "model_metrics.csv", index=False)
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, best_predictions, zero_division=0),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": int(len(frame)),
                "features": int(features.shape[1]),
                "target": args.target,
                "best_model": best_row["model"],
                "best_metrics": {
                    "accuracy": float(best_row["accuracy"]),
                    "precision": float(best_row["precision"]),
                    "recall": float(best_row["recall"]),
                    "f1_score": float(best_row["f1_score"]),
                    "roc_auc": None if pd.isna(best_row["roc_auc"]) else float(best_row["roc_auc"]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    joblib.dump(best_model, output_dir / "best_model.joblib")

    print("Training complete.")
    print(f"Saved results to: {output_dir}")
    print()
    print("Top models by F1-score:")
    for _, row in results_frame.head(5).iterrows():
        auc_value = "n/a" if pd.isna(row["roc_auc"]) else f"{row['roc_auc']:.4f}"
        print(
            f"- {row['model']}: accuracy={row['accuracy']:.4f}, f1={row['f1_score']:.4f}, auc={auc_value}"
        )
    print()
    print(f"Best model: {best_row['model']}")
    print("Generated files:")
    for filename in [
        "class_distribution.png",
        "model_comparison.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "model_metrics.csv",
        "classification_report.txt",
        "summary.json",
        "best_model.joblib",
    ]:
        if (output_dir / filename).exists():
            print(f"- {filename}")


if __name__ == "__main__":
    main()