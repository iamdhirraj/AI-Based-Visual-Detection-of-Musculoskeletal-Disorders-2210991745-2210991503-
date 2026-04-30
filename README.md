# AI-Based Visual Detection of Musculoskeletal Disorders with Personalized Exercise Recommendation

An intelligent platform combining radiographic analysis with personalized exercise recommendations for musculoskeletal disorder detection and rehabilitation.

## Authors

- Roll no 2210991745 — Kapil Tanwar
- Roll no 2210991503 — Dheeraj Sharma

**Project Title:** AI-Based Visual Detection of Musculoskeletal Disorders with Personalized Exercise Recommendation

## Files
- `index.html` - site entry point
- `styles.css` - visual design
- `script.js` - SVG charts rendered in the browser
- `generate_pdf.py` - creates the corrected PDF report at `assets/research-paper.pdf`

## Run locally
Open `index.html` in a browser, or serve the folder with a simple static server.

## Generate the corrected PDF

```bash
python3 generate_pdf.py
```

## Machine Learning Concepts

Machine learning projects often use different types of algorithms, metrics, and validation methods. A useful way to organize them is:

### 1. Supervised Learning

These models learn from labeled data.

Classification algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

Regression algorithms:
- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

### 2. Unsupervised Learning

These methods work without labeled outputs.

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis (PCA)

Note: PCA is mainly used for dimensionality reduction, not clustering.

### 3. Deep Learning

- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transformer
- Autoencoder

### 4. Model Evaluation Metrics

These are used to measure how well a model performs.

Classification metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- Fβ Score
- Confusion Matrix

Regression metrics:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE

Note: F1-score is a metric, not an algorithm.

F1-score formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

### 5. Validation Techniques

- Train-Test Split
- Cross-Validation

### Common Algorithms in Medical Prediction Papers

For disease prediction and medical classification, researchers often compare:

1. Logistic Regression
2. Support Vector Machine
3. Random Forest
4. Decision Tree
5. K-Nearest Neighbors

# AI-Based-Visual-Detection-of-Musculoskeletal-Disorders-2210991745-2210991503-

## How to run this project (quick)

1. Create and activate a Python virtual environment (macOS / Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo script to generate charts:

```bash
python demo.py
```

4. Open the web dashboard in a browser (double-click or serve the folder):

```bash
open index.html
# or serve: python -m http.server 8000
```

Notes:
- `demo.py` generates PNG charts in the `demo_output/` folder.
- `generate_pdf.py` creates `assets/research-paper.pdf`.
- If you already have a virtual environment in `/Users/dheerajsharma/Desktop/sys/.venv`, activate that instead of creating a new one.

## Train on your own dataset

If you give a CSV dataset with one target column, use `train_dataset.py` to train models and create graphs.

Example:

```bash
python train_dataset.py --data your_dataset.csv --target target
```

Quick test without your own dataset:

```bash
python train_dataset.py --sample
```

Outputs are saved in `training_output/`, including:
- `class_distribution.png`
- `model_comparison.png`
- `confusion_matrix.png`
- `roc_curve.png` for binary classification
- `model_metrics.csv`
- `classification_report.txt`
- `summary.json`
- `best_model.joblib`

## Predict from an X-ray image and recommend exercises

Use `muscle_ai_service.py` when you want image-based prediction plus exercise recommendations.

The model predicts the body part automatically from the uploaded image, then detects whether the image is normal or abnormal. If a disorder is detected, the backend fetches exercise recommendations server-side.

Suggested dataset layout for training:

```text
dataset/
	shoulder/
		normal/
			image1.png
		abnormal/
			image2.png
	wrist/
		normal/
		abnormal/
```

Train on a folder dataset:

```bash
python muscle_ai_service.py train --data-dir /path/to/dataset --output-dir muscle_ai_output
```

Quick demo with a synthetic image dataset:

```bash
python muscle_ai_service.py train --sample --output-dir muscle_ai_output
```

Predict one image and fetch exercises from ExerciseAPI.dev:

```bash
export EXERCISE_API_KEY="your_api_key_here"
python muscle_ai_service.py predict --image /path/to/xray.png --model muscle_ai_output/model_bundle.joblib
```

If the API key is missing, the script still works and falls back to safe local exercise recommendations. The user does not need to provide a body part manually.

## Run the web upload app

This is the browser version of the same workflow.

```bash
python web_app.py
```

Then open:

```bash
http://127.0.0.1:5000
```

If you want the app to use the live exercise API, set your key first:

```bash
export EXERCISE_API_KEY="your_api_key_here"
```

If the model bundle is missing, the app automatically builds a sample bundle for demo use.
