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
- `generate_pdf.py` - creates the corrected PDF report at `assets/mura-corrected-report.pdf`

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
- `generate_pdf.py` creates `assets/mura-corrected-report.pdf`.
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
