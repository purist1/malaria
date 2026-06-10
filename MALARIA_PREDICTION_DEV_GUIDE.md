# ML-Based Malaria Occurrence Prediction System
## Complete Development Guide

> **Project:** Development of a Machine Learning-Based Predictive Model for Forecasting Malaria Occurrence  
> **Stack:** Python · Scikit-learn · Pandas · NumPy · Matplotlib/Seaborn · Flask/Streamlit  
> **Target Outcome:** Trained, evaluated, and deployable malaria prediction system with comparative algorithm analysis

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset](#4-dataset)
5. [Phase 1 — Data Collection & Loading](#5-phase-1--data-collection--loading)
6. [Phase 2 — Exploratory Data Analysis (EDA)](#6-phase-2--exploratory-data-analysis-eda)
7. [Phase 3 — Data Preprocessing](#7-phase-3--data-preprocessing)
8. [Phase 4 — Feature Engineering & Selection](#8-phase-4--feature-engineering--selection)
9. [Phase 5 — Model Development](#9-phase-5--model-development)
10. [Phase 6 — Model Training](#10-phase-6--model-training)
11. [Phase 7 — Model Evaluation & Comparison](#11-phase-7--model-evaluation--comparison)
12. [Phase 8 — Visualization & Reporting](#12-phase-8--visualization--reporting)
13. [Phase 9 — Deployment (Streamlit/Flask)](#13-phase-9--deployment-streamlitflask)
14. [Ethical Considerations](#14-ethical-considerations)
15. [Testing Checklist](#15-testing-checklist)
16. [References](#16-references)

---

## 1. Project Overview

This system predicts the likelihood of malaria occurrence using supervised machine learning algorithms trained on historical environmental and epidemiological data. Four algorithms are implemented and compared: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)**.

### Objectives
- Collect and preprocess malaria-related datasets
- Identify key environmental and clinical factors affecting malaria transmission
- Develop, train, and evaluate multiple ML models
- Compare algorithm performance and select the best model
- Provide a usable prediction interface for healthcare decision-making

### System Architecture

```
Raw Data
   │
   ▼
Data Collection Layer
   │
   ▼
Data Preprocessing Layer  ──► Handle nulls, duplicates, encoding, scaling
   │
   ▼
Feature Selection Layer   ──► Correlation analysis, importance ranking
   │
   ▼
ML Training Layer         ──► LR | Decision Tree | Random Forest | SVM
   │
   ▼
Prediction Layer          ──► Output: Malaria Positive / Negative + Probability
   │
   ▼
Result Visualization Layer ──► Metrics, charts, confusion matrix
```

---

## 2. Repository Structure

```
malaria-prediction/
│
├── data/
│   ├── raw/                    # Original downloaded datasets
│   └── processed/              # Cleaned, preprocessed CSV files
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Cleaning and transformation
│   ├── 03_model_training.ipynb # Training all four models
│   └── 04_evaluation.ipynb     # Metrics, charts, comparison
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Reusable preprocessing functions
│   ├── train.py                # Model training scripts
│   ├── evaluate.py             # Evaluation and comparison utilities
│   └── predict.py              # Single-sample prediction logic
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
├── app/
│   ├── app.py                  # Streamlit or Flask web interface
│   └── templates/              # HTML templates (if using Flask)
│
├── reports/
│   ├── figures/                # Saved plots (confusion matrix, ROC, etc.)
│   └── model_comparison.csv    # Performance metrics table
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.9 or higher
- pip or conda

### Installation

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas==2.1.0
numpy==1.25.0
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
imbalanced-learn==0.11.0
joblib==1.3.2
streamlit==1.28.0
flask==3.0.0
jupyter==1.0.0
openpyxl==3.1.2
```

---

## 4. Dataset

### Recommended Dataset

**African Malaria Dataset — Kaggle**  
This dataset contains environmental factors (temperature, rainfall, humidity) and malaria case records suitable for binary classification.

🔗 **Download Link:**  
[https://www.kaggle.com/datasets/imdevskp/malaria-dataset](https://www.kaggle.com/datasets/imdevskp/malaria-dataset)

> **Note:** You need a free Kaggle account to download. After downloading, place the CSV file in `data/raw/`.

### Alternative/Supplementary Datasets

| Source | Description | Link |
|--------|-------------|------|
| WHO Global Malaria Data | Country-level annual incidence statistics | https://data.who.int/indicators/i/53799DF |
| Malaria Atlas Project | High-resolution spatial malaria data | https://malariaatlas.org/data-project/ |
| UCI ML Repository | Clinical malaria symptom data | https://archive.ics.uci.edu/dataset/351/malaria |
| Africa CDC Open Data | Regional health surveillance records | https://africacdc.org/resources/ |

### Dataset Features (Expected)

| Feature | Type | Description |
|---------|------|-------------|
| `temperature` | Float | Average monthly temperature (°C) |
| `rainfall` | Float | Monthly rainfall (mm) |
| `humidity` | Float | Relative humidity (%) |
| `mosquito_density` | Float | Estimated mosquito population index |
| `previous_cases` | Integer | Number of malaria cases in prior period |
| `population` | Integer | Population in the region |
| `age_group` | Categorical | Under-5, 5-14, 15+ |
| `region` | Categorical | Geographic region/state |
| `malaria_occurrence` | Integer | **Target** — 1 = Positive, 0 = Negative |

---

## 5. Phase 1 — Data Collection & Loading

### `src/preprocess.py` — Data Loader

```python
import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    df = load_data("data/raw/malaria_dataset.csv")
    print(df.head())
    print(df.info())
```

---

## 6. Phase 2 — Exploratory Data Analysis (EDA)

Run this in `notebooks/01_eda.ipynb`.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/malaria_dataset.csv")

# --- Basic Overview ---
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())

# --- Target Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='malaria_occurrence', data=df, palette='Set2')
plt.title("Malaria Occurrence Distribution")
plt.xlabel("Occurrence (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/figures/target_distribution.png")
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png")
plt.show()

# --- Feature Distributions ---
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'malaria_occurrence']

df[numeric_cols].hist(bins=20, figsize=(14, 10), color='steelblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("reports/figures/feature_distributions.png")
plt.show()

# --- Boxplots (features vs target) ---
for col in ['temperature', 'rainfall', 'humidity']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='malaria_occurrence', y=col, data=df, palette='Set1')
    plt.title(f"{col} vs Malaria Occurrence")
    plt.tight_layout()
    plt.savefig(f"reports/figures/boxplot_{col}.png")
    plt.show()
```

---

## 7. Phase 3 — Data Preprocessing

### `src/preprocess.py` — Full Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values."""
    print(f"[INFO] Missing values before:\n{df.isnull().sum()}\n")
    
    # Fill numeric columns with median
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"[INFO] Missing values after:\n{df.isnull().sum()}\n")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {before - len(df)} duplicate rows.")
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns."""
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"[INFO] Encoded column: {col}")
    return df

def scale_features(X_train, X_test):
    """Standardize numerical features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def split_data(df: pd.DataFrame, target_col: str = 'malaria_occurrence'):
    """Split into features and target, then train/test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def full_pipeline(filepath: str, target_col: str = 'malaria_occurrence'):
    """Run the complete preprocessing pipeline."""
    df = pd.read_csv(filepath)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categorical(df)
    
    # Save processed data
    df.to_csv("data/processed/malaria_processed.csv", index=False)
    print("[INFO] Processed data saved.")
    
    X_train, X_test, y_train, y_test = split_data(df, target_col)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
```

---

## 8. Phase 4 — Feature Engineering & Selection

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def select_features(X_train, y_train, feature_names, top_n=10):
    """Rank feature importance using Random Forest."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    importances[:top_n].plot(kind='barh', color='teal')
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png")
    plt.show()
    
    print(f"\nTop {top_n} Features:\n", importances[:top_n])
    return importances[:top_n].index.tolist()
```

### Feature Engineering Ideas

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new derived features."""
    
    # Heat-Humidity Index (proxy for mosquito breeding conditions)
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['heat_humidity_index'] = df['temperature'] * df['humidity'] / 100
    
    # Rainfall intensity category
    if 'rainfall' in df.columns:
        df['rainfall_category'] = pd.cut(
            df['rainfall'],
            bins=[0, 50, 150, 300, float('inf')],
            labels=[0, 1, 2, 3]  # Low, Moderate, High, Extreme
        ).astype(int)
    
    # Case rate (if population data available)
    if 'previous_cases' in df.columns and 'population' in df.columns:
        df['case_rate_per_1000'] = (df['previous_cases'] / df['population']) * 1000
    
    return df
```

---

## 9. Phase 5 — Model Development

### `src/train.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import os

def get_models():
    """Return a dictionary of all ML models to train."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        "SVM": SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    }

def train_all_models(X_train, y_train, save_dir="models/"):
    """Train all models and save to disk."""
    os.makedirs(save_dir, exist_ok=True)
    models = get_models()
    trained = {}
    
    for name, model in models.items():
        print(f"[INFO] Training: {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        
        # Save model
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(save_dir, filename))
        print(f"[INFO] Saved: {filename}")
    
    return trained
```

---

## 10. Phase 6 — Model Training

Run from `notebooks/03_model_training.ipynb` or directly:

```python
from src.preprocess import full_pipeline
from src.train import train_all_models

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = full_pipeline(
    "data/raw/malaria_dataset.csv",
    target_col="malaria_occurrence"
)

# Train all models
trained_models = train_all_models(X_train, y_train)

print("[INFO] All models trained successfully.")
```

---

## 11. Phase 7 — Model Evaluation & Comparison

### `src/evaluate.py`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
)

def evaluate_model(model, X_test, y_test, model_name: str):
    """Compute and display all evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        "Model": model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else "N/A"
    }
    
    print(f"\n{'='*50}")
    print(f"  Results for: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Malaria", "Malaria"]))
    
    return metrics, y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred, model_name: str):
    """Plot and save a styled confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["No Malaria", "Malaria"],
        yticklabels=["No Malaria", "Malaria"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"reports/figures/cm_{model_name.replace(' ', '_').lower()}.png")
    plt.show()

def plot_roc_curves(models_data: list):
    """Plot ROC curves for all models on one chart."""
    plt.figure(figsize=(9, 7))
    for name, fpr, tpr, auc_score in models_data:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curves_comparison.png")
    plt.show()

def compare_models(trained_models: dict, X_test, y_test):
    """Evaluate all models and produce a comparison table."""
    all_metrics = []
    roc_data = []
    
    for name, model in trained_models.items():
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        plot_confusion_matrix(y_test, y_pred, name)
        
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data.append((name, fpr, tpr, float(metrics["ROC-AUC"])))
    
    if roc_data:
        plot_roc_curves(roc_data)
    
    # Summary Table
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("\n[INFO] Comparison table saved to reports/model_comparison.csv")
    
    # Bar chart comparison
    plot_model_comparison(results_df)
    return results_df

def plot_model_comparison(results_df: pd.DataFrame):
    """Bar chart comparing all models across metrics."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_plot = results_df[["Model"] + metrics_to_plot].set_index("Model")
    
    df_plot.plot(kind='bar', figsize=(12, 6), colormap='Set2', edgecolor='black')
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison_bar.png")
    plt.show()
```

### Running the Evaluation

```python
from src.evaluate import compare_models

results = compare_models(trained_models, X_test, y_test)
print("\nFinal Comparison Table:")
print(results.to_string(index=False))

# Best model
best = results.loc[results['F1-Score'].idxmax()]
print(f"\nBest Model: {best['Model']} with F1-Score = {best['F1-Score']}")
```

---

## 12. Phase 8 — Visualization & Reporting

All plots are automatically saved to `reports/figures/`. Below is a summary of every chart produced:

| Chart | File | Purpose |
|-------|------|---------|
| Target distribution | `target_distribution.png` | Class balance check |
| Correlation heatmap | `correlation_heatmap.png` | Feature relationships |
| Feature distributions | `feature_distributions.png` | Skewness and range |
| Boxplots per feature | `boxplot_<feature>.png` | Feature vs target split |
| Feature importance | `feature_importance.png` | Top predictors |
| Confusion matrices | `cm_<model_name>.png` | TP/FP/TN/FN per model |
| ROC curves | `roc_curves_comparison.png` | AUC comparison |
| Model comparison bar | `model_comparison_bar.png` | Side-by-side scores |

---

## 13. Phase 9 — Deployment (Streamlit/Flask)

### Option A: Streamlit Web App (`app/app.py`)

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load best model and scaler
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Malaria Prediction System", page_icon="🦟")
st.title("🦟 Malaria Occurrence Prediction System")
st.markdown("Enter environmental and clinical data to predict the likelihood of malaria occurrence.")

st.sidebar.header("Input Features")

def get_user_input():
    temperature    = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, 30.0)
    rainfall       = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, 100.0)
    humidity       = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 65.0)
    previous_cases = st.sidebar.number_input("Previous Malaria Cases", 0, 10000, 200)
    mosquito_index = st.sidebar.slider("Mosquito Density Index", 0.0, 10.0, 5.0)
    
    data = {
        'temperature':    temperature,
        'rainfall':       rainfall,
        'humidity':       humidity,
        'previous_cases': previous_cases,
        'mosquito_index': mosquito_index,
    }
    return pd.DataFrame([data])

input_df = get_user_input()

st.subheader("Input Summary")
st.dataframe(input_df)

if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk: Malaria Likely** — Probability: {probability:.2%}")
    else:
        st.success(f"✅ **Low Risk: Malaria Unlikely** — Probability: {probability:.2%}")
    
    st.progress(float(probability))
    st.caption("This is a decision-support tool. Always consult qualified medical professionals.")
```

Run with:

```bash
streamlit run app/app.py
```

### Option B: Flask REST API (`app/app.py`)

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        data['temperature'],
        data['rainfall'],
        data['humidity'],
        data['previous_cases'],
        data['mosquito_index']
    ]])
    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])
    
    return jsonify({
        "prediction": prediction,
        "label": "Malaria Positive" if prediction == 1 else "Malaria Negative",
        "probability": round(probability, 4)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 14. Ethical Considerations

- **Anonymization:** Remove all personally identifiable information (PII) — names, patient IDs, phone numbers — before model training.
- **Bias Awareness:** Evaluate model fairness across age groups and regions. A model trained on data from one region may underperform in another.
- **No Replacement for Medical Diagnosis:** This system is a decision-support tool only. Predictions must not be used as standalone diagnoses.
- **Informed Use:** Outputs should be communicated with confidence intervals and limitations clearly stated to end users.
- **Data Privacy Compliance:** Handle all health records in line with Nigeria's NITDA Data Protection Regulation (NDPR) and WHO data governance guidelines.

---

## 15. Testing Checklist

### Data
- [ ] Dataset downloaded and placed in `data/raw/`
- [ ] No column with >30% missing values
- [ ] Target column (`malaria_occurrence`) confirmed as binary (0/1)
- [ ] Class imbalance checked; SMOTE applied if necessary

### Preprocessing
- [ ] Missing values handled (median/mode fill)
- [ ] Duplicates removed
- [ ] Categorical variables encoded
- [ ] Features scaled (StandardScaler)
- [ ] 80/20 stratified train-test split applied

### Modeling
- [ ] All 4 algorithms trained without errors
- [ ] Models saved as `.pkl` files in `models/`
- [ ] Scaler saved as `models/scaler.pkl`

### Evaluation
- [ ] Accuracy, Precision, Recall, F1-Score computed for all models
- [ ] ROC-AUC computed for all models
- [ ] Confusion matrices plotted and saved
- [ ] ROC curve comparison chart saved
- [ ] Model comparison table exported to `reports/model_comparison.csv`

### Deployment
- [ ] Streamlit/Flask app runs without errors
- [ ] Input form accepts all required features
- [ ] Prediction output shows label and probability
- [ ] Error handling for invalid inputs implemented

---

## 16. References

- WHO (2025). *World Malaria Report 2024*. World Health Organization.
- Nkiruka et al. (2020). Prediction of malaria incidence using climate variables. *Informatics in Medicine Unlocked*.
- Ahmed et al. (2021). Random Forest and Logistic Regression for malaria prediction. *Journal of Healthcare Informatics*.
- Akinyemi et al. (2021). Malaria prediction using Random Forest Algorithm. *Applied AI in Medicine*.
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Kaggle Malaria Dataset: https://www.kaggle.com/datasets/imdevskp/malaria-dataset

---

> Built for academic research. For questions or collaboration, contact the project author.
