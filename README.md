# Malaria Occurrence Prediction System

Machine learning project aligned to the development guide for binary malaria occurrence prediction (`Malaria Positive` / `Malaria Negative`).

The current implementation trains and compares four models:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM

## Dataset

Default dataset path:
- `dataset/Malaria Diseases dataset - .csv`

Raw columns are normalized automatically (for example `Hemoglobin(Hb%)` → `hemoglobin_hb_pct`, `Result` → `malaria_occurrence`).

Target labels in `Result` are converted to binary:
- `positive` / `Positive` / `1` → `1`
- `negative` / `Negative` / `0` → `0`

## Quick Start

```bash
cd /path/to/malaria-forcast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train + evaluate all four models
python scripts/train_model.py

# Run a single prediction (uses saved feature defaults when values are omitted)
python scripts/predict.py --sex Male --age 35 --hemoglobin-hb-pct 14.2

# Run batch predictions from a CSV
python scripts/predict.py --input-csv path/to/input.csv --output data/processed/predictions.csv

# Launch dashboard
streamlit run dashboard/app.py
```

## Validation Workflow

Run these commands from the repository root (`malaria-forcast`) to validate setup and model flow end-to-end:

```bash
.venv/bin/python scripts/train_model.py --config config/config.yaml
.venv/bin/python scripts/predict.py --config config/config.yaml --sex Male --age 35 --hemoglobin-hb-pct 14.2 --output data/processed/predictions.csv
```

Expected outputs include refreshed model artifacts in `models/`, comparison metrics in `reports/model_comparison.csv`, and predictions in `data/processed/predictions.csv`.

## Training Outputs

After training, the project writes:
- Model artifacts to `models/`
  - `logistic_regression.joblib`
  - `decision_tree.joblib`
  - `random_forest.joblib`
  - `svm.joblib`
  - `best_model.joblib`
  - `metadata.joblib`
- Comparison metrics to `reports/model_comparison.csv`
- Figures to `reports/figures/`
  - confusion matrices per model
  - ROC comparison curve
  - model metric bar chart
- Processed dataset to `data/processed/malaria_processed.csv`

## Dashboard

The Streamlit app (`dashboard/app.py`) provides:
- One-click model retraining
- Single-record prediction form
- Probability output with configurable threshold
- Model comparison table
- Dataset preview

## Configuration

Edit `config/config.yaml` to control:
- dataset path and target column
- train/test split and random seed
- model hyperparameters
- artifact/report output paths
- dashboard title and threshold
