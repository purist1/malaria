"""Unit tests for single and batch predictions using temporary mock pipelines.

Updated for the Malaria_Dataset.csv symptom-based feature set:
  Categorical: sex, residence_area
  Numeric: age, length_of_stay, fever, headache, abdominal_pain,
           general_body_malaise, dizziness, vomiting, confusion,
           backache, chest_pain, coughing, joint_pain
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.malaria_forecast.artifacts import save_artifact
from src.malaria_forecast.features import LABEL_MAP
from src.malaria_forecast.predict import predict_single_record, predict_batch_csv


# ---------------------------------------------------------------------------
# Canonical feature lists matching features.py
# ---------------------------------------------------------------------------
_CATEGORICAL = ["sex", "residence_area"]
_NUMERIC = [
    "age", "length_of_stay",
    "fever", "headache", "abdominal_pain", "general_body_malaise",
    "dizziness", "vomiting", "confusion", "backache",
    "chest_pain", "coughing", "joint_pain",
]


@pytest.fixture
def temp_model_pipeline(tmp_path: Path) -> Path:
    """Fixture: train a minimal mock classifier and save all required artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Fit scaler on 2-row dummy numeric data
    scaler = StandardScaler()
    dummy_num = pd.DataFrame(
        [[30.0, 4, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
         [45.0, 7, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0]],
        columns=_NUMERIC,
    )
    scaler.fit(dummy_num)

    # Fit encoder on 2-row dummy categorical data
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    dummy_cat = pd.DataFrame(
        [["Male", "Mangalore"],
         ["Female", "Udupi"]],
        columns=_CATEGORICAL,
    )
    encoder.fit(dummy_cat)

    encoded_cat_cols = list(encoder.get_feature_names_out(_CATEGORICAL))
    final_feature_order = _NUMERIC + encoded_cat_cols

    # Imputation defaults (must mirror keys used by predict.py)
    imputation_defaults: dict = {
        "sex": "Male",
        "residence_area": "Udupi",
        "age": 44.0,
        "length_of_stay": 5.0,
        "fever": 0.0,
        "headache": 0.0,
        "abdominal_pain": 0.0,
        "general_body_malaise": 1.0,
        "dizziness": 0.0,
        "vomiting": 0.0,
        "confusion": 0.0,
        "backache": 1.0,
        "chest_pain": 0.0,
        "coughing": 0.0,
        "joint_pain": 1.0,
    }

    # Fit a minimal LogisticRegression on the processed feature space
    clf = LogisticRegression()
    X_dummy = pd.DataFrame(
        [[0.0] * len(final_feature_order),
         [1.0] * len(final_feature_order)],
        columns=final_feature_order,
    )
    clf.fit(X_dummy, [0, 1])

    # Persist artifacts
    save_artifact(clf, model_dir / "best_model.joblib")
    save_artifact(clf, model_dir / "logistic_regression.joblib")
    save_artifact(
        {
            "scaler": scaler,
            "encoder": encoder,
            "imputation_defaults": imputation_defaults,
            "final_feature_order": final_feature_order,
            "best_model_name": "Logistic Regression",
            "best_model_file": "logistic_regression.joblib",
            "label_map": LABEL_MAP,
        },
        model_dir / "metadata.joblib",
    )

    return model_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_predict_single_record_all_fields(temp_model_pipeline: Path) -> None:
    """Full single-record prediction returns a valid label and probability."""
    record = {
        "sex": "Male",
        "residence_area": "Mangalore",
        "age": 35.0,
        "length_of_stay": 4.0,
        "fever": 1.0,
        "headache": 0.0,
        "abdominal_pain": 0.0,
        "general_body_malaise": 1.0,
        "dizziness": 0.0,
        "vomiting": 0.0,
        "confusion": 0.0,
        "backache": 1.0,
        "chest_pain": 0.0,
        "coughing": 0.0,
        "joint_pain": 1.0,
    }
    res = predict_single_record(
        record=record,
        model_dir=temp_model_pipeline,
        model_name="logistic_regression",
    )
    assert "label" in res
    assert res["label"] in list(LABEL_MAP.values())
    assert "probability" in res
    assert 0.0 <= res["probability"] <= 1.0
    assert "model_used" in res


def test_predict_single_record_partial_fields(temp_model_pipeline: Path) -> None:
    """Partial record (missing fields) must fall back to imputation defaults."""
    partial_record = {"sex": "Female", "age": 50.0, "fever": 1.0}
    res = predict_single_record(
        record=partial_record,
        model_dir=temp_model_pipeline,
    )
    assert res["label"] in list(LABEL_MAP.values())
    assert 0.0 <= res["probability"] <= 1.0


def test_predict_batch_csv(temp_model_pipeline: Path, tmp_path: Path) -> None:
    """Batch predictions are saved with predicted_label and predicted_probability."""
    input_df = pd.DataFrame([
        {
            "Sex": "Male", "Residence_Area": "Mangalore", "Age": 30,
            "DOA": "01-01-2020 08:00", "Discharge_Date": "05-01-2020 08:00",
            "Fever": 1, "Headache": 0, "Abdominal_Pain": 0,
            "General_Body_Malaise": 1, "Dizziness": 0, "Vomiting": 0,
            "Confusion": 0, "Backache": 1, "Chest_Pain": 0,
            "Coughing": 0, "Joint_Pain": 1,
            "IP_Number": "xx01", "Primary_Code": "B50.9",
            "Diagnosis_Type": "Mixed", "Target": 1, "Risk_Score": 7,
        },
        {
            "Sex": "Female", "Residence_Area": "Udupi", "Age": 45,
            "DOA": "02-01-2020 09:00", "Discharge_Date": "06-01-2020 09:00",
            "Fever": 0, "Headache": 1, "Abdominal_Pain": 1,
            "General_Body_Malaise": 0, "Dizziness": 1, "Vomiting": 1,
            "Confusion": 0, "Backache": 0, "Chest_Pain": 1,
            "Coughing": 1, "Joint_Pain": 0,
            "IP_Number": "xx02", "Primary_Code": "B54",
            "Diagnosis_Type": "Vivax", "Target": 0, "Risk_Score": 5,
        },
    ])
    input_csv = tmp_path / "batch_input.csv"
    input_df.to_csv(input_csv, index=False)
    output_csv = tmp_path / "batch_output.csv"

    predict_batch_csv(
        input_csv_path=input_csv,
        output_csv_path=output_csv,
        model_dir=temp_model_pipeline,
    )

    assert output_csv.exists()
    out_df = pd.read_csv(output_csv)
    assert "predicted_label" in out_df.columns
    assert "predicted_probability" in out_df.columns
    assert len(out_df) == 2


def test_missing_model_file_raises(tmp_path: Path) -> None:
    """predict_single_record must raise FileNotFoundError if model dir is empty."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        predict_single_record(record={"age": 30}, model_dir=empty_dir)
