"""Unit tests for single and batch predictions using temporary mock pipelines."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.malaria_forecast.artifacts import save_artifact
from src.malaria_forecast.features import LABEL_MAP
from src.malaria_forecast.predict import predict_single_record, predict_batch_csv


_CATEGORICAL = ["sex", "residence", "season"]
_NUMERIC = [
    "age_years", "hemoglobin_g_dl", "fever_days",
    "uses_mosquito_net", "has_fever", "has_chills",
    "has_headache", "has_vomiting", "has_diarrhea", "has_weakness",
]


@pytest.fixture
def temp_model_pipeline(tmp_path: Path) -> Path:
    """Fixture: train a minimal mock classifier and save all required artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    scaler = StandardScaler()
    dummy_num = pd.DataFrame(
        [[30.0, 10.5, 3, 1, 1, 1, 1, 0, 0, 1],
         [45.0, 13.5, 0, 0, 0, 0, 0, 0, 0, 0]],
        columns=_NUMERIC,
    )
    scaler.fit(dummy_num)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    dummy_cat = pd.DataFrame(
        [["Male", "Rural", "Rainy"],
         ["Female", "Urban", "Dry"]],
        columns=_CATEGORICAL,
    )
    encoder.fit(dummy_cat)

    encoded_cat_cols = list(encoder.get_feature_names_out(_CATEGORICAL))
    final_feature_order = _NUMERIC + encoded_cat_cols

    imputation_defaults: dict = {
        "sex": "Male",
        "residence": "Rural",
        "season": "Rainy",
        "age_years": 30.0,
        "hemoglobin_g_dl": 11.4,
        "fever_days": 1.0,
        "uses_mosquito_net": 1,
        "has_fever": 1,
        "has_chills": 0,
        "has_headache": 0,
        "has_vomiting": 0,
        "has_diarrhea": 0,
        "has_weakness": 1,
    }

    clf = LogisticRegression()
    X_dummy = pd.DataFrame(
        [[0.0] * len(final_feature_order),
         [1.0] * len(final_feature_order)],
        columns=final_feature_order,
    )
    clf.fit(X_dummy, [0, 1])

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


def test_predict_single_record_all_fields(temp_model_pipeline: Path) -> None:
    """Full single-record prediction returns a valid label and probability."""
    record = {
        "sex": "Male",
        "residence": "Rural",
        "season": "Rainy",
        "age_years": 35.0,
        "hemoglobin_g_dl": 10.2,
        "fever_days": 3.0,
        "uses_mosquito_net": 1,
        "has_fever": 1,
        "has_chills": 1,
        "has_headache": 1,
        "has_vomiting": 0,
        "has_diarrhea": 0,
        "has_weakness": 1,
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
    partial_record = {"sex": "Female", "age_years": 50.0, "has_fever": 1}
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
            "patient_id": "MAL001",
            "sex": "Male", "residence": "Rural", "season": "Rainy",
            "age_years": 30, "hemoglobin_g_dl": 10.5, "fever_days": 3,
            "uses_mosquito_net": True, "has_fever": True, "has_chills": True,
            "has_headache": True, "has_vomiting": False, "has_diarrhea": False,
            "has_weakness": True, "malaria_status": "Positive",
        },
        {
            "patient_id": "MAL002",
            "sex": "Female", "residence": "Urban", "season": "Dry",
            "age_years": 45, "hemoglobin_g_dl": 13.5, "fever_days": 0,
            "uses_mosquito_net": False, "has_fever": False, "has_chills": False,
            "has_headache": False, "has_vomiting": False, "has_diarrhea": False,
            "has_weakness": False, "malaria_status": "Negative",
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
        predict_single_record(record={"age_years": 30}, model_dir=empty_dir)
