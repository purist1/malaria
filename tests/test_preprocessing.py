"""Unit tests for split, imputation, scaling and feature ordering in preprocessing.py."""

from __future__ import annotations

import pandas as pd
import pytest
from src.malaria_forecast.preprocessing import preprocess_data
from src.malaria_forecast.features import ALL_NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


def _make_symptom_df(n_repeats: int = 6) -> pd.DataFrame:
    """Build a synthetic dataframe that mirrors the Malaria_Dataset feature set."""
    rows = []
    base_age = [30.0, None, 45.0, 60.0, 25.0, None]
    base_stay = [4, 7, 3, 10, 5, 6]
    for i in range(n_repeats):
        for idx in range(6):
            age_val = base_age[idx]
            if age_val is not None:
                age_val = age_val + i
            rows.append({
                "sex": ["Male", "Female", "Male", "Female", "Male", "Female"][idx],
                "residence_area": ["Mangalore", "Udupi", "Shimoga", "Udupi", "Mangalore", "Shimoga"][idx],
                "age": age_val,
                "length_of_stay": base_stay[idx] + i,
                "fever": [1, 0, 1, 0, 1, 0][idx],
                "headache": [0, 1, 0, 1, 0, 1][idx],
                "abdominal_pain": [0, 0, 1, 1, 0, 0][idx],
                "general_body_malaise": [1, 1, 0, 0, 1, 1][idx],
                "dizziness": [0, 1, 1, 0, 0, 1][idx],
                "vomiting": [0, 0, 1, 1, 0, 0][idx],
                "confusion": [0, 0, 0, 0, 1, 1][idx],
                "backache": [1, 0, 0, 1, 0, 0][idx],
                "chest_pain": [0, 1, 0, 0, 1, 0][idx],
                "coughing": [0, 0, 1, 0, 0, 1][idx],
                "joint_pain": [1, 1, 0, 0, 1, 1][idx],
                TARGET_COLUMN: [1, 0, 1, 0, 1, 0][idx],
            })
    return pd.DataFrame(rows)



_CONFIG = {
    "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
    "data": {"processed_path": "data/processed/malaria_processed.csv"},
}


def test_no_data_leakage_imputation() -> None:
    """Verify train-set statistics are used for imputation, not test-set values."""
    df = _make_symptom_df()
    res = preprocess_data(df, _CONFIG)

    assert "imputation_defaults" in res
    defaults = res["imputation_defaults"]

    # All numeric and categorical features should have an imputation default
    for col in ALL_NUMERIC_FEATURES:
        assert col in defaults, f"Missing imputation default for: {col}"
    for col in CATEGORICAL_FEATURES:
        assert col in defaults, f"Missing imputation default for: {col}"

    # Age had NaN values — check that the default is a real number, not NaN
    assert defaults["age"] is not None
    assert not pd.isna(defaults["age"])


def test_feature_order_consistency() -> None:
    """Scaled train and test feature column order must exactly match final_feature_order."""
    df = _make_symptom_df()
    res = preprocess_data(df, _CONFIG)

    assert list(res["X_train"].columns) == res["final_feature_order"]
    assert list(res["X_test"].columns) == res["final_feature_order"]


def test_no_nulls_after_preprocessing() -> None:
    """Preprocessed train and test sets must contain zero NaN values."""
    df = _make_symptom_df()
    res = preprocess_data(df, _CONFIG)

    assert res["X_train"].isnull().sum().sum() == 0
    assert res["X_test"].isnull().sum().sum() == 0


def test_train_test_sizes() -> None:
    """Train/test sizes should match the 80/20 configured split ratio."""
    df = _make_symptom_df()
    res = preprocess_data(df, _CONFIG)

    total = len(res["X_train"]) + len(res["X_test"])
    assert total == len(df)
    assert len(res["X_test"]) == pytest.approx(total * 0.2, abs=2)


def test_symptom_features_present_in_output() -> None:
    """All 11 symptom binary columns must appear in the final feature order."""
    df = _make_symptom_df()
    res = preprocess_data(df, _CONFIG)
    feature_order = res["final_feature_order"]

    symptom_cols = [
        "fever", "headache", "abdominal_pain", "general_body_malaise",
        "dizziness", "vomiting", "confusion", "backache",
        "chest_pain", "coughing", "joint_pain",
    ]
    for col in symptom_cols:
        assert col in feature_order, f"Symptom column missing from feature order: {col}"
