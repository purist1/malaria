"""Unit tests for split, imputation, scaling and feature ordering in preprocessing.py."""

from __future__ import annotations

import pandas as pd
import pytest
from src.malaria_forecast.preprocessing import preprocess_data
from src.malaria_forecast.features import ALL_NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


def _make_symptom_df(n_repeats: int = 6) -> pd.DataFrame:
    """Build a synthetic dataframe that mirrors the Africa-wide dataset feature set."""
    rows = []
    base_age = [30.0, None, 45.0, 60.0, 25.0, None]
    base_hb = [10.5, 12.0, None, 9.8, 13.5, 11.0]
    for i in range(n_repeats):
        for idx in range(6):
            age_val = 1.0 + (i * 6 + idx)
            rows.append({
                "sex": ["Male", "Female", "Male", "Female", "Male", "Female"][idx],
                "residence": ["Rural", "Urban", "Rural", "Urban", "Rural", "Urban"][idx],
                "season": ["Rainy", "Dry", "Rainy", "Dry", "Rainy", "Dry"][idx],
                "age_years": age_val,
                "hemoglobin_g_dl": base_hb[idx],
                "fever_days": [3, 0, 2, 0, 4, 0][idx],
                "uses_mosquito_net": [1, 0, 1, 0, 1, 0][idx],
                "has_fever": [1, 0, 1, 0, 1, 0][idx],
                "has_chills": [1, 0, 0, 0, 1, 0][idx],
                "has_headache": [1, 0, 1, 0, 0, 0][idx],
                "has_vomiting": [0, 0, 1, 0, 0, 0][idx],
                "has_diarrhea": [0, 0, 0, 0, 1, 0][idx],
                "has_weakness": [1, 0, 1, 0, 1, 0][idx],
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

    for col in ALL_NUMERIC_FEATURES:
        assert col in defaults, f"Missing imputation default for: {col}"
    for col in CATEGORICAL_FEATURES:
        assert col in defaults, f"Missing imputation default for: {col}"

    assert defaults["age_years"] is not None
    assert not pd.isna(defaults["age_years"])


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
    """All symptom binary columns must appear in the final feature order."""
    df = _make_symptom_df()
    res = preprocess_data(df, _CONFIG)
    feature_order = res["final_feature_order"]

    symptom_cols = [
        "uses_mosquito_net", "has_fever", "has_chills",
        "has_headache", "has_vomiting", "has_diarrhea", "has_weakness"
    ]
    for col in symptom_cols:
        assert col in feature_order, f"Symptom column missing from feature order: {col}"
