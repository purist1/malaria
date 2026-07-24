"""Unit tests for column normalization and target mapping in data_loader.py."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from src.malaria_forecast.data_loader import normalize_column_name, load_raw_dataset


def test_normalize_column_name_basic() -> None:
    """Test generic snake_case normalization for common column patterns."""
    assert normalize_column_name("Age_Years") == "age_years"
    assert normalize_column_name("Sex") == "sex"
    assert normalize_column_name("Residence") == "residence"
    assert normalize_column_name("Has_Fever") == "has_fever"
    assert normalize_column_name("Hemoglobin_g_dl") == "hemoglobin_g_dl"
    assert normalize_column_name("Patient_ID") == "patient_id"
    assert normalize_column_name("Malaria_Status") == "malaria_status"
    assert normalize_column_name("  Has_Chills  ") == "has_chills"


def test_normalize_column_name_special_chars() -> None:
    """Ensure non-alphanumeric characters collapse to single underscores."""
    assert normalize_column_name("Uses_Mosquito_Net") == "uses_mosquito_net"
    assert normalize_column_name("Fever_Days") == "fever_days"


def _make_minimal_csv(tmp_path: Path, extra: dict | None = None) -> Path:
    """Build a minimal 4-row CSV that matches the Africa-wide HF dataset structure."""
    rows = {
        "patient_id": ["MAL001", "MAL002", "MAL003", "MAL004"],
        "age_years": [30, 45, 60, 25],
        "age_months": [360, 540, 720, 300],
        "age_group": ["12+", "12+", "12+", "12+"],
        "sex": ["Male", "Female", "Male", "Female"],
        "residence": ["Rural", "Urban", "Rural", "Urban"],
        "season": ["Rainy", "Dry", "Rainy", "Dry"],
        "uses_mosquito_net": [True, False, True, False],
        "malaria_status": ["Positive", "Negative", "Positive", "Negative"],
        "parasitemia_level": ["High", None, "Low", None],
        "parasitemia_count": [10000, 0, 500, 0],
        "plasmodium_species": ["P. falciparum", None, "P. vivax", None],
        "hemoglobin_g_dl": [10.2, 13.5, 9.8, 14.1],
        "anemia_status": ["Moderate", None, "Moderate", None],
        "fever_days": [3, 0, 2, 0],
        "has_fever": [True, False, True, False],
        "has_chills": [True, False, False, False],
        "has_headache": [True, False, True, False],
        "has_vomiting": [False, False, True, False],
        "has_diarrhea": [False, False, False, False],
        "has_weakness": [True, False, True, False],
        "severe_malaria": [False, False, False, False],
        "cerebral_malaria": [False, False, False, False],
        "respiratory_distress": [False, False, False, False],
        "shock": [False, False, False, False],
        "acute_kidney_injury": [False, False, False, False],
        "outcome": ["Treated", "Healthy", "Treated", "Healthy"],
        "malaria_probability_score": [0.85, 0.12, 0.78, 0.05],
    }
    if extra:
        rows.update(extra)
    csv_file = tmp_path / "test_data.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    return csv_file


def test_load_raw_dataset_shape(tmp_path: Path) -> None:
    """Loaded dataframe drops leakage columns."""
    csv_file = _make_minimal_csv(tmp_path)
    df = load_raw_dataset(csv_file)

    # Should NOT contain leakage / administrative columns
    for banned in ("patient_id", "parasitemia_level", "parasitemia_count",
                   "plasmodium_species", "anemia_status", "outcome",
                   "malaria_probability_score", "severe_malaria"):
        assert banned not in df.columns, f"Banned column found: {banned}"

    # Target must be 0/1 integer
    assert set(df["malaria_status"].unique()).issubset({0, 1})

    # No nulls
    assert df.isnull().sum().sum() == 0


def test_boolean_casting(tmp_path: Path) -> None:
    """Boolean columns must be converted to 0/1 integers."""
    csv_file = _make_minimal_csv(tmp_path)
    df = load_raw_dataset(csv_file)
    assert set(df["uses_mosquito_net"].unique()).issubset({0, 1})
    assert set(df["has_fever"].unique()).issubset({0, 1})


def test_invalid_target_raises(tmp_path: Path) -> None:
    """Test that non-binary string target values raise a ValueError."""
    csv_file = _make_minimal_csv(tmp_path)
    df = pd.read_csv(csv_file)
    df.loc[0, "malaria_status"] = "unknown_status"
    bad_csv = tmp_path / "bad_target.csv"
    df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError):
        load_raw_dataset(bad_csv)
