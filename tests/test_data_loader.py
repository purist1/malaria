"""Unit tests for column normalization and target mapping in data_loader.py."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from src.malaria_forecast.data_loader import normalize_column_name, load_raw_dataset


def test_normalize_column_name_basic() -> None:
    """Test generic snake_case normalization for common column patterns."""
    assert normalize_column_name("Age") == "age"
    assert normalize_column_name("Sex") == "sex"
    assert normalize_column_name("Residence_Area") == "residence_area"
    assert normalize_column_name("General_Body_Malaise") == "general_body_malaise"
    assert normalize_column_name("Abdominal_Pain") == "abdominal_pain"
    assert normalize_column_name("IP_Number") == "ip_number"
    assert normalize_column_name("Target") == "target"
    assert normalize_column_name("Risk_Score") == "risk_score"
    assert normalize_column_name("  Fever  ") == "fever"  # strip whitespace


def test_normalize_column_name_special_chars() -> None:
    """Ensure non-alphanumeric characters collapse to single underscores."""
    assert normalize_column_name("Joint_Pain") == "joint_pain"
    assert normalize_column_name("DOA") == "doa"
    assert normalize_column_name("Discharge_Date") == "discharge_date"


def _make_minimal_csv(tmp_path: Path, extra: dict | None = None) -> Path:
    """Build a minimal 4-row CSV that matches the Malaria_Dataset structure."""
    rows = {
        "IP_Number": ["14xx01", "14xx02", "14xx03", "14xx04"],
        "Age": [30, 45, 60, 25],
        "Sex": ["Male", "Female", "Male", "Female"],
        "Residence_Area": ["Mangalore", "Udupi", "Shimoga", "Mangalore"],
        "DOA": ["01-01-2020 08:00", "02-01-2020 09:00", "03-01-2020 10:00", "04-01-2020 11:00"],
        "Discharge_Date": ["05-01-2020 08:00", "06-01-2020 09:00", "07-01-2020 10:00", "08-01-2020 11:00"],
        "Fever": [1, 0, 1, 0],
        "Headache": [0, 1, 0, 1],
        "Abdominal_Pain": [0, 0, 1, 1],
        "General_Body_Malaise": [1, 1, 0, 0],
        "Dizziness": [0, 1, 1, 0],
        "Vomiting": [0, 0, 1, 1],
        "Confusion": [0, 0, 0, 0],
        "Backache": [1, 0, 0, 1],
        "Chest_Pain": [0, 1, 0, 0],
        "Coughing": [0, 0, 1, 0],
        "Joint_Pain": [1, 1, 0, 0],
        "Primary_Code": ["B50.9", "B54", "B51.0", "B52.0"],
        "Diagnosis_Type": ["Mixed", "Vivax", "Falciparum", "Unspecified"],
        "Target": [1, 0, 1, 0],
        "Risk_Score": [7, 4, 5, 3],
    }
    if extra:
        rows.update(extra)
    csv_file = tmp_path / "test_data.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    return csv_file


def test_load_raw_dataset_shape(tmp_path: Path) -> None:
    """Loaded dataframe drops leakage columns and adds length_of_stay."""
    csv_file = _make_minimal_csv(tmp_path)
    df = load_raw_dataset(csv_file)

    # Should NOT contain leakage / administrative columns
    for banned in ("ip_number", "primary_code", "diagnosis_type", "risk_score",
                   "doa", "discharge_date"):
        assert banned not in df.columns, f"Banned column found: {banned}"

    # Must contain engineered feature
    assert "length_of_stay" in df.columns

    # Target must be 0/1 integer
    assert set(df["target"].unique()).issubset({0, 1})

    # No nulls
    assert df.isnull().sum().sum() == 0


def test_length_of_stay_engineering(tmp_path: Path) -> None:
    """Engineered length_of_stay must be non-negative integers."""
    csv_file = _make_minimal_csv(tmp_path)
    df = load_raw_dataset(csv_file)
    assert (df["length_of_stay"] >= 0).all()
    assert df["length_of_stay"].dtype in ("int64", "int32", int)


def test_invalid_target_raises(tmp_path: Path) -> None:
    """Test that non-binary string target values raise a ValueError."""
    csv_file = _make_minimal_csv(tmp_path)
    # Overwrite with a string-label target and unexpected value
    df = pd.read_csv(csv_file)
    df["Target"] = df["Target"].astype(str)
    df.loc[0, "Target"] = "unknown_status"
    bad_csv = tmp_path / "bad_target.csv"
    df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError):
        load_raw_dataset(bad_csv)
