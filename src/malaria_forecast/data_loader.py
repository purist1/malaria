"""Data loading and header normalization module.

Updated to handle Malaria_Dataset.csv:
  - Drops administrative/leakage columns (IP_Number, Primary_Code,
    Diagnosis_Type, Risk_Score).
  - Engineers `length_of_stay` (integer days) from DOA and Discharge_Date.
  - Normalizes remaining column names to snake_case.
  - Maps `Target` (0/1 integer) directly as the binary label.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Columns to discard before any feature engineering (administrative / leakage)
_DROP_COLUMNS = {
    "ip_number",       # patient ID — no predictive value
    "primary_code",    # ICD code — post-hoc / leakage
    "diagnosis_type",  # plain text diagnosis — post-hoc / leakage
    "risk_score",      # derived from symptom flags — leakage
}

# Raw column names for date engineering (before normalization)
_DATE_OF_ADMISSION_RAW = "DOA"
_DISCHARGE_DATE_RAW = "Discharge_Date"
_LOS_COLUMN = "length_of_stay"
_DATE_FORMAT = "%d-%m-%Y %H:%M"


def normalize_column_name(raw_name: str) -> str:
    """Normalize raw dataset column headers to canonical snake_case.

    Examples:
        'IP_Number'          -> 'ip_number'
        'General_Body_Malaise' -> 'general_body_malaise'
        'DOA'                -> 'doa'
    """
    name = raw_name.strip().lower()
    # Replace any non-alphanumeric character sequences with a single underscore
    name = re.sub(r"[^a-z0-9]+", "_", name)
    # Clean up duplicate underscores and trailing/leading underscores
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _engineer_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    """Compute length_of_stay (integer days) from DOA and Discharge_Date.

    Both columns are dropped after engineering. If parsing fails, the column
    is filled with 0 and a warning is logged.
    """
    try:
        doa = pd.to_datetime(df[_DATE_OF_ADMISSION_RAW], format=_DATE_FORMAT)
        discharge = pd.to_datetime(df[_DISCHARGE_DATE_RAW], format=_DATE_FORMAT)
        los = (discharge - doa).dt.days.clip(lower=0).fillna(0).astype(int)
        df[_LOS_COLUMN] = los
        logger.info("Engineered '%s': min=%d, max=%d, mean=%.2f days",
                    _LOS_COLUMN, int(los.min()), int(los.max()), float(los.mean()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse admission/discharge dates (%s). "
                       "Setting '%s' to 0.", exc, _LOS_COLUMN)
        df[_LOS_COLUMN] = 0

    df = df.drop(columns=[_DATE_OF_ADMISSION_RAW, _DISCHARGE_DATE_RAW], errors="ignore")
    return df


def load_raw_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load the Malaria_Dataset.csv, engineer features, normalize headers.

    Pipeline:
        1. Load CSV.
        2. Engineer ``length_of_stay`` from admission/discharge dates.
        3. Drop administrative and data-leakage columns.
        4. Normalize all remaining column names to snake_case.
        5. Validate that the target column is present and already binary (0/1).

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Ready-to-preprocess dataframe with normalized columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded raw dataset from %s (shape: %s)", path, df.shape)

    # --- Step 1: Engineer length_of_stay before renaming columns ---
    if _DATE_OF_ADMISSION_RAW in df.columns and _DISCHARGE_DATE_RAW in df.columns:
        df = _engineer_length_of_stay(df)
    else:
        logger.warning(
            "Admission/discharge date columns not found. "
            "Setting '%s' to 0.", _LOS_COLUMN
        )
        df[_LOS_COLUMN] = 0

    # --- Step 2: Compute raw-to-normalized column mapping ---
    mapping = {col: normalize_column_name(col) for col in df.columns}

    print("\n--- Dataset Column Normalization Mapping ---")
    for raw, norm in mapping.items():
        print(f"  '{raw}' -> '{norm}'")
    print("--------------------------------------------\n")

    df = df.rename(columns=mapping)

    # --- Step 3: Drop administrative / leakage columns ---
    drop_targets = _DROP_COLUMNS & set(df.columns)
    if drop_targets:
        logger.info("Dropping administrative/leakage columns: %s", sorted(drop_targets))
        df = df.drop(columns=list(drop_targets))

    # --- Step 4: Validate target column ---
    from src.malaria_forecast.features import TARGET_COLUMN  # avoid circular at module level

    if TARGET_COLUMN not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found after normalization. "
            f"Available columns: {available}"
        )

    raw_target = df[TARGET_COLUMN]

    # Accept integer 0/1 directly (Malaria_Dataset.csv) or string labels
    if raw_target.dtype in (int, "int64", "Int64"):
        unique_vals = set(raw_target.unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' contains non-binary integer values: {unique_vals}"
            )
        df[TARGET_COLUMN] = raw_target.astype(int)
    else:
        encoded = raw_target.astype(str).str.strip().str.lower()
        target_map = {"positive": 1, "1": 1, "negative": 0, "0": 0}
        unmapped = set(encoded.unique()) - set(target_map.keys())
        if unmapped:
            raise ValueError(
                f"Unrecognized target labels: {unmapped}. "
                "Expected binary positive/negative or 0/1."
            )
        df[TARGET_COLUMN] = encoded.map(target_map).astype(int)

    pos = int(df[TARGET_COLUMN].sum())
    neg = int((df[TARGET_COLUMN] == 0).sum())
    logger.info("Target distribution — Positive: %d, Negative: %d (total: %d)", pos, neg, len(df))

    return df
