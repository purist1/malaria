"""Data loading and header normalization module.

Supports the Africa-wide synthetic malaria dataset:
  electricsheepafrica/africa-synth-malaria-malaria-dataset-all

The loader:
  1. Reads the locally-cached CSV (dataset/africa_malaria_hf/train.csv or
     a merged train+validation file, depending on config).
  2. Drops all post-diagnosis leakage columns (defined in features.py).
  3. Converts boolean columns to int (0/1).
  4. Normalizes column names to snake_case.
  5. Encodes the target: malaria_status → Positive=1, Negative=0.
  6. Fills missing hemoglobin_g_dl with NaN (handled during preprocessing
     via median imputation on the training fold).
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Column name normalization ─────────────────────────────────────────────────

def normalize_column_name(raw_name: str) -> str:
    """Normalize raw dataset column headers to canonical snake_case.

    Examples:
        'Age_Years'   -> 'age_years'
        'Has_Fever'   -> 'has_fever'
        'HbG/dl'      -> 'hbg_dl'
    """
    name = raw_name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


# ── Boolean → int conversion ──────────────────────────────────────────────────

_BOOL_COLUMNS = {
    "uses_mosquito_net",
    "has_fever",
    "has_chills",
    "has_headache",
    "has_vomiting",
    "has_diarrhea",
    "has_weakness",
    "severe_malaria",
    "cerebral_malaria",
    "respiratory_distress",
    "shock",
    "acute_kidney_injury",
}


def _cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean / object True/False columns to integer 0/1."""
    for col in _BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: 1 if v is True or str(v).strip().lower() == "true" else 0
            ).astype(int)
    return df


# ── Main loader ───────────────────────────────────────────────────────────────

def load_raw_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load the Africa-wide malaria dataset CSV, clean and normalise.

    Pipeline:
        1. Load CSV.
        2. Normalize column names to snake_case.
        3. Drop post-diagnosis / leakage / administrative columns.
        4. Convert boolean columns to int (0/1).
        5. Ensure hemoglobin_g_dl is numeric (NaN where missing → imputed later).
        6. Validate and encode the target column (malaria_status → 0/1).

    Args:
        filepath: Path to the raw CSV file (locally cached HF export).

    Returns:
        pd.DataFrame: Ready-to-preprocess dataframe with normalised columns.
    """
    from src.malaria_forecast.features import HF_DROP_COLUMNS, TARGET_COLUMN, TARGET_VALUE_MAP

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at: {path}\n"
            "Run `python scripts/download_dataset.py` to download the dataset."
        )

    df = pd.read_csv(path)
    logger.info("Loaded dataset from %s  (shape: %s)", path, df.shape)

    # --- Step 1: Normalise column names ---
    mapping = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=mapping)

    # --- Step 2: Drop leakage / admin columns ---
    drop_targets = HF_DROP_COLUMNS & set(df.columns)
    if drop_targets:
        logger.info("Dropping leakage/admin columns: %s", sorted(drop_targets))
        df = df.drop(columns=list(drop_targets))

    # --- Step 3: Convert booleans to int ---
    df = _cast_booleans(df)

    # --- Step 4: Ensure hemoglobin is numeric; NaN rows will be imputed ---
    if "hemoglobin_g_dl" in df.columns:
        df["hemoglobin_g_dl"] = pd.to_numeric(df["hemoglobin_g_dl"], errors="coerce")

    # --- Step 5: Validate and encode target ---
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    raw_target = df[TARGET_COLUMN]

    if raw_target.dtype in (int, "int64", "Int64"):
        unique_vals = set(raw_target.dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' has non-binary values: {unique_vals}"
            )
        df[TARGET_COLUMN] = raw_target.astype(int)
    else:
        encoded = raw_target.astype(str).str.strip().str.lower()
        unmapped = set(encoded.unique()) - set(TARGET_VALUE_MAP.keys())
        if unmapped:
            raise ValueError(
                f"Unrecognized target labels: {unmapped}. "
                f"Expected one of: {set(TARGET_VALUE_MAP.keys())}"
            )
        df[TARGET_COLUMN] = encoded.map(TARGET_VALUE_MAP).astype(int)

    pos = int(df[TARGET_COLUMN].sum())
    neg = int((df[TARGET_COLUMN] == 0).sum())
    logger.info(
        "Target distribution — Positive: %d (%.1f%%), Negative: %d (%.1f%%) | Total: %d",
        pos, 100 * pos / len(df),
        neg, 100 * neg / len(df),
        len(df),
    )

    return df
