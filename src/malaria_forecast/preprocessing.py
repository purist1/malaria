"""Data preprocessing module for clinical risk prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.malaria_forecast.features import (
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def preprocess_data(
    df: pd.DataFrame,
    config: dict[str, Any]
) -> dict[str, Any]:
    """Perform train/test splitting, imputation, scaling, and encoding.

    Args:
        df: The normalized input dataframe.
        config: The configuration dictionary.

    Returns:
        dict: A dictionary containing preprocessed datasets and fitted pipeline estimators.
    """
    logger.info("Starting preprocessing pipeline...")

    # Drop rows where target is missing
    initial_rows = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    dropped_target = initial_rows - len(df)
    if dropped_target > 0:
        logger.warning("Dropped %d rows because the target column was missing.", dropped_target)

    # Remove duplicate rows
    pre_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dupes_removed = pre_dedup - len(df)
    if dupes_removed > 0:
        logger.warning("Removed %d duplicate rows.", dupes_removed)
    else:
        logger.info("No duplicate rows found.")

    # Log class imbalance
    positive = int(df[TARGET_COLUMN].sum())
    negative = int((df[TARGET_COLUMN] == 0).sum())
    total = len(df)
    logger.info(
        "Class distribution — Positive: %d (%.1f%%), Negative: %d (%.1f%%) | Total: %d",
        positive, 100 * positive / total,
        negative, 100 * negative / total,
        total,
    )

    # Separate features and target
    X = df[CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratified Train/Test split (80/20)
    split_cfg = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg["stratify"] else None
    )

    logger.info("Split data into train (size: %d) and test (size: %d)", len(X_train), len(X_test))

    # Reset indices to align concat
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Compute imputation defaults on training set only
    imputation_defaults: dict[str, Any] = {}
    for col in ALL_NUMERIC_FEATURES:
        imputation_defaults[col] = float(X_train[col].median())
    for col in CATEGORICAL_FEATURES:
        mode_val = X_train[col].mode()
        imputation_defaults[col] = str(mode_val.iloc[0]) if not mode_val.empty else "unknown"

    logger.info("Computed training set imputation defaults: %s", imputation_defaults)

    # Apply imputation to train and test
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()
    for col, val in imputation_defaults.items():
        if col in X_train_filled.columns:
            X_train_filled[col] = X_train_filled[col].fillna(val)
        if col in X_test_filled.columns:
            X_test_filled[col] = X_test_filled[col].fillna(val)

    # Standard scale numeric features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_filled[ALL_NUMERIC_FEATURES])
    X_test_num_scaled = scaler.transform(X_test_filled[ALL_NUMERIC_FEATURES])

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_encoded = encoder.fit_transform(X_train_filled[CATEGORICAL_FEATURES])
    X_test_cat_encoded = encoder.transform(X_test_filled[CATEGORICAL_FEATURES])

    encoded_cat_cols = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    final_feature_order = ALL_NUMERIC_FEATURES + encoded_cat_cols

    # Reconstruct preprocessed feature DataFrames
    X_train_processed = pd.concat([
        pd.DataFrame(X_train_num_scaled, columns=ALL_NUMERIC_FEATURES),
        pd.DataFrame(X_train_cat_encoded, columns=encoded_cat_cols)
    ], axis=1)

    X_test_processed = pd.concat([
        pd.DataFrame(X_test_num_scaled, columns=ALL_NUMERIC_FEATURES),
        pd.DataFrame(X_test_cat_encoded, columns=encoded_cat_cols)
    ], axis=1)

    # Save the fully processed dataset for auditability
    train_out = X_train_processed.copy()
    train_out[TARGET_COLUMN] = y_train

    test_out = X_test_processed.copy()
    test_out[TARGET_COLUMN] = y_test

    full_processed_df = pd.concat([train_out, test_out], axis=0, ignore_index=True)
    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    full_processed_df.to_csv(processed_path, index=False)
    logger.info("Saved fully processed dataset to: %s", processed_path)

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoder": encoder,
        "imputation_defaults": imputation_defaults,
        "final_feature_order": final_feature_order,
    }
