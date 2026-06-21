"""Inference orchestration module for single and batch predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.malaria_forecast.artifacts import load_artifact
from src.malaria_forecast.data_loader import normalize_column_name
from src.malaria_forecast.features import ALL_NUMERIC_FEATURES, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


def preprocess_prediction_input(
    input_df: pd.DataFrame,
    metadata: dict[str, Any]
) -> pd.DataFrame:
    """Standardize, impute, scale, and encode features for prediction.

    Args:
        input_df: The raw input dataframe (single row or batch).
        metadata: Pipeline metadata containing scaler, encoder, defaults, and order.

    Returns:
        pd.DataFrame: Fully processed dataframe aligned with final features.
    """
    # 1. Normalize column headers
    normalized_df = input_df.copy()
    normalized_df.columns = [normalize_column_name(col) for col in normalized_df.columns]

    imputation_defaults = metadata["imputation_defaults"]
    final_feature_order = metadata["final_feature_order"]

    # 2. Check for missing columns and log warnings if running batch
    required_features = CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES
    missing_cols = [col for col in required_features if col not in normalized_df.columns]

    if len(input_df) > 1 and missing_cols:
        for col in missing_cols:
            logger.warning(
                "Required feature column '%s' is missing in batch input. "
                "Applying imputation default.", col
            )

    # 3. Impute missing columns with saved training set defaults
    for col in CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES:
        if col not in normalized_df.columns:
            normalized_df[col] = imputation_defaults[col]
        else:
            # Handle pandas NA values
            normalized_df[col] = normalized_df[col].fillna(imputation_defaults[col])

    # 4. Standard scale numeric features
    scaler = metadata["scaler"]
    num_data = normalized_df[ALL_NUMERIC_FEATURES]
    num_scaled = scaler.transform(num_data)
    num_scaled_df = pd.DataFrame(num_scaled, columns=ALL_NUMERIC_FEATURES)

    # 5. One-hot encode categorical features
    encoder = metadata["encoder"]
    cat_data = normalized_df[CATEGORICAL_FEATURES]
    cat_encoded = encoder.transform(cat_data)
    encoded_cat_cols = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoded_cat_cols)

    # 6. Combine and re-order columns to match the trained pipeline order
    processed_df = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
    processed_df = processed_df[final_feature_order]

    return processed_df


def predict_single_record(
    record: dict[str, Any],
    model_dir: str | Path = "models",
    model_name: str | None = None
) -> dict[str, Any]:
    """Predict malaria occurrence for a single patient record dictionary.

    Args:
        record: Feature key-value dictionary.
        model_dir: Path to the models directory.
        model_name: Name of the specific model to use (default: best_model).

    Returns:
        dict: Result dictionary containing predicted label, probability, and classifier.
    """
    model_dir = Path(model_dir)
    metadata = load_artifact(model_dir / "metadata.joblib")

    if model_name:
        clf_file = f"{model_name}.joblib"
    else:
        clf_file = "best_model.joblib"

    clf = load_artifact(model_dir / clf_file)

    # Convert single dictionary record to a DataFrame row
    df = pd.DataFrame([record])
    processed_df = preprocess_prediction_input(df, metadata)

    # Execute predictions
    prediction = int(clf.predict(processed_df)[0])
    probability = 0.0
    if hasattr(clf, "predict_proba"):
        probability = float(clf.predict_proba(processed_df)[0][1])

    label_map = metadata["label_map"]
    label = label_map.get(prediction, "Unknown")

    return {
        "prediction": prediction,
        "label": label,
        "probability": probability,
        "model_used": model_name or metadata["best_model_name"],
    }


def predict_batch_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    model_dir: str | Path = "models",
    model_name: str | None = None
) -> None:
    """Predict malaria occurrence for a batch input CSV file.

    Args:
        input_csv_path: Path to the input CSV file.
        output_csv_path: Target path to save output prediction CSV.
        model_dir: Path to models directory.
        model_name: Specific model classifier to use (default: best_model).
    """
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Batch input CSV not found at: {input_path}")

    model_dir = Path(model_dir)
    metadata = load_artifact(model_dir / "metadata.joblib")

    if model_name:
        clf_file = f"{model_name}.joblib"
    else:
        clf_file = "best_model.joblib"

    clf = load_artifact(model_dir / clf_file)

    # Read original batch inputs
    df = pd.read_csv(input_path)
    logger.info("Loaded batch input CSV of shape: %s", df.shape)

    processed_df = preprocess_prediction_input(df, metadata)

    # Execute batch predictions
    predictions = clf.predict(processed_df).astype(int)
    probabilities = [0.0] * len(df)
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(processed_df)[:, 1]

    # Map labels
    label_map = metadata["label_map"]
    labels = [label_map.get(pred, "Unknown") for pred in predictions]

    # Add predicted columns to a copy of original dataframe
    out_df = df.copy()
    out_df["predicted_label"] = labels
    out_df["predicted_probability"] = [round(float(prob), 4) for prob in probabilities]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Saved batch predictions to: %s", output_path)
