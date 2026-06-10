"""Prediction helpers for malaria occurrence classifiers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import normalize_feature_columns
from src.validation import validate_prediction_dataframe


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _resolve_label(label_map: dict[Any, str], key: int, default: str) -> str:
    """Resolve integer or string keyed labels from metadata."""
    if key in label_map:
        return str(label_map[key])
    if str(key) in label_map:
        return str(label_map[str(key)])
    return default


def load_artifacts(
    model_dir: str | Path = "models",
    model_name: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load trained model pipeline plus metadata."""
    model_dir = Path(model_dir)
    metadata_path = model_dir / "metadata.joblib"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata missing at {metadata_path}. Run training first."
        )

    metadata = joblib.load(metadata_path)
    if model_name:
        model_filename = f"{_slugify(model_name)}.joblib"
    else:
        model_filename = metadata.get("best_model_file", "best_model.joblib")

    model_path = model_dir / model_filename
    if not model_path.exists() and not model_name:
        fallback = model_dir / "best_model.joblib"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file missing: {model_path}")
    elif not model_path.exists():
        raise FileNotFoundError(f"Requested model not found: {model_path}")

    model = joblib.load(model_path)
    return model, metadata


def prepare_input_frame(
    input_df: pd.DataFrame,
    metadata: dict[str, Any],
    validation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize and align incoming data frame to the trained feature schema."""
    normalized = normalize_feature_columns(input_df, include_target=False)
    feature_columns = metadata.get("feature_columns", [])
    validate_prediction_dataframe(
        df=normalized,
        validation_config=validation_config,
        known_features=feature_columns,
    )
    defaults = metadata.get("feature_defaults", {})

    for col in feature_columns:
        if col not in normalized.columns:
            normalized[col] = defaults.get(col)

    for col, value in defaults.items():
        if col in normalized.columns:
            normalized[col] = normalized[col].fillna(value)

    return normalized[feature_columns]


def predict_dataframe(
    input_df: pd.DataFrame,
    model_dir: str | Path = "models",
    model_name: str | None = None,
    validation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Predict malaria occurrence for one or more records."""
    model, metadata = load_artifacts(model_dir=model_dir, model_name=model_name)
    features = prepare_input_frame(
        input_df=input_df,
        metadata=metadata,
        validation_config=validation_config,
    )

    predictions = model.predict(features).astype(int)
    probabilities = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else None

    label_map = metadata.get("label_map", {})
    positive_label = _resolve_label(label_map=label_map, key=1, default="Malaria Positive")
    negative_label = _resolve_label(label_map=label_map, key=0, default="Malaria Negative")

    output = normalize_feature_columns(input_df, include_target=False).copy().reset_index(drop=True)
    output["prediction"] = predictions
    output["label"] = output["prediction"].map({0: negative_label, 1: positive_label})
    if probabilities is not None:
        output["probability"] = probabilities.round(4)
    return output


def predict_single(
    payload: dict[str, Any],
    model_dir: str | Path = "models",
    model_name: str | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a single record from a Python dictionary."""
    df = pd.DataFrame([payload])
    result = predict_dataframe(
        input_df=df,
        model_dir=model_dir,
        model_name=model_name,
        validation_config=validation_config,
    )
    return result.iloc[0].to_dict()
