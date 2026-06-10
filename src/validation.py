"""Validation helpers for training and prediction inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_REQUIRED_FEATURES = [
    "sex",
    "age",
    "hemoglobin_hb_pct",
    "total_wbc_count_cumm",
    "neutrophils",
    "lymphocytes",
    "total_cir_eosinophils",
    "htc_pcv_pct",
    "mch_pg",
    "mchc_g_dl",
    "rdw_cv_pct",
    "platelet_count",
]

DEFAULT_NUMERIC_RANGES: dict[str, dict[str, float]] = {
    "age": {"min": 0, "max": 120},
    "hemoglobin_hb_pct": {"min": 2, "max": 24},
    "total_wbc_count_cumm": {"min": 500, "max": 120_000},
    "neutrophils": {"min": 0, "max": 100},
    "lymphocytes": {"min": 0, "max": 100},
    "total_cir_eosinophils": {"min": 0, "max": 5_000},
    "htc_pcv_pct": {"min": 10, "max": 70},
    "mch_pg": {"min": 10, "max": 45},
    "mchc_g_dl": {"min": 20, "max": 45},
    "rdw_cv_pct": {"min": 5, "max": 35},
    "platelet_count": {"min": 5_000, "max": 1_500_000},
}

DEFAULT_ALLOWED_CATEGORICAL_VALUES: dict[str, set[str]] = {
    "sex": {"male", "female"},
}


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower()


def resolve_validation_rules(
    validation_config: dict[str, Any] | None,
) -> tuple[list[str], dict[str, dict[str, float | None]], dict[str, set[str]]]:
    """Resolve validation settings from config with defaults."""
    validation_config = validation_config or {}

    required_features = list(
        validation_config.get("required_features", DEFAULT_REQUIRED_FEATURES)
    )

    numeric_ranges_raw = validation_config.get("numeric_ranges", DEFAULT_NUMERIC_RANGES)
    numeric_ranges: dict[str, dict[str, float | None]] = {}
    for column, bounds in numeric_ranges_raw.items():
        if not isinstance(bounds, dict):
            continue
        min_value = bounds.get("min")
        max_value = bounds.get("max")
        numeric_ranges[column] = {
            "min": float(min_value) if min_value is not None else None,
            "max": float(max_value) if max_value is not None else None,
        }

    categorical_raw = validation_config.get(
        "categorical_allowed_values",
        {k: sorted(v) for k, v in DEFAULT_ALLOWED_CATEGORICAL_VALUES.items()},
    )
    categorical_allowed: dict[str, set[str]] = {}
    for column, values in categorical_raw.items():
        if values is None:
            continue
        normalized_values = {_normalize_token(value) for value in values}
        normalized_values = {value for value in normalized_values if value}
        if normalized_values:
            categorical_allowed[column] = normalized_values

    return required_features, numeric_ranges, categorical_allowed


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    context: str,
) -> None:
    """Validate required columns exist on the dataframe."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def validate_numeric_ranges(
    df: pd.DataFrame,
    numeric_ranges: dict[str, dict[str, float | None]],
    context: str,
) -> None:
    """Validate numeric columns are parseable and within configured ranges."""
    for column, bounds in numeric_ranges.items():
        if column not in df.columns:
            continue

        raw_series = df[column]
        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        invalid_numeric = raw_series[raw_series.notna() & numeric_series.isna()]
        if not invalid_numeric.empty:
            sample = invalid_numeric.head(5).tolist()
            raise ValueError(
                f"{context}: non-numeric values found in '{column}'. "
                f"Examples: {sample}"
            )

        min_value = bounds.get("min")
        max_value = bounds.get("max")
        out_of_range = pd.Series(False, index=df.index)

        if min_value is not None:
            out_of_range |= numeric_series.notna() & (numeric_series < float(min_value))
        if max_value is not None:
            out_of_range |= numeric_series.notna() & (numeric_series > float(max_value))

        if out_of_range.any():
            sample_values = numeric_series[out_of_range].head(5).tolist()
            raise ValueError(
                f"{context}: values out of range for '{column}' "
                f"(min={min_value}, max={max_value}). Examples: {sample_values}"
            )


def validate_categorical_values(
    df: pd.DataFrame,
    categorical_allowed_values: dict[str, set[str]],
    context: str,
) -> None:
    """Validate configured categorical columns against allowed vocabularies."""
    for column, allowed in categorical_allowed_values.items():
        if column not in df.columns or not allowed:
            continue
        values = df[column].dropna().astype(str).map(_normalize_token)
        unexpected = sorted(set(values.unique()) - allowed)
        if unexpected:
            raise ValueError(
                f"{context}: unexpected values in '{column}': {unexpected[:5]}. "
                f"Allowed values: {sorted(allowed)}"
            )


def validate_training_dataframe(
    df: pd.DataFrame,
    target_col: str = "malaria_occurrence",
    validation_config: dict[str, Any] | None = None,
) -> None:
    """Validate training dataset schema and value quality."""
    required_features, numeric_ranges, categorical_allowed = resolve_validation_rules(
        validation_config
    )
    required_columns = list(dict.fromkeys(required_features + [target_col]))
    validate_required_columns(
        df=df,
        required_columns=required_columns,
        context="Training dataset validation failed",
    )
    validate_numeric_ranges(
        df=df,
        numeric_ranges=numeric_ranges,
        context="Training dataset validation failed",
    )
    validate_categorical_values(
        df=df,
        categorical_allowed_values=categorical_allowed,
        context="Training dataset validation failed",
    )

    unique_targets = set(pd.to_numeric(df[target_col], errors="coerce").dropna().astype(int).tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            "Training dataset validation failed: target column must be binary 0/1, "
            f"got values: {sorted(unique_targets)}"
        )


def validate_prediction_dataframe(
    df: pd.DataFrame,
    validation_config: dict[str, Any] | None = None,
    known_features: list[str] | None = None,
) -> None:
    """Validate prediction payload values before inference."""
    _, numeric_ranges, categorical_allowed = resolve_validation_rules(validation_config)

    if known_features is not None:
        unknown = sorted(set(df.columns) - set(known_features))
        if unknown:
            raise ValueError(
                "Prediction validation failed: unknown feature columns supplied: "
                f"{unknown}"
            )

    validate_numeric_ranges(
        df=df,
        numeric_ranges=numeric_ranges,
        context="Prediction payload validation failed",
    )
    validate_categorical_values(
        df=df,
        categorical_allowed_values=categorical_allowed,
        context="Prediction payload validation failed",
    )
