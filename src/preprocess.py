"""Preprocessing utilities for malaria occurrence prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.validation import validate_training_dataframe

COLUMN_RENAMES = {
    "hemoglobin_hb": "hemoglobin_hb_pct",
    "htc_pcv": "htc_pcv_pct",
    "rdw_cv": "rdw_cv_pct",
    "result": "malaria_occurrence",
}

POSITIVE_LABELS = {"1", "positive", "pos", "yes", "true", "malaria positive"}
NEGATIVE_LABELS = {"0", "negative", "neg", "no", "false", "malaria negative"}


def canonicalize_column_name(name: str) -> str:
    """Normalize a raw column name into snake_case."""
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return COLUMN_RENAMES.get(normalized, normalized)


def normalize_feature_columns(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Normalize incoming dataset headers into canonical names."""
    renamed = {col: canonicalize_column_name(col) for col in df.columns}
    out = df.rename(columns=renamed).copy()
    if not include_target and "malaria_occurrence" in out.columns:
        out = out.drop(columns=["malaria_occurrence"])
    return out


def encode_target_column(df: pd.DataFrame, target_col: str = "malaria_occurrence") -> pd.DataFrame:
    """Encode positive/negative labels as binary target values."""
    out = df.copy()
    if target_col not in out.columns:
        raise ValueError(f"Target column not found: {target_col}")

    target = out[target_col]
    if pd.api.types.is_numeric_dtype(target):
        numeric_target = target.astype(float)
        unique_values = set(numeric_target.dropna().astype(int).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"Numeric target must contain only 0/1 values, got {sorted(unique_values)}")
        out[target_col] = numeric_target.astype(int)
        return out

    normalized = target.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        lambda value: (
            1
            if value in POSITIVE_LABELS
            else (0 if value in NEGATIVE_LABELS else pd.NA)
        )
    )

    unknown_values = normalized[mapped.isna()].unique().tolist()
    if unknown_values:
        preview = ", ".join(unknown_values[:5])
        raise ValueError(
            "Unrecognized target labels detected. "
            f"Example values: {preview}. Expected positive/negative style labels."
        )

    out[target_col] = mapped.astype(int)
    return out


def load_data(filepath: str | Path, target_col: str = "malaria_occurrence") -> pd.DataFrame:
    """Load CSV data, normalize columns, and encode the target."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    df = normalize_feature_columns(df)
    df = encode_target_column(df, target_col=target_col)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median and categorical with mode."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in out.columns if col not in numeric_cols]

    for col in numeric_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    for col in categorical_cols:
        if out[col].isna().any():
            mode = out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            out[col] = out[col].fillna(fill_value)

    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing transform for numerical and categorical columns."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(
    df: pd.DataFrame,
    target_col: str = "malaria_occurrence",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into stratified train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def compute_feature_defaults(
    df: pd.DataFrame,
    target_col: str = "malaria_occurrence",
) -> dict[str, Any]:
    """Compute median/mode defaults for each feature column."""
    defaults: dict[str, Any] = {}
    feature_df = df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()

    for col in feature_df.columns:
        if col in numeric_cols:
            defaults[col] = float(feature_df[col].median())
        else:
            mode = feature_df[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else "unknown"

    return defaults


def prepare_dataset(
    filepath: str | Path,
    target_col: str = "malaria_occurrence",
    test_size: float = 0.2,
    random_state: int = 42,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the complete preprocessing flow and return train/test-ready artifacts."""
    df = load_data(filepath=filepath, target_col=target_col)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    validate_training_dataframe(
        df=df,
        target_col=target_col,
        validation_config=validation_config,
    )

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = build_preprocessor(X_train)
    defaults = compute_feature_defaults(df=df, target_col=target_col)

    return {
        "dataset": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_columns": X_train.columns.tolist(),
        "feature_defaults": defaults,
    }
