"""Feature engineering for weekly malaria forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and calendar features from the date column."""
    out = df.copy()

    if "year" not in out.columns:
        out["year"] = out["date"].dt.isocalendar().year.astype(int)
    if "week" not in out.columns:
        out["week"] = out["date"].dt.isocalendar().week.astype(int)

    out["month"] = out["date"].dt.month
    out["week_sin"] = np.sin(2 * np.pi * out["week"] / 52)
    out["week_cos"] = np.cos(2 * np.pi * out["week"] / 52)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["is_rainy_season"] = out["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add lagged malaria case counts per area."""
    out = df.copy()
    out = out.sort_values(["area_id", "date"])

    for lag in lags:
        out[f"cases_lag_{lag}"] = out.groupby("area_id")["malaria_cases"].shift(lag)

    out["cases_rolling_4w"] = (
        out.groupby("area_id")["malaria_cases"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    )
    out["cases_rolling_8w"] = (
        out.groupby("area_id")["malaria_cases"]
        .transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    )
    return out


def add_climate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms between climate variables."""
    out = df.copy()
    out["rainfall_x_humidity"] = out["rainfall_mm"] * out["humidity_pct"]
    out["temp_x_humidity"] = out["temperature_c"] * out["humidity_pct"]
    out["rainfall_x_temp"] = out["rainfall_mm"] * out["temperature_c"]
    out["cases_per_10k"] = (out["malaria_cases"] / out["population"]) * 10_000
    return out


def assign_risk_level(cases: pd.Series, thresholds: dict[str, int]) -> pd.Series:
    """Classify malaria occurrence into risk tiers."""
    return pd.cut(
        cases,
        bins=[-1, thresholds["low"], thresholds["medium"], float("inf")],
        labels=["low", "medium", "high"],
    )


def build_feature_matrix(
    df: pd.DataFrame,
    lag_weeks: list[int],
    risk_thresholds: dict[str, int] | None = None,
    include_target: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the full feature matrix for training or inference.

    Returns the processed DataFrame and the list of feature column names.
    """
    out = df.copy()
    out = add_temporal_features(out)
    out = add_lag_features(out, lag_weeks)
    out = add_climate_interactions(out)

    if include_target and risk_thresholds is not None:
        out["risk_level"] = assign_risk_level(out["malaria_cases"], risk_thresholds)

    base_features = [
        "week",
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
        "is_rainy_season",
        "latitude",
        "longitude",
        "elevation_m",
        "population",
        "rainfall_mm",
        "temperature_c",
        "humidity_pct",
        "rainfall_x_humidity",
        "temp_x_humidity",
        "rainfall_x_temp",
    ]
    lag_features = [f"cases_lag_{lag}" for lag in lag_weeks]
    rolling_features = ["cases_rolling_4w", "cases_rolling_8w"]
    feature_cols = base_features + lag_features + rolling_features

    available = [c for c in feature_cols if c in out.columns]
    return out, available
