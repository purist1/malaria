"""Backtesting utilities for forecast vs actual comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_config, load_data
from src.features.engineering import build_feature_matrix
from src.models.predict import load_artifacts


def backtest_recent_weeks(
    n_weeks: int = 4,
    config_path: str | Path = "config/config.yaml",
    model_dir: str | Path = "models",
) -> pd.DataFrame:
    """
    Compare one-step-ahead predictions against actuals on the most recent weeks.

    For each area, holds out the last n_weeks and predicts using prior history.
    """
    config = load_config(config_path)
    artifacts = load_artifacts(model_dir)
    historical = load_data(config_path=config_path)
    results: list[dict] = []

    for area_id in historical["area_id"].unique():
        area_data = historical[historical["area_id"] == area_id].sort_values("date").copy()
        if len(area_data) <= n_weeks + max(config["training"]["lag_weeks"]):
            continue

        holdout = area_data.tail(n_weeks)
        train_slice = area_data.iloc[: -n_weeks].copy()

        for _, row in holdout.iterrows():
            working = pd.concat([train_slice, pd.DataFrame([row])], ignore_index=True)
            features, _ = build_feature_matrix(
                working,
                lag_weeks=config["training"]["lag_weeks"],
                include_target=False,
            )
            latest = features.iloc[[-1]]
            X = latest[artifacts["feature_cols"]]
            predicted = max(0, int(round(float(artifacts["regressor"].predict(X)[0]))))
            actual = int(row["malaria_cases"])

            results.append(
                {
                    "area_id": area_id,
                    "area_name": row.get("area_name", area_id),
                    "date": row["date"],
                    "actual_cases": actual,
                    "predicted_cases": predicted,
                    "error": predicted - actual,
                    "abs_error": abs(predicted - actual),
                    "pct_error": abs(predicted - actual) / max(actual, 1) * 100,
                }
            )
            train_slice = pd.concat([train_slice, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(results)
