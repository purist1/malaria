"""Model training orchestration module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.malaria_forecast.artifacts import save_artifact
from src.malaria_forecast.config import load_config
from src.malaria_forecast.data_loader import load_raw_dataset
from src.malaria_forecast.evaluate import (
    calculate_metrics,
    plot_metric_comparison,
    plot_roc_comparison,
    save_confusion_matrix,
)
from src.malaria_forecast.features import LABEL_MAP
from src.malaria_forecast.models import build_models
from src.malaria_forecast.preprocessing import preprocess_data

logger = logging.getLogger(__name__)


def train_all_models(
    config_path: str | Path = "config/config.yaml"
) -> tuple[pd.DataFrame, str]:
    """Execute the full end-to-end model training, evaluation, and serialization pipeline.

    Args:
        config_path: Path to configuration YAML.

    Returns:
        tuple[pd.DataFrame, str]: The metrics comparison dataframe and the selected best model name.
    """
    logger.info("Executing training pipeline using configuration: %s", config_path)

    # 1. Load configuration settings
    config = load_config(config_path)
    data_cfg = config["data"]
    artifacts_cfg = config["artifacts"]

    models_dir = Path(artifacts_cfg["models_dir"])
    reports_dir = Path(artifacts_cfg["reports_dir"])
    figures_dir = Path(artifacts_cfg["figures_dir"])

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset and normalize columns
    raw_df = load_raw_dataset(data_cfg["raw_path"])

    # 3. Preprocess and split dataset
    prep_data = preprocess_data(raw_df, config)

    X_train = prep_data["X_train"]
    X_test = prep_data["X_test"]
    y_train = prep_data["y_train"]
    y_test = prep_data["y_test"]

    # 4. Instantiate clinical models
    classifiers = build_models(config)

    # 5. Fit classifiers and evaluate
    metrics_records: list[dict[str, Any]] = []
    roc_curves_data: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for name, clf in classifiers.items():
        logger.info("Fitting classifier: %s...", name)
        clf.fit(X_train, y_train)
        fitted_models[name] = clf

        # Make predictions on test set
        y_pred = clf.predict(X_test)
        y_prob = None

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        metrics_records.append(metrics)

        # Plot and save confusion matrix
        save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            model_name=name,
            output_path=figures_dir / f"confusion_matrix_{name}.png"
        )

        # Collect ROC data
        if y_prob is not None:
            roc_curves_data.append({
                "model_name": name,
                "y_true": y_test,
                "y_prob": y_prob
            })

    # 6. Save model comparison metric CSV
    metrics_df = pd.DataFrame(metrics_records)
    # Order columns as required: model, accuracy, precision, recall, f1_score, roc_auc
    metrics_df = metrics_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    metrics_csv_path = reports_dir / "model_comparison.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Saved model comparison metrics to: %s", metrics_csv_path)

    # 7. Generate combined visualization comparison charts
    plot_roc_comparison(roc_curves_data, figures_dir / "roc_comparison.png")
    plot_metric_comparison(metrics_df, figures_dir / "metric_comparison_bar.png")

    # 8. Select the best model (F1-score primary, ROC-AUC secondary, Accuracy tertiary)
    sorted_metrics = metrics_df.sort_values(
        by=["f1_score", "roc_auc", "accuracy"],
        ascending=False
    ).reset_index(drop=True)

    best_model_name = sorted_metrics.iloc[0]["model"]
    logger.info("Selected Best Model: %s based on performance score hierarchy.", best_model_name)

    # 9. Save each fitted classifier and copy the best model
    for name, fitted_clf in fitted_models.items():
        save_artifact(fitted_clf, models_dir / f"{name}.joblib")

    best_model = fitted_models[best_model_name]
    save_artifact(best_model, models_dir / "best_model.joblib")

    # 10. Save pipeline metadata
    metadata = {
        "scaler": prep_data["scaler"],
        "encoder": prep_data["encoder"],
        "imputation_defaults": prep_data["imputation_defaults"],
        "final_feature_order": prep_data["final_feature_order"],
        "best_model_name": best_model_name,
        "best_model_file": f"{best_model_name}.joblib",
        "label_map": LABEL_MAP
    }
    save_artifact(metadata, models_dir / "metadata.joblib")

    # 11. Print summary table to stdout as required by specification
    print("\n=========================================================================")
    print("                    FINAL MODEL COMPARISON SUMMARY")
    print("=========================================================================")
    print(metrics_df.to_string(index=False))
    print("=========================================================================")
    print(f"  Best Selected Model: {best_model_name} (saved to best_model.joblib)")
    print("=========================================================================\n")

    return metrics_df, best_model_name
