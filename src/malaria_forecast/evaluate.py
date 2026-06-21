"""Model evaluation and metrics visualization module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None
) -> dict[str, float]:
    """Calculate Accuracy, Precision, Recall, F1, and ROC-AUC metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_prob: Predicted positive class probabilities.

    Returns:
        dict: Computed metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": 0.0,
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception as exc:
            logger.warning("Could not calculate ROC-AUC score: %s", exc)

    return metrics


def save_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    output_path: Path | str
) -> None:
    """Plot and save a styled confusion matrix for a specific model.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        model_name: The name of the model (for titles).
        output_path: The file path to save the generated figure.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Styling and rendering using ConfusionMatrixDisplay
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Malaria Negative", "Malaria Positive"]
    )
    display.plot(cmap="Blues", ax=ax, colorbar=False)

    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label", fontsize=10, labelpad=8)
    plt.tight_layout()

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix for %s to %s", model_name, path)


def plot_roc_comparison(
    roc_curves_data: list[dict[str, Any]],
    output_path: Path | str
) -> None:
    """Plot combined ROC curve comparing multiple models.

    Args:
        roc_curves_data: List of dicts, each with keys 'model_name', 'y_true', 'y_prob'.
        output_path: The file path to save the generated comparison chart.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    sns.set_theme(style="whitegrid")

    for data in roc_curves_data:
        name = data["model_name"]
        y_true = data["y_true"]
        y_prob = data["y_prob"]

        if y_prob is None:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, labelpad=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC comparison chart to %s", path)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path | str
) -> None:
    """Plot a grouped bar chart comparing multiple classification metrics across models.

    Args:
        metrics_df: Dataframe containing model metrics.
        output_path: The file path to save the comparison bar chart.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    plot_df = metrics_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sns.set_theme(style="whitegrid")

    # Map readable metric names for the legend
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score"
    }
    plot_df["Metric"] = plot_df["Metric"].map(metric_labels)

    sns.barplot(
        data=plot_df,
        x="model",
        y="Score",
        hue="Metric",
        ax=ax,
        palette="Set2",
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title("Model Performance Metrics Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Model Classifier", fontsize=11, labelpad=10)
    ax.set_ylabel("Performance Score", fontsize=11, labelpad=10)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    # Rotate model names on the X-axis for better visibility
    plt.xticks(rotation=15)
    plt.tight_layout()

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved performance metrics comparison chart to %s", path)
