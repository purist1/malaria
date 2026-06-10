"""Evaluation helpers for malaria occurrence classification models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
)


def slugify(value: str) -> str:
    """Convert model names into safe filenames."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    output_dir: str | Path,
) -> Path:
    """Save a confusion matrix heatmap to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Malaria", "Malaria"],
        yticklabels=["No Malaria", "Malaria"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    output_path = output_dir / f"cm_{slugify(model_name)}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_roc_curves(
    roc_entries: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot ROC curves for all models that provide probability outputs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    for entry in roc_entries:
        ax.plot(
            entry["fpr"],
            entry["tpr"],
            label=f"{entry['model']} (AUC = {entry['roc_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create a bar chart comparing model metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    plot_df = results_df[["model"] + metrics].set_index("model")

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black")
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    figures_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Evaluate one fitted model and return metrics plus optional ROC data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics: dict[str, Any] = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": None,
    }

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        model_name=model_name,
        output_dir=figures_dir,
    )

    roc_entry: dict[str, Any] | None = None
    if y_prob is not None:
        roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
        metrics["roc_auc"] = roc_auc
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_entry = {
            "model": model_name,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
        }

    return metrics, roc_entry
