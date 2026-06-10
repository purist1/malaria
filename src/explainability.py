"""Explainability utilities for trained malaria classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


def generate_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
    n_repeats: int = 8,
    random_state: int = 42,
    top_n: int = 15,
) -> dict[str, str]:
    """Create permutation importance CSV and chart for a trained classifier."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1",
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns.tolist(),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    csv_path = output_dir / "permutation_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    top_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, len(top_df) * 0.35)))
    ax.barh(
        top_df["feature"],
        top_df["importance_mean"],
        xerr=top_df["importance_std"],
        color="#2c7fb8",
        alpha=0.85,
    )
    ax.set_title("Permutation Importance (Top Features)")
    ax.set_xlabel("Importance (mean decrease in F1)")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    plot_path = output_dir / "permutation_importance_top_features.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
