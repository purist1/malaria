#!/usr/bin/env python3
"""EDA figure generation script.

Produces all exploratory data analysis charts required by the dev guide:
  - target_distribution.png        : Class balance bar chart
  - correlation_heatmap.png        : Numeric feature correlation matrix
  - feature_distributions.png     : Histogram grid for all numeric features
  - feature_importance.png         : Random Forest feature importance ranking
  - boxplot_<feature>.png          : Each symptom flag vs target (for key features)
  - class_imbalance_pie.png        : Pie chart of positive/negative split

All figures are saved to reports/figures/ by default.

Usage:
    python scripts/generate_eda.py
    python scripts/generate_eda.py --config config/config.yaml
    python scripts/generate_eda.py --output-dir reports/figures/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.malaria_forecast.config import load_config
from src.malaria_forecast.data_loader import load_raw_dataset
from src.malaria_forecast.features import ALL_NUMERIC_FEATURES, TARGET_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Shared plot style
sns.set_theme(style="whitegrid", palette="Set2")
SYMPTOM_FEATURES = [
    "fever", "headache", "abdominal_pain", "general_body_malaise",
    "dizziness", "vomiting", "confusion", "backache",
    "chest_pain", "coughing", "joint_pain",
]


# ---------------------------------------------------------------------------
# Individual chart functions
# ---------------------------------------------------------------------------

def plot_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of malaria positive vs negative counts."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Negative (0)", "Positive (1)"]
    colors = ["#4CAF50", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    axes[0].set_title("Malaria Occurrence Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Class Label", fontsize=11)
    axes[0].set_ylabel("Patient Count", fontsize=11)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold", fontsize=11)
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=[f"{l}\n({v}, {100*v/sum(counts.values):.1f}%)" for l, v in zip(labels, counts.values)],
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Class Proportion", fontsize=13, fontweight="bold")

    plt.suptitle("Target Variable: Malaria Occurrence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = out_dir / "target_distribution.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Correlation heatmap for all numeric features + target."""
    numeric_cols = ALL_NUMERIC_FEATURES + [TARGET_COLUMN]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle

    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
        square=True,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram grid for all numeric and symptom features."""
    features = ALL_NUMERIC_FEATURES  # age, length_of_stay + 11 symptoms
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5), dpi=120)
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        if col in SYMPTOM_FEATURES:
            # Binary: count plot
            counts = df[col].value_counts().sort_index()
            ax.bar(["No (0)", "Yes (1)"], counts.values,
                   color=["#78C8E0", "#FF8A65"], edgecolor="black", linewidth=0.7)
            ax.set_ylabel("Count", fontsize=8)
        else:
            # Continuous: histogram
            ax.hist(df[col].dropna(), bins=20, color="#5C85D6",
                    edgecolor="black", linewidth=0.6, alpha=0.85)
            ax.set_ylabel("Frequency", fontsize=8)

        ax.set_title(col.replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = out_dir / "feature_distributions.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_feature_importance(df: pd.DataFrame, out_dir: Path) -> None:
    """Random Forest feature importance ranking chart."""
    from sklearn.preprocessing import LabelEncoder

    # Minimal encoding for the importance calculation
    df_enc = df[ALL_NUMERIC_FEATURES + ["sex", "residence_area", TARGET_COLUMN]].copy()
    for col in ["sex", "residence_area"]:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    feature_cols = ALL_NUMERIC_FEATURES + ["sex", "residence_area"]
    X = df_enc[feature_cols]
    y = df_enc[TARGET_COLUMN]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    colors = ["#E57373" if imp > importances.median() else "#64B5F6" for imp in importances]
    bars = ax.barh(importances.index, importances.values, color=colors,
                   edgecolor="black", linewidth=0.6)

    ax.set_title("Random Forest Feature Importance", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score (Gini)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Value labels
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlim(0, importances.max() * 1.18)
    plt.tight_layout()

    path = out_dir / "feature_importance.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)

    # Print top features
    print("\nTop 5 Most Important Features:")
    for feat, imp in importances.sort_values(ascending=False).head(5).items():
        print(f"  {feat:<30} {imp:.4f}")


def plot_symptom_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar charts: each symptom flag prevalence split by target class."""
    # For binary symptom flags, stacked bar showing prevalence per class is more informative
    pos_df = df[df[TARGET_COLUMN] == 1]
    neg_df = df[df[TARGET_COLUMN] == 0]

    pos_rates = pos_df[SYMPTOM_FEATURES].mean() * 100  # % with symptom
    neg_rates = neg_df[SYMPTOM_FEATURES].mean() * 100

    x = np.arange(len(SYMPTOM_FEATURES))
    width = 0.38
    labels = [s.replace("_", "\n").title() for s in SYMPTOM_FEATURES]

    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    bars1 = ax.bar(x - width / 2, neg_rates, width, label="Malaria Negative",
                   color="#4CAF50", edgecolor="black", linewidth=0.6, alpha=0.9)
    bars2 = ax.bar(x + width / 2, pos_rates, width, label="Malaria Positive",
                   color="#F44336", edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Patients with Symptom (%)", fontsize=11)
    ax.set_xlabel("Clinical Symptom", fontsize=11)
    ax.set_title("Symptom Prevalence by Malaria Outcome", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=6.5, color="#2e7d32")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=6.5, color="#c62828")

    plt.tight_layout()
    path = out_dir / "boxplot_symptoms_by_target.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)

    # Also save individual boxplots for age and length_of_stay
    for cont_col in ["age", "length_of_stay"]:
        fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=150)
        df_plot = df[[cont_col, TARGET_COLUMN]].copy()
        df_plot["Outcome"] = df_plot[TARGET_COLUMN].map({0: "Negative", 1: "Positive"})
        sns.boxplot(
            data=df_plot, x="Outcome", y=cont_col, ax=ax2,
            palette={"Negative": "#4CAF50", "Positive": "#F44336"},
            width=0.5, linewidth=1.2,
        )
        ax2.set_title(f"{cont_col.replace('_', ' ').title()} vs Malaria Outcome",
                      fontsize=13, fontweight="bold")
        ax2.set_xlabel("Malaria Outcome", fontsize=11)
        ax2.set_ylabel(cont_col.replace("_", " ").title(), fontsize=11)
        plt.tight_layout()
        p2 = out_dir / f"boxplot_{cont_col}.png"
        fig2.savefig(p2, bbox_inches="tight", dpi=150)
        plt.close(fig2)
        logger.info("Saved: %s", p2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures for the malaria dataset.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = config["data"]["raw_path"]
    out_dir = Path(args.output_dir or config.get("eda", {}).get("figures_dir", "reports/figures/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from: %s", raw_path)
    df = load_raw_dataset(raw_path)

    logger.info("Generating EDA figures → %s", out_dir)

    plot_target_distribution(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_feature_distributions(df, out_dir)
    plot_feature_importance(df, out_dir)
    plot_symptom_boxplots(df, out_dir)

    print(f"\n✅  All EDA figures saved to: {out_dir.resolve()}")
    print("   Files generated:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"     {f.name}")


if __name__ == "__main__":
    main()
