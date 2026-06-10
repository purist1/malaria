"""Visualization utilities for malaria forecasting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loader import load_data


def plot_area_history(
    area_id: str,
    data_path: str | Path | None = None,
    output_path: str | Path = "outputs/area_history.png",
) -> Path:
    """Plot historical malaria cases and climate drivers for an area."""
    df = load_data(data_path)
    area = df[df["area_id"] == area_id].copy()

    if area.empty:
        raise ValueError(f"No data for area {area_id}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Malaria Trends — {area['area_name'].iloc[0]} ({area_id})", fontsize=14)

    axes[0].bar(area["date"], area["malaria_cases"], color="#c0392b", alpha=0.8, width=20)
    axes[0].set_ylabel("Malaria Cases")
    axes[0].set_title("Monthly Case Count")

    axes[1].plot(area["date"], area["rainfall_mm"], color="#2980b9", marker="o", markersize=3)
    axes[1].set_ylabel("Rainfall (mm)")
    axes[1].set_title("Rainfall")

    ax_temp = axes[2].twinx()
    axes[2].plot(area["date"], area["temperature_c"], color="#e67e22", label="Temperature")
    ax_temp.plot(area["date"], area["humidity_pct"], color="#27ae60", label="Humidity", alpha=0.7)
    axes[2].set_ylabel("Temperature (°C)", color="#e67e22")
    ax_temp.set_ylabel("Humidity (%)", color="#27ae60")
    axes[2].set_title("Temperature & Humidity")
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_forecasts(
    forecast_path: str | Path = "data/processed/forecasts.csv",
    output_path: str | Path = "outputs/forecasts.png",
) -> Path:
    """Plot predicted cases by area."""
    forecasts = pd.read_csv(forecast_path, parse_dates=["forecast_date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=forecasts,
        x="forecast_date",
        y="predicted_cases",
        hue="area_name",
        ax=ax,
    )
    ax.set_title("Malaria Case Forecasts by Area")
    ax.set_xlabel("Forecast Month")
    ax.set_ylabel("Predicted Cases")
    ax.tick_params(axis="x", rotation=45)
    plt.legend(title="Area", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
