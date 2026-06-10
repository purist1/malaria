"""Load and validate malaria incidence datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLUMNS = {
    "area_id",
    "date",
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "malaria_cases",
}


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def load_data(
    data_path: str | Path | None = None,
    config_path: str | Path = "config/config.yaml",
) -> pd.DataFrame:
    """Load malaria data from CSV and normalize column types."""
    config = load_config(config_path)

    if data_path is None:
        data_path = Path(config["data"]["raw_dir"]) / config["data"]["sample_file"]
    data_path = Path(data_path)

    if not data_path.exists():
        kaggle_path = Path(config["data"]["raw_dir"]) / config["data"]["kaggle_file"]
        if kaggle_path.exists():
            from src.data.kaggle_import import normalize_kaggle_data

            return normalize_kaggle_data(kaggle_path, config_path=config_path, output_path=data_path)

        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Run `python scripts/train_model.py --generate-data` or place Kaggle CSV at "
            f"{kaggle_path}"
        )

    df = pd.read_csv(data_path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    if "week" not in df.columns:
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
    if "year" not in df.columns:
        df["year"] = df["date"].dt.isocalendar().year.astype(int)

    df = df.sort_values(["area_id", "date"]).reset_index(drop=True)
    return df


def get_area_metadata(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    """Return configured area metadata as a DataFrame."""
    config = load_config(config_path)
    return pd.DataFrame(config["areas"])
