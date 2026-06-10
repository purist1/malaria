"""Normalize external malaria datasets (Kaggle, DHIS2 exports) to project schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.loader import load_config

COLUMN_ALIASES = {
    "area_id": ["area_id", "state_code", "state_id", "geo_code"],
    "area_name": ["area_name", "state", "state_name", "region"],
    "state": ["state", "state_name", "region"],
    "date": ["date", "week_start", "reporting_date", "period_start", "timestamp"],
    "malaria_cases": [
        "malaria_cases",
        "confirmed_cases",
        "positive_cases",
        "total_cases",
        "cases",
        "case_count",
    ],
    "rainfall_mm": ["rainfall_mm", "rainfall", "precipitation_mm", "rain_mm"],
    "temperature_c": [
        "temperature_c",
        "temperature_avg_c",
        "temperature",
        "temp_c",
        "avg_temp",
    ],
    "humidity_pct": ["humidity_pct", "humidity", "relative_humidity"],
    "population": ["population", "pop", "population_count"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "elevation_m": ["elevation_m", "elevation", "altitude_m"],
}

STATE_TO_ID = {
    "benue": "NG-BEN",
    "kogi": "NG-KOG",
    "kwara": "NG-KWA",
    "nasarawa": "NG-NAS",
    "niger": "NG-NIG",
    "plateau": "NG-PLA",
    "fct": "NG-FCT",
    "abuja": "NG-FCT",
    "federal capital territory": "NG-FCT",
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}

    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_cols:
                rename_map[lower_cols[alias.lower()]] = target
                break

    return df.rename(columns=rename_map)


def _resolve_area_id(row: pd.Series, area_lookup: dict[str, dict]) -> str | None:
    if pd.notna(row.get("area_id")):
        return str(row["area_id"])

    name = str(row.get("area_name") or row.get("state") or "").strip().lower()
    if name in STATE_TO_ID:
        return STATE_TO_ID[name]
    if name in area_lookup:
        return area_lookup[name]["id"]
    return None


def normalize_kaggle_data(
    input_path: str | Path,
    config_path: str | Path = "config/config.yaml",
    output_path: str | Path | None = None,
    aggregate_to_state: bool = True,
) -> pd.DataFrame:
    """
    Load a Kaggle or external CSV, map columns, and aggregate to state level.

    Drops rows outside North Central states defined in config.
    """
    config = load_config(config_path)
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = _rename_columns(df)

    if "date" not in df.columns:
        if {"year", "week"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-W" + df["week"].astype(str) + "-1")
        elif {"year", "month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
        else:
            raise ValueError("Could not infer date column. Provide 'date' or year/week.")

    df["date"] = pd.to_datetime(df["date"])

    areas = config["areas"]
    area_lookup = {a["name"].lower(): a for a in areas}
    area_lookup.update({a["state"].lower(): a for a in areas})

    df["area_id"] = df.apply(lambda r: _resolve_area_id(r, area_lookup), axis=1)
    df = df[df["area_id"].notna()].copy()

    meta = pd.DataFrame(areas).rename(columns={"id": "area_id", "name": "area_name"})
    meta_cols = ["area_id", "area_name", "state", "latitude", "longitude", "elevation_m", "population"]
    df = df.merge(meta[meta_cols], on="area_id", how="left", suffixes=("_raw", ""))

    if "area_name_raw" in df.columns:
        df["area_name"] = df["area_name"].fillna(df["area_name_raw"])
        df = df.drop(columns=["area_name_raw"], errors="ignore")

    if "malaria_cases" not in df.columns:
        raise ValueError(
            "No case count column found. Expected one of: "
            + ", ".join(COLUMN_ALIASES["malaria_cases"])
        )

    numeric_fill = {
        "rainfall_mm": df.groupby("area_id")["rainfall_mm"].transform("mean").mean()
        if "rainfall_mm" in df.columns
        else 20.0,
        "temperature_c": 28.0,
        "humidity_pct": 65.0,
    }

    for col, default in numeric_fill.items():
        if col not in df.columns:
            df[col] = default

    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    if aggregate_to_state and "lga" in df.columns:
        agg_cols = {
            "malaria_cases": "sum",
            "rainfall_mm": "mean",
            "temperature_c": "mean",
            "humidity_pct": "mean",
            "area_name": "first",
            "state": "first",
            "latitude": "first",
            "longitude": "first",
            "elevation_m": "first",
            "population": "first",
            "year": "first",
            "week": "first",
        }
        df = df.groupby(["area_id", "date"], as_index=False).agg(
            {k: v for k, v in agg_cols.items() if k in df.columns}
        )

    df = df.sort_values(["area_id", "date"]).reset_index(drop=True)

    if output_path is None:
        output_path = Path(config["data"]["raw_dir"]) / config["data"]["sample_file"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
