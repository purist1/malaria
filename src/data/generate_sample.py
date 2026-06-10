"""Generate realistic synthetic weekly malaria data for Nigeria North Central."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _weekly_rainfall(week: int, base: float, rng: np.random.Generator) -> float:
    """Nigeria rainy season peaks roughly May–October."""
    month = ((week - 1) // 4) % 12 + 1
    seasonal = base * (0.55 + 0.45 * np.sin(2 * np.pi * (month - 5) / 12))
    return max(0.0, seasonal / 4 + rng.normal(0, base * 0.05))


def _weekly_temperature(week: int, elevation_m: float, rng: np.random.Generator) -> float:
    month = ((week - 1) // 4) % 12 + 1
    base_temp = 30 - elevation_m * 0.004
    seasonal = 2.0 * np.sin(2 * np.pi * (month - 3) / 12)
    return base_temp + seasonal + rng.normal(0, 0.6)


def _humidity(rainfall: float, temperature: float, rng: np.random.Generator) -> float:
    base = 58 + rainfall * 0.25 - (temperature - 28) * 1.2
    return float(np.clip(base + rng.normal(0, 4), 35, 95))


def _weekly_cases(
    rainfall: float,
    temperature: float,
    humidity: float,
    week: int,
    elevation_m: float,
    population: int,
    malaria_burden: float,
    rng: np.random.Generator,
) -> int:
    month = ((week - 1) // 4) % 12 + 1
    temp_factor = np.exp(-((temperature - 27) ** 2) / 25)
    rain_factor = min(rainfall / 35, 2.5)
    humidity_factor = humidity / 75
    elevation_factor = max(0.15, 1 - elevation_m / 1800)
    seasonal_factor = 0.65 + 0.35 * np.sin(2 * np.pi * (month - 6) / 12)

    rate = (
        malaria_burden
        * 0.000012
        * population
        * temp_factor
        * rain_factor
        * humidity_factor
        * elevation_factor
        * seasonal_factor
    )
    return int(rng.poisson(max(rate, 2)))


def generate_sample_data(
    config_path: str | Path = "config/config.yaml",
    start_date: str = "2020-01-06",
    end_date: str = "2025-06-02",
    output_path: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate weekly malaria records for North Central Nigerian states."""
    config_path = Path(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    areas = config["areas"]
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")

    burden_map = {
        "Benue": 1.4,
        "Kogi": 1.1,
        "Kwara": 0.9,
        "Nasarawa": 1.2,
        "Niger": 1.0,
        "Plateau": 0.6,
        "FCT": 0.5,
    }

    for area in areas:
        base_rainfall = 90 if area["elevation_m"] < 500 else 70
        burden = burden_map.get(area["name"], 1.0)

        for date in dates:
            week = int(date.isocalendar().week)
            year = int(date.isocalendar().year)
            rainfall = _weekly_rainfall(week, base_rainfall, rng)
            temperature = _weekly_temperature(week, area["elevation_m"], rng)
            humidity = _humidity(rainfall, temperature, rng)
            cases = _weekly_cases(
                rainfall,
                temperature,
                humidity,
                week,
                area["elevation_m"],
                area["population"],
                burden,
                rng,
            )

            records.append(
                {
                    "area_id": area["id"],
                    "area_name": area["name"],
                    "state": area["state"],
                    "date": date,
                    "year": year,
                    "week": week,
                    "latitude": area["latitude"],
                    "longitude": area["longitude"],
                    "elevation_m": area["elevation_m"],
                    "population": area["population"],
                    "rainfall_mm": round(rainfall, 2),
                    "temperature_c": round(temperature, 2),
                    "humidity_pct": round(humidity, 2),
                    "malaria_cases": cases,
                }
            )

    df = pd.DataFrame(records).sort_values(["area_id", "date"]).reset_index(drop=True)

    if output_path is None:
        output_path = Path(config["data"]["raw_dir"]) / config["data"]["sample_file"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    data = generate_sample_data()
    print(f"Generated {len(data)} weekly records across {data['area_id'].nunique()} states")
    print(f"Saved to data/raw/malaria_nigeria_north_central.csv")
