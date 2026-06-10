"""Training run tracking utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return str(value)


def log_training_run(
    config: dict[str, Any],
    results_df: pd.DataFrame,
    metadata: dict[str, Any],
    experiments_dir: str | Path,
    explainability: dict[str, str] | None = None,
) -> dict[str, str]:
    """Persist a timestamped run summary and append it to an experiments index."""
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_dir = experiments_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    with config_snapshot_path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    metrics_path = run_dir / "metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    summary = {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "best_model_name": metadata.get("best_model_name"),
        "best_model_file": metadata.get("best_model_file"),
        "model_dir": metadata.get("model_dir"),
        "reports_dir": metadata.get("reports_dir"),
        "config_snapshot_path": str(config_snapshot_path),
        "metrics_path": str(metrics_path),
        "explainability": explainability or {},
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(_to_serializable(summary), file, indent=2)

    index_path = experiments_dir / "index.jsonl"
    with index_path.open("a") as file:
        file.write(json.dumps(_to_serializable(summary)) + "\n")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "config_snapshot_path": str(config_snapshot_path),
        "index_path": str(index_path),
    }
