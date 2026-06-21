"""Configuration loading and validation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file and validate required keys."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with path.open("r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML syntax in configuration file: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate structure and required keys in the configuration dictionary."""
    required_sections = ["data", "split", "models", "artifacts", "dashboard"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}'")

    # Data validations
    data = config["data"]
    if not isinstance(data, dict):
        raise ValueError("Section 'data' must be a dictionary.")
    for key in ["raw_path", "processed_path", "target_column"]:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in 'data' configuration.")

    # Split validations
    split = config["split"]
    if not isinstance(split, dict):
        raise ValueError("Section 'split' must be a dictionary.")
    for key in ["test_size", "random_state", "stratify"]:
        if key not in split:
            raise ValueError(f"Missing required key '{key}' in 'split' configuration.")

    # Models validations
    models = config["models"]
    if not isinstance(models, dict):
        raise ValueError("Section 'models' must be a dictionary.")
    required_models = ["logistic_regression", "decision_tree", "random_forest", "svm"]
    for model in required_models:
        if model not in models:
            raise ValueError(f"Missing required model configuration for: '{model}'")
        if not isinstance(models[model], dict):
            raise ValueError(f"Model config for '{model}' must be a dictionary.")

    # Artifacts validations
    artifacts = config["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("Section 'artifacts' must be a dictionary.")
    for key in ["models_dir", "reports_dir", "figures_dir"]:
        if key not in artifacts:
            raise ValueError(f"Missing required key '{key}' in 'artifacts' configuration.")

    # Dashboard validations
    dashboard = config["dashboard"]
    if not isinstance(dashboard, dict):
        raise ValueError("Section 'dashboard' must be a dictionary.")
    for key in ["title", "default_threshold"]:
        if key not in dashboard:
            raise ValueError(f"Missing required key '{key}' in 'dashboard' configuration.")
