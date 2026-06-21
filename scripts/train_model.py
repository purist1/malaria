#!/usr/bin/env python3
"""CLI wrapper script to train malaria prediction classifiers."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.malaria_forecast.train import train_all_models


def setup_logger() -> None:
    """Configure python logging configuration to write to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    setup_logger()

    parser = argparse.ArgumentParser(description="Train malaria occurrence prediction models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration settings file (default: config/config.yaml)"
    )

    args = parser.parse_args()

    try:
        train_all_models(config_path=args.config)
    except Exception as exc:
        logging.error("Failed to execute model training run: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
