#!/usr/bin/env python3
"""Wrapper script for prediction CLI."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli import predict_cli


def main() -> None:
    predict_cli()


if __name__ == "__main__":
    main()
