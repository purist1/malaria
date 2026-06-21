"""Artifact persistence and loading helper module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import joblib

logger = logging.getLogger(__name__)


def save_artifact(obj: Any, filepath: str | Path) -> None:
    """Save an object (model, dictionary, list) to disk as a joblib file.

    Args:
        obj: The object to save.
        filepath: Target location to save the artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact successfully to: %s", path)


def load_artifact(filepath: str | Path) -> Any:
    """Load a joblib artifact from disk.

    Args:
        filepath: Path to the joblib file.

    Returns:
        Any: The deserialized object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at: {path}")

    obj = joblib.load(path)
    logger.info("Loaded artifact successfully from: %s", path)
    return obj
