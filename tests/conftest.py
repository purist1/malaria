"""Pytest conftest setup file to configure PYTHONPATH."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to sys.path so pytest can locate 'src' package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
