#!/usr/bin/env python3
"""Download and cache the Africa-wide malaria HuggingFace dataset locally.

Usage:
    python scripts/download_dataset.py [--out-dir dataset/africa_malaria_hf]

Saves three CSV files:
    dataset/africa_malaria_hf/train.csv       (10 000 rows)
    dataset/africa_malaria_hf/validation.csv  ( 5 000 rows)
    dataset/africa_malaria_hf/test.csv        ( 2 000 rows)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def download(out_dir: Path, dataset_id: str) -> None:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        logging.error(
            "The 'datasets' package is not installed. "
            "Run: pip install datasets"
        )
        sys.exit(1)

    logging.info("Loading dataset '%s' from HuggingFace Hub...", dataset_id)
    ds = load_dataset(dataset_id)

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_ds in ds.items():
        out_path = out_dir / f"{split_name}.csv"
        df = split_ds.to_pandas()
        df.to_csv(out_path, index=False)
        logging.info("Saved %s split → %s  (%d rows)", split_name, out_path, len(df))

    logging.info("Dataset download complete. Files in: %s", out_dir)


def main() -> None:
    setup_logger()

    parser = argparse.ArgumentParser(
        description="Download Africa-wide malaria HF dataset and cache as local CSVs"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/africa_malaria_hf",
        help="Directory to save the downloaded CSV files (default: dataset/africa_malaria_hf)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="electricsheepafrica/africa-synth-malaria-malaria-dataset-all",
        help="HuggingFace dataset repository ID",
    )

    args = parser.parse_args()
    download(Path(args.out_dir), args.dataset_id)


if __name__ == "__main__":
    main()
