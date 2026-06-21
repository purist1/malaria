#!/usr/bin/env python3
"""Script to localize the Malaria dataset residence areas to Kogi State, Nigeria."""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path("dataset/Malaria_Dataset.csv")

MAPPING = {
    "Udupi": "Lokoja",
    "Kasargod": "Okene",
    "Shimoga": "Kabba",
    "Chickmagalur": "Anyigba",
    "Mangalore": "Idah"
}

def main() -> None:
    if not DATASET_PATH.exists():
        logger.error("Dataset not found at: %s", DATASET_PATH)
        return

    df = pd.read_csv(DATASET_PATH)
    logger.info("Loaded dataset from %s", DATASET_PATH)

    if "Residence_Area" not in df.columns:
        logger.error("Column 'Residence_Area' not found in dataset columns: %s", list(df.columns))
        return

    # Check unique values
    unique_before = list(df["Residence_Area"].unique())
    logger.info("Residence areas before mapping: %s", unique_before)

    # Perform mapping
    df["Residence_Area"] = df["Residence_Area"].replace(MAPPING)
    
    unique_after = list(df["Residence_Area"].unique())
    logger.info("Residence areas after mapping: %s", unique_after)

    # Save back to CSV
    df.to_csv(DATASET_PATH, index=False)
    logger.info("Successfully saved localized dataset to %s", DATASET_PATH)

if __name__ == "__main__":
    main()
