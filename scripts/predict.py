#!/usr/bin/env python3
"""CLI wrapper script to execute single or batch malaria risk predictions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
import pandas as pd

# Add project root to sys.path so we can import src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.malaria_forecast.artifacts import load_artifact
from src.malaria_forecast.predict import predict_batch_csv, predict_single_record


def setup_logger() -> None:
    """Configure python logging configuration to write to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    setup_logger()

    # Pre-parse --config and --model-dir to locate metadata.joblib
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config/config.yaml")
    pre_parser.add_argument("--model-dir", type=str, default="models")
    pre_args, remaining_args = pre_parser.parse_known_args()

    model_dir = Path(pre_args.model_dir)
    metadata_path = model_dir / "metadata.joblib"

    # Load dynamic features from metadata or fallback to standard list
    try:
        metadata = load_artifact(metadata_path)
        features = list(metadata["imputation_defaults"].keys())
    except FileNotFoundError:
        # Fallback list for the Africa-wide HF dataset schema
        features = [
            "sex", "residence", "season",
            "age_years", "hemoglobin_g_dl", "fever_days",
            "uses_mosquito_net",
            "has_fever", "has_chills", "has_headache",
            "has_vomiting", "has_diarrhea", "has_weakness",
        ]
        metadata = None

    # Main parser configuration
    parser = argparse.ArgumentParser(description="Predict malaria occurrence risk")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration yaml")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory where models are saved")
    parser.add_argument("--model-name", type=str, default=None, help="Use a specific model (default: best_model)")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to input CSV for batch prediction")
    parser.add_argument("--output", type=str, default="data/processed/predictions.csv", help="Path to save predictions")

    # Add dynamic argument flags based on feature names
    feature_dest_mapping: dict[str, str] = {}
    for feature in features:
        flag_name = f"--{feature.replace('_', '-')}"
        dest_name = feature
        feature_dest_mapping[dest_name] = flag_name
        parser.add_argument(flag_name, dest=dest_name, default=None, help=f"Value for {feature}")

    args = parser.parse_args()

    # If metadata is missing and we aren't doing batch prediction, warn and exit
    if metadata is None and not args.input_csv:
        logging.error(
            "Trained pipeline metadata not found at %s. Please run training first.",
            metadata_path
        )
        sys.exit(1)

    # 1. Batch Prediction flow
    if args.input_csv:
        logging.info("Running batch prediction...")
        try:
            predict_batch_csv(
                input_csv_path=args.input_csv,
                output_csv_path=args.output,
                model_dir=args.model_dir,
                model_name=args.model_name
            )
            print(f"Batch prediction results saved to: {args.output}")
        except Exception as exc:
            logging.error("Failed to run batch predictions: %s", exc, exc_info=True)
            sys.exit(1)

    # 2. Single Record Prediction flow
    else:
        # Collect dynamic flags into record
        record: dict[str, Any] = {}
        categorical_features = {"sex", "residence", "season"}
        for feature in features:
            val = getattr(args, feature)
            if val is not None:
                if feature not in categorical_features:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                record[feature] = val

        logging.info("Running single prediction with arguments: %s", record)
        try:
            res = predict_single_record(
                record=record,
                model_dir=args.model_dir,
                model_name=args.model_name
            )
            print("\n==================================================")
            print("         MALARIA OCCURRENCE PREDICTION RESULT")
            print("==================================================")
            print(f"  Prediction   : {res['label']}")
            print(f"  Probability  : {res['probability']:.2%}")
            print(f"  Model Used   : {res['model_used']}")
            print("==================================================\n")

            # Save prediction to output CSV if requested
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Map dynamic input fields + predicted labels to a single-row DataFrame
            row_dict = record.copy()
            row_dict["predicted_label"] = res["label"]
            row_dict["predicted_probability"] = round(res["probability"], 4)
            row_df = pd.DataFrame([row_dict])
            
            if output_path.exists():
                row_df.to_csv(output_path, mode="a", header=False, index=False)
            else:
                row_df.to_csv(output_path, index=False)
            logging.info("Saved single prediction result row to: %s", output_path)

        except Exception as exc:
            logging.error("Failed to run single prediction: %s", exc, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
