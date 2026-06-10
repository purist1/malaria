"""CLI entrypoints for model training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.predict import load_artifacts, predict_dataframe
from src.train import train_all_models

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open() as file:
        return yaml.safe_load(file)


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_file = Path(config_path).resolve()
    return (config_file.parents[1] / path).resolve()


def train_cli() -> None:
    parser = argparse.ArgumentParser(description="Train malaria occurrence prediction models")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    args = parser.parse_args()

    print("Training and evaluating models...")
    results_df, metadata = train_all_models(
        config_path=args.config,
        data_path=args.data_path,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
    )

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {metadata['best_model_name']}")
    print(f"Model artifacts: {metadata['model_dir']}")
    print(f"Reports: {metadata['reports_dir']}")


def _build_single_record(args: argparse.Namespace, defaults: dict) -> pd.DataFrame:
    record = defaults.copy()
    overrides = {
        "sex": args.sex,
        "age": args.age,
        "hemoglobin_hb_pct": args.hemoglobin_hb_pct,
        "total_wbc_count_cumm": args.total_wbc_count_cumm,
        "neutrophils": args.neutrophils,
        "lymphocytes": args.lymphocytes,
        "total_cir_eosinophils": args.total_cir_eosinophils,
        "htc_pcv_pct": args.htc_pcv_pct,
        "mch_pg": args.mch_pg,
        "mchc_g_dl": args.mchc_g_dl,
        "rdw_cv_pct": args.rdw_cv_pct,
        "platelet_count": args.platelet_count,
    }
    for key, value in overrides.items():
        if value is not None:
            record[key] = value
    return pd.DataFrame([record])


def predict_cli() -> None:
    parser = argparse.ArgumentParser(description="Predict malaria occurrence")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output", default="data/processed/predictions.csv")

    parser.add_argument("--sex", default=None)
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--hemoglobin-hb-pct", dest="hemoglobin_hb_pct", type=float, default=None)
    parser.add_argument("--total-wbc-count-cumm", dest="total_wbc_count_cumm", type=float, default=None)
    parser.add_argument("--neutrophils", type=float, default=None)
    parser.add_argument("--lymphocytes", type=float, default=None)
    parser.add_argument("--total-cir-eosinophils", dest="total_cir_eosinophils", type=float, default=None)
    parser.add_argument("--htc-pcv-pct", dest="htc_pcv_pct", type=float, default=None)
    parser.add_argument("--mch-pg", dest="mch_pg", type=float, default=None)
    parser.add_argument("--mchc-g-dl", dest="mchc_g_dl", type=float, default=None)
    parser.add_argument("--rdw-cv-pct", dest="rdw_cv_pct", type=float, default=None)
    parser.add_argument("--platelet-count", dest="platelet_count", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir = resolve_path(
        args.config,
        args.model_dir or config.get("artifacts", {}).get("model_dir", "models"),
    )
    validation_config = config.get("validation", {})

    _, metadata = load_artifacts(model_dir=model_dir, model_name=args.model_name)
    if args.input_csv:
        input_path = resolve_path(args.config, args.input_csv)
        input_df = pd.read_csv(input_path)
    else:
        input_df = _build_single_record(args=args, defaults=metadata.get("feature_defaults", {}))

    predictions = predict_dataframe(
        input_df=input_df,
        model_dir=model_dir,
        model_name=args.model_name,
        validation_config=validation_config,
    )

    output_path = resolve_path(args.config, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print("\nMalaria Occurrence Prediction Results")
    print("=" * 60)
    for idx, row in predictions.iterrows():
        prob = row.get("probability")
        prob_text = f"{float(prob):.2%}" if pd.notna(prob) else "N/A"
        print(f"Record {idx + 1}: {row['label']} (probability={prob_text})")
    print(f"\nPredictions saved to {output_path}")
