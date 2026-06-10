"""Training workflow for malaria occurrence prediction models."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.explainability import generate_permutation_importance

from src.evaluate import evaluate_model, plot_model_comparison, plot_roc_curves, slugify
from src.preprocess import prepare_dataset
from src.tracking import log_training_run


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load project configuration YAML."""
    with Path(config_path).open() as file:
        return yaml.safe_load(file)


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    """Resolve config-relative paths from project root."""
    value = Path(value)
    if value.is_absolute():
        return value

    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    return (project_root / value).resolve()


def get_models(config: dict[str, Any], random_state: int = 42) -> dict[str, Any]:
    """Build the four required classifier instances."""
    models_cfg = config.get("models", {})
    lr_cfg = models_cfg.get("logistic_regression", {})
    dt_cfg = models_cfg.get("decision_tree", {})
    rf_cfg = models_cfg.get("random_forest", {})
    svm_cfg = models_cfg.get("svm", {})

    svm_calibration_method = str(svm_cfg.get("calibration_method", "sigmoid"))
    svm_calibration_cv = int(svm_cfg.get("calibration_cv", 3))
    svm_estimator = SVC(
        kernel=svm_cfg.get("kernel", "rbf"),
        C=float(svm_cfg.get("c", 1.0)),
        probability=False,
        class_weight=svm_cfg.get("class_weight", "balanced"),
        random_state=random_state,
    )

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=int(lr_cfg.get("max_iter", 1000)),
            random_state=random_state,
            class_weight=lr_cfg.get("class_weight", "balanced"),
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=dt_cfg.get("max_depth", 5),
            random_state=random_state,
            class_weight=dt_cfg.get("class_weight", "balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth", 10),
            random_state=random_state,
            class_weight=rf_cfg.get("class_weight", "balanced"),
            n_jobs=-1,
        ),
        "SVM": CalibratedClassifierCV(
            estimator=svm_estimator,
            method=svm_calibration_method,
            cv=svm_calibration_cv,
        ),
    }


def train_all_models(
    config_path: str | Path = "config/config.yaml",
    data_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train all models, evaluate, and persist artifacts."""
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    artifacts_cfg = config.get("artifacts", {})
    prediction_cfg = config.get("prediction", {})
    validation_cfg = config.get("validation", {})
    explainability_cfg = config.get("explainability", {})
    tracking_cfg = config.get("tracking", {})

    dataset_path = resolve_path(
        config_path,
        data_path or data_cfg.get("dataset_path", "dataset/Malaria Diseases dataset - .csv"),
    )
    target_col = data_cfg.get("target_column", "malaria_occurrence")
    test_size = float(training_cfg.get("test_size", 0.2))
    random_state = int(training_cfg.get("random_state", 42))
    model_dir_path = resolve_path(config_path, model_dir or artifacts_cfg.get("model_dir", "models"))
    reports_dir_path = resolve_path(config_path, reports_dir or artifacts_cfg.get("reports_dir", "reports"))
    figures_dir_path = reports_dir_path / "figures"

    model_dir_path.mkdir(parents=True, exist_ok=True)
    reports_dir_path.mkdir(parents=True, exist_ok=True)
    figures_dir_path.mkdir(parents=True, exist_ok=True)

    prepared = prepare_dataset(
        filepath=dataset_path,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        validation_config=validation_cfg,
    )
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    preprocessor = prepared["preprocessor"]

    processed_dataset_path = None
    if data_cfg.get("processed_dataset_path"):
        processed_dataset_path = resolve_path(config_path, data_cfg["processed_dataset_path"])
        processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        prepared["dataset"].to_csv(processed_dataset_path, index=False)

    results: list[dict[str, Any]] = []
    roc_entries: list[dict[str, Any]] = []
    model_files: dict[str, str] = {}
    trained_models: dict[str, Pipeline] = {}

    for name, estimator in get_models(config=config, random_state=random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        model_filename = f"{slugify(name)}.joblib"
        model_path = model_dir_path / model_filename
        joblib.dump(pipeline, model_path)
        model_files[name] = model_filename
        trained_models[name] = pipeline

        metrics, roc_entry = evaluate_model(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            figures_dir=figures_dir_path,
        )
        results.append(metrics)
        if roc_entry is not None:
            roc_entries.append(roc_entry)

    results_df = pd.DataFrame(results)
    results_df["roc_auc_sort"] = results_df["roc_auc"].fillna(-1.0)
    results_df = (
        results_df.sort_values(
            by=["f1_score", "roc_auc_sort", "accuracy"],
            ascending=False,
        )
        .drop(columns=["roc_auc_sort"])
        .reset_index(drop=True)
    )

    comparison_csv = reports_dir_path / "model_comparison.csv"
    results_df.to_csv(comparison_csv, index=False)
    plot_model_comparison(results_df=results_df, output_path=figures_dir_path / "model_comparison_bar.png")

    if roc_entries:
        plot_roc_curves(
            roc_entries=roc_entries,
            output_path=figures_dir_path / "roc_curves_comparison.png",
        )

    best_model_name = str(results_df.iloc[0]["model"])
    best_model_file = model_files[best_model_name]
    shutil.copyfile(model_dir_path / best_model_file, model_dir_path / "best_model.joblib")

    explainability_artifacts: dict[str, str] = {}
    if bool(explainability_cfg.get("enabled", True)):
        explainability_dir = resolve_path(
            config_path,
            explainability_cfg.get("output_dir", "reports/explainability"),
        )
        explainability_artifacts = generate_permutation_importance(
            model=trained_models[best_model_name],
            X_test=X_test,
            y_test=y_test,
            output_dir=explainability_dir,
            n_repeats=int(explainability_cfg.get("n_repeats", 8)),
            random_state=random_state,
            top_n=int(explainability_cfg.get("top_n_features", 15)),
        )

    metadata: dict[str, Any] = {
        "best_model_name": best_model_name,
        "best_model_file": best_model_file,
        "model_files": model_files,
        "feature_columns": prepared["feature_columns"],
        "feature_defaults": prepared["feature_defaults"],
        "target_column": target_col,
        "label_map": {
            0: prediction_cfg.get("negative_label", "Malaria Negative"),
            1: prediction_cfg.get("positive_label", "Malaria Positive"),
        },
        "dataset_path": str(dataset_path),
        "processed_dataset_path": str(processed_dataset_path) if processed_dataset_path else None,
        "model_dir": str(model_dir_path),
        "reports_dir": str(reports_dir_path),
        "explainability": explainability_artifacts,
        "test_size": test_size,
        "random_state": random_state,
    }
    joblib.dump(metadata, model_dir_path / "metadata.joblib")

    if bool(tracking_cfg.get("enabled", True)):
        experiments_dir = resolve_path(
            config_path,
            tracking_cfg.get("experiments_dir", "reports/experiments"),
        )
        tracking_artifacts = log_training_run(
            config=config,
            results_df=results_df,
            metadata=metadata,
            experiments_dir=experiments_dir,
            explainability=explainability_artifacts,
        )
        metadata["tracking"] = tracking_artifacts
        joblib.dump(metadata, model_dir_path / "metadata.joblib")
    return results_df, metadata
