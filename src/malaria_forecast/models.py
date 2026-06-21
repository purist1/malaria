"""Model factory module for malaria occurrence classifiers."""

from __future__ import annotations

from typing import Any
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    """Build and configure the four model instances from configuration settings.

    All classifiers support ``class_weight='balanced'`` to compensate for the
    72/28 positive/negative class imbalance in the clinical dataset.

    Args:
        config: The model hyperparameter configurations.

    Returns:
        dict: Instantiated scikit-learn model algorithms.

    Note:
        SVC is wrapped in CalibratedClassifierCV to support predict_proba()
        without triggering the sklearn >=1.9 FutureWarning for probability=True.
        The class_weight is passed to the inner SVC estimator directly.
    """
    model_cfg = config["models"]

    return {
        "logistic_regression": LogisticRegression(
            max_iter=model_cfg["logistic_regression"]["max_iter"],
            C=model_cfg["logistic_regression"]["C"],
            class_weight=model_cfg["logistic_regression"].get("class_weight", "balanced"),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=model_cfg["decision_tree"]["max_depth"],
            min_samples_leaf=model_cfg["decision_tree"]["min_samples_leaf"],
            random_state=model_cfg["decision_tree"]["random_state"],
            class_weight=model_cfg["decision_tree"].get("class_weight", "balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=model_cfg["random_forest"]["n_estimators"],
            max_depth=model_cfg["random_forest"]["max_depth"],
            min_samples_leaf=model_cfg["random_forest"]["min_samples_leaf"],
            random_state=model_cfg["random_forest"]["random_state"],
            class_weight=model_cfg["random_forest"].get("class_weight", "balanced"),
        ),
        "svm": CalibratedClassifierCV(
            SVC(
                kernel=model_cfg["svm"]["kernel"],
                C=model_cfg["svm"]["C"],
                random_state=model_cfg["svm"]["random_state"],
                class_weight=model_cfg["svm"].get("class_weight", "balanced"),
            ),
            ensemble=False,
        ),
    }
