"""Unit tests for verifying build_models construction and hyperparameter propagation."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.malaria_forecast.models import build_models


def test_build_models_configuration() -> None:
    """Verify build_models instantiates expected classifiers with hyperparameters."""
    config = {
        "models": {
            "logistic_regression": {
                "max_iter": 500,
                "C": 0.5
            },
            "decision_tree": {
                "max_depth": 5,
                "min_samples_leaf": 2,
                "random_state": 10
            },
            "random_forest": {
                "n_estimators": 50,
                "max_depth": 6,
                "min_samples_leaf": 4,
                "random_state": 20
            },
            "svm": {
                "kernel": "linear",
                "C": 2.0,
                "random_state": 30
            }
        }
    }

    classifiers = build_models(config)

    assert len(classifiers) == 4
    assert "logistic_regression" in classifiers
    assert "decision_tree" in classifiers
    assert "random_forest" in classifiers
    assert "svm" in classifiers

    # Assert types
    assert isinstance(classifiers["logistic_regression"], LogisticRegression)
    assert isinstance(classifiers["decision_tree"], DecisionTreeClassifier)
    assert isinstance(classifiers["random_forest"], RandomForestClassifier)
    assert isinstance(classifiers["svm"], CalibratedClassifierCV)

    # Assert hyperparameter propagation
    assert classifiers["logistic_regression"].max_iter == 500
    assert classifiers["logistic_regression"].C == 0.5

    assert classifiers["decision_tree"].max_depth == 5
    assert classifiers["decision_tree"].min_samples_leaf == 2
    assert classifiers["decision_tree"].random_state == 10

    assert classifiers["random_forest"].n_estimators == 50
    assert classifiers["random_forest"].max_depth == 6
    assert classifiers["random_forest"].min_samples_leaf == 4
    assert classifiers["random_forest"].random_state == 20

    # SVM: params are accessed via the inner SVC estimator
    svm_estimator = classifiers["svm"].estimator
    assert isinstance(svm_estimator, SVC)
    assert svm_estimator.kernel == "linear"
    assert svm_estimator.C == 2.0
    assert svm_estimator.random_state == 30
