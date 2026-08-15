import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.automl.pipeline import run_automl_pipeline


def _build_regression_dataset() -> pd.DataFrame:
    X, y = make_regression(
        n_samples=250,
        n_features=8,
        n_informative=5,
        noise=8.0,
        random_state=42,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def _build_binary_dataset() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def _build_multiclass_dataset() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def test_regression_pipeline_success():
    df = _build_regression_dataset()

    result = run_automl_pipeline(df, target_column="target", problem_type="Regression")

    assert result["status"] == "success"
    assert result["target_column"] == "target"
    assert result["problem_type"] == "Regression"
    assert result["data"]["training_rows"] > 0
    assert result["data"]["test_rows"] > 0
    assert len(result["models"]["trained"]) > 0
    assert result["evaluation"]["status"] == "Completed"
    assert result["best_model"]["selection_metric"] == "rmse"
    assert "rmse" in result["best_model"]["metrics"]


def test_binary_classification_pipeline_success():
    df = _build_binary_dataset()

    result = run_automl_pipeline(df, target_column="target", problem_type="Binary Classification")

    assert result["status"] == "success"
    assert result["problem_type"] == "Binary Classification"
    assert len(result["models"]["trained"]) > 0
    assert result["evaluation"]["status"] == "Completed"
    assert result["best_model"]["selection_metric"] == "f1_score"
    assert "f1_score" in result["best_model"]["metrics"]


def test_multiclass_classification_pipeline_success():
    df = _build_multiclass_dataset()

    result = run_automl_pipeline(df, target_column="target", problem_type="Multi-class Classification")

    assert result["status"] == "success"
    assert result["problem_type"] == "Multi-class Classification"
    assert len(result["models"]["trained"]) > 0
    assert result["evaluation"]["status"] == "Completed"
    assert result["best_model"]["selection_metric"] == "f1_score"
    assert "f1_score" in result["best_model"]["metrics"]


def test_missing_target_raises_value_error():
    df = _build_binary_dataset()

    with pytest.raises(ValueError, match="Target column"):
        run_automl_pipeline(df, target_column="missing_target")


def test_unsupported_problem_type_raises_validation_error():
    df = _build_binary_dataset()

    with pytest.raises(ValueError, match="Unsupported problem type"):
        run_automl_pipeline(df, target_column="target", problem_type="Unsupported Task")


def test_all_model_failure_returns_controlled_failure(monkeypatch):
    df = _build_binary_dataset()

    def fake_train_models(*args, **kwargs):
        return {
            "trained_models": {},
            "model_names": [],
            "status": "Failed",
            "training_errors": [{"model_name": "synthetic", "error": "forced failure"}],
        }

    monkeypatch.setattr("services.automl.pipeline.train_models", fake_train_models)

    result = run_automl_pipeline(df, target_column="target", problem_type="Binary Classification")

    assert result["status"] == "failed"
    assert "All candidate models failed" in result["error"]
    assert result["models"]["trained"] == []
