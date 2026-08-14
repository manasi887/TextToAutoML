import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.automl.training import evaluate_models, select_best_model, train_models


def test_regression_training_and_selection():
    X, y = make_regression(
        n_samples=200,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )

    training_report = train_models(X_train, y_train, "regression")
    assert training_report["status"] == "Completed"
    assert len(training_report["trained_models"]) >= 2
    assert len(training_report["model_names"]) == len(training_report["trained_models"])
    assert not training_report["training_errors"]

    evaluation = evaluate_models(training_report["trained_models"], X_test, y_test, "regression")
    assert evaluation["status"] == "Completed"
    assert evaluation["results"]
    assert all("metrics" in item for item in evaluation["results"])

    selection = select_best_model(evaluation["results"], "regression")
    assert selection["best_model_name"] is not None
    assert selection["selection_metric"] == "rmse"
    assert selection["status"] == "Selected"


def test_classification_training_and_selection():
    X, y = make_classification(
        n_samples=220,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="label")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
    )

    training_report = train_models(X_train, y_train, "classification")
    assert training_report["status"] == "Completed"
    assert len(training_report["trained_models"]) >= 2

    evaluation = evaluate_models(training_report["trained_models"], X_test, y_test, "classification")
    assert evaluation["status"] == "Completed"
    assert evaluation["results"]
    assert all("metrics" in item for item in evaluation["results"])
    assert all("f1_score" in item["metrics"] for item in evaluation["results"])

    selection = select_best_model(evaluation["results"], "classification")
    assert selection["best_model_name"] is not None
    assert selection["selection_metric"] == "f1_score"
    assert selection["status"] == "Selected"


def test_one_failed_model_does_not_crash_pipeline():
    X, y = make_regression(
        n_samples=150,
        n_features=4,
        noise=0.2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=42
    )

    valid_model = LinearRegression()

    with patch("services.automl.training._build_model_specs", return_value=[
        ("LinearRegression", valid_model),
        ("BrokenModel", object()),
    ]):
        training_report = train_models(X_train, y_train, "regression")

    assert len(training_report["trained_models"]) == 1
    assert training_report["status"] == "Completed with errors"
    assert training_report["training_errors"]

    bad_models = {
        "LinearRegression": training_report["trained_models"]["LinearRegression"],
        "BrokenModel": object(),
    }
    evaluation = evaluate_models(bad_models, X_test, y_test, "regression")
    assert len(evaluation["results"]) == 1
    assert evaluation["errors"]
