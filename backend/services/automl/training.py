from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _build_model_specs(problem_type: str) -> List[Tuple[str, Any]]:
    """Create the baseline model suite for the detected problem type."""

    normalized_type = (problem_type or "").strip().lower()

    if "regression" in normalized_type:
        return [
            ("LinearRegression", LinearRegression()),
            ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
            (
                "RandomForestRegressor",
                RandomForestRegressor(random_state=42, n_estimators=200),
            ),
        ]

    if "classification" in normalized_type:
        return [
            ("LogisticRegression", LogisticRegression(max_iter=500, random_state=42)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
            (
                "RandomForestClassifier",
                RandomForestClassifier(random_state=42, n_estimators=200),
            ),
        ]

    raise ValueError(f"Unsupported problem type for training: {problem_type}")


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
) -> Dict[str, object]:
    """
    Train a baseline model suite for the detected task and continue on failures.

    The function intentionally tolerates per-model failures so the AutoML engine
    can still use the subset of models that successfully fit the data.
    """

    trained_models: Dict[str, Any] = {}
    model_names: List[str] = []
    training_errors: List[Dict[str, Any]] = []

    try:
        model_specs = _build_model_specs(problem_type)
    except ValueError as exc:
        return {
            "trained_models": {},
            "model_names": [],
            "status": "Failed",
            "training_errors": [{"model_name": None, "error": str(exc)}],
        }

    for model_name, model in model_specs:
        try:
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            model_names.append(model_name)
        except Exception as exc:  # pragma: no cover - defensive, but intentionally tolerated
            training_errors.append({"model_name": model_name, "error": str(exc)})

    if trained_models:
        status = "Completed" if not training_errors else "Completed with errors"
    else:
        status = "Failed"

    return {
        "trained_models": trained_models,
        "model_names": model_names,
        "status": status,
        "training_errors": training_errors,
    }


def _as_model_list(models: Dict[str, Any] | Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize evaluation results from either a dict or a list-like structure."""

    if isinstance(models, dict):
        return [{"model_name": name, "model": model} for name, model in models.items()]
    return list(models)


def evaluate_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> Dict[str, object]:
    """
    Evaluate each successfully trained model using task-specific metrics.

    Regression metrics: MAE, RMSE, R^2.
    Classification metrics: accuracy, precision, recall, F1.
    """

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    normalized_type = (problem_type or "").strip().lower()

    for model_name, model in models.items():
        try:
            predictions = model.predict(X_test)

            if "regression" in normalized_type:
                metrics = {
                    "mae": float(mean_absolute_error(y_test, predictions)),
                    "rmse": float(root_mean_squared_error(y_test, predictions)),
                    "r2": float(r2_score(y_test, predictions)),
                }
            elif "classification" in normalized_type:
                y_true = np.asarray(y_test)
                y_pred = np.asarray(predictions)
                metrics = {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(
                        precision_score(y_true, y_pred, average="weighted", zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_true, y_pred, average="weighted", zero_division=0)
                    ),
                    "f1_score": float(
                        f1_score(y_true, y_pred, average="weighted", zero_division=0)
                    ),
                }
            else:
                raise ValueError(f"Unsupported problem type for evaluation: {problem_type}")

            results.append({
                "model_name": model_name,
                "model": model,
                "metrics": metrics,
                "status": "Evaluated",
            })
        except Exception as exc:  # pragma: no cover - defensive failure recording
            errors.append({"model_name": model_name, "error": str(exc)})

    status = "Completed" if results else "Failed"

    return {
        "results": results,
        "status": status,
        "errors": errors,
    }


def select_best_model(
    evaluation_results: Dict[str, object] | List[Dict[str, Any]],
    problem_type: str,
) -> Dict[str, object]:
    """
    Select the best candidate using the task-specific primary metric.

    Regression: lowest RMSE.
    Classification: highest F1 score.
    """

    normalized_type = (problem_type or "").strip().lower()

    if isinstance(evaluation_results, dict):
        candidate_results = evaluation_results.get("results", [])
    else:
        candidate_results = evaluation_results

    if not candidate_results:
        return {
            "status": "Failed",
            "best_model_name": None,
            "best_model": None,
            "best_metrics": {},
            "selection_metric": None,
            "selection_reason": "No valid models were available for selection.",
        }

    if "regression" in normalized_type:
        metric_name = "rmse"
        selector = min
        reason = "lowest rmse"
    elif "classification" in normalized_type:
        metric_name = "f1_score"
        selector = max
        reason = "highest f1_score"
    else:
        raise ValueError(f"Unsupported problem type for model selection: {problem_type}")

    valid_results = [
        item for item in candidate_results if isinstance(item, dict) and metric_name in item.get("metrics", {})
    ]

    if not valid_results:
        return {
            "status": "Failed",
            "best_model_name": None,
            "best_model": None,
            "best_metrics": {},
            "selection_metric": metric_name,
            "selection_reason": f"No results had a valid {metric_name} metric.",
        }

    best_result = selector(valid_results, key=lambda item: float(item["metrics"][metric_name]))
    best_metrics = best_result.get("metrics", {})
    best_name = best_result.get("model_name")
    best_model = best_result.get("model")

    return {
        "status": "Selected",
        "best_model_name": best_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "selection_metric": metric_name,
        "selection_reason": f"Selected {best_name} because it has the {reason}.",
    }
