from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> Dict[str, object]:
    """
    Evaluate a set of trained models on a held-out test set.

    Strategy:
    - Use problem-specific metrics to capture the most important aspects of
      prediction quality.
    - For regression, evaluate goodness-of-fit, average error, and error
      dispersion.
    - For classification, evaluate overall correctness, class precision,
      recall, and balanced F1 performance.
    - Sort models by a primary metric so users can easily identify the best
      candidate for further tuning.
    """

    X = _to_numpy(X_test)
    y_true = _to_numpy(y_test)
    normalized_type = (problem_type or "").strip().lower()
    results: List[Dict[str, object]] = []

    for model_name, model in models.items():
        y_pred = _predict_model(model, X)

        if "regression" in normalized_type:
            metrics = {
                "r2_score": _r2_score(y_true, y_pred),
                "mae": _mean_absolute_error(y_true, y_pred),
                "rmse": _root_mean_squared_error(y_true, y_pred),
            }
            sort_key = metrics["r2_score"]
        elif "classification" in normalized_type:
            metrics = {
                "accuracy": _accuracy_score(y_true, y_pred),
                "precision": _precision_score(y_true, y_pred),
                "recall": _recall_score(y_true, y_pred),
                "f1_score": _f1_score(y_true, y_pred),
            }
            sort_key = metrics["f1_score"]
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        results.append({
            "model": model_name,
            "metrics": metrics,
            "sort_key": sort_key,
        })

    results.sort(key=lambda item: item["sort_key"], reverse=True)
    for item in results:
        del item["sort_key"]

    return {
        "results": results,
    }


def select_best_model(results: List[Dict[str, object]], problem_type: str) -> Dict[str, object]:
    """
    Select the best model based on the primary evaluation metric.

    Selection criteria:
    - Regression: choose the model with the highest R² score.
    - Classification: choose the model with the highest F1 score.

    This follows common practice because R² measures explained variance for
    numeric regression, while F1 balances precision and recall for classification.
    """

    if not results:
        raise ValueError("No model results provided for selection.")

    normalized_type = (problem_type or "").strip().lower()
    if "regression" in normalized_type:
        key = "r2_score"
    elif "classification" in normalized_type:
        key = "f1_score"
    else:
        raise ValueError(f"Unsupported problem type for model selection: {problem_type}")

    best = max(
        results,
        key=lambda item: float(item.get("metrics", {}).get(key, float("-inf")))
    )

    return {
        "best_model": best["model"],
        "metrics": best["metrics"],
    }


def _predict_model(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    raise AttributeError("Model object must implement a predict(X) method.")


def _to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy()
    return np.asarray(data)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    residual = np.sum((y_true - y_pred) ** 2)
    if total_variance == 0:
        return 1.0 if residual == 0 else 0.0
    return 1.0 - (residual / total_variance)


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions: List[float] = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        precisions.append(float(tp / (tp + fp)) if tp + fp > 0 else 0.0)
    return float(np.mean(precisions))


def _recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    recalls: List[float] = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        recalls.append(float(tp / (tp + fn)) if tp + fn > 0 else 0.0)
    return float(np.mean(recalls))


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision = _precision_score(y_true, y_pred)
    recall = _recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)
