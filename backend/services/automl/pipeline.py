from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from services.automl.problem_detection import detect_problem_type
from services.automl.trainer import prepare_training_data, split_dataset
from services.automl.training import evaluate_models, select_best_model, train_models


_SUPPORTED_PROBLEM_TYPES = {
    "binary classification",
    "multi-class classification",
    "multi class classification",
    "regression",
}


def _normalize_problem_type(problem_type: str | None, df: pd.DataFrame, target_column: str) -> str:
    """Normalize a task label into the supported supervised problem types."""

    if problem_type is None:
        problem_report = detect_problem_type(df, target_column)
        normalized = problem_report.get("problem_type")
        if normalized is None:
            raise ValueError("No supervised problem type could be inferred for the supplied target column.")
        problem_type = normalized

    normalized = str(problem_type).strip()
    lowered = normalized.lower()

    if "binary" in lowered and "class" in lowered:
        return "Binary Classification"
    if "multi" in lowered and "class" in lowered:
        return "Multi-class Classification"
    if "regression" in lowered:
        return "Regression"
    if lowered == "classification":
        return "Multi-class Classification"

    raise ValueError(
        "Unsupported problem type '"
        f"{problem_type}'. Supported types are: Binary Classification, Multi-class Classification, Regression."
    )


def _prepare_training_features(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Call the frozen training preparation logic and fall back if it strips all usable features."""

    preparation = prepare_training_data(df, target_column)
    X = preparation["X"]
    y = preparation["y"]

    if X.empty or X.shape[1] == 0:
        raw_X = df.drop(columns=[target_column], errors="ignore")
        if raw_X.empty or raw_X.shape[1] == 0:
            return preparation
        y_raw = df[target_column].dropna().reset_index(drop=True)
        return {
            "X": raw_X.reset_index(drop=True),
            "y": y_raw,
            "feature_names": raw_X.columns.tolist(),
            "encoders": {},
            "preprocessing": {
                **preparation.get("preprocessing", {}),
                "fallback_used": True,
                "fallback_reason": "The frozen preprocessing step removed all features; falling back to raw feature matrix.",
            },
        }

    return preparation


def run_automl_pipeline(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Run the end-to-end AutoML training workflow for a validated supervised task."""

    if df is None or df.empty:
        raise ValueError("Input dataset is empty.")

    if target_column is None or str(target_column).strip() == "":
        raise ValueError("A target column must be explicitly supplied to the AutoML pipeline.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the DataFrame.")

    normalized_problem_type = _normalize_problem_type(problem_type, df, target_column)

    preparation = _prepare_training_features(df, target_column)
    X = preparation["X"]
    y = preparation["y"]
    feature_names = preparation.get("feature_names", X.columns.tolist())
    encoders = preparation.get("encoders", {})

    if y.empty:
        raise ValueError(f"Target column '{target_column}' became empty after dropping missing target values.")
    if X.empty or X.shape[1] == 0:
        raise ValueError(f"No usable features remain for training after preprocessing the target '{target_column}'.")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        problem_type=normalized_problem_type,
        random_state=random_state,
    )

    training_report = train_models(X_train, y_train, normalized_problem_type)
    trained_models = training_report.get("trained_models", {})
    training_errors = training_report.get("training_errors", [])

    if not trained_models:
        return {
            "status": "failed",
            "target_column": target_column,
            "problem_type": normalized_problem_type,
            "data": {
                "original_rows": int(len(df)),
                "original_columns": int(len(df.columns)),
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "feature_count": int(X.shape[1]),
            },
            "preprocessing": preparation.get("preprocessing", {}),
            "models": {
                "trained": [],
                "failed": training_errors,
            },
            "evaluation": {"status": "Failed", "results": [], "errors": []},
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": "All candidate models failed to train.",
            },
            "error": "All candidate models failed to train for the selected problem type.",
        }

    evaluation = evaluate_models(trained_models, X_test, y_test, normalized_problem_type)
    evaluation_results = evaluation.get("results", [])
    evaluation_errors = evaluation.get("errors", [])

    if not evaluation_results:
        return {
            "status": "failed",
            "target_column": target_column,
            "problem_type": normalized_problem_type,
            "data": {
                "original_rows": int(len(df)),
                "original_columns": int(len(df.columns)),
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "feature_count": int(X.shape[1]),
            },
            "preprocessing": preparation.get("preprocessing", {}),
            "models": {
                "trained": list(trained_models.keys()),
                "failed": training_errors,
            },
            "evaluation": {
                "status": "Failed",
                "results": [],
                "errors": evaluation_errors,
            },
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": "Evaluation produced no valid model results.",
            },
            "error": "Evaluation produced no valid model results for the selected problem type.",
        }

    best_model_report = select_best_model(evaluation_results, normalized_problem_type)
    if best_model_report.get("status") == "Failed" or best_model_report.get("best_model") is None:
        return {
            "status": "failed",
            "target_column": target_column,
            "problem_type": normalized_problem_type,
            "data": {
                "original_rows": int(len(df)),
                "original_columns": int(len(df.columns)),
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "feature_count": int(X.shape[1]),
            },
            "preprocessing": preparation.get("preprocessing", {}),
            "models": {
                "trained": list(trained_models.keys()),
                "failed": training_errors,
            },
            "evaluation": {
                "status": evaluation.get("status", "Completed"),
                "results": evaluation_results,
                "errors": evaluation_errors,
            },
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": best_model_report.get("selection_reason", "Best model selection failed."),
            },
            "error": "Best model selection failed for the selected problem type.",
        }

    best_model = best_model_report["best_model"]
    best_name = best_model_report.get("best_model_name")
    metric_name = best_model_report.get("selection_metric")
    metrics = best_model_report.get("best_metrics", {})
    reason = best_model_report.get("selection_reason", "Selected using the task-specific metric.")

    if normalized_problem_type == "Regression":
        selection_metric = "rmse"
    else:
        selection_metric = "f1_score"

    return {
        "status": "success",
        "target_column": target_column,
        "problem_type": normalized_problem_type,
        "data": {
            "original_rows": int(len(df)),
            "original_columns": int(len(df.columns)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(X.shape[1]),
        },
        "preprocessing": {
            **preparation.get("preprocessing", {}),
            "feature_names": feature_names,
            "encoders": encoders,
        },
        "models": {
            "trained": list(trained_models.keys()),
            "failed": training_errors,
        },
        "evaluation": {
            "status": evaluation.get("status", "Completed"),
            "results": evaluation_results,
            "errors": evaluation_errors,
        },
        "best_model": {
            "name": best_name,
            "selection_metric": selection_metric,
            "metrics": metrics,
            "reason": reason,
        },
        "_internal_best_model": best_model,
    }
