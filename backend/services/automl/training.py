from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _build_model_specs(
    problem_type: str,
    recommended_model_order: List[str] | None = None,
) -> List[Tuple[str, Any]]:
    """Create the baseline model suite for the detected problem type."""

    normalized_type = (problem_type or "").strip().lower()

    if "regression" in normalized_type:
        model_specs = [
            ("LinearRegression", LinearRegression()),
            ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
            (
                "RandomForestRegressor",
                RandomForestRegressor(random_state=42, n_estimators=200),
            ),
        ]

    elif "classification" in normalized_type:
        model_specs = [
            ("LogisticRegression", LogisticRegression(max_iter=500, random_state=42)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
            (
                "RandomForestClassifier",
                RandomForestClassifier(random_state=42, n_estimators=200),
            ),
        ]

    else:
        raise ValueError(f"Unsupported problem type for training: {problem_type}")

    if not recommended_model_order:
        return model_specs

    specs_by_name = dict(model_specs)
    ordered_names: list[str] = []
    for model_name in recommended_model_order:
        if model_name in specs_by_name and model_name not in ordered_names:
            ordered_names.append(model_name)
    ordered_names.extend(
        model_name for model_name, _ in model_specs if model_name not in ordered_names
    )
    return [(model_name, specs_by_name[model_name]) for model_name in ordered_names]


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    recommended_model_order: List[str] | None = None,
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
        model_specs = _build_model_specs(problem_type, recommended_model_order)
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


def _clustering_score_sample(X: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    """Return a deterministic sample for silhouette scoring on large datasets."""
    if len(X) <= max_rows:
        return X.copy()
    return X.sample(n=max_rows, random_state=42).copy()


def train_clustering_models(X_train: pd.DataFrame) -> Dict[str, object]:
    """Train deterministic KMeans candidates on the prepared feature matrix."""
    if X_train is None or X_train.empty:
        return {
            "trained_models": {},
            "model_names": [],
            "status": "Failed",
            "training_errors": [{"model_name": None, "error": "Clustering input is empty."}],
        }

    n_samples = len(X_train)
    if n_samples < 2:
        return {
            "trained_models": {},
            "model_names": [],
            "status": "Failed",
            "training_errors": [{"model_name": None, "error": "Clustering requires at least two rows."}],
        }

    max_k = min(10, n_samples - 1)
    candidate_ks = list(range(2, max_k + 1))
    if not candidate_ks:
        return {
            "trained_models": {},
            "model_names": [],
            "status": "Failed",
            "training_errors": [{"model_name": None, "error": "No valid KMeans cluster counts are available."}],
        }

    trained_models: Dict[str, Any] = {}
    training_errors: List[Dict[str, Any]] = []

    for k in candidate_ks:
        model_name = f"KMeans_k{k}"
        try:
            model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42, n_init=10))
            model.fit(X_train)

            labels = model.predict(X_train)
            unique_labels = np.unique(labels)
            if unique_labels.size < 2 or unique_labels.size >= n_samples:
                raise ValueError(f"KMeans(k={k}) produced an invalid silhouette configuration.")

            trained_models[model_name] = model
        except Exception as exc:  # pragma: no cover - defensive failure recording
            training_errors.append({"model_name": model_name, "error": str(exc)})

    return {
        "trained_models": trained_models,
        "model_names": list(trained_models),
        "status": "Completed" if trained_models else "Failed",
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
    clustering_score_sample = (
        _clustering_score_sample(X_test) if "clustering" in normalized_type else None
    )

    for model_name, model in models.items():
        try:
            if "clustering" in normalized_type:
                assert clustering_score_sample is not None
                labels = np.asarray(model.predict(clustering_score_sample))
                unique_labels = set(labels.tolist())
                if len(unique_labels) < 2 or len(unique_labels) >= len(clustering_score_sample):
                    raise ValueError("Clustering evaluation requires at least two non-trivial clusters.")
                score = float(silhouette_score(clustering_score_sample, labels))
                if not np.isfinite(score):
                    raise ValueError("Clustering evaluation produced a non-finite silhouette score.")
                metrics = {"silhouette_score": score}
            elif "regression" in normalized_type:
                predictions = model.predict(X_test)
                metrics = {
                    "mae": float(mean_absolute_error(y_test, predictions)),
                    "rmse": float(root_mean_squared_error(y_test, predictions)),
                    "r2": float(r2_score(y_test, predictions)),
                }
            elif "classification" in normalized_type:
                predictions = model.predict(X_test)
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


def _safe_cv_folds(y: pd.Series, problem_type: str) -> int | None:
    """Return a safe fold count for supervised CV without crashing on tiny datasets."""
    normalized_type = (problem_type or "").strip().lower()

    if "classification" in normalized_type:
        value_counts = y.dropna().value_counts()
        if value_counts.empty:
            return None
        min_class_count = int(value_counts.min())
        if min_class_count < 2:
            return None
        return min(5, min_class_count)

    if "regression" in normalized_type:
        if len(y) < 2:
            return None
        return min(5, len(y))

    return None


def evaluate_models_cv(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
) -> Dict[str, object]:
    """Cross-validate models on the training split only using task-specific metrics."""
    normalized_type = (problem_type or "").strip().lower()
    if "clustering" in normalized_type:
        return {
            "results": [],
            "status": "Skipped",
            "errors": [{"model_name": None, "error": "Clustering CV is intentionally not enabled."}],
        }

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for model_name, model in models.items():
        try:
            n_splits = _safe_cv_folds(y_train, problem_type)
            if n_splits is None:
                errors.append({
                    "model_name": model_name,
                    "error": "Dataset is too small for 5-fold supervised CV; skipping CV scoring.",
                })
                continue

            if "classification" in normalized_type:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                fold_scores = []
                for train_idx, valid_idx in cv.split(X_train, y_train):
                    fold_model = clone(model)
                    fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    y_pred = fold_model.predict(X_train.iloc[valid_idx])
                    fold_scores.append(
                        float(
                            f1_score(
                                y_train.iloc[valid_idx],
                                y_pred,
                                average="weighted",
                                zero_division=0,
                            )
                        )
                    )
                metrics = {"f1_score": float(np.mean(fold_scores))}
            elif "regression" in normalized_type:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                fold_scores = []
                for train_idx, valid_idx in cv.split(X_train):
                    fold_model = clone(model)
                    fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    y_pred = fold_model.predict(X_train.iloc[valid_idx])
                    fold_scores.append(
                        float(root_mean_squared_error(y_train.iloc[valid_idx], y_pred))
                    )
                metrics = {"rmse": float(np.mean(fold_scores))}
            else:
                raise ValueError(f"Unsupported problem type for CV evaluation: {problem_type}")

            results.append({
                "model_name": model_name,
                "model": model,
                "metrics": metrics,
                "status": "CV Evaluated",
            })
        except Exception as exc:  # pragma: no cover - defensive failure recording
            errors.append({"model_name": model_name, "error": str(exc)})

    return {
        "results": results,
        "status": "Completed" if results else "Failed",
        "errors": errors,
    }


def tune_selected_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
) -> Dict[str, object]:
    """Run a small CV-only search for the selected supervised model family."""
    normalized_type = (problem_type or "").strip().lower()
    if "clustering" in normalized_type:
        return {"model": model, "status": "Skipped", "errors": []}

    search_spaces = {
        "LogisticRegression": {"C": [0.1, 1.0]},
        "DecisionTreeClassifier": {"max_depth": [None, 5]},
        "RandomForestClassifier": {"n_estimators": [100, 200]},
        "LinearRegression": {"fit_intercept": [True, False]},
        "DecisionTreeRegressor": {"max_depth": [None, 5]},
        "RandomForestRegressor": {"n_estimators": [100, 200]},
    }
    parameter_grid = search_spaces.get(model_name)
    if parameter_grid is None or (
        "classification" not in normalized_type
        and "regression" not in normalized_type
    ):
        return {"model": model, "status": "Skipped", "errors": []}

    candidates = []
    errors: List[Dict[str, Any]] = []
    try:
        for parameters in ParameterGrid(parameter_grid):
            candidate = clone(model)
            candidate.set_params(**parameters)
            evaluation = evaluate_models_cv(
                {model_name: candidate}, X_train, y_train, problem_type
            )
            results = evaluation.get("results", [])
            if results:
                candidates.append((results[0]["metrics"], candidate, parameters))
            else:
                errors.extend(evaluation.get("errors", []))
    except Exception as exc:  # pragma: no cover - defensive HPO fallback
        errors.append({"model_name": model_name, "error": str(exc)})

    if not candidates:
        return {"model": model, "status": "Fallback", "errors": errors}

    if "classification" in normalized_type:
        best_metrics, best_model, best_parameters = max(
            candidates, key=lambda item: item[0]["f1_score"]
        )
    else:
        best_metrics, best_model, best_parameters = min(
            candidates, key=lambda item: item[0]["rmse"]
        )

    return {
        "model": best_model,
        "status": "Completed",
        "errors": errors,
        "parameters": best_parameters,
        "metrics": best_metrics,
    }


def select_best_model(
    evaluation_results: Dict[str, object] | List[Dict[str, Any]],
    problem_type: str,
) -> Dict[str, object]:
    """
    Select the best candidate using the task-specific primary metric.

    Regression: lowest RMSE.
    Clustering: highest silhouette score.
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

    if "clustering" in normalized_type:
        metric_name = "silhouette_score"
        selector = max
        reason = "highest silhouette_score"
    elif "regression" in normalized_type:
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
