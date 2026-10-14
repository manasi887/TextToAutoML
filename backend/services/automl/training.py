from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import Birch, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ParameterGrid,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    precision_recall_fscore_support,
    r2_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .trainer import (
    prepare_training_data,
    transform_training_features,
    transform_training_target,
)
from .optimizer import (
    ensure_optuna_available,
    optimize_with_optuna,
    optimize_with_random_search,
)


_BAYESIAN_SEARCH_SPACES = {
    "LogisticRegression": {
        "C": {"kind": "float", "low": 1e-3, "high": 100.0, "log": True},
        "class_weight": {"kind": "categorical", "choices": [None, "balanced"]},
    },
    "DecisionTreeClassifier": {
        "max_depth": {"kind": "categorical", "choices": [None, 3, 5, 10, 20, 30]},
        "min_samples_split": {"kind": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"kind": "int", "low": 1, "high": 10},
        "class_weight": {"kind": "categorical", "choices": [None, "balanced"]},
    },
    "RandomForestClassifier": {
        "n_estimators": {"kind": "int", "low": 50, "high": 300, "step": 25},
        "max_depth": {"kind": "categorical", "choices": [None, 5, 10, 20, 30]},
        "min_samples_split": {"kind": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"kind": "int", "low": 1, "high": 10},
        "max_features": {"kind": "categorical", "choices": ["sqrt", "log2", None]},
        "class_weight": {
            "kind": "categorical",
            "choices": [None, "balanced", "balanced_subsample"],
        },
    },
    "LinearRegression": {
        "fit_intercept": {"kind": "categorical", "choices": [True, False]},
    },
    "DecisionTreeRegressor": {
        "criterion": {
            "kind": "categorical",
            "choices": ["squared_error", "friedman_mse", "absolute_error"],
        },
        "max_depth": {"kind": "categorical", "choices": [None, 3, 5, 10, 20, 30]},
        "min_samples_split": {"kind": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"kind": "int", "low": 1, "high": 10},
    },
    "RandomForestRegressor": {
        "n_estimators": {"kind": "int", "low": 50, "high": 300, "step": 25},
        "max_depth": {"kind": "categorical", "choices": [None, 5, 10, 20, 30]},
        "min_samples_split": {"kind": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"kind": "int", "low": 1, "high": 10},
        "max_features": {"kind": "categorical", "choices": [1.0, "sqrt", "log2"]},
    },
}

CLASS_IMBALANCE_RATIO_THRESHOLD = 2.0
"""Use balanced accuracy when the training largest/smallest class ratio is >= 2:1."""
CLUSTERING_EVALUATION_SAMPLE_SIZE = 3000
MAX_CLUSTER_COUNT_CANDIDATES = 12


def _classification_selection_metric(target: pd.Series) -> str:
    """Choose a class-sensitive metric when the training labels are imbalanced."""
    counts = pd.Series(target).dropna().value_counts()
    imbalance_ratio = float(counts.max() / counts.min()) if len(counts) >= 2 else 1.0
    return (
        "balanced_accuracy"
        if imbalance_ratio >= CLASS_IMBALANCE_RATIO_THRESHOLD
        else "f1_score"
    )


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


def _clustering_score_sample(
    X: pd.DataFrame,
    max_rows: int = CLUSTERING_EVALUATION_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Return a deterministic sample for silhouette scoring on large datasets."""
    if len(X) <= max_rows:
        return X.copy()
    return X.sample(n=max_rows, random_state=42).copy()


def train_clustering_models(
    X_train: pd.DataFrame,
) -> Dict[str, object]:
    """Fit comparable, prediction-compatible clustering candidates on one sample."""
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

    max_k = min(
        MAX_CLUSTER_COUNT_CANDIDATES,
        int(np.sqrt(n_samples)),
        n_samples - 1,
    )
    candidate_ks = list(range(2, max_k + 1))
    if not candidate_ks:
        return {
            "trained_models": {},
            "model_names": [],
            "status": "Failed",
            "training_errors": [{"model_name": None, "error": "No valid KMeans cluster counts are available."}],
            "candidate_cluster_counts": [],
            "candidate_algorithms": ["KMeans", "BIRCH"],
        }

    trained_models: Dict[str, Any] = {}
    training_errors: List[Dict[str, Any]] = []
    candidate_sample = _clustering_score_sample(X_train)

    for k in candidate_ks:
        candidates = (
            (
                f"KMeans_k{k}",
                KMeans(n_clusters=k, random_state=42, n_init=10),
            ),
            (f"Birch_k{k}", Birch(n_clusters=k)),
        )
        for model_name, model in candidates:
            try:
                model.fit(candidate_sample)
                labels = np.asarray(model.predict(candidate_sample))
                unique_labels = np.unique(labels)
                if unique_labels.size < 2 or unique_labels.size >= len(candidate_sample):
                    raise ValueError(
                        f"{model_name} produced an invalid silhouette configuration."
                    )
                if unique_labels.size != k:
                    raise ValueError(
                        f"{model_name} produced {unique_labels.size} clusters instead of {k}."
                    )
                trained_models[model_name] = model
            except Exception as exc:  # pragma: no cover - defensive failure recording
                training_errors.append({"model_name": model_name, "error": str(exc)})

    return {
        "trained_models": trained_models,
        "model_names": list(trained_models),
        "status": "Completed" if trained_models else "Failed",
        "training_errors": training_errors,
        "candidate_sample_size": len(candidate_sample),
        "candidate_cluster_counts": candidate_ks,
        "candidate_algorithms": ["KMeans", "BIRCH"],
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
        _clustering_score_sample(X_test)
        if "clustering" in normalized_type
        else None
    )

    for model_name, model in models.items():
        try:
            clustering_details: dict[str, Any] = {}
            if "clustering" in normalized_type:
                assert clustering_score_sample is not None
                labels = np.asarray(model.predict(clustering_score_sample))
                unique_labels, cluster_counts = np.unique(labels, return_counts=True)
                if len(unique_labels) < 2 or len(unique_labels) >= len(clustering_score_sample):
                    raise ValueError("Clustering evaluation requires at least two non-trivial clusters.")
                scoring_features = clustering_score_sample
                model_steps = getattr(model, "steps", [])
                if len(model_steps) > 1:
                    scoring_features = model[:-1].transform(clustering_score_sample)
                try:
                    metrics = {
                        "silhouette_score": float(
                            silhouette_score(scoring_features, labels)
                        ),
                        "calinski_harabasz_score": float(
                            calinski_harabasz_score(scoring_features, labels)
                        ),
                        "davies_bouldin_score": float(
                            davies_bouldin_score(scoring_features, labels)
                        ),
                    }
                except ValueError as exc:
                    raise ValueError(
                        f"Clustering metrics could not be calculated: {exc}"
                    ) from exc
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise ValueError("Clustering evaluation produced a non-finite metric.")
                clustering_details = {
                    "cluster_sizes": {
                    str(label): int(count)
                    for label, count in zip(unique_labels, cluster_counts)
                    },
                    "small_cluster_count": int(
                        np.count_nonzero(
                            cluster_counts
                            < max(2, int(np.ceil(0.01 * len(clustering_score_sample))))
                        )
                    ),
                }
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
                labels = np.unique(np.concatenate((y_true, y_pred)))
                class_precision, class_recall, class_f1, class_support = (
                    precision_recall_fscore_support(
                        y_true,
                        y_pred,
                        labels=labels,
                        average=None,
                        zero_division=0,
                    )
                )
                matrix = (
                    confusion_matrix(y_true, y_pred, labels=labels).tolist()
                    if len(labels) <= 100
                    else None
                )
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
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "macro_precision": float(
                        precision_score(y_true, y_pred, average="macro", zero_division=0)
                    ),
                    "macro_recall": float(
                        recall_score(y_true, y_pred, average="macro", zero_division=0)
                    ),
                    "macro_f1_score": float(
                        f1_score(y_true, y_pred, average="macro", zero_division=0)
                    ),
                    "confusion_matrix": matrix,
                    "confusion_matrix_labels": [str(label) for label in labels],
                    "per_class_metrics": {
                        str(label): {
                            "precision": float(class_precision[index]),
                            "recall": float(class_recall[index]),
                            "f1_score": float(class_f1[index]),
                            "support": int(class_support[index]),
                        }
                        for index, label in enumerate(labels)
                    },
                }
            else:
                raise ValueError(f"Unsupported problem type for evaluation: {problem_type}")

            result = {
                "model_name": model_name,
                "model": model,
                "metrics": metrics,
                "status": "Evaluated",
            }
            result.update(clustering_details)
            results.append(result)
        except Exception as exc:  # pragma: no cover - defensive failure recording
            errors.append({"model_name": model_name, "error": str(exc)})

    status = "Completed" if results else "Failed"

    return {
        "results": results,
        "status": status,
        "errors": errors,
    }


def _safe_cv_folds(
    y: pd.Series,
    problem_type: str,
    groups: pd.Series | None = None,
) -> int | None:
    """Return a safe fold count for supervised CV without crashing on tiny datasets."""
    normalized_type = (problem_type or "").strip().lower()
    if groups is not None:
        y = y.reset_index(drop=True)
        groups = pd.Series(groups).reset_index(drop=True)
        if len(y) != len(groups) or groups.isna().any():
            return None
        group_count = int(groups.nunique())
        if group_count < 2:
            return None
    else:
        group_count = len(y)

    if "classification" in normalized_type:
        if groups is not None:
            class_group_counts = pd.DataFrame({"label": y, "group": groups})
            class_group_counts = class_group_counts.drop_duplicates().groupby("label")["group"].nunique()
            if class_group_counts.empty or int(class_group_counts.min()) < 2:
                return None
            return min(5, group_count, int(class_group_counts.min()))
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
        return min(3, group_count)

    return None


def evaluate_models_cv(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    groups: pd.Series | None = None,
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
            n_splits = _safe_cv_folds(y_train, problem_type, groups)
            if n_splits is None:
                errors.append({
                    "model_name": model_name,
                    "error": "Dataset is too small for supervised CV; skipping CV scoring.",
                })
                continue

            if "classification" in normalized_type:
                cv = (
                    StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    if groups is not None
                    else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                )
                fold_scores = []
                fold_balanced_accuracy = []
                split_iterator = cv.split(X_train, y_train, groups) if groups is not None else cv.split(X_train, y_train)
                for train_idx, valid_idx in split_iterator:
                    fold_model = clone(model)
                    fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    y_pred = fold_model.predict(X_train.iloc[valid_idx])
                    y_true = y_train.iloc[valid_idx]
                    fold_scores.append(
                        float(
                            f1_score(
                                y_true,
                                y_pred,
                                average="weighted",
                                zero_division=0,
                            )
                        )
                    )
                    fold_balanced_accuracy.append(
                        float(balanced_accuracy_score(y_true, y_pred))
                    )
                metrics = {
                    "f1_score": float(np.mean(fold_scores)),
                    "balanced_accuracy": float(np.mean(fold_balanced_accuracy)),
                }
            elif "regression" in normalized_type:
                cv = (
                    GroupKFold(n_splits=n_splits)
                    if groups is not None
                    else KFold(n_splits=n_splits, shuffle=True, random_state=42)
                )
                fold_scores = []
                split_iterator = cv.split(X_train, y_train, groups) if groups is not None else cv.split(X_train)
                for train_idx, valid_idx in split_iterator:
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
    *,
    raw_training_df: pd.DataFrame | None = None,
    target_column: str | None = None,
    preparation_function: Any | None = None,
    group_column: str | None = None,
    tuning_strategy: str = "grid",
    n_trials: int = 8,
    random_state: int = 42,
) -> Dict[str, object]:
    """Tune one estimator using the existing CV-only strategy or Optuna TPE."""
    normalized_type = (problem_type or "").strip().lower()
    normalized_strategy = (tuning_strategy or "").strip().lower()
    if normalized_strategy not in {"grid", "bayesian", "random", "none"}:
        raise ValueError("tuning_strategy must be 'grid', 'bayesian', 'random', or 'none'.")
    if "clustering" in normalized_type:
        return {"model": model, "status": "Skipped", "errors": []}
    if normalized_strategy == "none":
        return {
            "model": model,
            "status": "Skipped",
            "strategy": "none",
            "errors": [],
        }
    if normalized_strategy == "bayesian":
        ensure_optuna_available()

    search_spaces = {
        "LogisticRegression": {
            "C": [0.1, 1.0],
            "class_weight": [None, "balanced"],
        },
        "DecisionTreeClassifier": {
            "max_depth": [None, 5],
            "class_weight": [None, "balanced"],
        },
        "RandomForestClassifier": {
            "n_estimators": [100, 200],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        "LinearRegression": {"fit_intercept": [True, False]},
        "DecisionTreeRegressor": {"max_depth": [None, 5]},
        "RandomForestRegressor": {"n_estimators": [50, 100]},
    }
    parameter_grid = search_spaces.get(model_name)
    if parameter_grid is None or (
        "classification" not in normalized_type
        and "regression" not in normalized_type
    ):
        return {"model": model, "status": "Skipped", "errors": []}

    candidates = []
    errors: List[Dict[str, Any]] = []
    cv_splitter = None
    groups = None
    if group_column is not None:
        if raw_training_df is None or group_column not in raw_training_df.columns:
            return {
                "model": model,
                "status": "Failed",
                "errors": [{
                    "model_name": model_name,
                    "error": f"Group-aware CV requires group column '{group_column}' in the raw training data.",
                }],
            }
        groups = raw_training_df[group_column].reset_index(drop=True)
        if groups.isna().any():
            return {
                "model": model,
                "status": "Failed",
                "errors": [{"model_name": model_name, "error": f"Group column '{group_column}' contains missing values."}],
            }
    if raw_training_df is not None and target_column is not None:
        n_splits = _safe_cv_folds(raw_training_df[target_column], problem_type, groups)
        if n_splits is None:
            return {
                "model": model,
                "status": "Failed" if group_column is not None else "Fallback",
                "errors": [{
                    "model_name": model_name,
                    "error": (
                        "Insufficient groups for group-aware CV. At least two groups are required, "
                        "and each class must occur in at least two groups."
                        if group_column is not None
                        else "Dataset is too small for supervised CV; using the baseline model."
                    ),
                }],
            }
        if "classification" in normalized_type:
            cv_splitter = (
                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                if groups is not None
                else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            )
        else:
            cv_splitter = (
                GroupKFold(n_splits=n_splits)
                if groups is not None
                else KFold(n_splits=n_splits, shuffle=True, random_state=42)
            )

    if normalized_strategy in {"bayesian", "random"}:
        search_space = _BAYESIAN_SEARCH_SPACES.get(model_name)
        if search_space is None:
            return {
                "model": model,
                "status": "Skipped",
                "strategy": normalized_strategy,
                "errors": [{"model_name": model_name, "error": f"No {normalized_strategy} search space is defined for this estimator."}],
            }

        metric_name = (
            _classification_selection_metric(
                raw_training_df[target_column]
                if raw_training_df is not None and target_column is not None
                else y_train
            )
            if "classification" in normalized_type
            else "rmse"
        )
        direction = "maximize" if metric_name == "f1_score" else "minimize"
        fold_metric_cache: dict[tuple[tuple[str, Any], ...], dict[str, float]] = {}

        def evaluate_parameters(parameters: dict[str, Any]) -> float:
            candidate = clone(model)
            candidate.set_params(**parameters)
            if "random_state" in candidate.get_params(deep=False):
                candidate.set_params(random_state=random_state)
            if cv_splitter is not None and raw_training_df is not None and target_column is not None:
                evaluation = _evaluate_model_cv_with_preprocessing(
                    model_name,
                    candidate,
                    raw_training_df,
                    target_column,
                    problem_type,
                    cv_splitter,
                    preparation_function or prepare_training_data,
                    group_column=group_column,
                )
            else:
                evaluation = evaluate_models_cv(
                    {model_name: candidate}, X_train, y_train, problem_type
                )
            results = evaluation.get("results", [])
            if not results:
                details = evaluation.get("errors", [])
                raise ValueError(
                    f"Bayesian CV evaluation failed for {model_name}: {details}"
                )
            metrics = results[0]["metrics"]
            cache_key = tuple(sorted(parameters.items()))
            fold_metric_cache[cache_key] = metrics
            return float(metrics[metric_name])

        optimizer = (
            optimize_with_optuna
            if normalized_strategy == "bayesian"
            else optimize_with_random_search
        )
        optimization = optimizer(
            model,
            search_space,
            evaluate_parameters,
            direction=direction,
            n_trials=n_trials,
            random_state=random_state,
        )
        best_parameters = {
            name: value
            for name, value in optimization["parameters"].items()
            if name != "random_state"
        }
        best_metrics = fold_metric_cache.get(tuple(sorted(best_parameters.items())))
        if best_metrics is None:
            best_metrics = {metric_name: optimization["value"]}
        return {
            "model": optimization["model"],
            "status": "Completed",
            "strategy": normalized_strategy,
            "errors": [],
            "parameters": best_parameters,
            "metrics": best_metrics,
            "trials_run": optimization["trials_run"],
        }

    try:
        for parameters in ParameterGrid(parameter_grid):
            candidate = clone(model)
            candidate.set_params(**parameters)
            if cv_splitter is not None and raw_training_df is not None and target_column is not None:
                evaluation = _evaluate_model_cv_with_preprocessing(
                    model_name,
                    candidate,
                    raw_training_df,
                    target_column,
                    problem_type,
                    cv_splitter,
                    preparation_function or prepare_training_data,
                    group_column=group_column,
                )
            else:
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

    metric_name = (
        _classification_selection_metric(
            raw_training_df[target_column]
            if raw_training_df is not None and target_column is not None
            else y_train
        )
        if "classification" in normalized_type
        else "rmse"
    )
    if "classification" in normalized_type:
        best_metrics, best_model, best_parameters = max(
            candidates, key=lambda item: item[0][metric_name]
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


def _evaluate_model_cv_with_preprocessing(
    model_name: str,
    model: Any,
    raw_training_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    cv: Any,
    preparation_function: Any,
    group_column: str | None = None,
) -> Dict[str, object]:
    """Evaluate a candidate with imputation/encoding fitted inside each CV fold."""
    fold_metrics: list[dict[str, float]] = []
    errors: list[dict[str, Any]] = []
    normalized_type = (problem_type or "").strip().lower()
    y_raw = raw_training_df[target_column].reset_index(drop=True)
    groups = (
        raw_training_df[group_column].reset_index(drop=True)
        if group_column is not None
        else None
    )

    try:
        if groups is not None:
            split_iterator = cv.split(raw_training_df, y_raw, groups)
        elif "classification" in normalized_type:
            split_iterator = cv.split(raw_training_df, y_raw)
        else:
            split_iterator = cv.split(raw_training_df)
        for train_indices, validation_indices in split_iterator:
            fold_train = raw_training_df.iloc[train_indices].reset_index(drop=True)
            fold_validation = raw_training_df.iloc[validation_indices].reset_index(drop=True)
            preparation = preparation_function(fold_train, target_column)
            X_validation = transform_training_features(
                fold_validation.drop(columns=[target_column]), preparation
            )
            y_validation = transform_training_target(
                fold_validation[target_column], preparation
            )
            fold_model = clone(model)
            fold_model.fit(preparation["X"], preparation["y"])
            predictions = np.asarray(fold_model.predict(X_validation))

            if "regression" in normalized_type:
                metrics = {
                    "rmse": float(root_mean_squared_error(y_validation, predictions)),
                    "mae": float(mean_absolute_error(y_validation, predictions)),
                }
                if len(y_validation) >= 2:
                    metrics["r2"] = float(r2_score(y_validation, predictions))
            elif "classification" in normalized_type:
                metrics = {
                    "f1_score": float(
                        f1_score(
                            y_validation,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_validation, predictions)
                    ),
                }
            else:
                raise ValueError(f"Unsupported problem type for CV evaluation: {problem_type}")
            fold_metrics.append(metrics)
    except Exception as exc:
        errors.append({"model_name": model_name, "error": str(exc)})

    if not fold_metrics:
        return {"results": [], "status": "Failed", "errors": errors}

    metrics = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics if metric in fold]))
        for metric in fold_metrics[0]
    }
    return {
        "results": [{
            "model_name": model_name,
            "model": model,
            "metrics": metrics,
            "status": "CV Evaluated",
        }],
        "status": "Completed",
        "errors": errors,
    }


def select_best_model(
    evaluation_results: Dict[str, object] | List[Dict[str, Any]],
    problem_type: str,
    *,
    training_target: pd.Series | None = None,
) -> Dict[str, object]:
    """
    Select the best candidate using the task-specific primary metric.

    Regression: lowest RMSE.
    Clustering: best mean rank across silhouette, Calinski-Harabasz, and
    Davies-Bouldin scores.
    Classification: highest weighted F1 for balanced training classes, otherwise
    highest balanced accuracy when the training largest/smallest class ratio is
    at least ``CLASS_IMBALANCE_RATIO_THRESHOLD``.
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
        selector = min
        reason = (
            "best mean metric rank across silhouette, Calinski-Harabasz, "
            "and Davies-Bouldin scores"
        )
    elif "regression" in normalized_type:
        metric_name = "rmse"
        selector = min
        reason = "lowest rmse"
    elif "classification" in normalized_type:
        metric_name = _classification_selection_metric(
            training_target
            if isinstance(training_target, pd.Series)
            else pd.Series(dtype="int64")
        )
        selector = max
        reason = f"highest {metric_name}"
    else:
        raise ValueError(f"Unsupported problem type for model selection: {problem_type}")

    valid_results = []
    for item in candidate_results:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics")
        value = metrics.get(metric_name) if isinstance(metrics, dict) else None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            continue
        if not np.isfinite(float(value)):
            continue
        valid_results.append(item)

    if not valid_results:
        return {
            "status": "Failed",
            "best_model_name": None,
            "best_model": None,
            "best_metrics": {},
            "selection_metric": metric_name,
            "selection_reason": f"No results had a valid {metric_name} metric.",
        }

    valid_results.sort(key=lambda item: str(item.get("model_name", "")))
    if "clustering" in normalized_type:
        metric_directions = {
            "silhouette_score": -1,
            "calinski_harabasz_score": -1,
            "davies_bouldin_score": 1,
        }
        rank_totals = {id(item): 0.0 for item in valid_results}
        rank_counts = {id(item): 0 for item in valid_results}
        for metric, direction in metric_directions.items():
            available = [
                item for item in valid_results
                if isinstance(item.get("metrics"), dict)
                and isinstance(item["metrics"].get(metric), (int, float, np.number))
                and np.isfinite(float(item["metrics"][metric]))
            ]
            ranked = sorted(
                available,
                key=lambda item: (
                    direction * float(item["metrics"][metric]),
                    str(item.get("model_name", "")),
                ),
            )
            for rank, item in enumerate(ranked, start=1):
                rank_totals[id(item)] += float(rank)
                rank_counts[id(item)] += 1

        best_result = min(
            valid_results,
            key=lambda item: (
                rank_totals[id(item)] / max(rank_counts[id(item)], 1),
                -float(item["metrics"][metric_name]),
                str(item.get("model_name", "")),
            ),
        )
    else:
        best_result = selector(
            valid_results,
            key=lambda item: float(item["metrics"][metric_name]),
        )
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
