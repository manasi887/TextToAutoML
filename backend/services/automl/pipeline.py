from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import clone

from services.automl.persistence import save_model_package
from services.automl.problem_detection import detect_problem_type
from services.automl.trainer import (
    prepare_clustering_data,
    prepare_training_data,
    split_dataset_train_validation_test,
    encode_features,
    transform_training_features,
    transform_training_target,
)
from services.automl.training import (
    evaluate_models,
    select_best_model,
    train_clustering_models,
    train_models,
    tune_selected_model,
    _BAYESIAN_SEARCH_SPACES,
    _safe_cv_folds,
)
from services.automl.optimizer import (
    MAX_OPTUNA_TRIALS,
    is_optuna_available,
    select_auto_optimizer,
)
from services.automl.optimizer import MAX_OPTUNA_TRIALS
from services.metalearning.meta_features import extract_meta_features
from services.metalearning.openml_loader import load_cached_openml_records
from services.metalearning.recommender import recommend_models


_SUPPORTED_PROBLEM_TYPES = {
    "binary classification",
    "multi-class classification",
    "multi class classification",
    "regression",
    "clustering",
}


def _safe_diagnostic_error(value: Any) -> str:
    """Return a short single-line error without exposing local filesystem paths."""
    reason = str(value or "").strip()
    reason = reason.splitlines()[0] if reason else "Candidate processing failed."
    path_marker = re.search(r"(?i)(?:[A-Z]:[\\/]|\\\\|(?<![\w:.])/(?!/))", reason)
    if path_marker:
        reason = f"{reason[:path_marker.start()]}[local path]"
    reason = re.sub(r"\s+", " ", reason)
    return reason[:240] or "Candidate processing failed."


def _diagnostic_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _build_validation_diagnostics(
    candidate_names: List[str],
    training_errors: List[Dict[str, Any]],
    tuning_results: Dict[str, Dict[str, Any]],
    validation_results: List[Dict[str, Any]],
    validation_errors: List[Dict[str, Any]],
    selection_metric: str | None,
    training_results: List[Dict[str, Any]] | None = None,
    training_metric_errors: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Build candidate diagnostics from training and validation outcomes only."""
    training_results = training_results or []
    training_metric_errors = training_metric_errors or []
    ordered_names = list(dict.fromkeys(name for name in candidate_names if name))
    for entries in (training_errors, validation_errors, training_metric_errors):
        for entry in entries:
            name = entry.get("model_name") if isinstance(entry, dict) else None
            if isinstance(name, str) and name and name not in ordered_names:
                ordered_names.append(name)
    for name in tuning_results:
        if name not in ordered_names:
            ordered_names.append(name)

    training_failures = {
        entry.get("model_name"): entry
        for entry in training_errors
        if isinstance(entry, dict) and entry.get("model_name")
    }
    validation_failures = {
        entry.get("model_name"): entry
        for entry in validation_errors
        if isinstance(entry, dict) and entry.get("model_name")
    }
    evaluated = {
        entry.get("model_name"): entry
        for entry in validation_results
        if isinstance(entry, dict) and entry.get("model_name")
    }
    training_evaluated = {
        entry.get("model_name"): entry
        for entry in training_results
        if isinstance(entry, dict) and entry.get("model_name")
    }
    training_metric_failures = {
        entry.get("model_name"): entry
        for entry in training_metric_errors
        if isinstance(entry, dict) and entry.get("model_name")
    }

    diagnostics = []
    for name in ordered_names:
        result = evaluated.get(name)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        training_result = training_evaluated.get(name)
        training_metrics = (
            training_result.get("metrics", {})
            if isinstance(training_result, dict)
            else None
        )
        if not isinstance(training_metrics, dict):
            training_metrics = None
        training_metric_error = training_metric_failures.get(name)

        error_reason = None
        tuning = tuning_results.get(name, {})
        tuning_status = tuning.get("status")
        tuning_errors = tuning.get("errors", [])
        if name in training_failures:
            status = "training_failed"
            error_reason = _safe_diagnostic_error(training_failures[name].get("error"))
        elif name in validation_failures:
            status = "validation_failed"
            error_reason = _safe_diagnostic_error(validation_failures[name].get("error"))
        elif result is None:
            status = "not_evaluated"
            error_reason = "Candidate did not produce validation metrics."
        elif not selection_metric or _diagnostic_number(metrics.get(selection_metric)) is None:
            status = "missing_metrics"
            if tuning_status in {"Failed", "Fallback"} and tuning_errors:
                error_reason = _safe_diagnostic_error(
                    tuning_errors[0].get("error")
                    if isinstance(tuning_errors[0], dict)
                    else tuning_errors[0]
                )
            else:
                error_reason = "Validation result did not include a finite selection metric."
        elif tuning_status in {"Failed", "Fallback"}:
            status = "evaluated_with_tuning_fallback"
            error_reason = _safe_diagnostic_error(
                tuning_errors[0].get("error")
                if tuning_errors and isinstance(tuning_errors[0], dict)
                else "Hyperparameter tuning did not complete; baseline parameters were evaluated."
            )
        elif tuning_errors:
            status = "evaluated_with_tuning_warnings"
            error_reason = _safe_diagnostic_error(
                tuning_errors[0].get("error")
                if isinstance(tuning_errors[0], dict)
                else tuning_errors[0]
            )
        else:
            status = "evaluated"

        per_class = metrics.get("per_class_metrics", {})
        per_class_recall_support = {}
        if isinstance(per_class, dict):
            for label, values in per_class.items():
                if isinstance(values, dict):
                    per_class_recall_support[str(label)] = {
                        "recall": _diagnostic_number(values.get("recall")),
                        "support": values.get("support")
                        if isinstance(values.get("support"), int)
                        else None,
                    }

        diagnostics.append({
            "candidate_name": name,
            "status": status,
            "error_reason": error_reason,
            "training_metrics": training_metrics,
            "training_metrics_error": (
                _safe_diagnostic_error(training_metric_error.get("error"))
                if training_metric_error
                else None
            ),
            "validation_metrics": metrics if result is not None else None,
            "selection_metric": selection_metric,
            "selection_value": _diagnostic_number(metrics.get(selection_metric))
            if selection_metric
            else None,
            "weighted_f1": _diagnostic_number(metrics.get("f1_score")),
            "balanced_accuracy": _diagnostic_number(metrics.get("balanced_accuracy")),
            "macro_f1": _diagnostic_number(metrics.get("macro_f1_score")),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "confusion_matrix_labels": metrics.get("confusion_matrix_labels"),
            "per_class_metrics": per_class_recall_support,
        })
    return diagnostics


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


def _prepare_training_features(
    df: pd.DataFrame,
    target_column: str,
    group_column: str | None = None,
    feature_columns: List[str] | None = None,
) -> Dict[str, Any]:
    """Call the frozen training preparation logic and fall back if it strips all usable features."""

    preparation = prepare_training_data(
        df,
        target_column,
        group_column=group_column,
        feature_columns=feature_columns,
    )
    X = preparation["X"]
    y = preparation["y"]

    if X.empty or X.shape[1] == 0:
        columns_to_drop = [target_column]
        if group_column is not None:
            columns_to_drop.append(group_column)
        raw_X = df.drop(columns=columns_to_drop, errors="ignore")
        if feature_columns is not None:
            raw_X = raw_X.loc[:, feature_columns].copy()
        if raw_X.empty or raw_X.shape[1] == 0:
            return preparation
        raw_X = raw_X.copy()
        numeric_values: dict[str, float] = {}
        for column in raw_X.select_dtypes(include=["number"]).columns:
            fill_value = float(raw_X[column].mean())
            raw_X[column] = raw_X[column].fillna(fill_value)
            numeric_values[column] = fill_value
        categorical_values: dict[str, Any] = {}
        for column in raw_X.select_dtypes(include=["object", "string", "category"]).columns:
            modes = raw_X[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "<missing>"
            raw_X[column] = raw_X[column].fillna(fill_value)
            categorical_values[column] = fill_value
        valid_target_rows = df[target_column].notna()
        raw_X = raw_X.loc[valid_target_rows].reset_index(drop=True)
        y_raw = df.loc[valid_target_rows, target_column].reset_index(drop=True)
        encoded_X, encoders = encode_features(raw_X)
        return {
            "X": encoded_X.reset_index(drop=True),
            "y": y_raw,
            "feature_names": encoded_X.columns.tolist(),
            "encoders": encoders,
            "preprocessing": {
                **preparation.get("preprocessing", {}),
                "removed_columns": {
                    **preparation.get("preprocessing", {}).get("removed_columns", {}),
                    "group_columns": [group_column] if group_column is not None else [],
                },
                "raw_feature_names": raw_X.columns.tolist(),
                "imputed_columns": {
                    "numeric": list(numeric_values),
                    "categorical": list(categorical_values),
                },
                "imputation_values": {
                    "numeric": numeric_values,
                    "categorical": categorical_values,
                },
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
    excluded_target_columns: List[str] | None = None,
    group_column: str | None = None,
    feature_columns: List[str] | None = None,
    tuning_strategy: str = "grid",
    tuning_trials: int = 8,
) -> Dict[str, object]:
    """Run the end-to-end AutoML training workflow for a validated supervised task."""

    if df is None or df.empty:
        raise ValueError("Input dataset is empty.")

    if target_column is None and str(problem_type or "").strip().lower() != "clustering":
        raise ValueError("A target column must be explicitly supplied to the AutoML pipeline.")

    if str(problem_type or "").strip().lower() != "clustering" and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the DataFrame.")
    if group_column is not None and group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' is not present in the DataFrame.")
    if group_column is not None and group_column == target_column:
        raise ValueError("Group column must be different from the target column.")
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or any(not isinstance(column, str) for column in feature_columns):
            raise ValueError("feature_columns must be a list of column names.")
        missing_columns = [column for column in feature_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Feature column(s) not present in the DataFrame: {', '.join(missing_columns)}")
        excluded_features = set(excluded_target_columns or [])
        if target_column in df.columns:
            excluded_features.add(target_column)
        if group_column is not None:
            excluded_features.add(group_column)
        feature_columns = list(dict.fromkeys(
            column for column in feature_columns if column not in excluded_features
        ))

    normalized_problem_type = "Clustering" if str(problem_type or "").strip().lower() == "clustering" else _normalize_problem_type(problem_type, df, target_column)
    normalized_tuning_strategy = str(tuning_strategy or "").strip().lower()
    if normalized_tuning_strategy not in {"grid", "bayesian", "auto"}:
        raise ValueError("tuning_strategy must be 'grid', 'bayesian', or 'auto'.")
    if isinstance(tuning_trials, bool) or not isinstance(tuning_trials, int):
        raise ValueError("tuning_trials must be an integer.")
    if not 1 <= tuning_trials <= MAX_OPTUNA_TRIALS:
        raise ValueError(f"tuning_trials must be between 1 and {MAX_OPTUNA_TRIALS}.")
    if normalized_problem_type == "Clustering" and normalized_tuning_strategy == "bayesian":
        raise ValueError("Bayesian optimization is available only for supervised classification and regression.")

    if normalized_problem_type == "Clustering":
        clustering_exclusions = list(excluded_target_columns or [])
        if group_column is not None:
            clustering_exclusions.append(group_column)
        preparation = prepare_clustering_data(
            df,
            excluded_target_columns=clustering_exclusions,
            feature_columns=feature_columns,
        )
        X = preparation["X"]
        feature_names = preparation["feature_names"]
        encoders = preparation["encoders"]
        if X.empty or X.shape[1] == 0:
            raise ValueError("No usable features remain for clustering after preprocessing.")
        training_report = train_clustering_models(X)
        trained_models = training_report.get("trained_models", {})
        training_errors = training_report.get("training_errors", [])
        evaluation = evaluate_models(trained_models, X, pd.Series(index=X.index, dtype=float), normalized_problem_type)
        evaluation_results = evaluation.get("results", [])
        if not evaluation_results:
            raise ValueError("No clustering model produced a valid evaluation.")
        best_model_report = select_best_model(evaluation_results, normalized_problem_type)
        best_model = best_model_report["best_model"]
        best_name = best_model_report["best_model_name"]
        metrics = best_model_report["best_metrics"]
        excluded_targets = preparation["preprocessing"].get(
            "excluded_target_columns", []
        )
        preprocessing_target = excluded_targets[0] if excluded_targets else None
        packaged_model = save_model_package(
            model=best_model,
            encoders=encoders,
            feature_names=feature_names,
            target_column="",
            problem_type=normalized_problem_type,
            preprocessing={
                **preparation["preprocessing"],
                "target_column": preprocessing_target,
                "feature_names": feature_names,
                "encoders": encoders,
            },
            model_name=best_name,
            selection_metric="silhouette_score",
            metrics=metrics,
        )
        return {
            "status": "success",
            "target_column": None,
            "problem_type": normalized_problem_type,
            "data": {"original_rows": int(len(df)), "original_columns": int(len(df.columns)), "training_rows": int(len(X)), "test_rows": 0, "feature_count": int(X.shape[1])},
            "preprocessing": {
                **preparation["preprocessing"],
                "target_column": preprocessing_target,
                "feature_names": feature_names,
                "encoders": encoders,
            },
            "models": {"trained": list(trained_models), "failed": training_errors},
            "evaluation": {"status": evaluation.get("status", "Completed"), "results": evaluation_results, "errors": evaluation.get("errors", [])},
            "best_model": {"name": best_name, "selection_metric": "silhouette_score", "metrics": metrics, "reason": best_model_report["selection_reason"]},
            "model": {"model_id": packaged_model["model_id"], "name": packaged_model["model_name"], "problem_type": normalized_problem_type, "target_column": None, "selection_metric": "silhouette_score", "metrics": metrics},
            **({
                "tuning": select_auto_optimizer(
                    training_rows=len(df),
                    feature_count=int(X.shape[1]),
                    problem_type=normalized_problem_type,
                    candidate_models=[],
                    cv_folds=None,
                    group_count=None,
                    optuna_available=is_optuna_available(),
                    bayesian_models=set(_BAYESIAN_SEARCH_SPACES),
                    random_models=set(_BAYESIAN_SEARCH_SPACES),
                    maximum_trials=tuning_trials,
                ),
            } if normalized_tuning_strategy == "auto" else {}),
            "_internal_best_model": best_model,
        }

    train_df, validation_df, test_df = split_dataset_train_validation_test(
        df,
        target_column,
        test_size=test_size,
        problem_type=normalized_problem_type,
        random_state=random_state,
        group_column=group_column,
    )
    duplicate_rows_removed = int(
        df[target_column].notna().sum()
        - len(train_df)
        - len(validation_df)
        - len(test_df)
    )
    preparation = _prepare_training_features(
        train_df, target_column, group_column, feature_columns
    )
    X_train = preparation["X"]
    y_train = preparation["y"]
    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError(f"No usable features remain for training after preprocessing the target '{target_column}'.")

    X_validation = transform_training_features(
        validation_df.drop(columns=[target_column]), preparation
    )
    y_validation = transform_training_target(
        validation_df[target_column], preparation
    )

    recommended_model_order: List[str] | None = None
    try:
        meta_features = extract_meta_features(
            (
                train_df.loc[:, list(dict.fromkeys(feature_columns + [target_column]))]
                if feature_columns is not None
                else train_df.drop(columns=[group_column], errors="ignore") if group_column else train_df
            ),
            target_column=target_column,
            problem_type=normalized_problem_type,
        )
        openml_records = load_cached_openml_records()
        recommendation = recommend_models(meta_features, openml_records=openml_records)
        if recommendation.get("status") == "recommended":
            recommended_model_order = [
                item["model_family"]
                for item in recommendation.get("recommendations", [])
                if isinstance(item, dict) and isinstance(item.get("model_family"), str)
            ] or None
    except Exception:  # pragma: no cover - recommender must never block baseline training
        recommended_model_order = None

    training_report = train_models(
        X_train,
        y_train,
        normalized_problem_type,
        recommended_model_order=recommended_model_order,
    )
    trained_models = training_report.get("trained_models", {})
    training_errors = training_report.get("training_errors", [])

    optimizer_decision: dict[str, Any] | None = None
    selected_tuning_strategy = normalized_tuning_strategy
    selected_tuning_trials = tuning_trials
    if normalized_tuning_strategy == "auto":
        training_groups = (
            train_df[group_column].reset_index(drop=True)
            if group_column is not None
            else None
        )
        cv_folds = _safe_cv_folds(
            train_df[target_column],
            normalized_problem_type,
            training_groups,
        )
        optimizer_decision = select_auto_optimizer(
            training_rows=len(train_df),
            feature_count=int(X_train.shape[1]),
            problem_type=normalized_problem_type,
            candidate_models=list(trained_models),
            cv_folds=cv_folds,
            group_count=(
                int(training_groups.nunique(dropna=False))
                if training_groups is not None
                else None
            ),
            optuna_available=is_optuna_available(),
            bayesian_models=set(_BAYESIAN_SEARCH_SPACES),
            random_models=set(_BAYESIAN_SEARCH_SPACES),
            maximum_trials=tuning_trials,
        )
        selected_tuning_strategy = optimizer_decision["selected_strategy"]
        selected_tuning_trials = optimizer_decision["trial_budget"]

    if not trained_models:
        validation_diagnostics = _build_validation_diagnostics(
            [], training_errors, {}, [], [], None
        )
        return {
            "status": "failed",
            "target_column": target_column,
            "problem_type": normalized_problem_type,
            "data": {
                "original_rows": int(len(df)),
                "duplicate_rows_removed": duplicate_rows_removed,
                "original_columns": int(len(df.columns)),
                "training_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "test_rows": int(len(test_df)),
                "feature_count": int(X_train.shape[1]),
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
            "validation_diagnostics": validation_diagnostics,
            "error": "All candidate models failed to train for the selected problem type.",
            **({"tuning": optimizer_decision} if optimizer_decision is not None else {}),
        }

    validation_models: Dict[str, Any] = {}
    cross_validation_metrics: Dict[str, Dict[str, float]] = {}
    tuning_results: Dict[str, Dict[str, Any]] = {}
    tuning_errors: List[Dict[str, Any]] = []
    validation_model_errors: List[Dict[str, Any]] = []
    tuning_duration_seconds = 0.0
    for model_name, model in trained_models.items():
        tuning_options: dict[str, Any] = {}
        if normalized_tuning_strategy == "auto":
            tuning_options = {
                "tuning_strategy": selected_tuning_strategy,
                "n_trials": selected_tuning_trials,
                "random_state": random_state,
            }
        elif normalized_tuning_strategy == "bayesian":
            tuning_options = {
                "tuning_strategy": normalized_tuning_strategy,
                "n_trials": tuning_trials,
                "random_state": random_state,
            }
        tuning_started_at = time.perf_counter()
        try:
            tuning = tune_selected_model(
                model_name,
                model,
                X_train,
                y_train,
                normalized_problem_type,
                raw_training_df=train_df,
                target_column=target_column,
                preparation_function=lambda fold_df, fold_target: _prepare_training_features(
                    fold_df, fold_target, group_column, feature_columns
                ),
                group_column=group_column,
                **tuning_options,
            )
        except Exception as exc:
            tuning = {
                "model": model,
                "status": "Failed",
                "errors": [{
                    "model_name": model_name,
                    "error": _safe_diagnostic_error(exc),
                }],
            }
        model_tuning_duration = time.perf_counter() - tuning_started_at
        tuning_duration_seconds += model_tuning_duration
        tuning_results[model_name] = {
            **{key: value for key, value in tuning.items() if key != "model"},
            "duration_seconds": round(model_tuning_duration, 6),
        }
        tuning_errors.extend(tuning.get("errors", []))
        if tuning.get("status") == "Completed":
            cross_validation_metrics[model_name] = tuning.get("metrics", {})
        tuning_model = (
            tuning.get("model")
            if tuning.get("status") == "Completed"
            else model
        )
        try:
            validation_model = clone(tuning_model)
            validation_model.fit(X_train, y_train)
            validation_models[model_name] = validation_model
        except Exception as exc:
            validation_error = {"model_name": model_name, "error": str(exc)}
            validation_model_errors.append(validation_error)
            tuning_errors.append(validation_error)

    training_evaluation = evaluate_models(
        validation_models,
        X_train,
        y_train,
        normalized_problem_type,
    )
    training_metric_results = training_evaluation.get("results", [])
    training_metric_errors = training_evaluation.get("errors", [])
    validation_evaluation = evaluate_models(
        validation_models,
        X_validation,
        y_validation,
        normalized_problem_type,
    )
    validation_results = validation_evaluation.get("results", [])
    validation_errors = [
        *validation_evaluation.get("errors", []),
        *validation_model_errors,
    ]
    tuning_summary = {
        **(optimizer_decision or {}),
        "requested_mode": normalized_tuning_strategy,
        "selected_strategy": selected_tuning_strategy,
        "strategy": selected_tuning_strategy,
        "trial_budget": selected_tuning_trials,
        "trials_per_model": selected_tuning_trials,
        "tuning_duration_seconds": round(tuning_duration_seconds, 6),
        "models": tuning_results,
    }
    include_tuning_summary = normalized_tuning_strategy in {"bayesian", "auto"}
    if not validation_results:
        validation_diagnostics = _build_validation_diagnostics(
            list(trained_models),
            training_errors,
            tuning_results,
            [],
            validation_errors,
            None,
            training_metric_results,
            training_metric_errors,
        )
        return {
            "status": "failed",
            "target_column": target_column,
            "problem_type": normalized_problem_type,
            "data": {
                "original_rows": int(len(df)),
                "duplicate_rows_removed": duplicate_rows_removed,
                "original_columns": int(len(df.columns)),
                "training_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "test_rows": int(len(test_df)),
                "feature_count": int(X_train.shape[1]),
            },
            "preprocessing": preparation.get("preprocessing", {}),
            "models": {
                "trained": list(trained_models.keys()),
                "failed": training_errors + tuning_errors,
            },
            "evaluation": {"status": "Failed", "results": [], "errors": validation_evaluation.get("errors", [])},
            "validation_diagnostics": validation_diagnostics,
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": "Validation produced no valid model results.",
            },
            **({"tuning": tuning_summary} if include_tuning_summary else {}),
        }

    best_model_report = select_best_model(
        validation_results,
        normalized_problem_type,
        training_target=(
            y_train if "classification" in normalized_problem_type.lower() else None
        ),
    )
    validation_diagnostics = _build_validation_diagnostics(
        list(trained_models),
        training_errors,
        tuning_results,
        validation_results,
        validation_errors,
        best_model_report.get("selection_metric"),
        training_metric_results,
        training_metric_errors,
    )
    if best_model_report.get("status") == "Failed" or best_model_report.get("best_model") is None:
        return {
            "status": "failed",
            "target_column": target_column,
            "problem_type": normalized_problem_type,
            "data": {
                "original_rows": int(len(df)),
                "duplicate_rows_removed": duplicate_rows_removed,
                "original_columns": int(len(df.columns)),
                "training_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "test_rows": int(len(test_df)),
                "feature_count": int(X_train.shape[1]),
            },
            "preprocessing": preparation.get("preprocessing", {}),
            "models": {
                "trained": list(trained_models.keys()),
                "failed": training_errors + tuning_errors,
            },
            "evaluation": {
                "status": validation_evaluation.get("status", "Completed"),
                "results": validation_results,
                "errors": validation_evaluation.get("errors", []),
            },
            "validation_diagnostics": validation_diagnostics,
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": best_model_report.get("selection_reason", "Best model selection failed."),
            },
            "error": "Best model selection failed for the selected problem type.",
            **({"tuning": tuning_summary} if include_tuning_summary else {}),
        }

    best_name = best_model_report["best_model_name"]
    validation_metrics = best_model_report["best_metrics"]
    selected_validation_model = best_model_report["best_model"]
    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_preparation = _prepare_training_features(
        train_validation_df, target_column, group_column, feature_columns
    )
    final_model = clone(selected_validation_model)
    final_model.fit(final_preparation["X"], final_preparation["y"])

    X_test = transform_training_features(
        test_df.drop(columns=[target_column]), final_preparation
    )
    y_test = transform_training_target(test_df[target_column], final_preparation)
    evaluation = evaluate_models(
        {best_name: final_model}, X_test, y_test, normalized_problem_type
    )
    evaluation_results = evaluation.get("results", [])
    if not evaluation_results:
        return {
            "status": "failed",
            "target_column": target_column,
            "problem_type": normalized_problem_type,
            "data": {
                "original_rows": int(len(df)),
                "duplicate_rows_removed": duplicate_rows_removed,
                "original_columns": int(len(df.columns)),
                "training_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "test_rows": int(len(test_df)),
                "final_fit_rows": int(len(train_validation_df)),
                "feature_count": int(final_preparation["X"].shape[1]),
            },
            "preprocessing": final_preparation.get("preprocessing", {}),
            "models": {
                "trained": list(trained_models.keys()),
                "failed": training_errors + tuning_errors,
            },
            "evaluation": {
                "status": "Failed",
                "results": [],
                "errors": evaluation.get("errors", []),
            },
            "validation_diagnostics": validation_diagnostics,
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": "Holdout evaluation produced no valid result.",
            },
            "error": "Holdout evaluation produced no valid model result.",
            **({"tuning": tuning_summary} if include_tuning_summary else {}),
        }

    best_model = final_model
    best_name = best_model_report.get("best_model_name")
    metrics = evaluation_results[0]["metrics"]
    reason = best_model_report["selection_reason"]

    selection_metric = best_model_report["selection_metric"]

    packaged_model = save_model_package(
        model=best_model,
        encoders=final_preparation.get("encoders", {}),
        feature_names=final_preparation.get("feature_names", []),
        target_column=target_column,
        problem_type=normalized_problem_type,
        preprocessing={
            **final_preparation.get("preprocessing", {}),
            "feature_names": final_preparation.get("feature_names", []),
            "encoders": final_preparation.get("encoders", {}),
        },
        model_name=best_name,
        selection_metric=selection_metric,
        metrics=metrics,
    )

    return {
        "status": "success",
        "target_column": target_column,
        "problem_type": normalized_problem_type,
        "data": {
            "original_rows": int(len(df)),
            "duplicate_rows_removed": duplicate_rows_removed,
            "original_columns": int(len(df.columns)),
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "final_fit_rows": int(len(train_validation_df)),
            "feature_count": int(final_preparation["X"].shape[1]),
        },
        "preprocessing": {
            **final_preparation.get("preprocessing", {}),
            "feature_names": final_preparation.get("feature_names", []),
            "encoders": final_preparation.get("encoders", {}),
        },
        "models": {
            "trained": list(trained_models.keys()),
            "failed": training_errors + tuning_errors,
        },
        "evaluation": {
            "status": evaluation.get("status", "Completed"),
            "results": evaluation_results,
            "errors": evaluation.get("errors", []),
        },
        "validation_diagnostics": validation_diagnostics,
        "best_model": {
            "name": best_name,
            "selection_metric": selection_metric,
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "cross_validation_metrics": cross_validation_metrics.get(best_name, {}),
            "reason": reason,
        },
        **({"tuning": tuning_summary} if include_tuning_summary else {}),
        "model": {
            "model_id": packaged_model["model_id"],
            "name": packaged_model["model_name"],
            "problem_type": packaged_model["problem_type"],
            "target_column": packaged_model["target_column"],
            "selection_metric": packaged_model["selection_metric"],
            "metrics": packaged_model["metrics"],
        },
        "_internal_best_model": best_model,
    }
