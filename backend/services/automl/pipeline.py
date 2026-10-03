from __future__ import annotations

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
)
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
) -> Dict[str, Any]:
    """Call the frozen training preparation logic and fall back if it strips all usable features."""

    preparation = prepare_training_data(
        df, target_column, group_column=group_column
    )
    X = preparation["X"]
    y = preparation["y"]

    if X.empty or X.shape[1] == 0:
        columns_to_drop = [target_column]
        if group_column is not None:
            columns_to_drop.append(group_column)
        raw_X = df.drop(columns=columns_to_drop, errors="ignore")
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

    normalized_problem_type = "Clustering" if str(problem_type or "").strip().lower() == "clustering" else _normalize_problem_type(problem_type, df, target_column)

    if normalized_problem_type == "Clustering":
        clustering_exclusions = list(excluded_target_columns or [])
        if group_column is not None:
            clustering_exclusions.append(group_column)
        preparation = prepare_clustering_data(
            df, excluded_target_columns=clustering_exclusions
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
    preparation = _prepare_training_features(train_df, target_column, group_column)
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
            train_df.drop(columns=[group_column], errors="ignore") if group_column else train_df,
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

    if not trained_models:
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
            "error": "All candidate models failed to train for the selected problem type.",
        }

    validation_models: Dict[str, Any] = {}
    cross_validation_metrics: Dict[str, Dict[str, float]] = {}
    tuning_errors: List[Dict[str, Any]] = []
    for model_name, model in trained_models.items():
        tuning = tune_selected_model(
            model_name,
            model,
            X_train,
            y_train,
            normalized_problem_type,
            raw_training_df=train_df,
            target_column=target_column,
            preparation_function=lambda fold_df, fold_target: _prepare_training_features(
                fold_df, fold_target, group_column
            ),
            group_column=group_column,
        )
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
            tuning_errors.append({"model_name": model_name, "error": str(exc)})

    validation_evaluation = evaluate_models(
        validation_models,
        X_validation,
        y_validation,
        normalized_problem_type,
    )
    validation_results = validation_evaluation.get("results", [])
    if not validation_results:
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
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": "Validation produced no valid model results.",
            },
        }

    best_model_report = select_best_model(validation_results, normalized_problem_type)
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
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": best_model_report.get("selection_reason", "Best model selection failed."),
            },
            "error": "Best model selection failed for the selected problem type.",
        }

    best_name = best_model_report["best_model_name"]
    validation_metrics = best_model_report["best_metrics"]
    selected_validation_model = best_model_report["best_model"]
    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_preparation = _prepare_training_features(
        train_validation_df, target_column, group_column
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
            "best_model": {
                "name": None,
                "selection_metric": None,
                "metrics": {},
                "reason": "Holdout evaluation produced no valid result.",
            },
            "error": "Holdout evaluation produced no valid model result.",
        }

    best_model = final_model
    best_name = best_model_report.get("best_model_name")
    metrics = evaluation_results[0]["metrics"]
    reason = best_model_report["selection_reason"]

    if normalized_problem_type == "Regression":
        selection_metric = "rmse"
    else:
        selection_metric = "f1_score"

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
        "best_model": {
            "name": best_name,
            "selection_metric": selection_metric,
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "cross_validation_metrics": cross_validation_metrics.get(best_name, {}),
            "reason": reason,
        },
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
