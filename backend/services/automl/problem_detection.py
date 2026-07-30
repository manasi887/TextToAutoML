from __future__ import annotations

from typing import Dict, List

import pandas as pd

from services.dataset.intelligence import (
    detect_constant_columns,
    detect_identifier_columns,
    generate_dataset_recommendations,
)


def detect_target_candidates(df: pd.DataFrame) -> Dict[str, object]:
    """
    Detect columns that are reasonable prediction target candidates.

    The function excludes identifier and constant columns, then evaluates
    numeric and categorical columns for meaningful variation. Each candidate
    receives a score that combines unique value richness, missing value
    completeness, and data type suitability.
    """

    identifier_report = detect_identifier_columns(df)
    constant_report = detect_constant_columns(df)
    exclude_columns = set(identifier_report["identifier_columns"]) | set(
        constant_report["constant_columns"]
    )

    row_count = len(df)
    candidates: List[Dict[str, object]] = []

    for column in df.columns:
        if column in exclude_columns:
            continue

        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if _is_numeric_target_series(series):
            datatype = "numeric"
        elif _is_categorical_target_series(series):
            datatype = "categorical"
        else:
            continue

        non_null_series = series.dropna()
        unique_count = int(non_null_series.nunique(dropna=True))
        if unique_count < 2:
            continue

        missing_percentage = _missing_percentage(series, row_count)
        unique_ratio = unique_count / row_count if row_count else 0.0
        score = _score_target_candidate(datatype, unique_count, unique_ratio, missing_percentage)

        candidates.append(
            {
                "column": column,
                "datatype": datatype,
                "score": round(score, 4),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return {
        "status": "Completed",
        "target_candidates": candidates,
    }


def detect_problem_type(df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    """
    Determine the best matching machine learning problem type for a selected target.

    Decision rules:
    - If no target is selected or the target column is missing, the task is
      treated as clustering because there is no supervised label.
    - Numeric targets are mapped to regression.
    - Boolean or binary categorical targets are mapped to binary classification.
    - Multi-class categorical targets are mapped to multi-class classification.
    """

    if not target_column or target_column.strip() == "" or target_column not in df.columns:
        return {
            "status": "Completed",
            "problem_type": "Clustering",
            "reason": "No target column was selected or the specified target column was not found.",
        }

    series = df[target_column]

    if _is_numeric_target_series(series):
        return {
            "status": "Completed",
            "problem_type": "Regression",
            "reason": "Selected target is numeric, which typically indicates a regression task.",
        }

    if pd.api.types.is_bool_dtype(series):
        return {
            "status": "Completed",
            "problem_type": "Binary Classification",
            "reason": "Selected target is boolean, which indicates a binary classification task.",
        }

    if _is_categorical_target_series(series):
        unique_classes = int(series.dropna().nunique(dropna=True))
        if unique_classes == 2:
            return {
                "status": "Completed",
                "problem_type": "Binary Classification",
                "reason": "Selected target is categorical with exactly two classes.",
            }
        if unique_classes > 2:
            return {
                "status": "Completed",
                "problem_type": "Multi-class Classification",
                "reason": "Selected target is categorical with more than two classes.",
            }

        return {
            "status": "Completed",
            "problem_type": "Clustering",
            "reason": "Selected target has no meaningful class variation.",
        }

    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "status": "Completed",
            "problem_type": "Regression",
            "reason": "Selected target is datetime-like, which is often treated as regression in AutoML workflows.",
        }

    return {
        "status": "Completed",
        "problem_type": "Clustering",
        "reason": "Selected target type is not numeric or categorical, so a clustering task is the safest fallback.",
    }


def generate_automl_recommendation(df: pd.DataFrame) -> Dict[str, object]:
    """
    Generate a combined AutoML recommendation report for the dataset.

    Architecture:
    - Use dataset intelligence to identify identifier, constant, and high-cardinality columns.
    - Use target detection to find promising supervised target candidates.
    - Use problem detection to infer the likely ML problem type from the best target.
    - Use model recommendation to suggest algorithms for the inferred problem.

    The function keeps the orchestration logic separate from the
    underlying detectors to preserve modularity and maintainability.
    """

    intelligence_report = generate_dataset_recommendations(df)
    target_report = detect_target_candidates(df)
    candidates = target_report.get("target_candidates", [])
    recommended_target = candidates[0]["column"] if candidates else None

    if recommended_target:
        problem_report = detect_problem_type(df, recommended_target)
        problem_type = problem_report["problem_type"]
    else:
        problem_report = {
            "status": "Completed",
            "problem_type": "Clustering",
            "reason": "No target candidates were detected, so unsupervised clustering is the safest default.",
        }
        problem_type = "Clustering"

    model_report = recommend_models(problem_type)

    recommendations: List[str] = []
    recommendations.extend(intelligence_report.get("recommendations", []))

    if recommended_target:
        recommendations.append(
            f"Consider using '{recommended_target}' as the target column for the AutoML pipeline."
        )
        recommendations.append(
            f"The dataset appears suited for a {problem_type} task based on the selected target."
        )
    else:
        recommendations.append(
            "No strong supervised target was identified. Consider selecting a label column or using clustering techniques."
        )

    if model_report["recommended_models"]:
        recommendations.append(
            f"Recommended model families for {problem_type}: {', '.join([item.split(' - ')[0] for item in model_report['recommended_models']])}."
        )

    return {
        "status": "Completed",
        "problem_type": problem_type,
        "recommended_target": recommended_target,
        "target_candidates": candidates,
        "recommended_models": model_report["recommended_models"],
        "recommendations": recommendations,
        "dataset_intelligence": intelligence_report,
        "problem_detection": problem_report,
    }


def recommend_models(problem_type: str) -> Dict[str, object]:
    """
    Recommend machine learning algorithms based on the detected problem type.

    The recommendations are chosen to cover simple baselines and powerful
    tree-based learners for supervised tasks, while clustering tasks use
    commonly applied unsupervised algorithms.
    """

    normalized_type = (problem_type or "").strip().lower()
    if normalized_type == "regression":
        recommended_models = [
            "Linear Regression - a strong baseline for numeric targets and simple relationships.",
            "Decision Tree Regressor - captures nonlinear relationships and interactions.",
            "Random Forest Regressor - robust ensemble model for improved accuracy and stability.",
            "XGBoost Regressor - high-performance gradient boosting for complex regression tasks.",
        ]
    elif normalized_type in {"binary classification", "multi-class classification", "classification"}:
        recommended_models = [
            "Logistic Regression - a reliable baseline for classification with interpretable coefficients.",
            "Decision Tree Classifier - easy to understand and handles nonlinear decision boundaries.",
            "Random Forest Classifier - an ensemble method that reduces overfitting and improves generalization.",
            "XGBoost Classifier - powerful gradient boosting suited for complex classification problems.",
        ]
    elif normalized_type == "clustering":
        recommended_models = [
            "K-Means - efficient for partitioning data into spherical clusters.",
            "DBSCAN - detects clusters of arbitrary shape and isolates noise.",
            "Agglomerative Clustering - hierarchical clustering suitable for nested group structures.",
        ]
    else:
        recommended_models = [
            "No model recommendations available for the specified problem type.",
        ]

    return {
        "status": "Completed",
        "recommended_models": recommended_models,
    }


def _is_numeric_target_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical_target_series(series: pd.Series) -> bool:
    return pd.api.types.is_categorical_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _missing_percentage(series: pd.Series, row_count: int) -> float:
    if row_count == 0:
        return 0.0
    return float(series.isna().sum() / row_count)


def _score_target_candidate(
    datatype: str,
    unique_count: int,
    unique_ratio: float,
    missing_percentage: float,
) -> float:
    """
    Compute a heuristic score for a potential target column.

    Numeric targets receive a larger base weight, while categorical targets
    use a smaller base weight. Higher unique ratio and lower missing value
    percentage increase the score.
    """

    if datatype == "numeric":
        base_score = 1.0
        unique_score = min(unique_ratio, 1.0) * 4.0
    else:
        base_score = 0.8
        unique_score = min(unique_ratio, 1.0) * 3.0

    completeness_score = max(0.0, 1.0 - missing_percentage) * 2.0
    return base_score + unique_score + completeness_score
