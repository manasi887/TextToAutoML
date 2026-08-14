from __future__ import annotations

import math
import re
from typing import Dict, List

import pandas as pd

from services.dataset.intelligence import (
    detect_constant_columns,
    detect_identifier_columns,
    generate_dataset_recommendations,
)


def detect_target_candidates(
    df: pd.DataFrame,
    generated_columns: List[str] | None = None,
) -> Dict[str, object]:
    """Rank likely target columns using multiple positive and negative signals."""

    identifier_report = detect_identifier_columns(df)
    constant_report = detect_constant_columns(df)
    exclude_columns = set(identifier_report["identifier_columns"]) | set(
        constant_report["constant_columns"]
    )
    if generated_columns:
        exclude_columns |= set(generated_columns)

    row_count = len(df)
    candidates: List[Dict[str, object]] = []

    for column in df.columns:
        if column in exclude_columns:
            continue

        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_null_series = series.dropna()
        if non_null_series.empty or non_null_series.nunique(dropna=True) < 2:
            continue

        datatype = _infer_candidate_datatype(series)
        if datatype is None:
            continue

        unique_count = int(non_null_series.nunique(dropna=True))
        unique_ratio = unique_count / row_count if row_count else 0.0
        missing_percentage = _missing_percentage(series, row_count)
        target_name_signal = _target_name_strength(column)
        score, confidence, reasons, problem_type = _score_target_candidate(
            series,
            column,
            datatype,
            unique_count,
            unique_ratio,
            missing_percentage,
            target_name_signal,
        )

        if score <= 0:
            continue

        candidates.append(
            {
                "column": column,
                "datatype": datatype,
                "score": round(score, 4),
                "confidence": round(confidence, 4),
                "inferred_problem_type": problem_type,
                "reasons": reasons,
            }
        )

    candidates.sort(key=lambda item: (item["score"], item["confidence"]), reverse=True)

    recommended_target = candidates[0]["column"] if candidates else None
    requires_confirmation = False
    if recommended_target is not None:
        if len(candidates) > 1:
            top_score = candidates[0]["score"]
            second_score = candidates[1]["score"]
            if abs(top_score - second_score) < 2.0:
                requires_confirmation = True
            if (
                candidates[0]["datatype"] == "numeric"
                and candidates[1]["datatype"] == "numeric"
                and "target_like_name" in candidates[0]["reasons"]
                and "target_like_name" in candidates[1]["reasons"]
            ):
                requires_confirmation = True
        if candidates[0]["confidence"] < 0.7:
            requires_confirmation = True

    return {
        "status": "Completed",
        "recommended_target": recommended_target,
        "requires_user_confirmation": requires_confirmation,
        "target_candidates": candidates,
    }


def detect_problem_type(df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    """Infer the problem type from target characteristics rather than raw dtype alone."""

    if not target_column or target_column.strip() == "" or target_column not in df.columns:
        return {
            "status": "Completed",
            "problem_type": "Clustering",
            "confidence": 0.0,
            "reason": "No target column was selected or the specified target column was not found.",
        }

    series = df[target_column]
    non_null_series = series.dropna()
    if non_null_series.empty:
        return {
            "status": "Completed",
            "problem_type": "Clustering",
            "confidence": 0.0,
            "reason": "The selected target contains no usable values.",
        }

    unique_classes = int(non_null_series.nunique(dropna=True))
    is_numeric = _is_numeric_target_series(series)
    is_categorical = _is_categorical_target_series(series)
    is_bool = pd.api.types.is_bool_dtype(series)

    if is_bool:
        return {
            "status": "Completed",
            "problem_type": "Binary Classification",
            "confidence": 0.96,
            "reason": "The target is boolean, which strongly indicates a binary classification task.",
        }

    if is_numeric:
        if unique_classes == 2 and _is_binary_numeric_target(non_null_series):
            return {
                "status": "Completed",
                "problem_type": "Binary Classification",
                "confidence": 0.92,
                "reason": "The target is numeric with exactly two discrete classes, which strongly indicates a binary classification task.",
            }

        return {
            "status": "Completed",
            "problem_type": "Regression",
            "confidence": 0.9,
            "reason": "The target is numeric, which strongly indicates a regression task.",
        }

    if is_categorical:
        if unique_classes <= 2:
            return {
                "status": "Completed",
                "problem_type": "Binary Classification",
                "confidence": 0.9,
                "reason": "The target is categorical with exactly two classes.",
            }
        if unique_classes > 2:
            return {
                "status": "Completed",
                "problem_type": "Multi-class Classification",
                "confidence": 0.88,
                "reason": "The target is categorical with more than two classes.",
            }

        return {
            "status": "Completed",
            "problem_type": "Clustering",
            "confidence": 0.2,
            "reason": "The target has insufficient class variation to support supervised learning.",
        }

    return {
        "status": "Completed",
        "problem_type": "Clustering",
        "confidence": 0.0,
        "reason": "The selected target does not provide enough supervised signal for a reliable task classification.",
    }


def generate_automl_recommendation(
    df: pd.DataFrame,
    generated_columns: List[str] | None = None,
) -> Dict[str, object]:
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
    target_report = detect_target_candidates(df, generated_columns=generated_columns)
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
            f"Consider using '{recommended_target}' as the target column"
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
        "confidence": round(target_report.get("target_candidates", [{}])[0].get("confidence", 0.0), 4) if target_report.get("target_candidates") else 0.0,
        "requires_user_confirmation": target_report.get("requires_user_confirmation", False),
        "problem_type": problem_type,
        "problem_confidence": problem_report.get("confidence", 0.0),
        "target_candidates": candidates,
        "recommended_models": model_report["recommended_models"],
        "recommendations": recommendations,
        "dataset_intelligence": intelligence_report,
        "problem_detection": problem_report,
    }


def recommend_models(problem_type: str) -> Dict[str, object]:
    """Recommend baseline algorithms for the inferred task type."""

    normalized_type = (problem_type or "").strip().lower()
    if "regression" in normalized_type:
        recommended_models = [
            "Linear Regression - a strong baseline for numeric targets and simple relationships.",
            "Decision Tree Regressor - captures nonlinear relationships and interactions.",
            "Random Forest Regressor - robust ensemble model for improved accuracy and stability.",
            "XGBoost Regressor - high-performance gradient boosting for complex regression tasks.",
        ]
    elif "binary classification" in normalized_type or "multi-class classification" in normalized_type or "classification" in normalized_type:
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


def _infer_candidate_datatype(series: pd.Series) -> str | None:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if _is_categorical_target_series(series):
        return "categorical"
    return None


def _is_numeric_target_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical_target_series(series: pd.Series) -> bool:
    return (
        pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    )


def _missing_percentage(series: pd.Series, row_count: int) -> float:
    if row_count == 0:
        return 0.0
    return float(series.isna().sum() / row_count)


def _score_target_candidate(
    series: pd.Series,
    column_name: str,
    datatype: str,
    unique_count: int,
    unique_ratio: float,
    missing_percentage: float,
    target_name_signal: int,
) -> tuple[float, float, List[str], str]:
    """Compute a robust score and confidence for one target candidate."""

    reasons: List[str] = []
    score = 0.0

    normalized = str(column_name).lower()
    if _looks_like_identifier_like_column(column_name):
        score -= 6.0
        reasons.append("identifier_like_name")

    if "fnlwgt" in normalized or "weight_count" in normalized or "sample_weight" in normalized:
        score -= 5.0
        reasons.append("weight_like_metadata")

    if _is_constant_series(series):
        score -= 4.0
        reasons.append("constant_column")

    if pd.api.types.is_datetime64_any_dtype(series):
        score -= 5.0
        reasons.append("datetime_column")

    if target_name_signal > 0:
        score += 0.75 + min(target_name_signal, 2) * 0.75
        reasons.append("target_like_name")

    if datatype == "numeric":
        score += 1.5
        reasons.append("numeric_dtype")
        if unique_count == 2:
            score += 6.0
            reasons.append("binary_numeric_target_like")
        elif unique_count <= 10:
            score += 2.0
            reasons.append("low_cardinality_numeric")
        if unique_count > 10:
            score += 1.5 + min(unique_ratio, 1.0) * 1.5
            reasons.append("continuous_numeric")
        if unique_count > 20 and not _looks_like_target_name(column_name):
            score -= 2.5
            reasons.append("continuous_feature_penalty")
        score += min(unique_ratio, 1.0) * 2.0
        if unique_count > 10 and not _looks_like_target_name(column_name):
            score -= 2.0
            reasons.append("measurement_feature_penalty")
    else:
        score += 1.0
        reasons.append("categorical_dtype")
        if unique_count <= 2:
            score += 3.5
            reasons.append("binary_categorical_target_like")
        elif 2 < unique_count <= 20:
            score += 2.5
            reasons.append("multi_class_target_like")
        if unique_count > 2 and target_name_signal <= 1:
            score -= 2.5
            reasons.append("ordinary_multiclass_feature_penalty")
        if not _looks_like_target_name(column_name):
            score -= 2.5
            reasons.append("generic_feature_penalty")
        if unique_count > 10:
            score -= 1.5
            reasons.append("high_cardinality_feature_penalty")
        score += min(unique_ratio, 1.0) * 1.5
        if unique_count <= 20 and not _looks_like_identifier_like_column(column_name):
            score += 0.75
            reasons.append("plausible_class_target")

    if missing_percentage < 0.1:
        score += 1.0
        reasons.append("complete_column")
    else:
        score -= min(missing_percentage, 1.0) * 2.0
        reasons.append("missing_values")

    if datatype == "numeric" and unique_count > 20 and unique_count < 0.9 * len(series) and not _looks_like_target_name(column_name):
        score -= 1.0
        reasons.append("not_target_like")

    problem_type = "Regression" if datatype == "numeric" else "Multi-class Classification"
    if datatype != "numeric" and unique_count <= 2:
        problem_type = "Binary Classification"

    confidence = min(0.98, max(0.15, 0.4 + max(score, 0.0) / 10.0))
    return score, confidence, reasons, problem_type


def _looks_like_identifier_like_column(column_name: str) -> bool:
    normalized = str(column_name).lower()
    return any(token in normalized for token in ["id", "row", "customerid", "userid", "account", "transaction", "employee", "invoice", "code", "key", "index"])


def _looks_like_target_name(column_name: str) -> bool:
    normalized = str(column_name).lower().replace(" ", "_")
    target_tokens = [
        "target", "label", "class", "outcome", "result", "status", "income",
        "sales", "price", "cost", "amount", "value", "revenue", "profit",
        "score", "rating", "stroke", "churn", "exited", "purchase",
        "default", "risk", "loan", "disease", "approved", "survived",
    ]
    return any(re.search(rf"(^|_){re.escape(token)}($|_)", normalized) for token in target_tokens)


def _target_name_strength(column_name: str) -> int:
    normalized = str(column_name).lower().replace(" ", "_")
    if re.search(r"(^|_)(target|label|class|outcome|result)(_|$)", normalized):
        return 4
    if re.search(r"(^|_)(income|sales|price|value|revenue|profit)(_|$)", normalized):
        return 3
    if re.search(r"(^|_)(status)(_|$)", normalized):
        return 1
    if re.search(r"(^|_)(churn|exited|stroke|default|approved|purchased|survived|score|rating)(_|$)", normalized):
        return 4
    if re.search(r"(^|_)(id|row|customer|user)(_|$)", normalized):
        return -4
    return 0


def _is_binary_numeric_target(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = int(non_null.nunique(dropna=True))
    if unique_count != 2:
        return False

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        return False

    rounded = numeric.round()
    if not numeric.eq(rounded).all():
        return False

    return True


def _is_constant_series(series: pd.Series) -> bool:
    return series.dropna().nunique(dropna=True) <= 1


def _looks_like_discrete_numeric_target(series: pd.Series) -> bool:
    unique_count = int(series.nunique(dropna=True))
    if unique_count <= 2:
        return True
    return unique_count <= 10 and series.dropna().round().nunique(dropna=True) <= 10


def _looks_like_continuous_numeric_target(series: pd.Series) -> bool:
    unique_count = int(series.nunique(dropna=True))
    return unique_count > 10 and series.dropna().std(ddof=0) > 0
