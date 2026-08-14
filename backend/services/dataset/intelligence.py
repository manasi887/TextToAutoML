from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import pandas as pd


TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_REGRESSION = "regression"
TASK_TYPE_UNSUPERVISED = "unsupervised"
TASK_TYPE_UNKNOWN = "unknown"


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Return a high-level summary of the dataset."""

    column_types = df.dtypes.astype(str).value_counts().to_dict()
    missing_values = int(df.isnull().sum().sum())
    datetime_columns = _get_datetime_columns(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing_values,
        "column_types": column_types,
        "datetime_columns": datetime_columns,
        "datetime_column_count": len(datetime_columns),
    }


def analyze_features(df: pd.DataFrame) -> Dict[str, object]:
    """Analyze features and return metadata useful for preprocessing."""

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    constant_columns = [
        column for column in df.columns if _is_constant_column(df[column])
    ]
    high_cardinality_report = detect_high_cardinality(df, threshold=0.5)
    high_cardinality_columns = [
        item["column"] for item in high_cardinality_report["high_cardinality_columns"]
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "high_cardinality_details": high_cardinality_report["high_cardinality_columns"],
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "constant_column_count": len(constant_columns),
        "high_cardinality_column_count": len(high_cardinality_columns),
    }


def detect_constant_columns(df: pd.DataFrame) -> Dict[str, object]:
    """Detect columns where every non-null value is identical."""

    constant_columns = [
        column for column in df.columns if _is_constant_column(df[column])
    ]

    return {
        "status": "Completed",
        "constant_columns": constant_columns,
        "count": len(constant_columns),
    }


def detect_identifier_columns(df: pd.DataFrame) -> Dict[str, object]:
    """Detect likely identifier columns using name and data-based signals."""

    identifier_columns: List[str] = []
    metadata: Dict[str, Dict[str, object]] = {}
    row_count = len(df)

    for column in df.columns:
        detection = _evaluate_identifier_column(df[column], column, row_count)
        if detection["is_identifier"]:
            identifier_columns.append(column)
            metadata[column] = detection["details"]

    return {
        "status": "Completed",
        "identifier_columns": identifier_columns,
        "count": len(identifier_columns),
        "details": metadata,
    }


def detect_high_cardinality(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Detect high-cardinality categorical columns with dataset-size-aware thresholds."""

    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be a float between 0 and 1.")

    row_count = len(df)
    candidate_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    high_cardinality_columns = []

    for column in candidate_columns:
        series = df[column]
        non_null_values = series.dropna()
        if non_null_values.empty:
            continue

        unique_count = int(non_null_values.nunique(dropna=True))
        unique_ratio = float(unique_count / row_count) if row_count else 0.0
        threshold_count = max(20, int(math.ceil(row_count * threshold)))

        if unique_count >= threshold_count or unique_ratio >= threshold:
            high_cardinality_columns.append(
                {
                    "column": column,
                    "unique_count": unique_count,
                    "unique_ratio": round(unique_ratio, 4),
                    "threshold_used": round(threshold, 4),
                    "threshold_count": threshold_count,
                    "dtype": str(series.dtype),
                    "reason": "High ratio of unique values relative to dataset size.",
                }
            )

    return {
        "status": "Completed",
        "high_cardinality_columns": high_cardinality_columns,
        "count": len(high_cardinality_columns),
    }


def generate_dataset_recommendations(
    df: pd.DataFrame,
    high_cardinality_threshold: float = 0.5,
) -> Dict[str, object]:
    """Generate a dataset intelligence report with actionable recommendations."""

    identifier_report = detect_identifier_columns(df)
    constant_report = detect_constant_columns(df)
    high_cardinality_report = detect_high_cardinality(
        df,
        threshold=high_cardinality_threshold,
    )

    recommendations: List[str] = []

    if identifier_report["count"] > 0:
        identifier_columns = identifier_report["identifier_columns"]
        recommendations.append(
            f"Remove {len(identifier_columns)} identifier column(s) before training: {', '.join(identifier_columns)}."
        )

    if constant_report["count"] > 0:
        constant_columns = constant_report["constant_columns"]
        recommendations.append(
            f"Remove {len(constant_columns)} constant column(s) that contain only a single value: {', '.join(constant_columns)}."
        )

    high_cardinality_columns = high_cardinality_report["high_cardinality_columns"]
    if high_cardinality_columns:
        column_names = [col["column"] for col in high_cardinality_columns]
        recommendations.append(
            f"Review {len(column_names)} high-cardinality categorical column(s): {', '.join(column_names)}. "
            "Consider feature hashing, target encoding, or dropping these columns if they are not predictive."
        )

    if not recommendations:
        recommendations.append(
            "No obvious identifier, constant, or high-cardinality issues were detected. "
            "Proceed with the standard preprocessing pipeline."
        )

    return {
        "identifier_columns": identifier_report,
        "constant_columns": constant_report,
        "high_cardinality_columns": high_cardinality_report,
        "recommendations": recommendations,
    }


def analyze_target(df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    """Analyze a target column and infer the problem type."""

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    target_series = df[target_column]
    unique_values = target_series.dropna().unique().tolist()
    unique_count = int(target_series.nunique(dropna=True))
    missing_count = int(target_series.isnull().sum())
    sample_type = str(target_series.dtype)
    distribution = _get_value_counts(target_series)
    problem_type = _infer_problem_type(target_series)

    return {
        "target_column": target_column,
        "target_dtype": sample_type,
        "unique_value_count": unique_count,
        "missing_values": missing_count,
        "sample_values": unique_values[:10],
        "distribution": distribution,
        "problem_type": problem_type,
    }


def recommend_model_family(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, object]:
    """Recommend a model family based on dataset characteristics."""

    if target_column is None:
        return {
            "task_type": TASK_TYPE_UNSUPERVISED,
            "recommendation": "No target column provided. Consider unsupervised learning or specify a target.",
        }

    analysis = analyze_target(df, target_column)
    problem_type = analysis["problem_type"]
    recommendation = _get_model_recommendation(problem_type)

    return {
        "task_type": problem_type,
        "recommendation": recommendation,
    }


def recommend_preprocessing(df: pd.DataFrame) -> Dict[str, object]:
    """
    Recommend preprocessing steps for the dataset.
    """

    summary = summarize_dataset(df)
    feature_analysis = analyze_features(df)
    steps: List[str] = []

    if summary["missing_values"] > 0:
        steps.append("impute_missing_values")

    if feature_analysis["categorical_column_count"] > 0:
        steps.append("encode_categorical_features")

    if summary["datetime_column_count"] > 0:
        steps.append("extract_date_features")

    if feature_analysis["high_cardinality_column_count"] > 0:
        steps.append("reduce_cardinality")

    if feature_analysis["constant_column_count"] > 0:
        steps.append("drop_constant_columns")

    return {
        "recommended_steps": steps,
        "recommendation_count": len(steps),
    }


def generate_dataset_intelligence(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Dict[str, object]:
    """
    Generate a complete dataset intelligence report.
    """

    intelligence = {
        "dataset_summary": summarize_dataset(df),
        "feature_analysis": analyze_features(df),
        "preprocessing_recommendations": recommend_preprocessing(df),
    }

    if target_column is not None:
        intelligence["target_analysis"] = analyze_target(df, target_column)
        intelligence["model_recommendation"] = recommend_model_family(df, target_column)

    return intelligence


def _get_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that are datetime-like.
    """

    return [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]


def _is_constant_column(series: pd.Series) -> bool:
    """
    Return True when a column contains at most one unique non-null value.
    """

    non_null_values = series.dropna()
    return non_null_values.nunique(dropna=False) <= 1


def _get_value_counts(series: pd.Series) -> Dict[str, int]:
    """
    Return a dictionary of the most frequent values and their counts.
    """

    counts = series.value_counts(dropna=False).head(10)
    return counts.astype(int).to_dict()


def _infer_problem_type(series: pd.Series) -> str:
    """
    Infer regression or classification based on the target series.
    """

    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
        return TASK_TYPE_REGRESSION

    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_string_dtype(series):
        return TASK_TYPE_CLASSIFICATION

    return TASK_TYPE_UNKNOWN


def _matches_identifier_name(column_name: str) -> bool:
    """Use name heuristics to detect obvious identifier-like fields."""

    normalized = re.sub(r"[^a-z0-9]+", " ", column_name.lower()).strip()
    if not normalized:
        return False

    if _looks_like_measurement_column(normalized):
        return False

    if "postal" in normalized or "zip" in normalized:
        return False

    identifier_patterns = [
        r"(?:row|order|customer|user|account|employee|transaction|invoice|ticket|shipment|session|product)[ _-]?(?:id|identifier|number|key|code)",
        r"(?:row|order|customer|user|account|employee|transaction|invoice|ticket|shipment|session|product)(?:id|identifier|number|key|code)",
        r"\b(?:id|identifier|uuid|guid)\b",
        r"(?:customer|user|account|employee|transaction|order|invoice|product)[ _-]?(?:no|num|number|id)",
    ]

    return any(re.search(pattern, normalized) for pattern in identifier_patterns)


def _matches_identifier_data(
    series: pd.Series,
    column_name: str,
    row_count: int,
) -> bool:
    """Use uniqueness and key-like data patterns to detect identifiers."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    non_null_series = series.dropna()
    if non_null_series.empty:
        return False

    unique_count = int(non_null_series.nunique(dropna=True))
    unique_ratio = unique_count / row_count if row_count else 0.0
    null_ratio = 1 - (len(non_null_series) / row_count) if row_count else 1.0

    if unique_count < 10 or unique_ratio < 0.8 or null_ratio > 0.1:
        return False

    normalized = re.sub(r"[^a-z0-9]+", " ", column_name.lower()).strip()
    if _looks_like_measurement_column(normalized):
        return False
    if "postal" in normalized or "zip" in normalized:
        return False
    if not re.search(r"\b(?:id|number|key|user|customer|account|employee|transaction|row|product|order)\b", normalized):
        return False

    sample = non_null_series.astype(str).head(50)
    if sample.empty:
        return False

    key_like = sample.str.fullmatch(r"[A-Za-z0-9_-]+", na=False)
    if len(key_like) and not key_like.all():
        return False

    return True


def _evaluate_identifier_column(
    series: pd.Series,
    column_name: str,
    row_count: int,
) -> Dict[str, object]:
    """Evaluate one column for identifier-likeness and return reasons."""

    details: Dict[str, object] = {
        "column": column_name,
        "unique_count": 0,
        "unique_ratio": 0.0,
        "null_ratio": 0.0,
        "reasons": [],
    }

    if series.empty or pd.api.types.is_datetime64_any_dtype(series):
        return {"is_identifier": False, "details": details}

    normalized = re.sub(r"[^a-z0-9]+", " ", str(column_name).lower()).strip()
    if _looks_like_measurement_column(normalized):
        details["reasons"].append("measurement_column")
        return {"is_identifier": False, "details": details}
    if "postal" in normalized or "zip" in normalized:
        details["reasons"].append("postal_code")
        return {"is_identifier": False, "details": details}

    non_null_series = series.dropna()
    if non_null_series.empty:
        return {"is_identifier": False, "details": details}

    unique_count = int(non_null_series.nunique(dropna=True))
    unique_ratio = float(unique_count / row_count) if row_count else 0.0
    null_ratio = float((len(series) - len(non_null_series)) / row_count) if row_count else 0.0

    details["unique_count"] = unique_count
    details["unique_ratio"] = round(unique_ratio, 4)
    details["null_ratio"] = round(null_ratio, 4)

    if _matches_identifier_name(column_name):
        details["reasons"].append("name_pattern")

    if unique_count >= 10 and unique_ratio >= 0.8:
        details["reasons"].append("near_unique")

    if _matches_identifier_data(series, column_name, row_count):
        details["reasons"].append("data_pattern")

    is_identifier = bool(details["reasons"]) and (unique_ratio >= 0.8 or unique_count >= 10)
    return {"is_identifier": is_identifier, "details": details}


def _looks_like_measurement_column(column_name: str) -> bool:
    """Return True for ordinary measurements that should not be treated as identifiers."""

    normalized = re.sub(r"[^a-z0-9]+", " ", str(column_name).lower()).strip()
    if not normalized:
        return False

    measurement_tokens = {
        "salary", "estimatedsalary", "income", "age", "balance", "score",
        "height", "weight", "amount", "price", "cost", "total", "value",
        "quantity", "duration", "time", "distance", "percent", "ratio",
    }

    return any(token in normalized for token in measurement_tokens)


def _get_model_recommendation(task_type: str) -> str:
    """
    Return a simple recommendation string for the detected task type.
    """

    if task_type == TASK_TYPE_REGRESSION:
        return "Use regression algorithms such as linear regression, random forest regression, or gradient boosting."
    if task_type == TASK_TYPE_CLASSIFICATION:
        return "Use classification algorithms such as logistic regression, random forest, or gradient boosting."
    if task_type == TASK_TYPE_UNSUPERVISED:
        return "Use unsupervised algorithms such as clustering or dimensionality reduction."

    return "Task type could not be determined. Consider specifying a target column or reviewing the target data."
