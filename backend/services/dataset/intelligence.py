from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd


TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_REGRESSION = "regression"
TASK_TYPE_UNSUPERVISED = "unsupervised"
TASK_TYPE_UNKNOWN = "unknown"


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """
    Return a high-level summary of the dataset.

    The summary includes row and column counts, missing value totals,
    column dtype counts, and the number of datetime-like columns.
    """

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
    """
    Analyze features and return metadata useful for preprocessing.
    """

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    constant_columns = [
        column
        for column in df.columns
        if _is_constant_column(df[column])
    ]
    high_cardinality_columns = [
        column
        for column in categorical_columns
        if df[column].nunique(dropna=False) > 50
    ]

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "constant_column_count": len(constant_columns),
        "high_cardinality_column_count": len(high_cardinality_columns),
    }


def detect_constant_columns(df: pd.DataFrame) -> Dict[str, object]:
    """
    Detect columns where every non-null value is identical.

    Constant columns provide no predictive power and should usually be
    removed before training. NaN values are ignored when determining
    whether a column contains a single repeating value.
    """

    constant_columns = [
        column
        for column in df.columns
        if _is_constant_column(df[column])
    ]

    return {
        "status": "Completed",
        "constant_columns": constant_columns,
        "count": len(constant_columns),
    }


def detect_identifier_columns(df: pd.DataFrame) -> Dict[str, object]:
    """
    Detect columns that are likely identifiers and should not be used as features.

    Detection uses column name heuristics and data characteristics to
    identify common identifier columns such as Order ID, Customer ID,
    Product ID, and Invoice ID.
    """

    identifier_columns: List[str] = []
    row_count = len(df)

    for column in df.columns:
        if _matches_identifier_name(column):
            identifier_columns.append(column)
            continue

        if _matches_identifier_data(df[column], column, row_count):
            identifier_columns.append(column)

    return {
        "status": "Completed",
        "identifier_columns": identifier_columns,
        "count": len(identifier_columns),
    }


def detect_high_cardinality(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Detect categorical columns with too many unique values.

    High-cardinality categorical columns often create many encoding
    dimensions, which can increase model complexity and overfitting risk.
    """

    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be a float between 0 and 1.")

    row_count = len(df)
    candidate_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    high_cardinality_columns = []

    for column in candidate_columns:
        non_null_values = df[column].dropna()
        unique_count = int(non_null_values.nunique(dropna=True))
        unique_percentage = float(unique_count / row_count) if row_count else 0.0

        if unique_percentage >= threshold:
            high_cardinality_columns.append(
                {
                    "column": column,
                    "unique_values": unique_count,
                    "unique_percentage": unique_percentage,
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
    """
    Generate a dataset intelligence report with actionable recommendations.

    The function combines identifier, constant, and high-cardinality
    detection and generates recommendations dynamically based on the
    detected issues.
    """

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
    """
    Analyze a target column and infer the problem type.
    """

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
    """
    Recommend a model family based on dataset characteristics.
    """

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
    """
    Use column name heuristics to identify likely identifier columns.
    """

    normalized = re.sub(r"[^a-z0-9]+", " ", column_name.lower()).strip()
    if not normalized:
        return False

    identifier_patterns = [
        r"\b(?:row|order|customer|product|employee|transaction|invoice|ticket|shipment)\b.*\b(?:id|number|key)\b",
        r"\b(?:id|identifier|key)\b",
        r"\b(?:order|customer|product|employee|transaction|invoice|ticket|shipment)\b.*\b(?:id|number)\b",
    ]

    return any(re.search(pattern, normalized) for pattern in identifier_patterns)


def _matches_identifier_data(
    series: pd.Series,
    column_name: str,
    row_count: int,
) -> bool:
    """
    Use data characteristics to identify likely identifier columns.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    non_null_series = series.dropna()
    if non_null_series.empty:
        return False

    unique_count = non_null_series.nunique(dropna=True)
    unique_ratio = unique_count / row_count if row_count else 0
    null_ratio = 1 - len(non_null_series) / row_count if row_count else 1

    if unique_count < 10 or unique_ratio < 0.8 or null_ratio > 0.1:
        return False

    normalized = re.sub(r"[^a-z0-9]+", " ", column_name.lower()).strip()
    if not re.search(r"\b(?:id|number|code|key)\b", normalized):
        return False

    sample = non_null_series.astype(str).head(50)
    if not sample.str.fullmatch(r"[A-Za-z0-9_-]+").all():
        return False

    return True


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
