"""Canonical, JSON-safe dataset meta-feature extraction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from services.automl.problem_detection import detect_problem_type
from services.dataset.intelligence import (
    analyze_features,
    analyze_target,
    detect_high_cardinality,
    detect_identifier_columns,
    summarize_dataset,
)


SCHEMA_VERSION = "1.0"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float, replacing invalid values with a safe default."""
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return converted if math.isfinite(converted) else default


def _safe_int(value: Any, default: int = 0) -> int:
    """Return a JSON-safe integer."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _safe_float(numerator / denominator)


def _target_defaults() -> dict[str, Any]:
    return {
        "target_exists": False,
        "target_dtype": None,
        "target_unique_count": None,
        "target_unique_ratio": None,
        "target_missing_ratio": None,
        "target_is_numeric": None,
        "target_is_categorical": None,
        "target_is_binary": None,
        "target_class_count": None,
        "target_majority_class_ratio": None,
        "target_entropy": None,
        "target_skewness": None,
        "target_problem_type": None,
    }


def _target_features(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Extract target descriptors without exposing target values or names."""
    target_series = df[target_column]
    target_report = analyze_target(df, target_column)
    non_null = target_series.dropna()
    unique_count = _safe_int(target_report.get("unique_value_count"))
    row_count = len(df)
    counts = non_null.value_counts(normalize=True)

    target_is_numeric = bool(pd.api.types.is_numeric_dtype(target_series))
    target_is_categorical = bool(
        isinstance(target_series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(target_series)
        or pd.api.types.is_string_dtype(target_series)
    )
    target_problem_type = detect_problem_type(df, target_column).get("problem_type")

    result = _target_defaults()
    result.update(
        {
            "target_exists": True,
            "target_dtype": str(target_series.dtype),
            "target_unique_count": unique_count,
            "target_unique_ratio": _ratio(unique_count, len(non_null)),
            "target_missing_ratio": _ratio(int(target_series.isna().sum()), row_count),
            "target_is_numeric": target_is_numeric,
            "target_is_categorical": target_is_categorical,
            "target_is_binary": unique_count == 2,
            "target_class_count": unique_count if "Classification" in str(target_problem_type) else None,
            "target_majority_class_ratio": _safe_float(counts.iloc[0]) if "Classification" in str(target_problem_type) and not counts.empty else None,
            "target_problem_type": target_problem_type,
        }
    )

    if target_is_numeric and len(non_null) > 0:
        result["target_skewness"] = _safe_float(non_null.skew(), default=0.0)

    if "Classification" in str(target_problem_type) and not counts.empty:
        probabilities = counts.to_numpy(dtype=float)
        result["target_entropy"] = _safe_float(-(probabilities * np.log2(probabilities)).sum())

    return result


def extract_meta_features(
    df: pd.DataFrame,
    target_column: str | None = None,
    problem_type: str | None = None,
) -> dict[str, Any]:
    """Return deterministic, JSON-safe meta-features for a tabular dataset."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    summary = summarize_dataset(df)
    target_is_valid = isinstance(target_column, str) and target_column in df.columns
    feature_df = df.drop(columns=[target_column], errors="ignore") if target_is_valid else df
    feature_analysis = analyze_features(feature_df)
    identifier_report = detect_identifier_columns(feature_df)
    cardinality_report = detect_high_cardinality(feature_df)

    row_count = _safe_int(summary.get("rows"))
    column_count = _safe_int(summary.get("columns"))
    feature_count = len(feature_df.columns)
    total_cells = row_count * column_count
    missing_count = _safe_int(summary.get("missing_values"))
    columns_with_missing = int(feature_df.isna().any().sum())
    max_missing_ratio = max(
        (_safe_float(feature_df[column].isna().mean()) for column in feature_df.columns),
        default=0.0,
    )
    duplicate_count = _safe_int(df.duplicated().sum())

    numeric_unique_ratios = [
        _ratio(int(feature_df[column].dropna().nunique()), len(feature_df[column].dropna()))
        for column in feature_analysis["numeric_columns"]
    ]
    categorical_cardinalities = [
        _safe_int(feature_df[column].dropna().nunique())
        for column in feature_analysis["categorical_columns"]
    ]
    numeric_count = _safe_int(feature_analysis["numeric_column_count"])
    categorical_count = _safe_int(feature_analysis["categorical_column_count"])
    datetime_count = sum(
        pd.api.types.is_datetime64_any_dtype(feature_df[column])
        for column in feature_df.columns
    )
    boolean_count = sum(pd.api.types.is_bool_dtype(feature_df[column]) for column in feature_df.columns)
    safe_problem_type = problem_type
    if safe_problem_type is None and target_is_valid:
        safe_problem_type = detect_problem_type(df, target_column).get("problem_type")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "problem_type": safe_problem_type,
        "row_count": row_count,
        "column_count": column_count,
        "feature_count": feature_count,
        "rows_per_feature": _safe_float(row_count / feature_count) if feature_count else 0.0,
        "numeric_column_count": numeric_count,
        "categorical_column_count": categorical_count,
        "datetime_column_count": datetime_count,
        "boolean_column_count": boolean_count,
        "numeric_ratio": _ratio(numeric_count, feature_count),
        "categorical_ratio": _ratio(categorical_count, feature_count),
        "datetime_ratio": _ratio(datetime_count, feature_count),
        "boolean_ratio": _ratio(boolean_count, feature_count),
        "missing_value_count": missing_count,
        "missing_value_ratio": _ratio(missing_count, total_cells),
        "columns_with_missing_count": columns_with_missing,
        "max_column_missing_ratio": max_missing_ratio,
        "duplicate_row_count": duplicate_count,
        "duplicate_row_ratio": _ratio(duplicate_count, row_count),
        "constant_column_count": _safe_int(feature_analysis["constant_column_count"]),
        "identifier_column_count": _safe_int(identifier_report["count"]),
        "high_cardinality_column_count": _safe_int(cardinality_report["count"]),
        "mean_numeric_unique_ratio": _safe_float(sum(numeric_unique_ratios) / len(numeric_unique_ratios)) if numeric_unique_ratios else 0.0,
        "max_numeric_unique_ratio": max(numeric_unique_ratios, default=0.0),
        "mean_categorical_cardinality": _safe_float(sum(categorical_cardinalities) / len(categorical_cardinalities)) if categorical_cardinalities else 0.0,
        "max_categorical_cardinality": max(categorical_cardinalities, default=0),
        "high_cardinality_ratio": _ratio(_safe_int(cardinality_report["count"]), categorical_count),
        "requires_imputation": columns_with_missing > 0,
        "requires_categorical_encoding": categorical_count > 0,
        "requires_datetime_features": datetime_count > 0,
        "recommended_preprocessing_count": int(
            (columns_with_missing > 0)
            + (categorical_count > 0)
            + (datetime_count > 0)
            + (_safe_int(feature_analysis["constant_column_count"]) > 0)
            + (_safe_int(cardinality_report["count"]) > 0)
        ),
    }
    result.update(_target_features(df, target_column) if target_is_valid else _target_defaults())
    return result
