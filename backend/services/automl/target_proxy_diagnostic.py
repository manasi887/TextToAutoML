"""Training-partition diagnostics for exact target proxy features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_missing(value: Any) -> bool:
    """Return whether a scalar is a pandas-recognized missing value."""
    try:
        result = pd.isna(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _values_equal(left: Any, right: Any) -> bool:
    """Compare scalar values safely, treating two missing values as equal."""
    left_missing = _is_missing(left)
    right_missing = _is_missing(right)
    if left_missing or right_missing:
        return left_missing and right_missing

    try:
        result = left == right
        if isinstance(result, (np.ndarray, list, tuple)):
            return bool(np.asarray(result).all())
        return bool(result)
    except (TypeError, ValueError):
        return False


def _is_exact_target_copy(feature: pd.Series, target: pd.Series) -> tuple[bool, int, int]:
    """Return whether all row values match, plus compared and missing counts."""
    matching_values = 0
    aligned_missing_values = 0

    for feature_value, target_value in zip(feature, target):
        if not _values_equal(feature_value, target_value):
            return False, matching_values, aligned_missing_values
        if _is_missing(target_value):
            aligned_missing_values += 1
        else:
            matching_values += 1

    return True, matching_values, aligned_missing_values


def _has_bijective_mapping(
    feature: pd.Series,
    target: pd.Series,
) -> tuple[bool, int, int, int]:
    """Check whether observed non-missing values form an unambiguous bijection."""
    target_present = target.notna()
    feature_values = feature.loc[target_present]
    target_values = target.loc[target_present]
    rows_checked = len(target_values)

    # Missing candidate values leave the mapping ambiguous for those target rows.
    if rows_checked == 0 or feature_values.isna().any():
        return False, rows_checked, int(feature_values.nunique(dropna=True)), int(target_values.nunique(dropna=True))

    try:
        feature_codes, feature_uniques = pd.factorize(feature_values, sort=False)
        target_codes, target_uniques = pd.factorize(target_values, sort=False)
    except (TypeError, ValueError):
        return False, rows_checked, 0, 0

    feature_count = len(feature_uniques)
    target_count = len(target_uniques)
    if feature_count < 2 or target_count < 2:
        return False, rows_checked, feature_count, target_count

    feature_to_target: dict[int, int] = {}
    target_to_feature: dict[int, int] = {}
    for feature_code, target_code in zip(feature_codes, target_codes):
        prior_target = feature_to_target.setdefault(int(feature_code), int(target_code))
        prior_feature = target_to_feature.setdefault(int(target_code), int(feature_code))
        if prior_target != int(target_code) or prior_feature != int(feature_code):
            return False, rows_checked, feature_count, target_count

    return feature_count == target_count, rows_checked, feature_count, target_count


def diagnose_target_proxy_features(
    training_df: pd.DataFrame,
    target_column: str,
    group_column: str | None = None,
) -> dict[str, Any]:
    """Report exact target copies and observed one-to-one encodings.

    The caller must provide training-partition data only. This helper accepts no
    validation or test frames, does not fit a model, and never modifies features.
    Warnings identify relationships in the supplied rows; they do not prove the
    feature is unavailable at prediction time or should be removed.
    """
    if not isinstance(training_df, pd.DataFrame):
        raise TypeError("training_df must be a pandas DataFrame")
    if target_column not in training_df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the training DataFrame.")
    if group_column is not None and group_column not in training_df.columns:
        raise ValueError(f"Group column '{group_column}' is not present in the training DataFrame.")
    if group_column == target_column:
        raise ValueError("Group column must be different from the target column.")

    target = training_df[target_column]
    excluded_columns = {target_column}
    if group_column is not None:
        excluded_columns.add(group_column)

    warnings: list[dict[str, Any]] = []
    for feature_name in training_df.columns:
        if feature_name in excluded_columns:
            continue

        feature = training_df[feature_name]
        is_copy, matching_values, aligned_missing_values = _is_exact_target_copy(feature, target)
        if is_copy and target.notna().any():
            warnings.append({
                "feature": str(feature_name),
                "detection_type": "exact_target_copy",
                "evidence": {
                    "rows_checked": int(len(training_df)),
                    "matching_non_missing_rows": matching_values,
                    "aligned_missing_rows": aligned_missing_values,
                },
                "message": (
                    "This feature exactly matches the target in the supplied training rows. "
                    "Confirm its provenance and prediction-time availability."
                ),
            })
            continue

        is_bijective, rows_checked, feature_count, target_count = _has_bijective_mapping(
            feature,
            target,
        )
        if is_bijective:
            warnings.append({
                "feature": str(feature_name),
                "detection_type": "one_to_one_target_encoding",
                "evidence": {
                    "rows_checked": rows_checked,
                    "distinct_feature_values": feature_count,
                    "distinct_target_values": target_count,
                    "mapping_is_bijective": True,
                },
                "message": (
                    "Observed feature values map one-to-one to target values in the supplied "
                    "training rows. This does not establish feature provenance or availability."
                ),
            })

    return {
        "target_column": target_column,
        "warnings": warnings,
    }