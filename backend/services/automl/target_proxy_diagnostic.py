"""Training-partition diagnostics for exact target proxies and associations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


_FDR_ALPHA = 0.05
_MIN_EXPECTED_CELL_COUNT = 5.0
_MIN_GROUP_SAMPLE_COUNT = 2
_NO_UNIQUE_MODE = object()


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
    if rows_checked == 0 or feature_values.isna().any():
        return (
            False,
            rows_checked,
            int(feature_values.nunique(dropna=True)),
            int(target_values.nunique(dropna=True)),
        )

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


def _unique_mode(values: pd.Series) -> Any:
    """Return the unique mode, or a sentinel when the mode is tied or empty."""
    counts = values.value_counts(dropna=True, sort=False)
    if counts.empty or int((counts == counts.max()).sum()) != 1:
        return _NO_UNIQUE_MODE
    return counts.index[int(np.argmax(counts.to_numpy()))]


def _group_summaries(
    feature: pd.Series,
    target: pd.Series,
    groups: pd.Series,
    *,
    feature_is_numeric: bool,
    target_is_classification: bool,
) -> tuple[pd.Series, pd.Series]:
    """Reduce repeated rows to one unambiguous summary for each group.

    This avoids treating rows from one entity as independent. Inference still
    assumes groups are independent and does not model dependence between groups;
    aggregation also discards within-group variation.
    """
    values = pd.DataFrame({"feature": feature, "target": target, "group": groups})
    summaries: list[tuple[Any, Any]] = []
    for _, group_frame in values.groupby("group", sort=False, dropna=True):
        if target_is_classification:
            target_value = _unique_mode(group_frame["target"])
        else:
            numeric_target = pd.to_numeric(group_frame["target"], errors="coerce")
            target_value = float(numeric_target.mean()) if numeric_target.notna().any() else np.nan

        if feature_is_numeric:
            numeric_feature = pd.to_numeric(group_frame["feature"], errors="coerce")
            feature_value = float(numeric_feature.mean()) if numeric_feature.notna().any() else np.nan
        else:
            feature_value = _unique_mode(group_frame["feature"])

        if target_value is _NO_UNIQUE_MODE or feature_value is _NO_UNIQUE_MODE:
            continue
        summaries.append((feature_value, target_value))

    if not summaries:
        return pd.Series(dtype="object"), pd.Series(dtype="object")
    return pd.Series([pair[0] for pair in summaries]), pd.Series([pair[1] for pair in summaries])


def _association_statistic(
    feature: pd.Series,
    target: pd.Series,
    problem_type: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Calculate a task-appropriate association, effect size, and p-value.

    The boolean indicates a skipped test due to sparse categories or groups.
    """
    is_classification = "classification" in problem_type.lower()
    feature_is_numeric = (
        pd.api.types.is_numeric_dtype(feature.dtype)
        and not pd.api.types.is_bool_dtype(feature.dtype)
        and not pd.api.types.is_complex_dtype(feature.dtype)
    )

    if is_classification:
        valid = feature.notna() & target.notna()
        feature_values = feature.loc[valid].reset_index(drop=True)
        target_values = target.loc[valid].reset_index(drop=True)
    else:
        numeric_target = pd.to_numeric(target, errors="coerce")
        valid = feature.notna() & numeric_target.notna()
        feature_values = feature.loc[valid].reset_index(drop=True)
        target_values = numeric_target.loc[valid].reset_index(drop=True).astype(float)

    if feature_is_numeric:
        numeric_feature = pd.to_numeric(feature_values, errors="coerce").astype(float)
        finite_feature = np.isfinite(numeric_feature.to_numpy())
        feature_values = numeric_feature.iloc[np.flatnonzero(finite_feature)].reset_index(drop=True)
        target_values = target_values.iloc[np.flatnonzero(finite_feature)].reset_index(drop=True)
        if not is_classification:
            finite_target = np.isfinite(target_values.to_numpy(dtype=float))
            feature_values = feature_values.iloc[np.flatnonzero(finite_target)].reset_index(drop=True)
            target_values = target_values.iloc[np.flatnonzero(finite_target)].reset_index(drop=True)

    rows_used = len(target_values)
    if (
        rows_used < 3
        or feature_values.nunique(dropna=True) < 2
        or target_values.nunique(dropna=True) < 2
    ):
        return None, False

    try:
        if is_classification and feature_is_numeric:
            target_codes, target_levels = pd.factorize(target_values, sort=False)
            samples = [
                feature_values.iloc[np.flatnonzero(target_codes == code)].to_numpy(dtype=float)
                for code in range(len(target_levels))
            ]
            if any(len(sample) < _MIN_GROUP_SAMPLE_COUNT for sample in samples):
                return None, True
            statistic, p_value = stats.kruskal(*samples)
            group_count = len(samples)
            effect_size = max(0.0, (float(statistic) - group_count + 1) / (rows_used - group_count))
            method = "Kruskal-Wallis H test"
            effect_size_name = "epsilon_squared"
            details: dict[str, Any] = {"target_classes": group_count}
        elif is_classification:
            feature_codes, feature_levels = pd.factorize(feature_values, sort=False)
            target_codes, target_levels = pd.factorize(target_values, sort=False)
            if len(feature_levels) < 2 or len(target_levels) < 2:
                return None, False
            contingency = np.zeros((len(feature_levels), len(target_levels)), dtype=np.int64)
            np.add.at(contingency, (feature_codes, target_codes), 1)
            statistic, asymptotic_p, degrees, expected = stats.chi2_contingency(
                contingency,
                correction=False,
            )
            minimum_expected = float(expected.min())
            if minimum_expected < _MIN_EXPECTED_CELL_COUNT:
                if contingency.shape != (2, 2):
                    return None, True
                _, p_value = stats.fisher_exact(contingency, alternative="two-sided")
                method = "Fisher exact test"
            else:
                p_value = asymptotic_p
                method = "Pearson chi-square test"
            dimensions = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            effect_size = math.sqrt(float(statistic) / (rows_used * dimensions))
            effect_size_name = "cramers_v"
            details = {
                "feature_categories": len(feature_levels),
                "target_classes": len(target_levels),
                "degrees_of_freedom": int(degrees),
                "minimum_expected_cell_count": minimum_expected,
                "association_statistic": "pearson_chi_square",
            }
        elif feature_is_numeric:
            statistic, p_value = stats.spearmanr(
                feature_values.to_numpy(dtype=float),
                target_values.to_numpy(dtype=float),
            )
            effect_size = abs(float(statistic))
            effect_size_name = "absolute_spearman_rho"
            method = "Spearman rank correlation test"
            details = {"spearman_rho": float(statistic)}
        else:
            feature_codes, feature_levels = pd.factorize(feature_values, sort=False)
            samples = [
                target_values.iloc[np.flatnonzero(feature_codes == code)].to_numpy(dtype=float)
                for code in range(len(feature_levels))
            ]
            if len(samples) < 2:
                return None, False
            if any(len(sample) < _MIN_GROUP_SAMPLE_COUNT for sample in samples):
                return None, True
            statistic, p_value = stats.kruskal(*samples)
            group_count = len(samples)
            effect_size = max(0.0, (float(statistic) - group_count + 1) / (rows_used - group_count))
            effect_size_name = "epsilon_squared"
            method = "Kruskal-Wallis H test"
            details = {"feature_categories": group_count}
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return None, False

    if (
        not math.isfinite(float(p_value))
        or not math.isfinite(float(statistic))
        or not math.isfinite(float(effect_size))
    ):
        return None, False
    return {
        "method": method,
        "statistic": float(statistic),
        "effect_size_name": effect_size_name,
        "effect_size": float(min(1.0, effect_size)),
        "p_value": float(p_value),
        "rows_used": int(rows_used),
        **details,
    }, False


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    count = len(p_values)
    adjusted = [1.0] * count
    running_minimum = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running_minimum = min(running_minimum, p_values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, running_minimum)
    return adjusted


def diagnose_target_proxy_features(
    training_df: pd.DataFrame,
    target_column: str,
    group_column: str | None = None,
    problem_type: str | None = None,
) -> dict[str, Any]:
    """Report exact proxies and exploratory univariate associations.

    The caller provides only outer-training-partition data. Statistical tests
    use Benjamini-Hochberg false-discovery control at 5%, are not repeated in
    each CV fold, do not alter input data or feature selection, and do not prove
    leakage or prediction-time unavailability. Grouped tests use one summary
    per group, assume independent groups, and are not cluster-robust.
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
    statistical_results: list[dict[str, Any]] = []
    sparse_tests_skipped = 0
    normalized_problem_type = str(problem_type or "").strip().lower()
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

        if not (
            "classification" in normalized_problem_type
            or "regression" in normalized_problem_type
        ):
            continue

        feature_values = feature
        target_values = target
        rows_unit = "rows"
        if group_column is not None:
            groups = training_df[group_column]
            valid_groups = groups.notna()
            feature_values = feature.loc[valid_groups]
            target_values = target.loc[valid_groups]
            groups = groups.loc[valid_groups]
            if feature_values.empty:
                continue
            feature_is_numeric = (
                pd.api.types.is_numeric_dtype(feature_values.dtype)
                and not pd.api.types.is_bool_dtype(feature_values.dtype)
                and not pd.api.types.is_complex_dtype(feature_values.dtype)
            )
            try:
                feature_values, target_values = _group_summaries(
                    feature_values,
                    target_values,
                    groups,
                    feature_is_numeric=feature_is_numeric,
                    target_is_classification="classification" in normalized_problem_type,
                )
            except (TypeError, ValueError):
                continue
            rows_unit = "groups"

        result, was_skipped_for_sparse_categories = _association_statistic(
            feature_values,
            target_values,
            normalized_problem_type,
        )
        sparse_tests_skipped += int(was_skipped_for_sparse_categories)
        if result is not None:
            result["feature"] = str(feature_name)
            result["rows_unit"] = rows_unit
            if rows_unit == "groups":
                result["group_inference_note"] = (
                    "One summary per group; assumes independent groups and is not cluster-robust. "
                    "Within-group variation is discarded."
                )
            statistical_results.append(result)

    adjusted_p_values = _benjamini_hochberg(
        [result["p_value"] for result in statistical_results]
    )
    for result, q_value in zip(statistical_results, adjusted_p_values):
        result["q_value"] = q_value
        result["false_discovery_rate_alpha"] = _FDR_ALPHA
        if q_value <= _FDR_ALPHA:
            group_note = (
                " Group summaries assume independent groups and do not account for dependence "
                "between groups."
                if result["rows_unit"] == "groups"
                else ""
            )
            warnings.append({
                "feature": result["feature"],
                "detection_type": "statistical_association",
                "evidence": {
                    key: value
                    for key, value in result.items()
                    if key != "feature"
                },
                "message": (
                    "This feature has a statistically unusual univariate relationship with the target "
                    "in the outer training partition. This exploratory warning is not proof of leakage "
                    "or prediction-time unavailability and is not evaluated within each CV fold."
                    f"{group_note}"
                ),
            })

    return {
        "target_column": target_column,
        "scope": "exploratory_training_partition",
        "statistical_tests_run": len(statistical_results),
        "statistical_tests_skipped_for_sparse_categories": sparse_tests_skipped,
        "warnings": warnings,
    }