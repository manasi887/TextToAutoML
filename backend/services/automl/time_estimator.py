"""Deterministic training-time estimates for the baseline AutoML workflow."""

from __future__ import annotations

from numbers import Real
from typing import Any


_PROBLEM_TYPE_MULTIPLIERS = {
    "binary classification": 1.0,
    "classification": 1.0,
    "multi-class classification": 1.2,
    "multi class classification": 1.2,
    "regression": 1.0,
    "clustering": 0.9,
}


def _validate_non_negative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative number.")
    converted = float(value)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative number.")
    return converted


def estimate_training_time(
    rows: int,
    features: int,
    problem_type: str,
    model_count: int,
) -> dict[str, float]:
    """Estimate training duration using deterministic dataset-size heuristics.

    The estimate uses one second of fixed setup time, a small cost based on
    rows multiplied by features, and two seconds per candidate model. The
    problem type applies a fixed multiplier: multi-class classification is
    slightly more expensive and clustering slightly less expensive than the
    baseline. The returned range is 80% to 125% of the central estimate.
    """
    row_count = _validate_non_negative(rows, "rows")
    feature_count = _validate_non_negative(features, "features")
    candidate_count = _validate_non_negative(model_count, "model_count")
    if not isinstance(problem_type, str) or not problem_type.strip():
        raise ValueError("problem_type must be a non-empty string.")

    normalized_type = problem_type.strip().lower()
    multiplier = _PROBLEM_TYPE_MULTIPLIERS.get(normalized_type, 1.0)
    workload_seconds = (row_count * feature_count) / 1_000_000
    estimated_seconds = (1.0 + workload_seconds + (2.0 * candidate_count)) * multiplier
    estimated_seconds = round(estimated_seconds, 2)

    return {
        "estimated_seconds": estimated_seconds,
        "min_seconds": round(estimated_seconds * 0.8, 2),
        "max_seconds": round(estimated_seconds * 1.25, 2),
    }
