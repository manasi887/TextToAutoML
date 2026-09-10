"""Build a compact, JSON-safe report from NLP and AutoML results."""

from __future__ import annotations

import math
from typing import Any


_DEFAULT_PROBLEM_TYPE = "Unknown"


def _json_safe(value: Any) -> Any:
    """Convert common scalar and container values to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _non_negative_int(value: Any) -> int:
    """Return a safe non-negative integer, defaulting invalid values to zero."""
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(converted, 0)


def _list_value(value: Any) -> list[Any]:
    """Return list-like values as JSON-safe lists."""
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return []


def _summary(
    problem_type: str,
    target_column: Any,
    trained_count: int,
    best_name: str | None,
    selection_metric: Any,
) -> str:
    """Create a deterministic task-specific training summary."""
    target_text = str(target_column) if target_column else "unspecified target"
    model_text = best_name or "no best model"
    metric_text = str(selection_metric) if selection_metric else "task metric"
    normalized = problem_type.lower()
    if "classification" in normalized:
        return (
            f"Classification training for target '{target_text}' completed with "
            f"{trained_count} trained model(s); best model: {model_text} "
            f"selected by {metric_text}."
        )
    if "regression" in normalized:
        return (
            f"Regression training for target '{target_text}' completed with "
            f"{trained_count} trained model(s); best model: {model_text} "
            f"selected by {metric_text}."
        )
    return (
        f"{problem_type} training for target '{target_text}' completed with "
        f"{trained_count} trained model(s); best model: {model_text} "
        f"selected by {metric_text}."
    )


def generate_training_report(
    nlp_resolution: dict[str, Any], automl_training: dict[str, Any]
) -> dict[str, Any]:
    """Generate a deterministic JSON-safe report from training outputs."""
    if not isinstance(nlp_resolution, dict):
        raise TypeError("nlp_resolution must be a dictionary")
    if not isinstance(automl_training, dict):
        raise TypeError("automl_training must be a dictionary")

    target_column = nlp_resolution.get("target_column")
    problem_type = nlp_resolution.get("problem_type") or automl_training.get(
        "problem_type", _DEFAULT_PROBLEM_TYPE
    )
    problem_type = str(problem_type)
    if not problem_type.strip():
        problem_type = _DEFAULT_PROBLEM_TYPE

    dataset = automl_training.get("data")
    if not isinstance(dataset, dict):
        dataset = {}
    preprocessing = automl_training.get("preprocessing")
    if not isinstance(preprocessing, dict):
        preprocessing = {}
    models = automl_training.get("models")
    if not isinstance(models, dict):
        models = {}
    best_model = automl_training.get("best_model")
    if not isinstance(best_model, dict):
        best_model = {}
    model = automl_training.get("model")
    if not isinstance(model, dict):
        model = {}

    trained = _list_value(models.get("trained"))
    failed = _list_value(models.get("failed"))
    raw_feature_names = _list_value(preprocessing.get("raw_feature_names"))
    best_name = best_model.get("name")
    selection_metric = best_model.get("selection_metric")
    if best_name is not None:
        best_name = str(best_name)

    report = {
        "summary": _summary(
            problem_type,
            target_column,
            len(trained),
            best_name,
            selection_metric,
        ),
        "model_id": _json_safe(model.get("model_id")),
        "task": {
            "target": _json_safe(target_column),
            "problem_type": problem_type,
        },
        "dataset": {
            "rows": _non_negative_int(dataset.get("original_rows")),
            "features": len(raw_feature_names),
            "training_rows": _non_negative_int(dataset.get("training_rows")),
            "test_rows": _non_negative_int(dataset.get("test_rows")),
        },
        "models": {
            "trained": len(trained),
            "failed": len(failed),
        },
        "best_model": {
            "name": _json_safe(best_name),
            "selection_metric": _json_safe(best_model.get("selection_metric")),
            "metrics": _json_safe(best_model.get("metrics", {}))
            if isinstance(best_model.get("metrics", {}), dict)
            else {},
        },
        "warnings": {
            "failed_model_count": len(failed),
        },
    }
    return _json_safe(report)
