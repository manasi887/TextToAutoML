"""Integrate confirmed NLP dataset resolution with the AutoML pipeline."""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd

from services.automl.pipeline import run_automl_pipeline
from services.automl.run_history import record_training_run


def _json_safe(value: Any) -> Any:
    """Convert model/report values to JSON-safe built-in values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def integrate_nlp_with_automl(
    df: pd.DataFrame, nlp_result: dict[str, Any], dataset_name: str | None = None
) -> dict[str, Any]:
    """Run AutoML only when NLP resolution confirms training readiness."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(nlp_result, dict):
        raise TypeError("nlp_result must be a dictionary")

    resolution = nlp_result.get("dataset_resolution")
    if not isinstance(resolution, dict):
        raise ValueError("nlp_result must contain a dataset_resolution dictionary")

    safe_resolution = _json_safe(resolution)
    if not bool(nlp_result.get("ready_for_training", False)):
        return {
            "nlp_resolution": safe_resolution,
            "automl_training": None,
            "ready_for_training": False,
            "needs_clarification": True,
            "reason": "NLP resolution is not ready for training.",
        }

    target_column = resolution.get("target_column")
    problem_type = resolution.get("problem_type")
    if not isinstance(problem_type, str) or not problem_type.strip():
        raise ValueError("A resolved problem_type is required for AutoML training")
    if problem_type.strip().lower() != "clustering" and (not isinstance(target_column, str) or not target_column.strip()):
        raise ValueError("A resolved target_column is required for AutoML training")

    training_started_at = time.perf_counter()
    training_output = run_automl_pipeline(df, target_column, problem_type)
    training_time_seconds = round(time.perf_counter() - training_started_at, 6)
    training_output = {
        **training_output,
        "training_time_seconds": training_time_seconds,
    }
    run_record, comparison = record_training_run(
        df,
        training_output,
        dataset_name=dataset_name,
    )
    if run_record is not None:
        training_output["run"] = run_record
        training_output["comparison"] = comparison
    return {
        "nlp_resolution": safe_resolution,
        "automl_training": _json_safe(training_output),
        "ready_for_training": True,
        "needs_clarification": False,
        "reason": "AutoML training completed from the confirmed NLP resolution.",
    }
