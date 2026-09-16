"""Lightweight persistent history and comparison for AutoML training runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HISTORY_PATH = (
    Path(__file__).resolve().parents[2] / "storage" / "reports" / "training_history.json"
).resolve()


def _dataset_key(df: pd.DataFrame, dataset_name: str | None) -> str:
    if dataset_name and dataset_name.strip():
        return dataset_name.strip()
    payload = df.to_json(orient="split", date_format="iso")
    return f"dataframe:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _metric_direction(selection_metric: str) -> str:
    return "lower" if selection_metric.lower() == "rmse" else "higher"


def _comparison(previous: dict[str, Any] | None, metric: str, value: float | None) -> dict[str, Any]:
    if previous is None or value is None or previous.get("metric_value") is None:
        return {
            "status": "baseline",
            "previous_run_id": None,
            "metric": metric,
            "metric_delta": None,
        }

    delta = float(value) - float(previous["metric_value"])
    direction = _metric_direction(metric)
    if delta == 0:
        status = "unchanged"
    elif (direction == "higher" and delta > 0) or (direction == "lower" and delta < 0):
        status = "improved"
    else:
        status = "worsened"
    return {
        "status": status,
        "previous_run_id": previous.get("run_id"),
        "metric": metric,
        "metric_delta": delta,
    }


def record_training_run(
    df: pd.DataFrame,
    training_output: dict[str, Any],
    *,
    dataset_name: str | None = None,
    history_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Persist a successful training run and compare it with its previous sibling."""
    model = training_output.get("model")
    if not isinstance(model, dict) or not model.get("model_id"):
        return None, None

    best_model = training_output.get("best_model") or {}
    selection_metric = str(best_model.get("selection_metric") or "")
    metrics = best_model.get("metrics") or {}
    metric_value = metrics.get(selection_metric) if isinstance(metrics, dict) else None
    target_column = training_output.get("target_column")
    problem_type = str(training_output.get("problem_type") or "")
    dataset_key = _dataset_key(df, dataset_name)
    run_id = str(model["model_id"])
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    target_path = Path(history_path) if history_path is not None else HISTORY_PATH
    target_path = target_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = json.loads(target_path.read_text(encoding="utf-8")) if target_path.exists() else []
    except (OSError, json.JSONDecodeError):
        history = []
    if not isinstance(history, list):
        history = []

    previous = next(
        (
            item for item in reversed(history)
            if isinstance(item, dict)
            and item.get("dataset") == dataset_key
            and item.get("target") == target_column
            and item.get("problem_type") == problem_type
        ),
        None,
    )
    comparison = _comparison(previous, selection_metric, metric_value)
    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset": dataset_key,
        "target": target_column,
        "problem_type": problem_type,
        "best_model": best_model.get("name"),
        "selection_metric": selection_metric,
        "metric_value": metric_value,
        "training_time_seconds": training_output.get("training_time_seconds"),
    }
    history.append(record)
    target_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return record, comparison
