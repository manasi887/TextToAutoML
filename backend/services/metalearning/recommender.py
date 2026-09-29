"""Explainable recommendations from historical dataset meta-features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from services.automl.run_history import HISTORY_PATH


_DEFAULT_MIN_HISTORY = 2
_DEFAULT_TOP_K = 3
_DEFAULT_MIN_CONFIDENCE = 0.60
OPENML_SIMILARITY_WEIGHT_CAP = 0.25
_EXCLUDED_FEATURES = {"schema_version", "problem_type"}
_SUPPORTED_CLASSIFICATION_MODELS = {
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
}
_SUPPORTED_REGRESSION_MODELS = {
    "LinearRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
}


def load_historical_meta_records(
    history_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load dictionary records from the existing JSON run history."""
    path = Path(history_path) if history_path is not None else HISTORY_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []
    return [record for record in payload if isinstance(record, dict)]


def _normalized_problem_type(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if "classification" in text:
        return "classification"
    if "regression" in text:
        return "regression"
    if "clustering" in text:
        return "clustering"
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _historical_outcome(record: dict[str, Any]) -> tuple[str, float] | None:
    """Return only successful best-model outcomes from one history record."""
    model = record.get("best_model")
    metric = _finite_number(record.get("metric_value"))
    if not isinstance(model, str) or not model.strip() or metric is None:
        return None
    return model.strip(), metric


def _numeric_features(features: Any) -> dict[str, float]:
    if not isinstance(features, dict):
        return {}
    return {
        str(key): value
        for key, raw_value in features.items()
        if str(key) not in _EXCLUDED_FEATURES
        and (value := _finite_number(raw_value)) is not None
    }


def _feature_ranges(records: list[dict[str, Any]], current: dict[str, float]) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = {key: [value] for key, value in current.items()}
    for record in records:
        for key, value in _numeric_features(record.get("meta_features")).items():
            values.setdefault(key, []).append(value)
    return {key: (min(items), max(items)) for key, items in values.items()}


def _normalized_distance(
    current: dict[str, float], historical: dict[str, float], ranges: dict[str, tuple[float, float]]
) -> tuple[float, int]:
    distances: list[float] = []
    for key in sorted(set(current) & set(historical)):
        low, high = ranges[key]
        span = high - low
        if span == 0:
            current_value = historical_value = 0.0
        else:
            current_value = min(1.0, max(0.0, (current[key] - low) / span))
            historical_value = min(1.0, max(0.0, (historical[key] - low) / span))
        distances.append((current_value - historical_value) ** 2)
    if not distances:
        return float("inf"), 0
    return math.sqrt(sum(distances) / len(distances)), len(distances)


def _similarity(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def _record_source(record: dict[str, Any]) -> str | None:
    source = record.get("source", "local")
    return source if source in {"local", "openml"} else None


def _openml_record_is_compatible(record: dict[str, Any], problem_type: str | None) -> bool:
    if _record_source(record) != "openml":
        return False
    normalized_problem = _normalized_problem_type(
        (record.get("meta_features") or {}).get("problem_type", record.get("problem_type"))
    )
    if normalized_problem != problem_type:
        return False
    model = record.get("best_model")
    if "classification" in str(problem_type):
        supported_models = _SUPPORTED_CLASSIFICATION_MODELS
        supported_metric = "f1_score"
    elif "regression" in str(problem_type):
        supported_models = _SUPPORTED_REGRESSION_MODELS
        supported_metric = "rmse"
    else:
        return False
    return (
        isinstance(model, str)
        and model in supported_models
        and record.get("selection_metric") == supported_metric
        and _finite_number(record.get("metric_value")) is not None
    )


def _openml_deduplication_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("dataset_id"),
        record.get("dataset_version"),
        record.get("task_id"),
        record.get("run_id"),
    )


def _valid_openml_records(
    records: Any,
    problem_type: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    valid: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        if not isinstance(record, dict) or not _openml_record_is_compatible(record, problem_type):
            continue
        key = _openml_deduplication_key(record)
        if key in seen:
            continue
        seen.add(key)
        valid.append(record)
    return valid


def recommend_models(
    meta_features: dict[str, Any],
    *,
    history_path: str | Path | None = None,
    openml_records: list[dict[str, Any]] | None = None,
    min_history: int = _DEFAULT_MIN_HISTORY,
    top_k: int = _DEFAULT_TOP_K,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Rank historical best-model families for a current meta-feature record."""
    if not isinstance(meta_features, dict):
        raise TypeError("meta_features must be a dictionary")

    current_problem_type = _normalized_problem_type(meta_features.get("problem_type"))
    current_numeric = _numeric_features(meta_features)
    local_records = load_historical_meta_records(history_path)
    compatible_local_records = [
        record
        for record in local_records
        if _record_source(record) == "local"
        and _normalized_problem_type(
            (record.get("meta_features") or {}).get("problem_type", record.get("problem_type"))
        )
        == current_problem_type
    ]
    compatible_openml_records = _valid_openml_records(openml_records, current_problem_type)
    compatible_records = compatible_local_records + compatible_openml_records
    all_records = local_records + compatible_openml_records
    
    # OpenML records are optional evidence; they never count toward local history minimums.
    if not compatible_local_records and not compatible_openml_records:
        compatible_records = []

    legacy_compatible_records = [
        record
        for record in local_records
        if _normalized_problem_type(
            (record.get("meta_features") or {}).get("problem_type", record.get("problem_type"))
        )
        == current_problem_type
    ]

    base_result = {
        "problem_type": current_problem_type,
        "history_count": len(all_records),
        "compatible_history_count": len(legacy_compatible_records),
        "local_history_count": len(local_records),
        "openml_record_count": len(compatible_openml_records),
        "feature_count": len(current_numeric),
        "recommendations": [],
        "similar_datasets": [],
        "confidence": 0.0,
    }
    if not local_records and not compatible_openml_records:
        return {**base_result, "status": "no_history", "reason": "No historical run history was found."}
    if current_problem_type is None or not compatible_records:
        return {**base_result, "status": "no_compatible_history", "reason": "No historical records match the current problem type."}
    if len(compatible_local_records) < max(1, int(min_history)):
        return {**base_result, "status": "insufficient_history", "reason": "Not enough compatible historical records are available."}
    if len(current_numeric) == 0:
        return {**base_result, "status": "insufficient_features", "reason": "The current record has no comparable numeric meta-features."}

    ranges = _feature_ranges(compatible_records, current_numeric)
    similar: list[dict[str, Any]] = []
    for record in compatible_records:
        outcome = _historical_outcome(record)
        if outcome is None:
            continue
        historical_numeric = _numeric_features(record.get("meta_features"))
        distance, compared_count = _normalized_distance(current_numeric, historical_numeric, ranges)
        if compared_count == 0:
            continue
        similarity = _similarity(distance)
        similar.append(
            {
                "dataset": record.get("dataset"),
                "run_id": record.get("run_id"),
                "model_family": outcome[0],
                "source": _record_source(record),
                "dataset_id": record.get("dataset_id"),
                "dataset_version": record.get("dataset_version"),
                "task_id": record.get("task_id"),
                "similarity": similarity,
                "compared_feature_count": compared_count,
                "metric_value": outcome[1],
            }
        )

    local_weight = sum(item["similarity"] for item in similar if item["source"] == "local")
    openml_weight = sum(item["similarity"] for item in similar if item["source"] == "openml")
    openml_scale = (
        min(1.0, OPENML_SIMILARITY_WEIGHT_CAP * local_weight / openml_weight)
        if openml_weight and local_weight
        else 0.0
    )
    for item in similar:
        raw_similarity = item["similarity"]
        weighted_similarity = raw_similarity if item["source"] == "local" else raw_similarity * openml_scale
        item["similarity"] = round(raw_similarity, 6)
        item["weighted_similarity"] = weighted_similarity

    similar.sort(key=lambda item: (-item["weighted_similarity"], str(item.get("run_id") or "")))
    if not similar:
        return {**base_result, "status": "insufficient_outcomes", "reason": "Compatible history has no successful model outcomes."}

    model_votes: dict[str, list[dict[str, Any]]] = {}
    total_weight = sum(item["weighted_similarity"] for item in similar)
    for item in similar:
        model_votes.setdefault(item["model_family"], []).append(item)

    recommendations: list[dict[str, Any]] = []
    for model_family, evidence in model_votes.items():
        weight = sum(item["weighted_similarity"] for item in evidence)
        support = weight / total_weight if total_weight else 0.0
        local_evidence_count = sum(item["source"] == "local" for item in evidence)
        openml_evidence_count = sum(item["source"] == "openml" for item in evidence)
        recommendations.append(
            {
                "model_family": model_family,
                "recommendation_score": round(support, 6),
                "average_similarity": round(weight / len(evidence), 6),
                "evidence_count": len(evidence),
                "local_evidence_count": local_evidence_count,
                "openml_evidence_count": openml_evidence_count,
                "evidence": evidence[:top_k],
            }
        )
    recommendations.sort(
        key=lambda item: (
            -item["recommendation_score"],
            -item["average_similarity"],
            str(item["model_family"]),
        )
    )
    recommendations = recommendations[: max(1, int(top_k))]
    confidence = recommendations[0]["recommendation_score"] if recommendations else 0.0
    status = "recommended" if confidence >= float(min_confidence) else "low_confidence"
    reason = (
        "Recommendations are supported by similarity-weighted historical outcomes."
        if status == "recommended"
        else "Historical evidence is available but no model family has sufficient support."
    )
    return {
        **base_result,
        "status": status,
        "reason": reason,
        "confidence": round(confidence, 6),
        "similar_datasets": similar[:top_k],
        "recommendations": recommendations,
    }
