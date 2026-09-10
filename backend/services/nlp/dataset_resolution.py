"""Resolve NLP intent and dataset-derived target context."""

from __future__ import annotations

from typing import Any

import pandas as pd

from services.automl.problem_detection import (
    detect_problem_type,
    detect_target_candidates,
)


_CLEAR_TARGET_MARGIN = 2.0
_STRONG_TARGET_CONFIDENCE = 0.80


def _json_safe(value: Any) -> Any:
    """Convert detector output to JSON-safe built-in values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _nlp_problem_type(nlp_result: dict[str, Any]) -> str | None:
    task = nlp_result.get("task")
    if isinstance(task, dict) and task.get("problem_type"):
        return str(task["problem_type"]).strip().lower()

    intent = nlp_result.get("intent")
    if isinstance(intent, dict):
        intent = intent.get("intent")
    if isinstance(intent, str):
        return intent.strip().lower()
    return None


def _compatible(nlp_type: str | None, dataset_type: str | None) -> bool:
    if not nlp_type or not dataset_type:
        return False
    normalized_dataset_type = dataset_type.strip().lower()
    if nlp_type == "classification":
        return "classification" in normalized_dataset_type
    if nlp_type == "regression":
        return normalized_dataset_type == "regression"
    if nlp_type == "clustering":
        return normalized_dataset_type == "clustering"
    return False


def _candidate_is_clear(target_report: dict[str, Any]) -> bool:
    candidates = target_report.get("target_candidates", [])
    if not candidates or target_report.get("requires_user_confirmation", False):
        return False
    if len(candidates) == 1:
        return True
    top_score = float(candidates[0].get("score", 0.0))
    second_score = float(candidates[1].get("score", 0.0))
    return top_score - second_score >= _CLEAR_TARGET_MARGIN


def _result(
    target_column: str | None,
    problem_type: str,
    confidence: float,
    candidates: list[Any],
    needs_clarification: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "target_column": target_column,
        "problem_type": problem_type,
        "confidence": float(confidence),
        "candidates": _json_safe(candidates),
        "needs_clarification": bool(needs_clarification),
        "reason": reason,
    }


def resolve_dataset_context(
    nlp_result: dict[str, Any], df: pd.DataFrame
) -> dict[str, Any]:
    """Resolve a safe target and data-grounded problem type for an NLP request."""
    if not isinstance(nlp_result, dict):
        raise TypeError("nlp_result must be a dictionary")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    nlp_type = _nlp_problem_type(nlp_result)
    target_result = nlp_result.get("target", {})
    if not isinstance(target_result, dict):
        target_result = {}
    intent_analysis = nlp_result.get("intent_analysis", {})
    intent_needs_clarification = bool(
        isinstance(intent_analysis, dict)
        and intent_analysis.get("needs_clarification", False)
    )
    target_column = target_result.get("matched_column")
    target_is_clear = (
        isinstance(target_column, str)
        and target_column in df.columns
        and not target_result.get("needs_clarification", False)
    )

    target_report = _json_safe(detect_target_candidates(df))
    candidates = target_report.get("target_candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    if nlp_type == "clustering" and not intent_needs_clarification and not target_is_clear:
        return _result(
            None,
            "Clustering",
            0.0,
            candidates,
            False,
            "Clustering does not require a target column.",
        )

    if target_is_clear:
        problem_report = detect_problem_type(df, target_column)
        dataset_type = str(problem_report.get("problem_type", "Clustering"))
        if intent_needs_clarification or nlp_type in {None, "unknown"}:
            return _result(
                None,
                dataset_type,
                0.0,
                candidates,
                True,
                "The NLP result still requires clarification.",
            )
        target_confidence = float(
            target_result.get("confidence", 0.0)
        )
        if not _compatible(nlp_type, dataset_type) and target_confidence < _STRONG_TARGET_CONFIDENCE:
            return _result(
                None,
                dataset_type,
                0.0,
                candidates,
                True,
                "The NLP intent conflicts with the data-grounded problem type.",
            )
        return _result(
            target_column,
            dataset_type,
            float(problem_report.get("confidence", target_confidence)),
            candidates,
            False,
            (
                "Used the clear NLP-matched target and verified its data-grounded problem type."
                if _compatible(nlp_type, dataset_type)
                else "Used the strong explicit target and preferred its data-grounded problem type."
            ),
        )

    if intent_needs_clarification or nlp_type in {None, "unknown"}:
        return _result(
            None,
            "Clustering",
            0.0,
            candidates,
            True,
            "The NLP result requires clarification before selecting a target.",
        )

    if not candidates:
        return _result(
            None,
            "Clustering",
            0.0,
            candidates,
            True,
            "No dataset target candidate was detected.",
        )

    if not _candidate_is_clear(target_report):
        return _result(
            None,
            "Clustering",
            0.0,
            candidates,
            True,
            "Dataset target candidates are ambiguous and need confirmation.",
        )

    candidate = candidates[0]
    candidate_column = candidate.get("column")
    if not isinstance(candidate_column, str):
        return _result(
            None,
            "Clustering",
            0.0,
            candidates,
            True,
            "The dominant dataset target candidate conflicts with the NLP intent.",
        )

    problem_report = detect_problem_type(df, candidate_column)
    dataset_type = str(problem_report.get("problem_type", "Clustering"))
    if not _compatible(nlp_type, dataset_type):
        return _result(
            None,
            dataset_type,
            0.0,
            candidates,
            True,
            "The selected candidate conflicts with the final data-grounded problem type.",
        )
    return _result(
        candidate_column,
        dataset_type,
        float(problem_report.get("confidence", candidate.get("confidence", 0.0))),
        candidates,
        False,
        "Auto-selected the clearly dominant dataset target compatible with the NLP intent.",
    )
