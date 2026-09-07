"""Orchestration for the isolated NLP-to-AutoML decision pipeline."""

from typing import Any

import pandas as pd

from .clarification import analyze_intent_confidence
from .intent_detection import detect_user_intent
from .target_extraction import extract_target_reference
from .target_matching import match_target_column
from .task_mapping import map_intent_to_task


def process_nlp_request(user_text: str, df: pd.DataFrame) -> dict[str, Any]:
    """Run intent, confidence, task, target extraction, and target matching."""
    intent_result = detect_user_intent(user_text)
    intent_analysis = analyze_intent_confidence(intent_result)
    intent = intent_result["intent"]
    task_result = map_intent_to_task(intent)
    target_extraction = extract_target_reference(user_text)

    target_result: dict[str, Any] = {
        "target_reference": target_extraction["target_reference"],
        "target_found": target_extraction["target_found"],
        "matched_column": None,
        "confidence": 0.0,
        "candidates": [],
        "needs_clarification": target_extraction["needs_clarification"],
    }
    if target_extraction["target_found"]:
        target_result = match_target_column(target_extraction["target_reference"], df)
        target_result["target_found"] = True

    supervised_target_required = task_result["problem_type"] in {
        "classification",
        "regression",
    }
    target_needs_clarification = (
        supervised_target_required
        and (
            not target_result["target_found"]
            or target_result["matched_column"] is None
            or target_result["needs_clarification"]
        )
    )
    needs_clarification = bool(
        intent_analysis["needs_clarification"]
        or task_result["clarification_required"]
        or target_needs_clarification
    )

    return {
        "user_text": user_text,
        "intent": intent_result,
        "intent_analysis": intent_analysis,
        "task": {
            "problem_type": task_result["problem_type"],
            "clarification_required": task_result["clarification_required"],
        },
        "target": target_result,
        "needs_clarification": needs_clarification,
    }
