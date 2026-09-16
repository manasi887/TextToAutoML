"""Orchestration for the isolated NLP-to-AutoML decision pipeline."""

from typing import Any

import pandas as pd

from .clarification import analyze_intent_confidence
from .dataset_resolution import resolve_dataset_context
from .intent_detection import detect_user_intent
from .target_extraction import extract_target_reference
from .target_matching import match_target_column
from .task_mapping import map_intent_to_task
from services.automl.time_estimator import estimate_training_time


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
        target_result["target_found"] = target_result["matched_column"] is not None

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

    dataset_resolution = resolve_dataset_context(
        {
            "intent": intent_result,
            "task": {
                "problem_type": task_result["problem_type"],
                "clarification_required": task_result["clarification_required"],
            },
            "target": target_result,
            "needs_clarification": needs_clarification,
        },
        df,
    )
    needs_clarification = bool(dataset_resolution["needs_clarification"])
    ready_for_training = bool(
        not needs_clarification
        and (
            dataset_resolution["problem_type"] == "Clustering"
            or dataset_resolution["target_column"] is not None
        )
    )
    training_time_estimate = None
    if ready_for_training:
        target_column = dataset_resolution.get("target_column")
        feature_frame = df.drop(columns=[target_column], errors="ignore")
        training_time_estimate = estimate_training_time(
            rows=len(df),
            features=len(feature_frame.columns),
            problem_type=dataset_resolution["problem_type"],
            model_count=3,
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
        "dataset_resolution": dataset_resolution,
        "needs_clarification": needs_clarification,
        "ready_for_training": ready_for_training,
        "training_time_estimate": training_time_estimate,
    }
