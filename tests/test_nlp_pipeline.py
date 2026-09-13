import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.nlp.pipeline import process_nlp_request


def _nlp_result(*, needs_clarification, problem_type, target_column):
    return {
        "intent": "classification",
        "confidence": 0.9,
        "probabilities": {"classification": 0.9, "regression": 0.05, "clustering": 0.05},
    }, {
        "needs_clarification": needs_clarification,
        "problem_type": problem_type,
        "target_column": target_column,
        "confidence": 0.9,
        "candidates": [],
        "reason": "test",
    }


def test_estimates_time_after_successful_dataset_resolution():
    df = pd.DataFrame({"target": [0, 1, 0], "feature_a": [1, 2, 3], "feature_b": [4, 5, 6]})
    intent_result, resolution = _nlp_result(
        needs_clarification=False,
        problem_type="Binary Classification",
        target_column="target",
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result), patch(
        "services.nlp.pipeline.analyze_intent_confidence",
        return_value={"needs_clarification": False},
    ), patch(
        "services.nlp.pipeline.map_intent_to_task",
        return_value={"problem_type": "classification", "clarification_required": False},
    ), patch(
        "services.nlp.pipeline.extract_target_reference",
        return_value={"target_reference": "target", "target_found": True, "needs_clarification": False},
    ), patch(
        "services.nlp.pipeline.match_target_column",
        return_value={"matched_column": "target", "confidence": 0.9, "needs_clarification": False, "candidates": []},
    ), patch(
        "services.nlp.pipeline.resolve_dataset_context", return_value=resolution,
    ), patch(
        "services.nlp.pipeline.estimate_training_time",
        return_value={"estimated_seconds": 3.0, "min_seconds": 2.4, "max_seconds": 3.75},
    ) as estimate_mock:
        result = process_nlp_request("Predict target", df)

    estimate_mock.assert_called_once_with(
        rows=3,
        features=2,
        problem_type="Binary Classification",
        model_count=3,
    )
    assert result["training_time_estimate"]["estimated_seconds"] == 3.0


def test_does_not_estimate_when_clarification_is_required():
    df = pd.DataFrame({"target": [0, 1], "feature": [1, 2]})
    intent_result, resolution = _nlp_result(
        needs_clarification=True,
        problem_type="Clustering",
        target_column=None,
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result), patch(
        "services.nlp.pipeline.analyze_intent_confidence",
        return_value={"needs_clarification": True},
    ), patch(
        "services.nlp.pipeline.map_intent_to_task",
        return_value={"problem_type": None, "clarification_required": True},
    ), patch(
        "services.nlp.pipeline.extract_target_reference",
        return_value={"target_reference": None, "target_found": False, "needs_clarification": True},
    ), patch(
        "services.nlp.pipeline.resolve_dataset_context", return_value=resolution,
    ), patch("services.nlp.pipeline.estimate_training_time") as estimate_mock:
        result = process_nlp_request("Analyze this dataset", df)

    estimate_mock.assert_not_called()
    assert result["ready_for_training"] is False
    assert result["training_time_estimate"] is None