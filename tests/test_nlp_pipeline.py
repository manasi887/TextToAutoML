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


def _mock_intent(intent, probabilities):
    return {
        "intent": intent,
        "confidence": probabilities[intent],
        "probabilities": probabilities,
    }


def _housing_dataframe():
    return pd.DataFrame(
        {
            "median_income": [3.1, 2.8, 4.2, 5.1],
            "housing_median_age": [20, 30, 40, 50],
            "median_house_value": [220000, 250000, 300000, 310000],
        }
    )


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


def test_house_prices_resolves_to_regression_target():
    df = pd.DataFrame(
        {
            "median_income": [3.1, 2.8, 4.2, 5.1],
            "median_house_value": [220000, 250000, 300000, 310000],
        }
    )
    intent_result = {
        "intent": "regression",
        "confidence": 0.9,
        "probabilities": {"classification": 0.05, "regression": 0.9, "clustering": 0.05},
    }

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result), patch(
        "services.nlp.pipeline.analyze_intent_confidence",
        return_value={"needs_clarification": False},
    ), patch(
        "services.nlp.pipeline.map_intent_to_task",
        return_value={"problem_type": "regression", "clarification_required": False},
    ), patch(
        "services.nlp.pipeline.estimate_training_time",
        return_value={"estimated_seconds": 3.0},
    ):
        result = process_nlp_request("Predict house prices", df)

    assert result["target"]["target_found"] is True
    assert result["target"]["matched_column"] == "median_house_value"
    assert result["target"]["needs_clarification"] is False
    assert result["dataset_resolution"]["target_column"] == "median_house_value"
    assert result["dataset_resolution"]["problem_type"] == "Regression"
    assert result["needs_clarification"] is False
    assert result["ready_for_training"] is True


def test_full_explicit_housing_prompt_resolves_regression_despite_clustering_prediction():
    dataframe = _housing_dataframe()
    task = (
        "This is a regression task. Predict the numerical value in the "
        "median_house_value column using the other columns in the dataset. "
        "Use median_house_value as the target column. Do not predict any other column."
    )
    intent_result = _mock_intent(
        "clustering",
        {
            "classification": 0.22872005,
            "clustering": 0.60402733,
            "regression": 0.10184219,
            "unknown": 0.06541049,
        },
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request(task, dataframe)

    assert result["intent"]["intent"] == "clustering"
    assert result["dataset_resolution"]["problem_type"] == "Regression"
    assert result["dataset_resolution"]["target_column"] == "median_house_value"
    assert result["needs_clarification"] is False
    assert result["ready_for_training"] is True


def test_explicit_missing_target_does_not_match_a_similar_existing_column():
    dataframe = pd.DataFrame(
        {
            "house_value": [220000, 250000, 300000, 180000],
            "median_income": [3.1, 2.8, 4.2, 3.0],
        }
    )
    intent_result = _mock_intent(
        "regression",
        {
            "classification": 0.05,
            "clustering": 0.05,
            "regression": 0.85,
            "unknown": 0.05,
        },
    )
    task = "This is a regression task. Use median_house_value as the target column."

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request(task, dataframe)

    assert result["target"]["matched_column"] is None
    assert result["dataset_resolution"]["target_column"] is None
    assert result["needs_clarification"] is True


def test_explicit_classification_request_resolves_with_clustering_prediction():
    dataframe = pd.DataFrame(
        {
            "age": [22, 37, 46, 58],
            "balance": [1200, 900, 700, 300],
            "outcome": ["approved", "declined", "approved", "declined"],
        }
    )
    intent_result = _mock_intent(
        "clustering",
        {
            "classification": 0.20,
            "clustering": 0.65,
            "regression": 0.10,
            "unknown": 0.05,
        },
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request(
            "This is a classification task. Classify records using outcome as "
            "the target column.",
            dataframe,
        )

    assert result["intent"]["intent"] == "clustering"
    assert result["dataset_resolution"]["problem_type"] == "Binary Classification"
    assert result["dataset_resolution"]["target_column"] == "outcome"
    assert result["needs_clarification"] is False


def test_explicit_binary_classification_with_income_target_overrides_clustering_prediction():
    dataframe = pd.DataFrame(
        {
            "age": [22, 37, 46, 58, 29, 41, 35, 53],
            "education": ["HS", "College", "HS", "College"] * 2,
            "income": ["<=50K", ">50K"] * 4,
        }
    )
    intent_result = _mock_intent(
        "clustering",
        {
            "classification": 0.20,
            "clustering": 0.65,
            "regression": 0.10,
            "unknown": 0.05,
        },
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request(
            "I want binary classification with income as the target.",
            dataframe,
        )

    assert result["intent"]["intent"] == "clustering"
    assert result["task"]["problem_type"] == "classification"
    assert result["target"]["target_reference"] == "income"
    assert result["target"]["matched_column"] == "income"
    assert result["dataset_resolution"]["problem_type"] == "Binary Classification"
    assert result["dataset_resolution"]["target_column"] == "income"
    assert result["needs_clarification"] is False
    assert result["ready_for_training"] is True


def test_explicit_binary_classification_with_missing_target_requests_clarification():
    dataframe = pd.DataFrame(
        {
            "age": [22, 37, 46, 58],
            "income": ["<=50K", ">50K", "<=50K", ">50K"],
        }
    )
    intent_result = _mock_intent(
        "clustering",
        {
            "classification": 0.20,
            "clustering": 0.65,
            "regression": 0.10,
            "unknown": 0.05,
        },
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request(
            "I want binary classification with salary_band as the target.",
            dataframe,
        )

    assert result["task"]["problem_type"] == "classification"
    assert result["target"]["matched_column"] is None
    assert result["dataset_resolution"]["target_column"] is None
    assert result["needs_clarification"] is True
    assert result["ready_for_training"] is False


def test_ambiguous_task_keeps_classifier_clarification_behavior():
    dataframe = _housing_dataframe()
    intent_result = _mock_intent(
        "unknown",
        {
            "classification": 0.15,
            "clustering": 0.10,
            "regression": 0.20,
            "unknown": 0.55,
        },
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request("Analyze this dataset.", dataframe)

    assert result["intent"]["intent"] == "unknown"
    assert result["intent_analysis"]["needs_clarification"] is True
    assert result["dataset_resolution"]["target_column"] is None
    assert result["needs_clarification"] is True
    assert result["ready_for_training"] is False


def test_indirect_task_wording_keeps_classifier_prediction():
    dataframe = _housing_dataframe()
    intent_result = _mock_intent(
        "clustering",
        {
            "classification": 0.20,
            "clustering": 0.65,
            "regression": 0.10,
            "unknown": 0.05,
        },
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request(
            "Find useful patterns in this housing dataset.", dataframe
        )

    assert result["intent"]["intent"] == "clustering"
    assert result["task"]["problem_type"] == "clustering"
    assert result["dataset_resolution"]["problem_type"] == "Clustering"
    assert result["needs_clarification"] is False


def test_conflicting_task_instructions_preserve_clarification_behavior():
    dataframe = _housing_dataframe()
    intent_result = _mock_intent(
        "regression",
        {
            "classification": 0.05,
            "clustering": 0.03,
            "regression": 0.90,
            "unknown": 0.02,
        },
    )
    task = (
        "Predict the numerical value in median_house_value, but also classify "
        "that same value as a category."
    )

    with patch("services.nlp.pipeline.detect_user_intent", return_value=intent_result):
        result = process_nlp_request(task, dataframe)

    assert result["intent"]["intent"] == "regression"
    assert result["target"]["matched_column"] == "median_house_value"
    assert result["needs_clarification"] is True
    assert result["ready_for_training"] is False