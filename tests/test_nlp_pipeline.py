import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

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


def _regression_intent():
    probabilities = {
        "classification": 0.01,
        "regression": 0.98,
        "clustering": 0.005,
        "unknown": 0.005,
    }
    return {
        "intent": "regression",
        "confidence": probabilities["regression"],
        "probabilities": probabilities,
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


@pytest.mark.parametrize(
    "prompt",
    [
        "Predict body performance class",
        "Predict the class",
    ],
)
def test_predict_class_label_wording_resolves_categorical_target_despite_regression_classifier(
    prompt,
):
    dataframe = pd.DataFrame(
        {
            "class": ["A", "B", "C", "D"] * 5,
            "body performance": [index / 10 for index in range(20)],
            "age": [20 + index for index in range(20)],
        }
    )

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=_regression_intent(),
    ):
        result = process_nlp_request(prompt, dataframe)

    assert result["intent"]["intent"] == "regression"
    assert result["task"]["problem_type"] == "classification"
    assert result["dataset_resolution"]["target_column"] == "class"
    assert result["dataset_resolution"]["problem_type"] == "Multi-class Classification"
    assert result["needs_clarification"] is False
    assert result["ready_for_training"] is True


def test_predict_category_cue_resolves_an_exact_categorical_category_column():
    dataframe = pd.DataFrame(
        {
            "body_performance_category": ["A", "B", "C", "D"] * 5,
            "feature": [index / 10 for index in range(20)],
        }
    )

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=_regression_intent(),
    ):
        result = process_nlp_request("Predict body performance category", dataframe)

    assert result["task"]["problem_type"] == "classification"
    assert result["dataset_resolution"]["target_column"] == "body_performance_category"
    assert result["dataset_resolution"]["problem_type"] == "Multi-class Classification"
    assert result["ready_for_training"] is True


def test_numeric_target_named_class_is_not_forced_to_classification():
    dataframe = pd.DataFrame(
        {
            "class": [float(index * 10 + 1) for index in range(20)],
            "feature": [index % 4 for index in range(20)],
        }
    )

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=_regression_intent(),
    ):
        result = process_nlp_request("Predict the class", dataframe)

    assert result["intent"]["intent"] == "regression"
    assert result["task"]["problem_type"] == "classification"
    assert result["dataset_resolution"]["target_column"] is None
    assert result["needs_clarification"] is True
    assert result["ready_for_training"] is False


def test_ambiguous_class_like_targets_still_require_clarification():
    dataframe = pd.DataFrame(
        {
            "class_label": ["A", "B", "C", "D"] * 5,
            "class_type": ["W", "X", "Y", "Z"] * 5,
            "age": [20 + index for index in range(20)],
        }
    )

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=_regression_intent(),
    ):
        result = process_nlp_request("Predict the class", dataframe)

    assert result["task"]["problem_type"] == "classification"
    assert result["dataset_resolution"]["target_column"] is None
    assert result["needs_clarification"] is True
    assert result["ready_for_training"] is False


def test_explicit_regression_and_numeric_prediction_wording_takes_precedence():
    dataframe = pd.DataFrame(
        {
            "Sales": [float(index * 10 + 1) for index in range(20)],
            "feature": [index % 4 for index in range(20)],
        }
    )
    requests = (
        "This is a regression task. Predict the class in Sales.",
        "Make a numeric prediction for Sales.",
    )

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=_regression_intent(),
    ):
        for request in requests:
            result = process_nlp_request(request, dataframe)
            assert result["task"]["problem_type"] == "regression"
            assert result["dataset_resolution"]["problem_type"] == "Regression"
            assert result["dataset_resolution"]["target_column"] == "Sales"
            assert result["ready_for_training"] is True


def test_explicit_regression_prevents_class_cue_from_overriding_numeric_target():
    dataframe = pd.DataFrame(
        {
            "class": [float(index * 10 + 1) for index in range(20)],
            "feature": [index % 4 for index in range(20)],
        }
    )

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=_regression_intent(),
    ):
        result = process_nlp_request(
            "This is a regression task. Predict the class", dataframe
        )

    assert result["task"]["problem_type"] == "regression"
    assert result["dataset_resolution"]["target_column"] == "class"
    assert result["dataset_resolution"]["problem_type"] == "Regression"
    assert result["ready_for_training"] is True


def test_explicit_classify_request_still_uses_existing_classification_behavior():
    dataframe = pd.DataFrame(
        {
            "class": ["A", "B", "C", "D"] * 5,
            "feature": [index % 6 for index in range(20)],
        }
    )

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=_regression_intent(),
    ):
        result = process_nlp_request("Classify body performance class", dataframe)

    assert result["task"]["problem_type"] == "classification"
    assert result["dataset_resolution"]["problem_type"] == "Multi-class Classification"
    assert result["dataset_resolution"]["target_column"] == "class"
    assert result["ready_for_training"] is True


def test_prediction_class_cue_does_not_change_clustering_or_generic_ambiguity():
    dataframe = pd.DataFrame(
        {
            "feature_a": [index % 7 for index in range(20)],
            "feature_b": [index % 11 for index in range(20)],
        }
    )
    clustering_probabilities = {
        "classification": 0.02,
        "regression": 0.03,
        "clustering": 0.94,
        "unknown": 0.01,
    }
    clustering_intent = {
        "intent": "clustering",
        "confidence": 0.94,
        "probabilities": clustering_probabilities,
    }
    unknown_probabilities = {
        "classification": 0.1,
        "regression": 0.1,
        "clustering": 0.1,
        "unknown": 0.7,
    }
    unknown_intent = {
        "intent": "unknown",
        "confidence": 0.7,
        "probabilities": unknown_probabilities,
    }

    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=clustering_intent,
    ):
        clustering = process_nlp_request("Group records by pattern", dataframe)
    with patch(
        "services.nlp.pipeline.detect_user_intent",
        return_value=unknown_intent,
    ):
        ambiguous = process_nlp_request("Analyze this dataset", dataframe)

    assert clustering["task"]["problem_type"] == "clustering"
    assert clustering["dataset_resolution"]["problem_type"] == "Clustering"
    assert clustering["ready_for_training"] is True
    assert ambiguous["intent"]["intent"] == "unknown"
    assert ambiguous["needs_clarification"] is True
    assert ambiguous["ready_for_training"] is False