import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.nlp.target_extraction import extract_target_reference
from services.nlp.target_matching import match_target_column


def test_extracts_action_and_question_target_wording():
    requests = [
        "Predict whether a customer will leave the bank.",
        "Will this customer churn?",
        "Determine if the customer exits",
        "Classify customers who are likely to leave",
    ]

    for request in requests:
        result = extract_target_reference(request)

        assert result["target_found"] is True
        assert result["needs_clarification"] is False
        assert isinstance(result["target_reference"], str)


def test_extracts_clear_regression_question_and_action():
    for request in (
        "Estimate annual revenue",
        "How much will the property sell for?",
    ):
        result = extract_target_reference(request)

        assert result["target_found"] is True
        assert result["needs_clarification"] is False


def test_vague_request_does_not_produce_a_target():
    result = extract_target_reference("What should I predict?")

    assert result["target_reference"] is None
    assert result["target_found"] is False
    assert result["needs_clarification"] is True


def test_extracts_explicit_target_column_from_middle_of_regression_prompt():
    result = extract_target_reference(
        "This is a regression task. Predict the numerical value in the "
        "median_house_value column using the other columns in the dataset. "
        "Use median_house_value as the target column. Do not predict any other column."
    )

    assert result["target_found"] is True
    assert result["needs_clarification"] is False
    assert result["target_reference"] == "median_house_value"


def test_extracts_target_named_with_using_as_target_wording():
    result = extract_target_reference("Predict Exited using Exited as the target.")

    assert result["target_reference"] == "exited"
    assert result["target_found"] is True
    assert result["explicit_target_column"] is True


def test_extracts_target_from_use_sales_to_predict_wording():
    result = extract_target_reference("Use Sales to predict.")

    assert result["target_reference"] == "sales"
    assert result["target_found"] is True
    assert result["needs_clarification"] is False


def test_supported_target_phrases_resolve_to_the_named_column():
    dataframe = pd.DataFrame(
        {
            "Exited": [0, 1, 0, 1],
            "Sales": [10.0, 20.0, 30.0, 40.0],
        }
    )
    requests = (
        "The target is Exited.",
        "The target column is Exited.",
        "Use Exited as the target.",
        "Using Exited as the target.",
        "Predict Exited.",
        "Predict the Exited column.",
        "Forecast Sales.",
        "Use Sales to predict.",
        "Use 'Exited' as the target column.",
    )

    for request in requests:
        extracted = extract_target_reference(request)
        matched = match_target_column(
            extracted["target_reference"],
            dataframe,
            require_exact_match=extracted.get("explicit_target_column", False),
        )
        expected_column = "Sales" if "Sales" in request else "Exited"

        assert extracted["target_found"] is True
        assert matched["matched_column"] == expected_column
        assert matched["needs_clarification"] is False


def test_extracts_explicit_target_after_common_qualifiers():
    for qualifier in ("strictly", "specifically", "explicitly"):
        result = extract_target_reference(
            f"Build a binary classification model. The target column is {qualifier} Exited."
        )

        assert result["target_reference"] == "exited"
        assert result["target_found"] is True
        assert result["explicit_target_column"] is True