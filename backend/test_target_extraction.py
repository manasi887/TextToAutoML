import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.nlp.target_extraction import extract_target_reference


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
    assert "median_house_value" in result["target_reference"]