import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.nlp.pipeline import process_nlp_request
from services.nlp.target_matching import match_target_column


def _intent(intent):
    probabilities = {
        "classification": 0.9 if intent == "classification" else 0.03,
        "regression": 0.9 if intent == "regression" else 0.03,
        "clustering": 0.03,
        "unknown": 0.04,
    }
    return {
        "intent": intent,
        "confidence": probabilities[intent],
        "probabilities": probabilities,
    }


def _analyze(text, dataframe, intent):
    with patch("services.nlp.pipeline.detect_user_intent", return_value=_intent(intent)):
        return process_nlp_request(text, dataframe)


def test_iris_species_request_resolves_variety_end_to_end():
    dataframe = pd.DataFrame(
        {
            "sepal.length": [5.1, 4.9, 5.0, 6.1, 5.9, 6.0, 6.5, 6.3, 6.4],
            "sepal.width": [3.5, 3.0, 3.2, 2.8, 3.0, 2.9, 3.0, 3.3, 3.1],
            "petal.length": [1.4, 1.4, 1.5, 4.7, 4.2, 4.5, 5.2, 5.0, 5.1],
            "petal.width": [0.2, 0.2, 0.2, 1.2, 1.5, 1.5, 2.0, 1.9, 2.0],
            "variety": ["setosa"] * 3 + ["versicolor"] * 3 + ["virginica"] * 3,
        }
    )

    result = _analyze(
        "Determine which species each flower belongs to.",
        dataframe,
        "classification",
    )

    assert result["intent"]["intent"] == "classification"
    assert result["target"]["target_reference"] == "which species each flower belongs to"
    assert result["dataset_resolution"]["target_column"] == "variety"
    assert result["dataset_resolution"]["problem_type"] == "Multi-class Classification"
    assert result["needs_clarification"] is False
    assert result["ready_for_training"] is True
    assert result["dataset_resolution"]["target_match_score"] >= 0.92
    assert result["dataset_resolution"]["target_match_margin"] >= 0.01


def test_churn_request_resolves_exit_end_to_end():
    dataframe = pd.DataFrame(
        {
            "exit": [True, False] * 10,
            "active_member": [True, True, False, False] * 5,
            "risk_score": [index / 20 for index in range(20)],
            "risk_segment": ["high", "low"] * 10,
        }
    )

    result = _analyze("Will this customer churn?", dataframe, "classification")

    assert result["dataset_resolution"]["target_column"] == "exit"
    assert result["dataset_resolution"]["problem_type"] == "Binary Classification"
    assert result["needs_clarification"] is False
    assert result["dataset_resolution"]["target_match_score"] >= 0.92
    assert result["dataset_resolution"]["target_match_margin"] >= 0.01


def test_bank_outcome_request_prefers_exit_over_generic_customer_segment():
    dataframe = pd.DataFrame(
        {
            "exit": [True, False] * 12,
            "active_member": [True, True, False, False] * 6,
            "risk_segment": ["high", "low"] * 12,
            "customer_segment": [
                "segment_a", "segment_b", "segment_c", "segment_d",
                "segment_e", "segment_f", "segment_g", "segment_h",
            ] * 3,
            "risk_score": [index / 24 for index in range(24)],
        }
    )

    result = _analyze(
        "Predict whether a customer will leave the bank.",
        dataframe,
        "classification",
    )

    assert result["target"]["matched_column"] == "exit"
    assert result["dataset_resolution"]["target_column"] == "exit"
    assert result["dataset_resolution"]["problem_type"] == "Binary Classification"
    assert result["needs_clarification"] is False


def test_bank_churn_structural_evidence_overrides_semantic_hard_negative():
    from services.nlp.target_matching import _apply_structural_preference

    dataframe = pd.DataFrame(
        {
            "exit": [0, 1] * 12,
            "customer_segment": [
                "segment_a", "segment_b", "segment_c", "segment_d",
                "segment_e", "segment_f", "segment_g", "segment_h",
            ] * 3,
        }
    )
    candidates = [
        {"column": "customer_segment", "score": 0.9766},
        {"column": "exit", "score": 0.0168},
    ]

    reranked = _apply_structural_preference(
        "Predict whether a customer will leave the bank.", dataframe, candidates
    )

    assert reranked[0]["column"] == "exit"
    assert reranked[0]["score"] >= 0.92
    assert reranked[0]["score"] - reranked[1]["score"] >= 0.01


def test_house_price_request_resolves_value_not_house_age():
    dataframe = pd.DataFrame(
        {
            "median_house_value": [220000, 250000, 300000, 180000],
            "house_age": [20, 30, 40, 50],
            "median_income": [3.1, 2.8, 4.2, 3.0],
        }
    )

    result = _analyze("What is the house price?", dataframe, "regression")

    assert result["dataset_resolution"]["target_column"] == "median_house_value"
    assert result["dataset_resolution"]["target_column"] != "house_age"
    assert result["needs_clarification"] is False


def test_hard_negative_columns_do_not_beat_the_requested_target():
    cases = [
        (
            "What is the house price?",
            "median_house_value",
            "house_age",
            {"median_house_value": [220000, 250000, 300000, 180000], "house_age": [20, 30, 40, 50], "median_income": [3.1, 2.8, 4.2, 3.0]},
        ),
        (
            "How many passengers are on this trip?",
            "passenger_count",
            "trip_distance",
            {"passenger_count": [1, 2, 3, 4], "trip_distance": [3.0, 4.5, 2.0, 9.0], "fare": [10.0, 20.0, 30.0, 40.0]},
        ),
        (
            "Does the patient have a disease?",
            "diagnosis",
            "patient_age",
            {"diagnosis": ["yes", "no", "yes", "no"], "patient_age": [20, 30, 40, 50], "hospital": ["a", "b", "a", "b"]},
        ),
    ]

    for request, expected, hard_negative, data in cases:
        dataframe = pd.DataFrame(data)
        result = match_target_column(request, dataframe)
        assert result["matched_column"] == expected
        assert result["matched_column"] != hard_negative


def test_single_hard_negative_candidate_without_target_requires_clarification():
    negative_cases = (
        ("What is the house price?", {"house_age": [20, 30, 40, 50]}),
        ("How many passengers are on this trip?", {"trip_distance": [3.0, 4.5, 2.0, 9.0]}),
        ("Does the patient have a disease?", {"patient_age": [20, 30, 40, 50]}),
    )

    for request, data in negative_cases:
        result = match_target_column(request, pd.DataFrame(data))

        assert result["matched_column"] is None
        assert result["needs_clarification"] is True
        assert result["margin"] == 0.0


def test_lexically_ambiguous_targets_still_require_clarification():
    dataframe = pd.DataFrame(
        {
            "customer_status": ["active", "closed", "active", "closed"],
            "account_status": ["open", "closed", "open", "closed"],
        }
    )

    result = match_target_column("status", dataframe)

    assert result["matched_column"] is None
    assert result["needs_clarification"] is True
    assert result["match_type"] == "lexical"
