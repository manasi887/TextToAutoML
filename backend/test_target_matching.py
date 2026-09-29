"""Comprehensive tests for target matching functionality."""

import sys
from unittest.mock import patch

import pandas as pd

# Add backend to path for imports
sys.path.insert(0, '.')

from services.nlp.target_matching import match_target_column
from services.nlp.target_extraction import extract_target_reference


def test_target_matching():
    """Run comprehensive tests for target matching functionality."""
    # Test DataFrame with various column naming styles
    test_df = pd.DataFrame({
        "Customer_ID": [1, 2, 3],
        "Age": [25, 30, 35],
        "Salary": [50000, 60000, 70000],
        "Exited": [0, 1, 0],
        "House_Price": [100000, 150000, 200000],
        "Customer_Segment": ["A", "B", "C"],
        "Churn": [0, 1, 0],
        "Product_Name": ["X", "Y", "Z"],
        "Total_Revenue": [1000, 2000, 3000],
    })
    
    test_cases = [
        # Exact matches
        ("Exited", "Exited", True, "exact match"),
        ("Age", "Age", True, "exact match simple"),
        ("House_Price", "House_Price", True, "exact match with underscore"),
        ("Product_Name", "Product_Name", True, "exact match with underscore"),
        
        # Natural language descriptions - generic phrase cleanup
        ("price of a house", "House_Price", True, "natural language - price of house"),
        ("price of house", "House_Price", True, "natural language - price variant"),
        ("house price", "House_Price", True, "natural language - direct phrase"),
        ("customer segment", "Customer_Segment", True, "natural language - segment"),
        ("product name", "Product_Name", True, "natural language - product name"),
        ("total revenue", "Total_Revenue", True, "natural language - revenue"),
        
        # Natural language with predictive patterns removed
        ("predict the age", "Age", True, "natural language - predict age"),
        ("predict house price", "House_Price", True, "natural language - predict house price"),
        ("predict the salary", "Salary", True, "natural language - predict salary"),
        
        # Token-based matches
        ("customer id", "Customer_ID", True, "token match with separator"),
        ("churn", "Churn", True, "simple token match"),
        ("salary", "Salary", True, "case insensitive"),
        ("age", "Age", True, "lowercase exact match"),
        
        # Token-based partial matches (single token in multi-token column)
        ("segment", "Customer_Segment", True, "partial token match - unique"),
        ("revenue", "Total_Revenue", True, "partial token match - revenue"),
        ("price", "House_Price", True, "partial token match - price"),
        
        # Ambiguous cases (multiple similar columns or tied scores)
        ("customer", None, False, "ambiguous - multiple customer columns"),
        
        # Unrelated targets (no match)
        ("income", None, False, "unrelated - no matching column"),
        ("department", None, False, "unrelated - different domain"),
        ("category", None, False, "unrelated - not in dataframe"),
        ("country", None, False, "unrelated - geographical term"),
        
        # Edge cases
        ("  Salary  ", "Salary", True, "whitespace handling"),
        ("HOUSE_PRICE", "House_Price", True, "uppercase normalization"),
        ("customer-id", "Customer_ID", True, "hyphen separator"),
    ]
    
    print("=" * 80)
    print("TARGET MATCHING COMPREHENSIVE TESTS")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    errors = 0
    
    for target, expected_match, should_match, description in test_cases:
        try:
            result = match_target_column(target, test_df)
            matched_column = result["matched_column"]
            needs_clarification = result["needs_clarification"]
            confidence = result["confidence"]
            
            # Check if match expectation is met
            success = False
            if should_match:
                success = matched_column == expected_match and not needs_clarification
            else:
                success = matched_column is None and needs_clarification
            
            if success:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
            
            print(f"{status} | {description}")
            print(f"  Input: '{target}'")
            print(f"  Expected: {expected_match if should_match else 'No match (clarification needed)'}")
            print(f"  Got: {matched_column} (confidence: {confidence:.2f}, needs_clarification: {needs_clarification})")
            if result["candidates"]:
                top_candidates = result["candidates"][:3]
                candidates_str = ", ".join([f"{c['column']}({c['score']:.2f})" for c in top_candidates])
                print(f"  Candidates: {candidates_str}")
            print()
            
        except Exception as e:
            print(f"✗ ERROR | {description}")
            print(f"  Input: '{target}'")
            print(f"  Exception: {e}")
            print()
            errors += 1
            failed += 1
    
    print("=" * 80)
    total = passed + failed
    print(f"RESULTS: {passed} passed, {failed} failed ({errors} errors) out of {total} tests")
    print("=" * 80)

    assert failed == 0


def test_min_candidate_score_includes_exact_boundary_only():
    dataframe = pd.DataFrame({"boundary": [1], "below": [2]})

    def fake_score(target, column):
        return {"boundary": 0.50, "below": 0.499999}[column]

    with patch("services.nlp.target_matching._score_match", side_effect=fake_score), patch(
        "services.nlp.target_matching._hybrid_candidate_scores", return_value=[]
    ):
        result = match_target_column("requested", dataframe)

    assert [item["column"] for item in result["candidates"]] == ["boundary"]


def test_min_match_score_accepts_exact_boundary_and_rejects_just_below():
    boundary_df = pd.DataFrame({"boundary": [1]})
    below_df = pd.DataFrame({"below": [1]})

    with patch(
        "services.nlp.target_matching._score_match", return_value=0.80
    ), patch("services.nlp.target_matching._hybrid_candidate_scores", return_value=[]):
        boundary_result = match_target_column("requested", boundary_df)
    with patch(
        "services.nlp.target_matching._score_match", return_value=0.799999
    ), patch("services.nlp.target_matching._hybrid_candidate_scores", return_value=[]):
        below_result = match_target_column("requested", below_df)

    assert boundary_result["matched_column"] == "boundary"
    assert below_result["matched_column"] is None
    assert below_result["needs_clarification"] is True


def test_min_score_margin_accepts_exact_boundary_and_rejects_just_below():
    dataframe = pd.DataFrame({"best": [1], "second": [2]})

    def exact_margin_score(target, column):
        return {"best": 0.90, "second": 0.65}[column]

    def below_margin_score(target, column):
        return {"best": 0.90, "second": 0.650001}[column]

    with patch(
        "services.nlp.target_matching._score_match", side_effect=exact_margin_score
    ):
        exact_result = match_target_column("requested", dataframe)
    with patch(
        "services.nlp.target_matching._score_match", side_effect=below_margin_score
    ):
        below_result = match_target_column("requested", dataframe)

    assert exact_result["matched_column"] == "best"
    assert below_result["matched_column"] is None
    assert below_result["needs_clarification"] is True


def test_abbreviated_target_names_require_clarification_without_synonyms():
    abbreviation_cases = [
        ("amt", "amount"),
        ("cust_ID", "customer ID"),
        ("sales_amt", "sales amount"),
    ]

    for target_reference, column_name in abbreviation_cases:
        result = match_target_column(
            target_reference,
            pd.DataFrame({column_name: [1]}),
        )

        assert result["matched_column"] is None
        assert result["needs_clarification"] is True
        assert result["margin"] == 0.0


def test_single_word_semantic_guess_requires_clarification_but_income_token_matches():
    dataframe = pd.DataFrame(
        {
            "monthly_income": [3000.0, 4000.0, 5000.0],
            "customer_segment": ["a", "b", "a"],
        }
    )

    income_result = match_target_column("income", dataframe)
    unrelated_result = match_target_column("country", dataframe)

    assert income_result["matched_column"] == "monthly_income"
    assert income_result["match_type"] == "lexical"
    assert unrelated_result["matched_column"] is None
    assert unrelated_result["needs_clarification"] is True


def test_generic_column_matches_only_when_reference_contains_exact_name():
    dataframe = pd.DataFrame({"col_1": [1, 2], "col_2": [3, 4], "col_3": [5, 6]})

    exact_result = match_target_column("predict col_2", dataframe)
    descriptive_result = match_target_column("predict the second column", dataframe)

    assert exact_result["matched_column"] == "col_2"
    assert exact_result["needs_clarification"] is False
    assert descriptive_result["matched_column"] is None
    assert descriptive_result["needs_clarification"] is True


def test_house_price_paraphrases_match_house_value_column():
    dataframe = pd.DataFrame(
        {
            "median_income": [3.1, 2.8, 4.2],
            "median_house_value": [220000, 250000, 300000],
        }
    )

    for target_reference in ("house prices", "value of each house", "value of a house"):
        result = match_target_column(target_reference, dataframe)

        assert result["matched_column"] == "median_house_value"
        assert result["target_reference"] == target_reference
        assert result["needs_clarification"] is False


def test_customer_outcome_paraphrases_match_exit_with_a_clear_margin():
    dataframe = pd.DataFrame(
        {
            "exit": [True, False, True, False, True, False, True, False],
            "active_member": [True, True, False, False, True, True, False, False],
            "risk_segment": ["high", "low"] * 4,
            "risk_score": [0.2, 0.4, 0.8, 0.5, 0.7, 0.1, 0.3, 0.6],
        }
    )
    requests = (
        "Predict whether a customer will leave the bank.",
        "Will this customer churn?",
        "Determine if the customer exits",
    )

    for request in requests:
        target = extract_target_reference(request)
        result = match_target_column(target["target_reference"], dataframe)

        assert result["matched_column"] == "exit"
        assert result["needs_clarification"] is False
        assert result["score"] >= 0.92
        assert result["margin"] >= 0.01
        assert result["margin"] < result["score"]
        assert result["match_type"] in {"lexical", "hybrid"}


def test_semantically_near_tied_candidates_require_clarification():
    dataframe = pd.DataFrame(
        {
            "exit": [True, False, True, False],
            "exits": [False, True, True, False],
        }
    )

    result = match_target_column("whether a customer will leave", dataframe)

    assert result["matched_column"] is None
    assert result["needs_clarification"] is True
    assert result["score"] < 0.92


def test_clear_regression_target_uses_lexical_fast_path():
    dataframe = pd.DataFrame(
        {
            "annual_revenue": [10.0, 20.0, 30.0, 40.0],
            "headcount": [1, 2, 3, 4],
        }
    )
    target = extract_target_reference("Estimate annual revenue")

    result = match_target_column(target["target_reference"], dataframe)

    assert result["matched_column"] == "annual_revenue"
    assert result["match_type"] == "lexical"
    assert result["score"] == 1.0
    assert result["margin"] == 1.0


def test_vague_target_wording_does_not_semantically_select_a_column():
    dataframe = pd.DataFrame(
        {
            "exit": [True, False, True, False],
            "risk_score": [0.2, 0.8, 0.4, 0.6],
        }
    )
    target = extract_target_reference("What should I predict?")

    assert target["target_found"] is False
    assert match_target_column("Tell me about this dataset", dataframe)["needs_clarification"] is True


if __name__ == "__main__":
    test_target_matching()
    sys.exit(0)
