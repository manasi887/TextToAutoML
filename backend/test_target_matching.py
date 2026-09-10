"""Comprehensive tests for target matching functionality."""

import sys
import pandas as pd

# Add backend to path for imports
sys.path.insert(0, '.')

from services.nlp.target_matching import match_target_column


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
    
    return failed == 0


if __name__ == "__main__":
    success = test_target_matching()
    sys.exit(0 if success else 1)
