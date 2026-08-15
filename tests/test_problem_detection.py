import sys
from pathlib import Path

import pandas as pd

# Ensure the backend package is importable from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.automl.problem_detection import (
    detect_problem_type,
    detect_target_candidates,
    generate_automl_recommendation,
    recommend_models,
)
from services.dataset.intelligence import detect_identifier_columns


def test_detect_target_candidates_excludes_identifier_and_constant_columns():
    df = pd.DataFrame(
        {
            "Order ID": [1001, 1002, 1003],
            "Country": ["India", "India", "India"],
            "Sales": [100.0, 150.0, 200.0],
            "Segment": ["Consumer", "Corporate", "Home Office"],
        }
    )

    report = detect_target_candidates(df)

    assert report["status"] == "Completed"
    assert all(
        column not in ["Order ID", "Country"]
        for column in [candidate["column"] for candidate in report["target_candidates"]]
    )
    assert any(candidate["column"] == "Sales" for candidate in report["target_candidates"])
    assert any(candidate["column"] == "Segment" for candidate in report["target_candidates"])


def test_detect_target_candidates_ranks_candidates_by_score():
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "category": ["a", "b", "b"],
            "id": [1, 2, 3],
        }
    )

    report = detect_target_candidates(df)
    candidates = report["target_candidates"]

    assert report["status"] == "Completed"
    assert len(candidates) == 2
    assert candidates[0]["column"] == "value"
    assert candidates[1]["column"] == "category"
    assert candidates[0]["score"] > candidates[1]["score"]


def test_detect_problem_type_returns_regression_for_numeric_target():
    df = pd.DataFrame(
        {
            "Sales": [100.0, 150.0, 200.0],
            "Category": ["A", "B", "A"],
        }
    )

    report = detect_problem_type(df, "Sales")

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Regression"
    assert "numeric" in report["reason"].lower()


def test_detect_problem_type_returns_binary_classification_for_discrete_zero_one_target():
    df = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1, 0, 1]})

    report = detect_problem_type(df, "label")

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Binary Classification"


def test_detect_problem_type_keeps_continuous_numeric_target_as_regression():
    df = pd.DataFrame({"price": [10.2, 12.4, 18.3, 20.1, 14.6, 22.7, 25.0, 27.4]})

    report = detect_problem_type(df, "price")

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Regression"


def test_detect_problem_type_returns_binary_classification_for_binary_numeric_target():
    for column, values in {
        "Exited": [0, 1, 0, 1, 1],
        "stroke": [0, 1, 1, 0, 1],
    }.items():
        df = pd.DataFrame({column: values})
        report = detect_problem_type(df, column)
        assert report["status"] == "Completed"
        assert report["problem_type"] == "Binary Classification"
        assert "binary" in report["reason"].lower() or "two" in report["reason"].lower()


def test_detect_problem_type_returns_binary_classification_for_binary_categorical_target():
    df = pd.DataFrame(
        {
            "Outcome": ["Yes", "No", "Yes"],
        }
    )

    report = detect_problem_type(df, "Outcome")

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Binary Classification"
    assert "two classes" in report["reason"].lower()


def test_detect_identifier_columns_ignores_measurements_and_postal_codes():
    df = pd.DataFrame(
        {
            "CustomerId": [1, 2, 3, 4, 5],
            "RowNumber": [10, 11, 12, 13, 14],
            "EstimatedSalary": [35000, 42000, 51000, 45000, 61000],
            "Postal Code": [10001, 10002, 10003, 10004, 10005],
            "Age": [25, 30, 28, 35, 45],
        }
    )

    identifier_report = detect_identifier_columns(df)
    identifiers = identifier_report["identifier_columns"]

    assert "CustomerId" in identifiers
    assert "RowNumber" in identifiers
    assert "EstimatedSalary" not in identifiers
    assert "Postal Code" not in identifiers


def test_detect_target_candidates_ranks_income_over_workclass():
    df = pd.DataFrame(
        {
            "income": [25000, 42000, 31000, 56000, 59000, 48000, 42000, 65000],
            "workclass": ["Private", "Private", "Self-emp", "Private", "Local-gov", "Private", "Self-emp", "Local-gov"],
            "age": [30, 42, 35, 50, 58, 44, 40, 60],
        }
    )

    report = detect_target_candidates(df)
    columns = [candidate["column"] for candidate in report["target_candidates"]]

    assert report["status"] == "Completed"
    assert "income" in columns
    assert columns[0] == "income"


def test_detect_target_candidates_prioritizes_strong_categorical_target_over_measurement_columns():
    df = pd.DataFrame(
        {
            "weight_kg": [72.5, 69.0, 74.8, 71.1, 70.9, 77.3, 68.2, 76.5, 73.0, 75.9, 69.8, 71.5, 74.1, 72.0, 78.4],
            "class": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C"],
            "height_cm": [172.2, 169.8, 181.0, 174.8, 170.5, 178.1, 168.9, 180.0, 171.2, 177.8, 175.9, 173.4, 182.1, 170.7, 179.6],
        }
    )

    report = detect_target_candidates(df)
    columns = [candidate["column"] for candidate in report["target_candidates"]]

    assert report["status"] == "Completed"
    assert columns[0] == "class"
    assert any(candidate["column"] == "class" and candidate["inferred_problem_type"] == "Multi-class Classification" for candidate in report["target_candidates"])


def test_detect_target_candidates_prioritizes_binary_numeric_outcome_over_multiclass_feature():
    df = pd.DataFrame(
        {
            "stroke": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "smoking_status": [
                "never smoked",
                "formerly smoked",
                "smokes",
                "never smoked",
                "formerly smoked",
                "smokes",
                "never smoked",
                "formerly smoked",
                "smokes",
                "never smoked",
                "formerly smoked",
                "smokes",
                "never smoked",
                "formerly smoked",
                "smokes",
            ],
        }
    )

    report = detect_target_candidates(df)
    columns = [candidate["column"] for candidate in report["target_candidates"]]

    assert report["status"] == "Completed"
    assert columns[0] == "stroke"
    assert detect_problem_type(df, "stroke")["problem_type"] == "Binary Classification"


def test_detect_target_candidates_requires_confirmation_for_ambiguous_numeric_targets():
    df = pd.DataFrame(
        {
            "median_income": [3.1, 2.8, 4.2, 5.1, 3.6, 2.9, 4.8, 5.4],
            "median_house_value": [220000, 250000, 300000, 310000, 270000, 240000, 330000, 350000],
            "total_rooms": [1200, 1500, 2200, 2600, 1800, 1400, 2100, 2400],
        }
    )

    report = detect_target_candidates(df)

    assert report["status"] == "Completed"
    assert report["requires_user_confirmation"] is True


def test_detect_problem_type_returns_multiclass_classification_for_categorical_target():
    df = pd.DataFrame(
        {
            "Label": ["Red", "Green", "Blue"],
        }
    )

    report = detect_problem_type(df, "Label")

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Multi-class Classification"
    assert "more than two" in report["reason"].lower()


def test_detect_problem_type_returns_clustering_if_target_missing():
    df = pd.DataFrame(
        {
            "Feature": [1, 2, 3],
        }
    )

    report = detect_problem_type(df, "")

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Clustering"
    assert "no target" in report["reason"].lower()


def test_recommend_models_for_regression():
    report = recommend_models("Regression")

    assert report["status"] == "Completed"
    assert any("Linear Regression" in item for item in report["recommended_models"])
    assert any("XGBoost Regressor" in item for item in report["recommended_models"])


def test_recommend_models_for_classification():
    report = recommend_models("Binary Classification")

    assert report["status"] == "Completed"
    assert any("Logistic Regression" in item for item in report["recommended_models"])
    assert any("XGBoost Classifier" in item for item in report["recommended_models"])


def test_recommend_models_for_clustering():
    report = recommend_models("Clustering")

    assert report["status"] == "Completed"
    assert any("K-Means" in item for item in report["recommended_models"])
    assert any("DBSCAN" in item for item in report["recommended_models"])


def test_generate_automl_recommendation_compiles_report():
    df = pd.DataFrame(
        {
            "Order Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Sales": [100.0, 150.0, 200.0],
            "Product": ["A", "B", "A"],
        }
    )

    report = generate_automl_recommendation(df)

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Regression"
    assert report["recommended_target"] == "Sales"
    assert report["target_candidates"]
    assert report["recommended_models"]
    assert "Consider using 'Sales' as the target column" in report["recommendations"]
    assert isinstance(report["dataset_intelligence"], dict)
    assert isinstance(report["problem_detection"], dict)


def test_generate_automl_recommendation_handles_no_candidates():
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "Country": ["India", "India", "India"],
        }
    )

    report = generate_automl_recommendation(df)

    assert report["status"] == "Completed"
    assert report["problem_type"] == "Clustering"
    assert report["recommended_target"] is None
    assert report["recommended_models"]
    assert any("No strong supervised target" in rec for rec in report["recommendations"])
