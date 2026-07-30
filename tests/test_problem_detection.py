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
