import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.metalearning.meta_features import extract_meta_features


def test_classification_meta_features_exclude_target_and_capture_dtype_counts():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "segment": ["A", "B", "A", "B"],
            "is_active": [True, False, True, True],
            "target": [0, 1, 0, 1],
        }
    )

    features = extract_meta_features(
        dataframe,
        target_column="target",
        problem_type="Binary Classification",
    )

    assert features["schema_version"] == "1.0"
    assert features["row_count"] == 4
    assert features["column_count"] == 4
    assert features["feature_count"] == 3
    assert features["numeric_column_count"] == 1
    assert features["categorical_column_count"] == 1
    assert features["boolean_column_count"] == 1
    assert features["target_exists"] is True
    assert features["target_unique_count"] == 2
    assert features["target_is_binary"] is True
    assert features["target_problem_type"] == "Binary Classification"
    json.dumps(features)


def test_regression_meta_features_include_target_statistics():
    dataframe = pd.DataFrame(
        {
            "rooms": [1.0, 2.0, 3.0, 4.0],
            "income": [10.0, 20.0, 30.0, 40.0],
            "price": [100.0, 200.0, 300.0, 400.0],
        }
    )

    features = extract_meta_features(dataframe, target_column="price")

    assert features["problem_type"] == "Regression"
    assert features["feature_count"] == 2
    assert features["target_is_numeric"] is True
    assert features["target_class_count"] is None
    assert features["target_unique_count"] == 4
    assert features["target_unique_ratio"] == 1.0
    assert features["requires_categorical_encoding"] is False


def test_clustering_meta_features_without_target_use_all_columns():
    dataframe = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [3.0, 2.0, 1.0],
        }
    )

    features = extract_meta_features(dataframe, problem_type="Clustering")

    assert features["problem_type"] == "Clustering"
    assert features["feature_count"] == 2
    assert features["column_count"] == 2
    assert features["target_exists"] is False
    assert features["target_unique_count"] is None


def test_meta_features_capture_missing_duplicates_and_cardinality():
    dataframe = pd.DataFrame(
        {
            "numeric": [1.0, None, 1.0, 4.0],
            "category": ["a", None, "a", "b"],
            "target": [0, 1, 0, 1],
        }
    )

    features = extract_meta_features(dataframe, target_column="target")

    assert features["missing_value_count"] == 2
    assert features["columns_with_missing_count"] == 2
    assert features["max_column_missing_ratio"] == 0.25
    assert features["duplicate_row_count"] == 1
    assert features["duplicate_row_ratio"] == 0.25
    assert features["requires_imputation"] is True
    assert features["requires_categorical_encoding"] is True
    assert features["max_categorical_cardinality"] == 2


def test_empty_dataframe_returns_stable_json_safe_defaults():
    features = extract_meta_features(pd.DataFrame(), problem_type="Clustering")

    assert features["row_count"] == 0
    assert features["column_count"] == 0
    assert features["feature_count"] == 0
    assert features["rows_per_feature"] == 0.0
    assert features["target_exists"] is False
    json.dumps(features, allow_nan=False)
