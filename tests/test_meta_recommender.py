import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.metalearning.recommender import (
    OPENML_SIMILARITY_WEIGHT_CAP,
    recommend_models,
)


def _features(problem_type, rows, numeric_ratio=1.0):
    return {
        "schema_version": "1.0",
        "problem_type": problem_type,
        "row_count": rows,
        "column_count": 4,
        "feature_count": 3,
        "numeric_ratio": numeric_ratio,
        "categorical_ratio": 1.0 - numeric_ratio,
        "missing_value_ratio": 0.0,
        "target_unique_count": 2 if "Classification" in problem_type else 100,
        "target_unique_ratio": 0.02 if "Classification" in problem_type else 1.0,
        "target_entropy": 1.0 if "Classification" in problem_type else None,
    }


def _record(run_id, model, features, metric=0.8, problem_type="Binary Classification"):
    return {
        "run_id": run_id,
        "dataset": f"dataset_{run_id}.csv",
        "target": "target",
        "problem_type": problem_type,
        "best_model": model,
        "selection_metric": "f1_score" if "Classification" in problem_type else "rmse",
        "metric_value": metric,
        "training_time_seconds": 1.0,
        "meta_features": features,
    }


def _write_history(tmp_path, records):
    path = tmp_path / "training_history.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_similar_historical_datasets_rank_model_family(tmp_path):
    current = _features("Binary Classification", 100, 0.75)
    history = [
        _record("one", "LogisticRegression", _features("Binary Classification", 98, 0.75)),
        _record("two", "LogisticRegression", _features("Binary Classification", 105, 0.75)),
        _record("three", "RandomForestClassifier", _features("Binary Classification", 900, 0.25)),
    ]

    result = recommend_models(current, history_path=_write_history(tmp_path, history))

    assert result["status"] == "recommended"
    assert result["recommendations"][0]["model_family"] == "LogisticRegression"
    assert result["recommendations"][0]["evidence_count"] == 2
    assert result["similar_datasets"][0]["dataset"] == "dataset_one.csv"
    assert "target" not in result["similar_datasets"][0]


def test_different_problem_types_are_filtered(tmp_path):
    current = _features("Regression", 100)
    history = [
        _record("classification", "LogisticRegression", _features("Binary Classification", 100), problem_type="Binary Classification"),
        _record("regression", "RandomForestRegressor", _features("Regression", 100), problem_type="Regression"),
    ]

    result = recommend_models(current, history_path=_write_history(tmp_path, history))

    assert result["status"] == "insufficient_history"
    assert result["compatible_history_count"] == 1
    assert result["recommendations"] == []


def test_no_history_is_safe(tmp_path):
    result = recommend_models(
        _features("Binary Classification", 100),
        history_path=tmp_path / "missing.json",
    )

    assert result["status"] == "no_history"
    assert result["recommendations"] == []
    assert result["confidence"] == 0.0


def test_insufficient_history_is_safe(tmp_path):
    history = [_record("one", "LogisticRegression", _features("Binary Classification", 100))]

    result = recommend_models(
        _features("Binary Classification", 100),
        history_path=_write_history(tmp_path, history),
    )

    assert result["status"] == "insufficient_history"
    assert result["recommendations"] == []


def test_equal_historical_support_returns_low_confidence(tmp_path):
    current = _features("Binary Classification", 100)
    history = [
        _record("one", "LogisticRegression", _features("Binary Classification", 100)),
        _record("two", "RandomForestClassifier", _features("Binary Classification", 100)),
    ]

    result = recommend_models(current, history_path=_write_history(tmp_path, history))

    assert result["status"] == "low_confidence"
    assert result["confidence"] == 0.5
    assert len(result["recommendations"]) == 2
    json.dumps(result, allow_nan=False)


def test_local_only_behavior_remains_unchanged_when_openml_is_empty(tmp_path):
    current = _features("Binary Classification", 100, 0.75)
    history = [
        _record("one", "LogisticRegression", _features("Binary Classification", 98, 0.75)),
        _record("two", "LogisticRegression", _features("Binary Classification", 105, 0.75)),
    ]
    history_path = _write_history(tmp_path, history)

    local_result = recommend_models(current, history_path=history_path)
    empty_openml_result = recommend_models(
        current,
        history_path=history_path,
        openml_records=[],
    )

    assert empty_openml_result == local_result


def test_openml_records_are_filtered_and_provenance_is_preserved(tmp_path):
    current = _features("Binary Classification", 100, 0.75)
    history = [
        _record("local", "LogisticRegression", _features("Binary Classification", 100, 0.75)),
        _record("local_two", "RandomForestClassifier", _features("Binary Classification", 100, 0.75)),
    ]
    openml_records = [
        {
            "source": "openml",
            "dataset_id": "61",
            "dataset_version": "2",
            "task_id": "7",
            "run_id": "11",
            "problem_type": "Binary Classification",
            "best_model": "LogisticRegression",
            "selection_metric": "f1_score",
            "metric_value": 0.9,
            "meta_features": _features("Binary Classification", 101, 0.75),
        },
        {
            "source": "openml",
            "dataset_id": "62",
            "task_id": "8",
            "run_id": "12",
            "problem_type": "Binary Classification",
            "best_model": "UnsupportedModel",
            "selection_metric": "f1_score",
            "metric_value": 0.99,
            "meta_features": _features("Binary Classification", 100, 0.75),
        },
        {
            "source": "openml",
            "dataset_id": "63",
            "task_id": "9",
            "run_id": "13",
            "problem_type": "Binary Classification",
            "best_model": "LogisticRegression",
            "selection_metric": "accuracy",
            "metric_value": 0.99,
            "meta_features": _features("Binary Classification", 100, 0.75),
        },
    ]

    result = recommend_models(
        current,
        history_path=_write_history(tmp_path, history),
        openml_records=openml_records,
    )

    assert result["openml_record_count"] == 1
    assert result["recommendations"][0]["openml_evidence_count"] == 1
    evidence = result["recommendations"][0]["evidence"]
    openml_evidence = next(item for item in evidence if item["source"] == "openml")
    assert openml_evidence["dataset_id"] == "61"
    assert openml_evidence["dataset_version"] == "2"
    assert openml_evidence["task_id"] == "7"
    assert openml_evidence["run_id"] == "11"


def test_duplicate_openml_records_are_counted_once(tmp_path):
    current = _features("Regression", 100)
    history = [
        _record("local_one", "LinearRegression", _features("Regression", 100), problem_type="Regression"),
        _record("local_two", "RandomForestRegressor", _features("Regression", 110), problem_type="Regression"),
    ]
    openml_record = {
        "source": "openml",
        "dataset_id": "61",
        "dataset_version": "1",
        "task_id": "7",
        "run_id": "11",
        "problem_type": "Regression",
        "best_model": "LinearRegression",
        "selection_metric": "rmse",
        "metric_value": 1.2,
        "meta_features": _features("Regression", 100),
    }

    result = recommend_models(
        current,
        history_path=_write_history(tmp_path, history),
        openml_records=[openml_record, dict(openml_record)],
    )

    assert result["openml_record_count"] == 1
    assert result["recommendations"][0]["openml_evidence_count"] == 1


def test_openml_similarity_weight_is_capped_relative_to_local(tmp_path):
    current = _features("Binary Classification", 100, 0.75)
    history = [
        _record("local_one", "RandomForestClassifier", _features("Binary Classification", 100, 0.75)),
        _record("local_two", "RandomForestClassifier", _features("Binary Classification", 101, 0.75)),
    ]
    openml_records = [
        {
            "source": "openml",
            "dataset_id": str(index),
            "dataset_version": "1",
            "task_id": str(index),
            "run_id": str(index),
            "problem_type": "Binary Classification",
            "best_model": "LogisticRegression",
            "selection_metric": "f1_score",
            "metric_value": 0.99,
            "meta_features": _features("Binary Classification", 100, 0.75),
        }
        for index in range(10)
    ]

    result = recommend_models(
        current,
        history_path=_write_history(tmp_path, history),
        openml_records=openml_records,
    )
    local_weight = sum(
        item["weighted_similarity"]
        for recommendation in result["recommendations"]
        for item in recommendation["evidence"]
        if item["source"] == "local"
    )
    openml_weight = sum(
        item["weighted_similarity"]
        for recommendation in result["recommendations"]
        for item in recommendation["evidence"]
        if item["source"] == "openml"
    )

    assert openml_weight <= OPENML_SIMILARITY_WEIGHT_CAP * local_weight


def test_invalid_openml_records_fall_back_to_local_history(tmp_path):
    current = _features("Binary Classification", 100, 0.75)
    history = [
        _record("one", "LogisticRegression", _features("Binary Classification", 100, 0.75)),
        _record("two", "LogisticRegression", _features("Binary Classification", 101, 0.75)),
    ]
    history_path = _write_history(tmp_path, history)

    local_result = recommend_models(current, history_path=history_path)
    fallback_result = recommend_models(
        current,
        history_path=history_path,
        openml_records=[None, {"source": "other"}, {"source": "openml", "best_model": "bad"}],
    )

    assert fallback_result == local_result
