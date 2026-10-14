import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.run_history as run_history
from services.automl.nlp_integration import integrate_nlp_with_automl
from services.reporting.training_report import generate_training_report


def test_repeated_training_runs_are_preserved_and_compared(tmp_path, monkeypatch):
    dataframe = pd.DataFrame({"target": [0, 1, 0], "feature": [1, 2, 3]})
    resolution = {
        "target_column": "target",
        "problem_type": "Binary Classification",
        "confidence": 0.9,
        "needs_clarification": False,
    }
    nlp_result = {"ready_for_training": True, "dataset_resolution": resolution}
    outputs = [
        {
            "status": "success",
            "model": {"model_id": "run_one", "name": "ModelOne"},
            "target_column": "target",
            "problem_type": "Binary Classification",
            "best_model": {
                "name": "ModelOne",
                "selection_metric": "f1_score",
                "metrics": {"f1_score": 0.80},
            },
        },
        {
            "status": "success",
            "model": {"model_id": "run_two", "name": "ModelTwo"},
            "target_column": "target",
            "problem_type": "Binary Classification",
            "best_model": {
                "name": "ModelTwo",
                "selection_metric": "f1_score",
                "metrics": {"f1_score": 0.90},
            },
        },
    ]
    history_path = tmp_path / "training_history.json"
    monkeypatch.setattr(run_history, "HISTORY_PATH", history_path)

    with patch(
        "services.automl.nlp_integration.run_automl_pipeline",
        side_effect=outputs,
    ), patch(
        "services.automl.nlp_integration.time.perf_counter",
        side_effect=[1.0, 1.25, 2.0, 2.5],
    ):
        first = integrate_nlp_with_automl(
            dataframe, nlp_result, dataset_name="same.csv"
        )
        second = integrate_nlp_with_automl(
            dataframe, nlp_result, dataset_name="same.csv"
        )

    assert first["automl_training"]["run"]["run_id"] == "run_one"
    assert first["automl_training"]["comparison"]["status"] == "baseline"
    assert first["automl_training"]["training_time_seconds"] == 0.25
    assert second["automl_training"]["run"]["run_id"] == "run_two"
    assert second["automl_training"]["comparison"]["status"] == "improved"
    assert second["automl_training"]["comparison"]["previous_run_id"] == "run_one"
    assert second["automl_training"]["comparison"]["metric"] == "f1_score"
    assert second["automl_training"]["comparison"]["metric_delta"] == pytest.approx(0.1)
    assert second["automl_training"]["training_time_seconds"] == 0.5

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert [item["run_id"] for item in history] == ["run_one", "run_two"]
    assert history[0]["meta_features"]["target_exists"] is True
    assert history[0]["meta_features"]["target_problem_type"] == "Binary Classification"
    assert history[1]["meta_features"]["feature_count"] == 1

    report = generate_training_report(
        second["nlp_resolution"], second["automl_training"]
    )
    assert report["run"]["run_id"] == "run_two"
    assert report["comparison"]["status"] == "improved"
    json.dumps(report)


def test_legacy_history_records_without_meta_features_remain_compatible(tmp_path):
    history_path = tmp_path / "training_history.json"
    legacy_record = {
        "run_id": "legacy_run",
        "timestamp": "2026-01-01T00:00:00Z",
        "dataset": "legacy.csv",
        "target": "target",
        "problem_type": "Regression",
        "best_model": "LinearRegression",
        "selection_metric": "rmse",
        "metric_value": 1.5,
        "training_time_seconds": 0.2,
    }
    history_path.write_text(json.dumps([legacy_record]) + "\n", encoding="utf-8")

    dataframe = pd.DataFrame({"target": [1.0, 2.0, 3.0], "feature": [2.0, 4.0, 6.0]})
    training_output = {
        "model": {"model_id": "new_run"},
        "target_column": "target",
        "problem_type": "Regression",
        "best_model": {
            "name": "LinearRegression",
            "selection_metric": "rmse",
            "metrics": {"rmse": 0.5},
        },
        "training_time_seconds": 0.3,
    }

    record, comparison = run_history.record_training_run(
        dataframe,
        training_output,
        dataset_name="legacy.csv",
        history_path=history_path,
    )

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history[0] == legacy_record
    assert "meta_features" not in history[0]
    assert record["meta_features"]["target_problem_type"] == "Regression"
    assert comparison["status"] == "improved"
    json.dumps(history, allow_nan=False)


def test_run_history_persists_optional_optimizer_decision_and_keeps_legacy_records(tmp_path):
    history_path = tmp_path / "training_history.json"
    legacy_record = {"run_id": "legacy", "dataset": "course.csv", "metric_value": 0.5}
    history_path.write_text(json.dumps([legacy_record]) + "\n", encoding="utf-8")
    decision = {
        "requested_mode": "auto",
        "selected_strategy": "random",
        "trial_budget": 3,
        "reason": "Bounded medium-workload search.",
        "tuning_duration_seconds": 0.25,
        "workload": {"candidate_model_count": 3, "cv_folds": 5},
    }
    dataframe = pd.DataFrame({"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]})
    training_output = {
        "model": {"model_id": "auto_run"},
        "target_column": "target",
        "problem_type": "Binary Classification",
        "best_model": {
            "name": "DecisionTreeClassifier",
            "selection_metric": "f1_score",
            "metrics": {"f1_score": 0.75},
        },
        "tuning": decision,
    }

    record, _ = run_history.record_training_run(
        dataframe,
        training_output,
        dataset_name="course.csv",
        history_path=history_path,
    )

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history[0] == legacy_record
    assert record["optimizer_decision"] == decision
    assert history[1]["optimizer_decision"]["selected_strategy"] == "random"
    json.dumps(history, allow_nan=False)


def test_clustering_run_history_does_not_claim_cross_run_score_improvements(tmp_path):
    history_path = tmp_path / "training_history.json"
    dataframe = pd.DataFrame({"usage": [1, 2, 3, 4]})
    output = {
        "model": {"model_id": "cluster_run"},
        "target_column": None,
        "problem_type": "Clustering",
        "best_model": {
            "name": "KMeans_k2",
            "selection_metric": "silhouette_score",
            "metrics": {
                "silhouette_score": 0.2,
                "calinski_harabasz_score": 10.0,
                "davies_bouldin_score": 0.8,
            },
            "selection_metrics": {"silhouette_score": 0.25},
        },
        "preprocessing": {
            "clustering_evaluation": {
                "quality_assessment": "Positive silhouette; business review required."
            }
        },
    }

    run_history.record_training_run(
        dataframe,
        output,
        dataset_name="segments.csv",
        history_path=history_path,
    )
    second_output = {
        **output,
        "model": {"model_id": "cluster_run_2"},
        "best_model": {
            **output["best_model"],
            "metrics": {
                **output["best_model"]["metrics"],
                "silhouette_score": 0.4,
            },
        },
    }
    record, comparison = run_history.record_training_run(
        dataframe,
        second_output,
        dataset_name="segments.csv",
        history_path=history_path,
    )

    assert comparison["status"] == "not_comparable"
    assert comparison["metric_delta"] is None
    assert "selection_metrics" in record
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history[0]["clustering_evaluation"]["quality_assessment"].startswith(
        "Positive silhouette"
    )
    json.dumps(history, allow_nan=False)