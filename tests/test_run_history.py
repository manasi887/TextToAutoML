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