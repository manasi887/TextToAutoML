import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app
from config import UPLOAD_DIR
from api.train import _serialize_training_result


client = TestClient(app)


@pytest.fixture
def upload_dir():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _make_regression_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature_1": [i * 1.5 for i in range(80)],
            "feature_2": [i * 2.0 for i in range(80)],
            "target": [i * 3.0 for i in range(80)],
        }
    )
    df.to_csv(path, index=False)


def _make_binary_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature_1": [i % 7 for i in range(120)],
            "feature_2": [((i * 3) % 5) for i in range(120)],
            "feature_3": [i * 0.5 for i in range(120)],
            "target": [1 if i % 2 == 0 else 0 for i in range(120)],
        }
    )
    df.to_csv(path, index=False)


def _make_multiclass_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature_1": [i % 10 for i in range(180)],
            "feature_2": [((i * 2) % 7) for i in range(180)],
            "feature_3": [i * 0.25 for i in range(180)],
            "target": [i % 3 for i in range(180)],
        }
    )
    df.to_csv(path, index=False)


def test_training_result_serializer_keeps_diagnostics_optional_and_backward_compatible():
    legacy_payload = {
        "status": "success",
        "data": {"validation_rows": 10},
        "models": {"trained": ["ModelA"], "failed": []},
        "evaluation": {"status": "Completed", "results": []},
        "best_model": {"name": "ModelA", "selection_metric": "f1_score"},
        "model": {"name": "ModelA"},
    }

    legacy_response = _serialize_training_result(legacy_payload)
    assert "validation_diagnostics" not in legacy_response
    assert legacy_response["best_model"]["name"] == "ModelA"
    assert legacy_response["evaluation"]["status"] == "Completed"

    diagnostics = [{
        "candidate_name": "ModelA",
        "status": "evaluated",
        "selection_metric": "balanced_accuracy",
        "selection_value": 0.75,
        "training_metrics": {"f1_score": 0.95},
        "validation_metrics": {"f1_score": 0.8},
    }]
    extended_response = _serialize_training_result({
        **legacy_payload,
        "validation_diagnostics": diagnostics,
    })
    assert extended_response["validation_diagnostics"] == diagnostics
    assert extended_response["validation_diagnostics"][0]["training_metrics"] != extended_response["validation_diagnostics"][0]["validation_metrics"]


def test_valid_regression_training_request(upload_dir):
    dataset_name = "api_regression_test.csv"
    _make_regression_dataset(upload_dir / dataset_name)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "target",
            "problem_type": "Regression",
            "test_size": 0.2,
            "random_state": 42,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["filename"] == dataset_name
    assert payload["target_column"] == "target"
    assert payload["problem_type"] == "Regression"
    assert payload["data"]["training_rows"] > 0
    assert payload["best_model"]["name"] is not None
    assert "model" in payload
    assert payload["model"]["model_id"]
    assert payload["model"]["name"] == payload["best_model"]["name"]


def test_training_request_accepts_group_column_and_excludes_it_from_features(upload_dir):
    dataset_name = "api_group_regression_test.csv"
    dataframe = pd.DataFrame(
        [
            {"group": f"g-{group_index}", "feature": group_index + row_index / 10, "target": group_index * 2 + row_index}
            for group_index in range(20)
            for row_index in range(5)
        ]
    )
    dataframe.to_csv(upload_dir / dataset_name, index=False)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "target",
            "group_column": "group",
            "problem_type": "Regression",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["preprocessing"]["removed_columns"]["group_columns"] == ["group"]


def test_training_request_forwards_selected_feature_columns(upload_dir):
    dataset_name = "api_feature_selection_test.csv"
    _make_regression_dataset(upload_dir / dataset_name)
    pipeline_result = {
        "status": "success",
        "target_column": "target",
        "problem_type": "Regression",
        "data": {},
        "preprocessing": {"raw_feature_names": ["feature_1"]},
        "models": {},
        "evaluation": {},
        "best_model": {},
        "model": {},
    }

    with patch("api.train.run_automl_pipeline", return_value=pipeline_result) as pipeline_mock:
        response = client.post(
            "/train/",
            json={
                "filename": dataset_name,
                "target_column": "target",
                "problem_type": "Regression",
                "feature_columns": ["feature_1"],
            },
        )

    assert response.status_code == 200
    assert pipeline_mock.call_args.kwargs["feature_columns"] == ["feature_1"]


def test_training_request_forwards_bayesian_strategy_and_trial_budget(upload_dir):
    dataset_name = "api_bayesian_test.csv"
    _make_regression_dataset(upload_dir / dataset_name)
    pipeline_result = {
        "status": "success",
        "target_column": "target",
        "problem_type": "Regression",
        "data": {},
        "preprocessing": {},
        "models": {},
        "evaluation": {},
        "best_model": {},
        "model": {},
        "tuning": {"strategy": "bayesian", "trials_per_model": 5, "models": {}},
    }

    with patch("api.train.run_automl_pipeline", return_value=pipeline_result) as pipeline_mock:
        response = client.post(
            "/train/",
            json={
                "filename": dataset_name,
                "target_column": "target",
                "problem_type": "Regression",
                "tuning_strategy": "bayesian",
                "tuning_trials": 5,
            },
        )

    assert response.status_code == 200
    assert response.json()["tuning"]["strategy"] == "bayesian"
    assert pipeline_mock.call_args.kwargs["tuning_strategy"] == "bayesian"
    assert pipeline_mock.call_args.kwargs["tuning_trials"] == 5


def test_training_request_forwards_auto_mode_and_returns_decision(upload_dir):
    dataset_name = "api_auto_optimizer_test.csv"
    _make_regression_dataset(upload_dir / dataset_name)
    decision = {
        "requested_mode": "auto",
        "selected_strategy": "random",
        "strategy": "random",
        "trial_budget": 3,
        "reason": "Automatic policy selected Random Search.",
        "workload": {"candidate_model_count": 3},
        "models": {},
    }
    pipeline_result = {
        "status": "success",
        "target_column": "target",
        "problem_type": "Regression",
        "data": {},
        "preprocessing": {},
        "models": {},
        "evaluation": {},
        "best_model": {},
        "model": {},
        "tuning": decision,
    }

    with patch("api.train.run_automl_pipeline", return_value=pipeline_result) as pipeline_mock:
        response = client.post(
            "/train/",
            json={
                "filename": dataset_name,
                "target_column": "target",
                "problem_type": "Regression",
                "tuning_strategy": "auto",
                "tuning_trials": 6,
            },
        )

    assert response.status_code == 200
    assert response.json()["tuning"] == decision
    assert pipeline_mock.call_args.kwargs["tuning_strategy"] == "auto"
    assert pipeline_mock.call_args.kwargs["tuning_trials"] == 6


@pytest.mark.parametrize(
    "settings",
    [
        {"tuning_strategy": "random"},
        {"tuning_trials": 31},
        {"tuning_trials": 0},
    ],
)
def test_training_request_rejects_invalid_tuning_settings(settings):
    response = client.post(
        "/train/",
        json={
            "filename": "unused.csv",
            "target_column": "target",
            **settings,
        },
    )

    assert response.status_code == 422


def test_valid_binary_classification_training_request(upload_dir):
    dataset_name = "api_binary_test.csv"
    _make_binary_dataset(upload_dir / dataset_name)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "target",
            "problem_type": "Binary Classification",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["problem_type"] == "Binary Classification"
    assert payload["best_model"]["selection_metric"] == "f1_score"


def test_valid_multiclass_classification_training_request(upload_dir):
    dataset_name = "api_multiclass_test.csv"
    _make_multiclass_dataset(upload_dir / dataset_name)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "target",
            "problem_type": "Multi-class Classification",
            "test_size": 0.25,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["problem_type"] == "Multi-class Classification"
    assert payload["best_model"]["selection_metric"] == "f1_score"


def test_missing_dataset_returns_404(upload_dir):
    response = client.post(
        "/train/",
        json={
            "filename": "missing.csv",
            "target_column": "target",
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_missing_target_column_is_validation_error():
    response = client.post(
        "/train/",
        json={
            "filename": "api_binary_test.csv",
        },
    )

    assert response.status_code == 422


def test_invalid_target_returns_400(upload_dir):
    dataset_name = "api_binary_test.csv"
    _make_binary_dataset(upload_dir / dataset_name)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "missing_target",
            "problem_type": "Binary Classification",
        },
    )

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"].lower()


def test_invalid_problem_type_returns_400(upload_dir):
    dataset_name = "api_binary_test.csv"
    _make_binary_dataset(upload_dir / dataset_name)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "target",
            "problem_type": "Unsupported Task",
        },
    )

    assert response.status_code == 400
    assert "unsupported problem type" in response.json()["detail"].lower()


def test_invalid_test_size_returns_400(upload_dir):
    dataset_name = "api_regression_test.csv"
    _make_regression_dataset(upload_dir / dataset_name)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "target",
            "problem_type": "Regression",
            "test_size": 1.5,
        },
    )

    assert response.status_code == 422


def test_successful_response_contains_only_json_serializable_fields(upload_dir):
    dataset_name = "api_binary_test.csv"
    _make_binary_dataset(upload_dir / dataset_name)

    response = client.post(
        "/train/",
        json={
            "filename": dataset_name,
            "target_column": "target",
            "problem_type": "Binary Classification",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert set(payload.keys()) == {"status", "filename", "target_column", "problem_type", "data", "preprocessing", "models", "evaluation", "best_model", "model", "validation_diagnostics"}
    assert isinstance(payload["data"], dict)
    assert isinstance(payload["models"], dict)
    assert isinstance(payload["evaluation"], dict)
    assert isinstance(payload["best_model"], dict)
    assert isinstance(payload["model"], dict)
    assert payload["model"]["model_id"]
