import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app


client = TestClient(app)


@pytest.fixture
def upload_dir():
    upload_dir = Path(__file__).resolve().parents[1] / "backend" / "storage" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


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

    assert set(payload.keys()) == {"status", "filename", "target_column", "problem_type", "data", "preprocessing", "models", "evaluation", "best_model", "model"}
    assert isinstance(payload["data"], dict)
    assert isinstance(payload["models"], dict)
    assert isinstance(payload["evaluation"], dict)
    assert isinstance(payload["best_model"], dict)
    assert isinstance(payload["model"], dict)
    assert payload["model"]["model_id"]
