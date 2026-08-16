import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.persistence as persistence
from main import app
from services.automl.persistence import save_model_package


class BinaryPredictModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return [1 if row.sum() >= 2.0 else 0 for row in X]


class RegressionPredictModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return [float(sum(row)) for row in X]


class DummyOneHotEncoder:
    def transform(self, X):
        X = np.asarray(X).ravel()
        result = []
        for value in X:
            v = str(value)
            if v == "France" or v == "Female":
                result.append([1, 0])
            else:
                result.append([0, 1])
        return np.asarray(result, dtype=float)


class ScalarClassificationModel:
    def predict(self, X):
        return np.int64(1)

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]], dtype=float)


class ScalarRegressionModel:
    def predict(self, X):
        return np.float64(42.5)


@pytest.fixture
def persisted_model(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    package = save_model_package(
        model=BinaryPredictModel(),
        encoders={
            "country": {"type": "onehot", "encoder": DummyOneHotEncoder(), "feature_names": ["country_France", "country_Germany"]},
            "gender": {"type": "onehot", "encoder": DummyOneHotEncoder(), "feature_names": ["gender_Female", "gender_Male"]},
        },
        feature_names=["Age", "Balance", "country_France", "country_Germany", "gender_Female", "gender_Male"],
        target_column="Exited",
        problem_type="Binary Classification",
        preprocessing={
            "removed_columns": {"identifier_columns": [], "constant_columns": []},
            "imputed_columns": {"numeric": ["Age", "Balance"], "categorical": ["country", "gender"]},
            "raw_feature_names": ["Age", "Balance", "country", "gender"],
            "imputation_values": {"numeric": {"Age": 35.0, "Balance": 25000.0}, "categorical": {"country": "France", "gender": "Female"}},
        },
        model_name="BinaryModel",
        selection_metric="f1_score",
        metrics={"f1_score": 0.9},
        storage_dir=tmp_path / "models",
    )
    return package


def test_predict_api_successful_request(persisted_model):
    client = TestClient(app)
    response = client.post(
        "/predict/",
        json={
            "model_id": persisted_model["model_id"],
            "data": [{"Age": 30, "Balance": 1000, "country": "France", "gender": "Female"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model_id"] == persisted_model["model_id"]
    assert body["prediction_count"] == 1
    assert body["predictions"] in ([1], [0])


def test_predict_api_missing_model_returns_404():
    client = TestClient(app)
    response = client.post("/predict/", json={"model_id": "missing_model_id", "data": [{"Age": 30}]})
    assert response.status_code == 404


def test_predict_api_invalid_input_returns_400():
    client = TestClient(app)
    response = client.post("/predict/", json={"model_id": "abc", "data": []})
    assert response.status_code == 400


def test_predict_api_empty_data_returns_400():
    client = TestClient(app)
    response = client.post("/predict/", json={"model_id": "abc", "data": []})
    assert response.status_code == 400


def test_predict_api_returns_classification_success_response(persisted_model):
    client = TestClient(app)
    response = client.post(
        "/predict/",
        json={
            "model_id": persisted_model["model_id"],
            "data": [{"Age": 30, "Balance": 1000, "country": "France", "gender": "Female"}, {"Age": 50, "Balance": 2000, "country": "Germany", "gender": "Male"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["problem_type"] == "Binary Classification"
    assert len(body["predictions"]) == 2


def test_predict_api_returns_regression_success_response(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    package = save_model_package(
        model=RegressionPredictModel(),
        encoders={"country": {"type": "onehot", "encoder": DummyOneHotEncoder(), "feature_names": ["country_France", "country_Germany"]}},
        feature_names=["Age", "country_France", "country_Germany"],
        target_column="target",
        problem_type="Regression",
        preprocessing={
            "removed_columns": {"identifier_columns": [], "constant_columns": []},
            "imputed_columns": {"numeric": ["Age"], "categorical": ["country"]},
            "raw_feature_names": ["Age", "country"],
            "imputation_values": {"numeric": {"Age": 30.0}, "categorical": {"country": "France"}},
        },
        model_name="RegressionModel",
        selection_metric="rmse",
        metrics={"rmse": 0.2},
        storage_dir=tmp_path / "models",
    )
    client = TestClient(app)
    response = client.post(
        "/predict/",
        json={"model_id": package["model_id"], "data": [{"Age": 30, "country": "France"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["problem_type"] == "Regression"
    assert body["predictions"]


def test_predict_api_serializes_numpy_scalar_outputs_to_native_python_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    classification_package = save_model_package(
        model=ScalarClassificationModel(),
        encoders={
            "country": {"type": "onehot", "encoder": DummyOneHotEncoder(), "feature_names": ["country_France", "country_Germany"]},
            "gender": {"type": "onehot", "encoder": DummyOneHotEncoder(), "feature_names": ["gender_Female", "gender_Male"]},
        },
        feature_names=["Age", "Balance", "country_France", "country_Germany", "gender_Female", "gender_Male"],
        target_column="Exited",
        problem_type="Binary Classification",
        preprocessing={
            "removed_columns": {"identifier_columns": [], "constant_columns": []},
            "imputed_columns": {"numeric": ["Age", "Balance"], "categorical": ["country", "gender"]},
            "raw_feature_names": ["Age", "Balance", "country", "gender"],
            "imputation_values": {"numeric": {"Age": 35.0, "Balance": 25000.0}, "categorical": {"country": "France", "gender": "Female"}},
        },
        model_name="ScalarBinaryModel",
        selection_metric="f1_score",
        metrics={"f1_score": 0.9},
        storage_dir=tmp_path / "models",
    )

    classification_response = TestClient(app).post(
        "/predict/",
        json={
            "model_id": classification_package["model_id"],
            "data": [{"Age": 30, "Balance": 1000, "country": "France", "gender": "Female"}],
        },
    )
    assert classification_response.status_code == 200
    classification_body = classification_response.json()
    assert classification_body["predictions"] == [1]
    assert isinstance(classification_body["predictions"][0], int)
    assert classification_body["probabilities"] == [[0.2, 0.8]]
    assert all(isinstance(value, float) for value in classification_body["probabilities"][0])

    regression_package = save_model_package(
        model=ScalarRegressionModel(),
        encoders={"country": {"type": "onehot", "encoder": DummyOneHotEncoder(), "feature_names": ["country_France", "country_Germany"]}},
        feature_names=["Age", "country_France", "country_Germany"],
        target_column="target",
        problem_type="Regression",
        preprocessing={
            "removed_columns": {"identifier_columns": [], "constant_columns": []},
            "imputed_columns": {"numeric": ["Age"], "categorical": ["country"]},
            "raw_feature_names": ["Age", "country"],
            "imputation_values": {"numeric": {"Age": 30.0}, "categorical": {"country": "France"}},
        },
        model_name="ScalarRegressionModel",
        selection_metric="rmse",
        metrics={"rmse": 0.2},
        storage_dir=tmp_path / "models",
    )

    regression_response = TestClient(app).post(
        "/predict/",
        json={"model_id": regression_package["model_id"], "data": [{"Age": 30, "country": "France"}]},
    )
    assert regression_response.status_code == 200
    regression_body = regression_response.json()
    assert regression_body["predictions"] == [42.5]
    assert isinstance(regression_body["predictions"][0], float)
