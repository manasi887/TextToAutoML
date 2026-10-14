from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.persistence as persistence
from main import app
from services.automl.persistence import save_model_package


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    directory = tmp_path / "models"
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", directory)
    return directory


def _package(store_dir: Path, problem_type: str) -> dict:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(np.array(["a", "b"]).reshape(-1, 1))
    encoded_category_names = encoder.get_feature_names_out(["category"]).tolist()
    encoded_features = ["score", *encoded_category_names]
    training_features = pd.DataFrame(
        {
            "score": [0.0, 1.0, 10.0, 11.0],
            encoded_category_names[0]: [1.0, 0.0, 1.0, 0.0],
            encoded_category_names[1]: [0.0, 1.0, 0.0, 1.0],
        }
    )

    if problem_type == "classification":
        model = DecisionTreeClassifier(max_depth=1, random_state=42)
        model.fit(training_features, [0, 0, 1, 1])
        metadata = {
            "target_column": "label",
            "problem_type": "Binary Classification",
            "target_encoded": True,
            "target_classes": ["negative", "positive"],
            "selection_metric": "accuracy",
            "metrics": {"accuracy": 1.0},
        }
    elif problem_type == "regression":
        model = LinearRegression()
        model.fit(training_features, [0.0, 2.0, 20.0, 22.0])
        metadata = {
            "target_column": "value",
            "problem_type": "Regression",
            "target_encoded": False,
            "target_classes": [],
            "selection_metric": "rmse",
            "metrics": {"rmse": 0.0},
        }
    else:
        model = make_pipeline(
            StandardScaler(),
            KMeans(n_clusters=2, n_init=10, random_state=42),
        )
        model.fit(training_features)
        metadata = {
            "target_column": "",
            "problem_type": "Clustering",
            "target_encoded": False,
            "target_classes": [],
            "selection_metric": "silhouette_score",
            "metrics": {"silhouette_score": 1.0},
        }

    return save_model_package(
        model=model,
        encoders={
            "category": {
                "type": "onehot",
                "encoder": encoder,
                "feature_names": encoded_category_names,
            }
        },
        feature_names=encoded_features,
        target_column=metadata["target_column"],
        problem_type=metadata["problem_type"],
        preprocessing={
            "raw_feature_names": ["score", "category"],
            "feature_names": encoded_features,
            "encoders": {},
            "imputed_columns": {
                "numeric": ["score"],
                "categorical": ["category"],
            },
            "imputation_values": {
                "numeric": {"score": 10.0},
                "categorical": {"category": "a"},
            },
            **{
                key: metadata[key]
                for key in ("target_encoded", "target_classes")
            },
        },
        model_name=f"{problem_type.title()}Model",
        selection_metric=metadata["selection_metric"],
        metrics=metadata["metrics"],
        storage_dir=store_dir,
    )


def _upload(client: TestClient, model_id: str, filename: str, contents: bytes):
    return client.post(
        "/predict/file",
        data={"model_id": model_id},
        files={"file": (filename, contents)},
    )


def _csv(rows: str) -> bytes:
    return rows.encode("utf-8")


def _xlsx(dataframe: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    dataframe.to_excel(buffer, index=False)
    return buffer.getvalue()


def test_classification_file_prediction_uses_saved_model_and_supports_repeated_rows(store_dir, monkeypatch):
    package = _package(store_dir, "classification")
    client = TestClient(app)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Prediction must not fit or retrain the classifier.")

    monkeypatch.setattr(DecisionTreeClassifier, "fit", forbid_fit)
    first_response = _upload(
        client,
        package["model_id"],
        "new_records.csv",
        _csv("category,score,unused\nb,10,ignored\na,0,ignored\nunknown,11,ignored\n"),
    )
    second_response = _upload(
        client,
        package["model_id"],
        "new_records.csv",
        _csv("score,category\n10,b\n0,a\n11,unknown\n"),
    )

    assert first_response.status_code == 200, first_response.text
    assert second_response.status_code == 200, second_response.text
    result = first_response.json()
    assert result["model_id"] == package["model_id"]
    assert result["problem_type"] == "Binary Classification"
    assert result["prediction_count"] == 3
    assert result["prediction_labels"] == ["positive", "negative", "positive"]
    assert second_response.json()["predictions"] == result["predictions"]


def test_regression_xlsx_prediction_reuses_saved_preprocessing(store_dir, monkeypatch):
    package = _package(store_dir, "regression")
    client = TestClient(app)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Prediction must not fit or retrain the regressor.")

    monkeypatch.setattr(LinearRegression, "fit", forbid_fit)
    response = _upload(
        client,
        package["model_id"],
        "new_records.xlsx",
        _xlsx(pd.DataFrame({"category": ["b", "unknown"], "score": [5.0, np.nan]})),
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["problem_type"] == "Regression"
    assert result["prediction_count"] == 2
    assert result["predictions"] == pytest.approx([10.0, 20.0])


def test_clustering_file_prediction_uses_fitted_scaler_and_clusterer(store_dir, monkeypatch):
    package = _package(store_dir, "clustering")
    client = TestClient(app)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Clustering inference must not refit preprocessing or the clusterer.")

    monkeypatch.setattr(StandardScaler, "fit", forbid_fit)
    monkeypatch.setattr(KMeans, "fit", forbid_fit)
    response = _upload(
        client,
        package["model_id"],
        "new_records.csv",
        _csv("category,score\nb,10.5\nb,11\na,0\na,0.5\n"),
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["problem_type"] == "Clustering"
    assert result["prediction_count"] == 4
    assert result["predictions"][0] == result["predictions"][1]
    assert result["predictions"][2] == result["predictions"][3]
    assert result["predictions"][0] != result["predictions"][2]


def test_saved_file_prediction_loads_package_from_disk_on_each_request(store_dir):
    package = _package(store_dir, "regression")
    client = TestClient(app)
    del package

    response = _upload(
        client,
        next(store_dir.glob("*.joblib")).stem,
        "new_records.csv",
        _csv("score,category\n5,b\n"),
    )

    assert response.status_code == 200, response.text
    assert response.json()["predictions"] == pytest.approx([10.0])


def test_saved_model_is_usable_in_a_fresh_backend_process(store_dir, tmp_path):
    package = _package(store_dir, "regression")
    prediction_file = tmp_path / "after_restart.csv"
    prediction_file.write_text("score,category\n5,b\n", encoding="utf-8")
    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
import services.automl.persistence as persistence
persistence.DEFAULT_STORAGE_DIR = Path(sys.argv[2])
from fastapi.testclient import TestClient
from main import app
with TestClient(app) as client:
    with open(sys.argv[3], "rb") as data_file:
        response = client.post(
            "/predict/file",
            data={"model_id": sys.argv[1]},
            files={"file": ("after_restart.csv", data_file, "text/csv")},
        )
print(json.dumps({"status_code": response.status_code, "body": response.json()}))
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            package["model_id"],
            str(store_dir),
            str(prediction_file),
            str(backend_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["status_code"] == 200
    assert result["body"]["model_id"] == package["model_id"]
    assert result["body"]["predictions"] == pytest.approx([10.0])


@pytest.mark.parametrize(
    ("model_id", "filename", "contents", "status_code", "message"),
    [
        ("missing_model_id", "data.csv", b"score,category\n1,a\n", 404, "not found"),
        ("../../outside", "data.csv", b"score,category\n1,a\n", 400, "unsafe"),
        ("valid", "data.csv", b"score\n1\n", 400, "missing required feature"),
        ("valid", "data.xlsx", b"not an excel file", 400, "Unable to read"),
        ("valid", "data.txt", b"score,category\n1,a\n", 400, "Only CSV and Excel"),
    ],
)
def test_file_prediction_rejects_invalid_ids_and_files(
    store_dir,
    model_id,
    filename,
    contents,
    status_code,
    message,
):
    package = _package(store_dir, "regression")
    if model_id == "valid":
        model_id = package["model_id"]
    response = _upload(TestClient(app), model_id, filename, contents)

    assert response.status_code == status_code
    assert message.lower() in response.json()["detail"].lower()
