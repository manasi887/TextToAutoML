import io
import re
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app
import api.cnn as cnn_api


client = TestClient(app)


@pytest.fixture
def isolated_cnn_storage(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    model_root = tmp_path / "models" / "cnn"
    prediction_upload_dir = tmp_path / "prediction_uploads"

    for directory in (dataset_root, model_root, prediction_upload_dir):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cnn_api, "CNN_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(cnn_api, "MODEL_ROOT", model_root)
    monkeypatch.setattr(cnn_api, "PREDICTION_UPLOAD_DIR", prediction_upload_dir)
    return model_root, prediction_upload_dir


def _valid_png_zip() -> io.BytesIO:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        for split in ("train", "validation", "test"):
            for class_name in ("class_a", "class_b"):
                archive.writestr(f"{split}/{class_name}/sample.png", b"png placeholder")
    archive_buffer.seek(0)
    return archive_buffer


def test_cnn_train_success(monkeypatch, isolated_cnn_storage):
    received_arguments = {}

    def fake_train_cnn_model(dataset_root, model_output_dir, **kwargs):
        received_arguments.update(kwargs)
        return {
            "class_names": ["class_a", "class_b"],
            "input_shape": [224, 224, 3],
            "test_metrics": {"accuracy": 0.95, "loss": 0.12},
            "training_history": [{"accuracy": 0.9}],
        }

    monkeypatch.setattr(cnn_api, "train_cnn_model", fake_train_cnn_model)

    response = client.post(
        "/cnn/train",
        files={"dataset_zip": ("dataset.zip", _valid_png_zip(), "application/zip")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert re.fullmatch(r"^[0-9a-f]{32}$", payload["model_id"])
    assert payload["class_names"] == ["class_a", "class_b"]
    assert payload["test_metrics"] == {"accuracy": 0.95, "loss": 0.12}
    assert payload["training_history"] == [{"accuracy": 0.9}]
    assert received_arguments["image_size"] == (224, 224)


def test_cnn_train_rejects_non_zip_upload(isolated_cnn_storage):
    response = client.post(
        "/cnn/train",
        files={"dataset_zip": ("dataset.txt", io.BytesIO(b"not a zip"), "text/plain")},
    )

    assert response.status_code == 400
    assert "zip" in response.json()["detail"].lower()


def test_cnn_train_rejects_corrupted_zip(isolated_cnn_storage):
    response = client.post(
        "/cnn/train",
        files={"dataset_zip": ("dataset.zip", io.BytesIO(b"not a zip"), "application/zip")},
    )

    assert response.status_code == 400
    assert "valid zip archive" in response.json()["detail"].lower()


def test_cnn_train_rejects_zip_path_traversal(isolated_cnn_storage):
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("../escape.png", b"png placeholder")
    archive_buffer.seek(0)

    response = client.post(
        "/cnn/train",
        files={"dataset_zip": ("dataset.zip", archive_buffer, "application/zip")},
    )

    assert response.status_code == 400


def test_cnn_train_rejects_oversized_upload(monkeypatch, isolated_cnn_storage):
    monkeypatch.setattr(cnn_api, "MAX_DATASET_UPLOAD_BYTES", 1)

    response = client.post(
        "/cnn/train",
        files={"dataset_zip": ("dataset.zip", _valid_png_zip(), "application/zip")},
    )

    assert response.status_code == 400
    assert "maximum allowed size" in response.json()["detail"].lower()


def test_cnn_predict_success(monkeypatch, isolated_cnn_storage):
    model_root, prediction_upload_dir = isolated_cnn_storage
    model_id = "0123456789abcdef0123456789abcdef"
    model_dir = model_root / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "best_cnn_model.keras").write_bytes(b"dummy model")
    (model_dir / "cnn_metadata.json").write_text("{}", encoding="utf-8")

    def fake_predict_image(model_path, image_path, metadata_path):
        return {
            "predicted_class": "class_a",
            "predicted_class_index": 0,
            "confidence": 0.97,
            "probabilities": {"class_a": 0.97, "class_b": 0.03},
        }

    monkeypatch.setattr(cnn_api, "predict_image", fake_predict_image)

    response = client.post(
        f"/cnn/predict/{model_id}",
        files={"image": ("sample.png", io.BytesIO(b"png placeholder"), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == model_id
    assert payload["predicted_class"] == "class_a"
    assert payload["confidence"] == 0.97
    assert payload["probabilities"] == {"class_a": 0.97, "class_b": 0.03}
    assert list(prediction_upload_dir.iterdir()) == []


def test_cnn_predict_rejects_invalid_model_id(isolated_cnn_storage):
    response = client.post(
        "/cnn/predict/not-a-valid-model-id",
        files={"image": ("sample.png", io.BytesIO(b"png placeholder"), "image/png")},
    )

    assert response.status_code == 400


def test_cnn_predict_returns_404_for_missing_model(isolated_cnn_storage):
    model_id = "fedcba9876543210fedcba9876543210"

    response = client.post(
        f"/cnn/predict/{model_id}",
        files={"image": ("sample.png", io.BytesIO(b"png placeholder"), "image/png")},
    )

    assert response.status_code == 404
