import json
import queue
import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
import api.train as train_api
import services.nlp.intent_detection as intent_detection
import services.nlp.target_matching as target_matching


client = TestClient(app)


def test_health_endpoint_reports_service_state():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["service"] == "text-to-automl"


def test_readiness_endpoint_exists_and_returns_valid_status():
    response = client.get("/ready")

    assert response.status_code in {200, 503}
    payload = response.json()
    assert payload["status"] in {"ok", "not_ready"}


def _write_valid_intent_artifacts(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "label_mapping.json").write_text(
        json.dumps({"id_to_label": {"0": "classification"}}),
        encoding="utf-8",
    )


def test_readiness_rejects_missing_intent_model_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(intent_detection, "MODEL_DIR", tmp_path / "missing")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["missing"] == ["intent_model_artifacts"]
    assert "model directory" in response.json()["artifact_details"]
    assert str(tmp_path) not in response.text


def test_readiness_rejects_empty_intent_model_directory(monkeypatch, tmp_path):
    empty_dir = tmp_path / "intent-model"
    empty_dir.mkdir()
    monkeypatch.setattr(intent_detection, "MODEL_DIR", empty_dir)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "config.json" in response.json()["artifact_details"]
    assert "model weights" in response.json()["artifact_details"]


def test_readiness_accepts_valid_intent_artifacts(monkeypatch, tmp_path):
    model_dir = tmp_path / "intent-model"
    _write_valid_intent_artifacts(model_dir)
    monkeypatch.setattr(intent_detection, "MODEL_DIR", model_dir)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["intent_model_artifacts"] is True


def test_readiness_does_not_require_optional_target_matching_models(monkeypatch, tmp_path):
    model_dir = tmp_path / "intent-model"
    _write_valid_intent_artifacts(model_dir)
    monkeypatch.setattr(intent_detection, "MODEL_DIR", model_dir)
    monkeypatch.setattr(target_matching, "_MODEL_COMPONENTS", None)
    monkeypatch.setattr(target_matching, "_MODEL_LOAD_FAILED", True)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["optional"]["target_matching_models"] == "unavailable"


def test_health_does_not_initialize_or_inspect_models(monkeypatch):
    with patch("backend.main.inspect_model_artifacts", side_effect=AssertionError("must not load models")):
        response = client.get("/health")

    assert response.status_code == 200


def test_health_endpoint_stays_responsive_while_training_runs(monkeypatch, tmp_path):
    dataset_path = tmp_path / "health_probe.csv"
    dataset_path.write_text("feature,target\n0,0\n1,1\n", encoding="utf-8")
    monkeypatch.setattr(train_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        train_api,
        "load_dataset",
        lambda _filename: pd.DataFrame({"feature": [0, 1], "target": [0, 1]}),
    )
    monkeypatch.setattr(train_api, "validate_dataset_df", lambda _df: {"valid": True})

    training_started = threading.Event()
    release_training = threading.Event()
    def slow_pipeline(*args, **kwargs):
        training_started.set()
        if not release_training.wait(timeout=5):
            raise TimeoutError("test did not release the training worker")
        return {
            "status": "success",
            "problem_type": "classification",
            "target_column": "target",
            "data": {"original_rows": 4, "feature_count": 1},
            "preprocessing": {"raw_feature_names": ["feature"], "removed_columns": []},
            "models": {},
            "evaluation": {"status": "success", "results": []},
            "best_model": {"name": "DecisionTreeClassifier", "selection_metric": "accuracy", "metrics": {"accuracy": 1.0}},
            "model": {"model_id": "probe-model", "name": "DecisionTreeClassifier", "problem_type": "classification", "target_column": "target", "selection_metric": "accuracy", "metrics": {"accuracy": 1.0}},
        }

    monkeypatch.setattr(train_api, "run_automl_pipeline", slow_pipeline)
    training_response = queue.Queue()

    def trigger_training():
        training_response.put(client.post(
            "/train/",
            json={"filename": "health_probe.csv", "target_column": "target", "problem_type": "classification"},
        ))

    train_thread = threading.Thread(target=trigger_training)
    train_thread.start()
    assert training_started.wait(timeout=2)

    health_response = queue.Queue()
    health_finished = threading.Event()

    def request_health():
        health_response.put(client.get("/health"))
        health_finished.set()

    health_thread = threading.Thread(target=request_health)
    health_thread.start()
    try:
        assert health_finished.wait(timeout=2), "health waited for the training worker"
    finally:
        release_training.set()

    train_thread.join(timeout=5)
    health_thread.join(timeout=5)

    assert not train_thread.is_alive()
    assert not health_thread.is_alive()
    assert training_response.get_nowait().status_code == 200
    assert health_response.get_nowait().status_code == 200
