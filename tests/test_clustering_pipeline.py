import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.persistence as persistence
from services.automl.nlp_integration import integrate_nlp_with_automl
from services.automl.persistence import load_model_package
from services.automl.predictor import predict_with_model
from services.reporting.training_report import generate_training_report


def _segmentation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annual_spend": [120, 125, 130, 900, 920, 880, 410, 430, 390, 700, 720, 680] * 4,
            "visits_per_month": [2, 3, 2, 15, 16, 14, 8, 9, 7, 12, 13, 11] * 4,
            "avg_order_value": [30, 32, 29, 110, 115, 105, 70, 72, 68, 95, 98, 92] * 4,
            "support_contacts": [5, 4, 6, 1, 2, 1, 3, 4, 3, 2, 1, 2] * 4,
        }
    )


def test_clustering_nlp_to_persistence_prediction_and_report(tmp_path, monkeypatch):
    dataframe = _segmentation_data()
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")

    nlp_result = {
        "ready_for_training": True,
        "needs_clarification": False,
        "dataset_resolution": {
            "target_column": None,
            "problem_type": "Clustering",
            "confidence": 0.0,
            "candidates": [],
            "needs_clarification": False,
            "reason": "Clustering does not require a target column.",
        },
    }

    integration = integrate_nlp_with_automl(dataframe, nlp_result)

    assert integration["ready_for_training"] is True
    assert integration["needs_clarification"] is False
    training = integration["automl_training"]
    assert training["status"] == "success"
    assert set(training["models"]["trained"]) == {"KMeans", "DBSCAN"}
    assert training["models"]["failed"] == []
    assert training["model"]["problem_type"] == "Clustering"
    assert training["model"]["target_column"] is None
    assert training["best_model"]["name"] == "KMeans"
    assert training["best_model"]["selection_metric"] == "silhouette_score"
    assert training["best_model"]["metrics"]["silhouette_score"] > 0.0

    package = load_model_package(training["model"]["model_id"])
    prediction = predict_with_model(package, dataframe.iloc[:3].to_dict(orient="records"))
    assert prediction["status"] == "success"
    assert prediction["problem_type"] == "Clustering"
    assert prediction["target_column"] == ""
    assert prediction["prediction_count"] == 3
    assert len(prediction["predictions"]) == 3

    report = generate_training_report(integration["nlp_resolution"], training)
    assert report["task"] == {"target": None, "problem_type": "Clustering"}
    assert report["model_id"] == training["model"]["model_id"]
    assert report["best_model"]["selection_metric"] == "silhouette_score"
    assert report["warnings"] == {"failed_model_count": 0}
    json.dumps(report)
