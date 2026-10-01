import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.persistence as persistence
from services.automl.nlp_integration import integrate_nlp_with_automl
from services.automl.persistence import load_model_package
from services.automl.predictor import predict_with_model
from services.reporting.training_report import generate_training_report
from services.automl.training import (
    _clustering_score_sample,
    evaluate_models,
    select_best_model,
    train_clustering_models,
)


def _segmentation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annual_spend": [120, 125, 130, 900, 920, 880, 410, 430, 390, 700, 720, 680] * 4,
            "visits_per_month": [2, 3, 2, 15, 16, 14, 8, 9, 7, 12, 13, 11] * 4,
            "avg_order_value": [30, 32, 29, 110, 115, 105, 70, 72, 68, 95, 98, 92] * 4,
            "support_contacts": [5, 4, 6, 1, 2, 1, 3, 4, 3, 2, 1, 2] * 4,
        }
    )


def test_clustering_score_sample_is_deterministic_and_capped():
    dataframe = pd.DataFrame({"value": range(6001)})

    first_sample = _clustering_score_sample(dataframe)
    second_sample = _clustering_score_sample(dataframe)

    assert len(first_sample) == 5000
    pd.testing.assert_frame_equal(first_sample, second_sample)


def test_clustering_evaluation_scores_candidates_on_capped_sample(monkeypatch):
    import services.automl.training as training

    prediction_indices = []

    class LabelModel:
        def __init__(self, cluster_count):
            self.cluster_count = cluster_count

        def predict(self, features):
            prediction_indices.append(features.index.copy())
            return np.arange(len(features)) % self.cluster_count

    scored_sizes = []

    def fake_silhouette_score(features, labels):
        scored_sizes.append(len(features))
        return 0.8 if len(np.unique(labels)) == 3 else 0.2

    monkeypatch.setattr(training, "silhouette_score", fake_silhouette_score)
    dataframe = pd.DataFrame({"value": range(6001)}, index=range(10000, 16001))
    models = {"KMeans_k2": LabelModel(2), "KMeans_k3": LabelModel(3)}

    evaluation = evaluate_models(
        models,
        dataframe,
        pd.Series(index=dataframe.index, dtype=float),
        "Clustering",
    )
    selected = select_best_model(evaluation, "Clustering")

    assert scored_sizes == [5000, 5000]
    expected_indices = _clustering_score_sample(dataframe).index
    assert len(prediction_indices) == len(models)
    assert all(indices.equals(expected_indices) for indices in prediction_indices)
    assert selected["best_model_name"] == "KMeans_k3"
    assert selected["best_metrics"]["silhouette_score"] == 0.8


def test_clustering_candidates_fit_on_full_training_dataset(monkeypatch):
    import services.automl.training as training

    fitted_indices = []

    class CapturingModel:
        def fit(self, features):
            fitted_indices.append(features.index.copy())
            return self

        def predict(self, features):
            return np.arange(len(features)) % 2

    monkeypatch.setattr(training, "make_pipeline", lambda *args: CapturingModel())
    dataframe = pd.DataFrame({"value": range(24)}, index=range(200, 224))

    result = train_clustering_models(dataframe)

    assert result["status"] == "Completed"
    assert result["model_names"]
    assert len(fitted_indices) == len(result["model_names"])
    assert all(indices.equals(dataframe.index) for indices in fitted_indices)


@pytest.mark.parametrize(
    "labels",
    [
        [0] * 12,
        list(range(12)),
    ],
    ids=["single-cluster", "one-cluster-per-row"],
)
def test_clustering_evaluation_records_degenerate_labels_without_crashing(labels):
    class LabelModel:
        def predict(self, features):
            return np.asarray(labels[: len(features)])

    dataframe = pd.DataFrame({"value": range(len(labels))})

    evaluation = evaluate_models(
        {"KMeans_invalid": LabelModel()},
        dataframe,
        pd.Series(index=dataframe.index, dtype=float),
        "Clustering",
    )

    assert evaluation["status"] == "Failed"
    assert evaluation["results"] == []
    assert evaluation["errors"][0]["model_name"] == "KMeans_invalid"
    assert "non-trivial clusters" in evaluation["errors"][0]["error"]


def test_clustering_evaluation_uses_full_silhouette_for_small_datasets(monkeypatch):
    import services.automl.training as training

    class LabelModel:
        def predict(self, features):
            return np.arange(len(features)) % 2

    scored_sizes = []

    def fake_silhouette_score(features, labels):
        scored_sizes.append(len(features))
        return 0.5

    monkeypatch.setattr(training, "silhouette_score", fake_silhouette_score)
    dataframe = pd.DataFrame({"value": range(12)})

    evaluation = evaluate_models(
        {"KMeans_k2": LabelModel()},
        dataframe,
        pd.Series(index=dataframe.index, dtype=float),
        "Clustering",
    )

    assert scored_sizes == [len(dataframe)]
    assert evaluation["results"][0]["metrics"]["silhouette_score"] == 0.5


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
    assert training["models"]["failed"] == []
    assert all(name.startswith("KMeans") for name in training["models"]["trained"])
    assert "DBSCAN" not in training["models"]["trained"]
    assert training["model"]["problem_type"] == "Clustering"
    assert training["model"]["target_column"] is None
    assert training["best_model"]["name"].startswith("KMeans")
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


def test_clustering_excludes_recommended_stroke_target_from_saved_features(tmp_path, monkeypatch):
    dataframe = _segmentation_data()
    dataframe["stroke"] = [0, 1] * (len(dataframe) // 2)
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")

    nlp_result = {
        "ready_for_training": True,
        "needs_clarification": False,
        "dataset_resolution": {
            "target_column": None,
            "problem_type": "Clustering",
            "confidence": 0.0,
            "candidates": [{"column": "stroke", "confidence": 0.91}],
            "needs_clarification": False,
            "reason": "Clustering does not require a target column.",
        },
    }

    integration = integrate_nlp_with_automl(dataframe, nlp_result)
    training = integration["automl_training"]
    package = load_model_package(training["model"]["model_id"])

    assert "stroke" not in training["preprocessing"]["raw_feature_names"]
    assert training["preprocessing"]["target_column"] == "stroke"
    prediction = predict_with_model(
        package,
        dataframe.drop(columns=["stroke"]).iloc[:3].to_dict(orient="records"),
    )
    assert prediction["status"] == "success"
    assert prediction["prediction_count"] == 3
