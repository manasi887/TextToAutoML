import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.persistence as persistence
from services.automl.nlp_integration import integrate_nlp_with_automl
from services.automl.persistence import load_model_package
from services.automl.predictor import predict_with_model
from services.automl.trainer import prepare_clustering_data
from services.automl.clustering_preprocessing import ClusteringPreprocessor
from services.reporting.training_report import generate_training_report
from services.automl.training import (
    _clustering_score_sample,
    evaluate_models,
    select_best_model,
    train_clustering_models,
)
from services.automl.pipeline import run_automl_pipeline


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

    assert len(first_sample) == 3000
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
    monkeypatch.setattr(
        training,
        "calinski_harabasz_score",
        lambda features, labels: 80.0 if len(np.unique(labels)) == 3 else 20.0,
    )
    monkeypatch.setattr(
        training,
        "davies_bouldin_score",
        lambda features, labels: 0.2 if len(np.unique(labels)) == 3 else 0.8,
    )
    dataframe = pd.DataFrame({"value": range(6001)}, index=range(10000, 16001))
    models = {"KMeans_k2": LabelModel(2), "KMeans_k3": LabelModel(3)}

    evaluation = evaluate_models(
        models,
        dataframe,
        pd.Series(index=dataframe.index, dtype=float),
        "Clustering",
    )
    selected = select_best_model(evaluation, "Clustering")

    assert scored_sizes == [3000, 3000]
    expected_indices = _clustering_score_sample(dataframe).index
    assert len(prediction_indices) == len(models)
    assert all(indices.equals(expected_indices) for indices in prediction_indices)
    assert selected["best_model_name"] == "KMeans_k3"
    assert selected["best_metrics"]["silhouette_score"] == 0.8


def test_clustering_silhouette_uses_the_fitted_feature_space(monkeypatch):
    import services.automl.training as training

    dataframe = pd.DataFrame({"value": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]})
    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=2, random_state=42, n_init=10),
    ).fit(dataframe)
    expected_features = model[:-1].transform(dataframe)
    seen_features = []

    def capture_silhouette(features, labels):
        seen_features.append(np.asarray(features))
        return 0.25

    monkeypatch.setattr(training, "silhouette_score", capture_silhouette)
    result = evaluate_models(
        {"KMeans_k2": model},
        dataframe,
        pd.Series(index=dataframe.index, dtype=float),
        "Clustering",
    )

    np.testing.assert_allclose(seen_features[0], expected_features)
    assert result["results"][0]["metrics"]["silhouette_score"] == 0.25


def test_clustering_candidates_fit_on_the_same_capped_evaluation_sample(monkeypatch):
    import services.automl.training as training

    fitted_indices = []

    class CapturingModel:
        def fit(self, features):
            fitted_indices.append(features.index.copy())
            return self

        def predict(self, features):
            return np.arange(len(features)) % 2

    monkeypatch.setattr(training, "KMeans", lambda **kwargs: CapturingModel())
    monkeypatch.setattr(training, "Birch", lambda **kwargs: CapturingModel())
    dataframe = pd.DataFrame({"value": range(24)}, index=range(200, 224))

    result = train_clustering_models(dataframe)

    assert result["status"] == "Completed"
    assert "KMeans_k2" in result["model_names"]
    assert "Birch_k2" in result["model_names"]
    expected_indices = _clustering_score_sample(dataframe).index
    assert all(indices.equals(expected_indices) for indices in fitted_indices)
    assert len(fitted_indices) == 6
    assert result["candidate_sample_size"] == len(expected_indices)


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


def test_nonfinite_silhouette_is_reported_as_an_invalid_candidate(monkeypatch):
    import services.automl.training as training

    class LabelModel:
        def predict(self, features):
            return np.arange(len(features)) % 2

    monkeypatch.setattr(training, "silhouette_score", lambda features, labels: np.nan)
    dataframe = pd.DataFrame({"value": np.arange(12, dtype=float)})
    evaluation = evaluate_models(
        {"invalid_metric": LabelModel()},
        dataframe,
        pd.Series(index=dataframe.index, dtype=float),
        "Clustering",
    )

    assert evaluation["status"] == "Failed"
    assert evaluation["results"] == []
    assert "non-finite metric" in evaluation["errors"][0]["error"]


def test_clustering_nlp_to_persistence_prediction_and_report(tmp_path, monkeypatch):
    import services.automl.run_history as run_history

    dataframe = _segmentation_data()
    dataframe["region"] = ["north", "south", "east", "west"] * 12
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")
    monkeypatch.setattr(run_history, "HISTORY_PATH", tmp_path / "training_history.json")

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
    assert all(
        name.startswith(("KMeans", "Birch"))
        for name in training["models"]["trained"]
    )
    assert training["model"]["problem_type"] == "Clustering"
    assert training["model"]["target_column"] is None
    assert training["target_column"] is None
    assert training["preprocessing"]["target_column"] is None
    assert training["best_model"]["name"].startswith("KMeans")
    assert training["best_model"]["selection_metric"] == "silhouette_score"
    assert training["best_model"]["metrics"]["silhouette_score"] > 0.0

    package = load_model_package(training["model"]["model_id"])
    assert package["target_column"] == ""
    assert package["preprocessing"]["target_column"] is None
    assert package["encoders"] == {}
    assert package["preprocessing"]["clustering_preprocessor"] is not None
    prediction = predict_with_model(package, dataframe.iloc[:3].to_dict(orient="records"))
    assert prediction["status"] == "success"
    assert prediction["problem_type"] == "Clustering"
    assert prediction["target_column"] == ""
    assert prediction["prediction_count"] == 3
    assert len(prediction["predictions"]) == 3
    preparation = prepare_clustering_data(
        dataframe,
        excluded_target_columns=training["preprocessing"]["excluded_target_columns"],
    )
    full_prediction = predict_with_model(
        package,
        dataframe.to_dict(orient="records"),
    )
    assert full_prediction["prediction_count"] == len(dataframe)
    assert full_prediction["predictions"] == package["model"].predict(
        preparation["X"]
    ).tolist()
    assert prediction["predictions"] == package["model"].predict(
        preparation["X"].iloc[:3]
    ).tolist()

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Saved-model prediction must not refit K-Means.")

    monkeypatch.setattr(KMeans, "fit", forbid_fit)
    reused_prediction = predict_with_model(
        load_model_package(training["model"]["model_id"]),
        dataframe.iloc[:3].to_dict(orient="records"),
    )
    assert reused_prediction["predictions"] == prediction["predictions"]

    report = generate_training_report(integration["nlp_resolution"], training)
    assert report["task"] == {"target": None, "problem_type": "Clustering"}
    assert report["model_id"] == training["model"]["model_id"]
    assert report["best_model"]["selection_metric"] == "silhouette_score"
    assert report["warnings"] == {"failed_model_count": 0}
    assert report["features"]["retained"]
    assert report["clustering"]["cluster_distribution"]
    assert report["clustering"]["quality_assessment"]
    json.dumps(report)


def test_clustering_excludes_recommended_stroke_target_from_saved_features(tmp_path, monkeypatch):
    dataframe = _segmentation_data()
    dataframe["stroke"] = [0, 1] * (len(dataframe) // 2)
    dataframe["candidate_label"] = [0, 1] * (len(dataframe) // 2)
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")

    nlp_result = {
        "ready_for_training": True,
        "needs_clarification": False,
        "dataset_resolution": {
            "target_column": None,
            "problem_type": "Clustering",
            "confidence": 0.0,
            "candidates": [
                {"column": "stroke", "confidence": 0.91},
                {"column": "candidate_label", "confidence": 0.82},
            ],
            "needs_clarification": False,
            "reason": "Clustering does not require a target column.",
        },
    }

    integration = integrate_nlp_with_automl(dataframe, nlp_result)
    training = integration["automl_training"]
    package = load_model_package(training["model"]["model_id"])

    assert "stroke" not in training["preprocessing"]["raw_feature_names"]
    assert training["preprocessing"]["target_column"] is None
    assert training["preprocessing"]["excluded_target_columns"] == [
        "stroke",
        "candidate_label",
    ]
    prediction = predict_with_model(
        package,
        dataframe.drop(columns=["stroke"]).iloc[:3].to_dict(orient="records"),
    )
    assert prediction["status"] == "success"
    assert prediction["prediction_count"] == 3


def test_clustering_excludes_outcomes_existing_labels_and_customer_names():
    dataframe = pd.DataFrame({
        "id": range(12),
        "full_name": [f"customer-{index % 4}" for index in range(12)],
        "age": [25, 30, 35, 40] * 3,
        "credit_score": [600, 650, 700, 750] * 3,
        "exit": [0, 1] * 6,
        "customer_segment": ["a", "b", "c"] * 4,
        "risk_score": [0.1, 0.9] * 6,
        "risk_segment": ["low", "high"] * 6,
        "engagement_score": [10, 20, 30, 40] * 3,
        "cluster_group": [1, 2, 3] * 4,
    })

    preparation = prepare_clustering_data(
        dataframe,
        feature_columns=list(dataframe.columns),
    )

    assert preparation["preprocessing"]["raw_feature_names"] == [
        "age",
        "credit_score",
        "risk_score",
        "engagement_score",
    ]
    assert preparation["preprocessing"]["removed_columns"]["identifier_columns"] == [
        "id",
        "full_name",
    ]
    assert preparation["preprocessing"]["removed_columns"]["clustering_label_columns"] == [
        "exit",
        "customer_segment",
        "risk_segment",
        "cluster_group",
    ]
    assert {
        "risk_score",
        "engagement_score",
    }.issubset(preparation["preprocessing"]["raw_feature_names"])


def test_numeric_only_separated_groups_produce_valid_cluster_metrics():
    rng = np.random.RandomState(19)
    dataframe = pd.DataFrame(
        np.vstack(
            (
                rng.normal(loc=(-5, -5), scale=0.25, size=(30, 2)),
                rng.normal(loc=(5, 5), scale=0.25, size=(30, 2)),
            )
        ),
        columns=["measure_a", "measure_b"],
    )
    preparation = prepare_clustering_data(dataframe)
    candidates = train_clustering_models(preparation["X"])
    evaluation = evaluate_models(
        candidates["trained_models"],
        preparation["X"],
        pd.Series(index=preparation["X"].index, dtype=float),
        "Clustering",
    )

    assert candidates["status"] == "Completed"
    assert any(name.startswith("Birch") for name in candidates["model_names"])
    assert evaluation["results"]
    assert all(
        set(
            (
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
            )
        ).issubset(result["metrics"])
        for result in evaluation["results"]
    )


def test_overlapping_numeric_groups_are_evaluated_without_quality_assumptions():
    rng = np.random.RandomState(23)
    dataframe = pd.DataFrame(
        {
            "left": rng.normal(0, 1, 80),
            "right": rng.normal(0, 1, 80),
        }
    )
    preparation = prepare_clustering_data(dataframe)
    candidates = train_clustering_models(preparation["X"])
    evaluation = evaluate_models(
        candidates["trained_models"],
        preparation["X"],
        pd.Series(index=preparation["X"].index, dtype=float),
        "Clustering",
    )

    assert evaluation["results"]
    assert all(
        -1.0 <= result["metrics"]["silhouette_score"] <= 1.0
        for result in evaluation["results"]
    )


def test_mixed_and_categorical_only_features_use_nominal_encodings():
    dataframe = pd.DataFrame(
        {
            "spend": [10.0, 11.0, 80.0, 82.0, 12.0, 79.0],
            "region": ["north", "north", "south", "south", "east", "east"],
            "is_member": [True, False, True, False, True, False],
        }
    )
    mixed = prepare_clustering_data(dataframe)
    categorical = prepare_clustering_data(dataframe.drop(columns=["spend"]))

    assert mixed["preprocessing"]["retained_features"] == [
        {"column": "spend", "type": "numeric"},
        {"column": "region", "type": "categorical"},
        {"column": "is_member", "type": "categorical"},
    ]
    assert all(
        not column.startswith("numeric__region")
        for column in mixed["feature_names"]
    )
    assert categorical["X"].shape[1] > 0
    assert categorical["X"].select_dtypes(include=["object"]).empty


def test_missing_values_unseen_categories_and_extra_columns_are_prediction_safe():
    dataframe = pd.DataFrame(
        {
            "amount": [1.0, 1.5, 9.0, 9.5, np.nan, 8.5],
            "group": ["a", "a", "b", None, "b", "b"],
        }
    )
    preparation = prepare_clustering_data(dataframe)
    model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(preparation["X"])
    package = {
        "model_id": "test-clustering",
        "model": model,
        "encoders": {},
        "feature_names": preparation["feature_names"],
        "target_column": "",
        "problem_type": "Clustering",
        "preprocessing": preparation["preprocessing"],
    }
    records = [
        {"amount": None, "group": "unseen-category", "unrelated": "ignored"},
        {"amount": 9.2},
    ]

    prediction = predict_with_model(package, records)

    assert prediction["prediction_count"] == len(records)
    transformed = preparation["preprocessing"]["clustering_preprocessor"].transform(
        pd.DataFrame(records)
    )
    assert prediction["predictions"] == model.predict(transformed).tolist()


def test_identifier_constants_dates_and_high_cardinality_are_handled_explicitly():
    row_count = 120
    dataframe = pd.DataFrame(
        {
            "record_code": [f"row-{index:04d}" for index in range(row_count)],
            "constant_flag": ["same"] * row_count,
            "created_date": pd.date_range("2021-01-01", periods=row_count).astype(str),
            "event_time": pd.date_range("2022-01-01", periods=row_count).astype(str),
            "postal_code": [f"{index % 20:05d}" for index in range(row_count)],
            "category_code": [f"category-{index % 80}" for index in range(row_count)],
            "amount": np.arange(row_count, dtype=float),
        }
    )
    preparation = prepare_clustering_data(dataframe)
    preprocessor = preparation["preprocessing"]["clustering_preprocessor"]
    excluded = {
        item["column"]: item["reason"]
        for item in preparation["preprocessing"]["excluded_features"]
    }

    assert "record_code" in excluded
    assert excluded["constant_flag"] == "constant column"
    assert preprocessor.feature_types_["created_date"] == "date"
    assert preprocessor.feature_types_["event_time"] == "date"
    assert preprocessor.feature_types_["postal_code"] == "categorical"
    assert preprocessor.feature_types_["category_code"] == "categorical"
    assert sum(
        name.startswith("categorical__category_code")
        for name in preparation["feature_names"]
    ) <= preprocessor.max_categories
    assert np.isfinite(preparation["X"].to_numpy()).all()


def test_target_and_existing_label_exclusions_keep_behavioral_scores():
    dataframe = pd.DataFrame(
        {
            "usage_score": [2, 3, 4, 5, 6, 7],
            "risk_score": [0.2, 0.4, 0.1, 0.8, 0.3, 0.9],
            "target": [0, 1, 0, 1, 0, 1],
            "segment": ["a", "b", "a", "b", "a", "b"],
            "cluster_id": [0, 1, 0, 1, 0, 1],
        }
    )
    preparation = prepare_clustering_data(
        dataframe,
        excluded_target_columns=["target"],
    )

    assert preparation["preprocessing"]["raw_feature_names"] == [
        "usage_score",
        "risk_score",
    ]
    reasons = {
        item["column"]: item["reason"]
        for item in preparation["preprocessing"]["excluded_features"]
    }
    assert reasons["segment"] == "target, outcome, segment, or existing cluster label"
    assert reasons["cluster_id"] == "target, outcome, segment, or existing cluster label"
    assert {
        item["column"]: item["reason"]
        for item in preparation["preprocessing"]["excluded_features"]
    }["target"] == "explicitly supplied target or group exclusion"


def test_outlier_and_unequal_cluster_sizes_remain_finite_and_reported():
    dataframe = pd.DataFrame(
        {
            "value": [0.0] * 35 + [10.0] * 8 + [10000.0],
            "activity": [1.0] * 35 + [7.0] * 8 + [50.0],
        }
    )
    preparation = prepare_clustering_data(dataframe)
    result = train_clustering_models(preparation["X"])
    evaluation = evaluate_models(
        result["trained_models"],
        preparation["X"],
        pd.Series(index=preparation["X"].index, dtype=float),
        "Clustering",
    )

    assert np.isfinite(preparation["X"].to_numpy()).all()
    assert evaluation["results"]
    assert all(
        result["small_cluster_count"] >= 0
        for result in evaluation["results"]
    )


def test_few_distinct_samples_fail_with_clear_clustering_error():
    dataframe = pd.DataFrame({"same": [1.0, 1.0]})

    with pytest.raises(ValueError, match="No usable features remain"):
        prepare_clustering_data(dataframe)


def test_candidate_evaluation_is_deterministic_and_reports_nonpositive_silhouette(
    tmp_path,
    monkeypatch,
):
    import services.automl.training as training

    dataframe = _segmentation_data().iloc[:12].copy()
    preparation = prepare_clustering_data(dataframe)
    first = train_clustering_models(preparation["X"])
    second = train_clustering_models(preparation["X"])
    assert first["model_names"] == second["model_names"]
    first_evaluation = evaluate_models(
        first["trained_models"],
        preparation["X"],
        pd.Series(index=preparation["X"].index, dtype=float),
        "Clustering",
    )
    second_evaluation = evaluate_models(
        second["trained_models"],
        preparation["X"],
        pd.Series(index=preparation["X"].index, dtype=float),
        "Clustering",
    )
    assert [
        result["metrics"] for result in first_evaluation["results"]
    ] == [
        result["metrics"] for result in second_evaluation["results"]
    ]

    monkeypatch.setattr(training, "silhouette_score", lambda features, labels: -0.1)
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")
    result = run_automl_pipeline(dataframe, None, "Clustering")
    report = generate_training_report(
        {"problem_type": "Clustering", "target_column": None},
        result,
    )

    assert result["best_model"]["metrics"]["silhouette_score"] == -0.1
    assert "No positive silhouette separation" in report["clustering"]["quality_assessment"]
    assert preparation["preprocessing"]["target_column"] is None
