import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.persistence as persistence
from services.automl.pipeline import run_automl_pipeline
from services.automl.persistence import load_model_package
from services.automl.predictor import predict_with_model
from services.automl.trainer import (
    prepare_training_data,
    split_dataset_train_validation_test,
    transform_training_features,
)
from services.automl.training import (
    evaluate_models,
    select_best_model,
    train_models as baseline_train_models,
    tune_selected_model,
)


def _build_regression_dataset() -> pd.DataFrame:
    X, y = make_regression(
        n_samples=250,
        n_features=8,
        n_informative=5,
        noise=8.0,
        random_state=42,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def _build_binary_dataset() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def _build_multiclass_dataset() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def test_train_validation_test_split_is_reproducible_and_fits_preprocessing_on_train_only():
    rng = np.random.RandomState(21)
    measurement = rng.choice([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], size=120).astype(float)
    measurement[::9] = np.nan
    dataframe = pd.DataFrame(
        {
            "row_key": [f"row-{index}" for index in range(120)],
            "measurement": measurement,
            "category": [f"group-{index % 3}" for index in range(120)],
            "target": rng.normal(size=120),
        }
    )

    train, validation, test = split_dataset_train_validation_test(
        dataframe, "target", problem_type="Regression", random_state=42
    )
    repeat = split_dataset_train_validation_test(
        dataframe, "target", problem_type="Regression", random_state=42
    )

    assert (len(train), len(validation), len(test)) == (72, 24, 24)
    assert set(train["row_key"]).isdisjoint(validation["row_key"])
    assert set(train["row_key"]).isdisjoint(test["row_key"])
    assert set(validation["row_key"]).isdisjoint(test["row_key"])
    for first, second in zip((train, validation, test), repeat):
        assert first["row_key"].tolist() == second["row_key"].tolist()

    preparation = prepare_training_data(train, "target")
    training_mean = float(train["measurement"].mean())
    assert preparation["preprocessing"]["imputation_values"]["numeric"]["measurement"] == pytest.approx(training_mean)
    validation_features = transform_training_features(
        validation.drop(columns="target"), preparation
    )
    assert validation_features.columns.tolist() == preparation["feature_names"]
    assert len(validation_features) == len(validation)


def test_classification_train_validation_test_splits_preserve_imbalanced_proportions():
    rng = np.random.RandomState(44)
    dataframe = pd.DataFrame(
        {
            "feature": rng.normal(size=200),
            "target": [0] * 190 + [1] * 10,
        }
    )

    train, validation, test = split_dataset_train_validation_test(
        dataframe,
        "target",
        problem_type="Binary Classification",
        random_state=19,
    )

    assert train["target"].value_counts().to_dict() == {0: 114, 1: 6}
    assert validation["target"].value_counts().to_dict() == {0: 38, 1: 2}
    assert test["target"].value_counts().to_dict() == {0: 38, 1: 2}


@pytest.mark.parametrize("problem_type", ["Binary Classification", "Regression"])
def test_exact_duplicate_rows_are_removed_before_train_validation_test_splitting(problem_type):
    dataframe = pd.DataFrame(
        {
            "feature": list(range(20)),
            "target": [index % 2 for index in range(20)]
            if problem_type == "Binary Classification"
            else [float(index) for index in range(20)],
        }
    )
    dataframe = pd.concat(
        [dataframe, pd.concat([dataframe.iloc[[7]]] * 8)],
        ignore_index=True,
    )

    train, validation, test = split_dataset_train_validation_test(
        dataframe,
        "target",
        problem_type=problem_type,
        random_state=42,
    )

    partitions = (train, validation, test)
    duplicate_occurrences = sum(
        int((partition["feature"] == 7).sum())
        for partition in partitions
    )
    duplicate_partitions = sum(
        bool((partition["feature"] == 7).any())
        for partition in partitions
    )

    assert sum(map(len, partitions)) == len(dataframe.drop_duplicates())
    assert duplicate_occurrences == 1
    assert duplicate_partitions == 1


def test_group_aware_split_remains_disjoint_with_exact_duplicates():
    rows = [
        {"group": f"g-{group_index}", "feature": row_index % 4, "target": row_index % 2}
        for group_index in range(12)
        for row_index in range(8)
    ]
    rows.extend([rows[0].copy(), rows[0].copy()])
    dataframe = pd.DataFrame(rows)

    train, validation, test = split_dataset_train_validation_test(
        dataframe,
        "target",
        problem_type="Binary Classification",
        random_state=42,
        group_column="group",
    )

    group_sets = [set(partition["group"]) for partition in (train, validation, test)]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert sum(map(len, (train, validation, test))) == len(dataframe.drop_duplicates())


@pytest.mark.parametrize("problem_type", ["Binary Classification", "Regression"])
def test_grouped_train_validation_test_split_has_disjoint_groups_and_excludes_group_feature(problem_type):
    rows = []
    for group_index in range(20):
        for row_index in range(6):
            target = row_index % 2 if problem_type == "Binary Classification" else group_index * 3 + row_index
            rows.append({"group": f"g-{group_index}", "feature": group_index + row_index / 10, "target": target})
    dataframe = pd.DataFrame(rows)

    train, validation, test = split_dataset_train_validation_test(
        dataframe,
        "target",
        problem_type=problem_type,
        random_state=42,
        group_column="group",
    )

    group_sets = [set(partition["group"]) for partition in (train, validation, test)]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    preparation = prepare_training_data(train, "target", group_column="group")
    assert "group" not in preparation["feature_names"]
    assert preparation["preprocessing"]["removed_columns"]["group_columns"] == ["group"]


def test_grouped_split_reports_insufficient_groups():
    dataframe = pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "feature": [1, 2, 3, 4], "target": [1.0, 2.0, 3.0, 4.0]}
    )

    with pytest.raises(ValueError, match="at least 3 distinct groups"):
        split_dataset_train_validation_test(
            dataframe,
            "target",
            problem_type="Regression",
            group_column="group",
        )


@pytest.mark.parametrize("problem_type", ["Binary Classification", "Regression"])
def test_grouped_cross_validation_keeps_groups_out_of_validation_folds(problem_type, monkeypatch):
    import services.automl.training as training_module

    rows = []
    for group_index in range(12):
        for row_index in range(10):
            target = row_index % 2 if problem_type == "Binary Classification" else group_index + row_index / 10
            rows.append({"group": f"g-{group_index}", "feature": row_index % 5, "target": target})
    dataframe = pd.DataFrame(rows)
    model_name = "LogisticRegression" if problem_type == "Binary Classification" else "LinearRegression"
    model = LogisticRegression(max_iter=500) if problem_type == "Binary Classification" else LinearRegression()
    training_groups = []
    validation_groups = []
    original_transform = training_module.transform_training_features

    def record_transform(features, preparation):
        validation_groups.append(set(features["group"]))
        return original_transform(features, preparation)

    def prepare_fold(fold_dataframe, target_column):
        training_groups.append(set(fold_dataframe["group"]))
        return prepare_training_data(fold_dataframe, target_column, group_column="group")

    monkeypatch.setattr(training_module, "transform_training_features", record_transform)
    result = tune_selected_model(
        model_name,
        model,
        pd.DataFrame(),
        dataframe["target"],
        problem_type,
        raw_training_df=dataframe,
        target_column="target",
        preparation_function=prepare_fold,
        group_column="group",
    )

    assert result["status"] == "Completed"
    assert len(training_groups) == len(validation_groups)
    assert all(train_groups.isdisjoint(valid_groups) for train_groups, valid_groups in zip(training_groups, validation_groups))


def test_grouped_cross_validation_reports_insufficient_groups():
    dataframe = pd.DataFrame(
        {"group": ["a"] * 8, "feature": np.arange(8), "target": [0, 1] * 4}
    )

    result = tune_selected_model(
        "LogisticRegression",
        baseline_train_models(
            dataframe[["feature"]], dataframe["target"], "Binary Classification"
        )["trained_models"]["LogisticRegression"],
        dataframe[["feature"]],
        dataframe["target"],
        "Binary Classification",
        raw_training_df=dataframe,
        target_column="target",
        group_column="group",
    )

    assert result["status"] == "Failed"
    assert "Insufficient groups" in result["errors"][0]["error"]


def test_majority_only_classifier_reports_minority_performance_and_confusion_matrix():
    class MajorityOnlyClassifier:
        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    labels = pd.Series([0] * 90 + [1] * 10)
    evaluation = evaluate_models(
        {"MajorityOnly": MajorityOnlyClassifier()},
        pd.DataFrame({"feature": np.arange(len(labels))}),
        labels,
        "Binary Classification",
    )
    metrics = evaluation["results"][0]["metrics"]

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["precision"] == pytest.approx(0.81)
    assert metrics["recall"] == pytest.approx(0.9)
    assert metrics["f1_score"] == pytest.approx(0.8526315789)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1_score"] < 0.5
    assert metrics["confusion_matrix_labels"] == ["0", "1"]
    assert metrics["confusion_matrix"] == [[90, 0], [10, 0]]
    assert metrics["per_class_metrics"]["1"]["recall"] == 0.0


def test_regression_tuning_uses_reproducible_training_cv():
    X, y = make_regression(n_samples=72, n_features=4, noise=4.0, random_state=17)
    features = pd.DataFrame(X, columns=[f"feature_{index}" for index in range(X.shape[1])])
    target = pd.Series(y)

    first = tune_selected_model(
        "LinearRegression", LinearRegression(), features, target, "Regression"
    )
    second = tune_selected_model(
        "LinearRegression", LinearRegression(), features, target, "Regression"
    )

    assert first["status"] == "Completed"
    assert first["parameters"] in ({"fit_intercept": True}, {"fit_intercept": False})
    assert first["metrics"]["rmse"] == pytest.approx(second["metrics"]["rmse"])


def test_regression_model_selection_minimizes_rmse():
    lower_rmse_model = LinearRegression()
    higher_r2_model = object()
    selection = select_best_model(
        [
            {
                "model_name": "LowerRMSE",
                "model": lower_rmse_model,
                "metrics": {"rmse": 10.0, "mae": 7.0, "r2": 0.80},
            },
            {
                "model_name": "HigherR2",
                "model": higher_r2_model,
                "metrics": {"rmse": 12.0, "mae": 6.0, "r2": 0.95},
            },
        ],
        "Regression",
    )

    assert selection["selection_metric"] == "rmse"
    assert selection["best_model_name"] == "LowerRMSE"
    assert selection["best_model"] is lower_rmse_model


def test_regression_pipeline_success(tmp_path, monkeypatch):
    df = _build_regression_dataset()
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")

    result = run_automl_pipeline(df, target_column="target", problem_type="Regression")

    assert result["status"] == "success"
    assert result["target_column"] == "target"
    assert result["problem_type"] == "Regression"
    assert result["data"]["training_rows"] > 0
    assert result["data"]["test_rows"] > 0
    assert len(result["models"]["trained"]) > 0
    assert result["evaluation"]["status"] == "Completed"
    assert result["best_model"]["selection_metric"] == "rmse"
    assert set(result["best_model"]["metrics"]) == {"rmse", "mae", "r2"}
    assert result["best_model"]["metrics"] == result["evaluation"]["results"][0]["metrics"]
    assert result["best_model"]["name"] == result["evaluation"]["results"][0]["model_name"]
    assert result["data"]["validation_rows"] > 0
    assert result["data"]["test_rows"] > 0
    assert (
        result["data"]["training_rows"]
        + result["data"]["validation_rows"]
        + result["data"]["test_rows"]
        == result["data"]["original_rows"]
    )
    assert result["data"]["final_fit_rows"] == (
        result["data"]["training_rows"] + result["data"]["validation_rows"]
    )
    assert result["best_model"]["validation_metrics"]["rmse"] >= 0
    assert result["best_model"]["cross_validation_metrics"]["rmse"] >= 0
    assert result["best_model"]["cross_validation_metrics"]["mae"] >= 0
    assert "r2" in result["best_model"]["cross_validation_metrics"]

    package = load_model_package(result["model"]["model_id"])
    assert package["model_name"] == result["best_model"]["name"]
    assert package["selection_metric"] == "rmse"
    assert package["metrics"] == result["best_model"]["metrics"]
    prediction = predict_with_model(
        package,
        df.drop(columns=["target"]).iloc[:3].to_dict(orient="records"),
    )
    assert prediction["status"] == "success"
    assert prediction["prediction_count"] == 3


def test_pipeline_reports_exact_duplicate_rows_removed(tmp_path, monkeypatch):
    dataframe = _build_binary_dataset()
    dataframe = pd.concat(
        [dataframe, dataframe.iloc[[5, 17, 29]]],
        ignore_index=True,
    )
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")

    result = run_automl_pipeline(
        dataframe,
        target_column="target",
        problem_type="Binary Classification",
    )

    assert result["status"] == "success"
    assert result["data"]["duplicate_rows_removed"] == 3
    assert (
        result["data"]["training_rows"]
        + result["data"]["validation_rows"]
        + result["data"]["test_rows"]
        == result["data"]["original_rows"] - 3
    )


def test_regression_holdout_evaluation_is_reproducible():
    X, y = make_regression(n_samples=80, n_features=4, noise=5.0, random_state=12)
    features = pd.DataFrame(X, columns=[f"feature_{index}" for index in range(X.shape[1])])
    target = pd.Series(y)
    model = LinearRegression().fit(features.iloc[:60], target.iloc[:60])
    test_features = features.iloc[60:].reset_index(drop=True)
    test_target = target.iloc[60:].reset_index(drop=True)

    first = evaluate_models({"LinearRegression": model}, test_features, test_target, "Regression")
    second = evaluate_models({"LinearRegression": model}, test_features, test_target, "Regression")

    assert first["results"][0]["metrics"] == second["results"][0]["metrics"]


def test_regression_evaluation_calculates_rmse_mae_and_r2_correctly():
    class FixedPredictionModel:
        def predict(self, features):
            return np.array([2.0, 2.0, 6.0])

    evaluation = evaluate_models(
        {"fixed": FixedPredictionModel()},
        pd.DataFrame({"feature": [0.0, 1.0, 2.0]}),
        pd.Series([1.0, 3.0, 5.0]),
        "Regression",
    )

    assert evaluation["results"][0]["metrics"] == {
        "mae": 1.0,
        "rmse": 1.0,
        "r2": 0.625,
    }


def test_tuning_fits_preprocessing_only_on_each_training_fold(monkeypatch):
    rng = np.random.RandomState(31)
    dataframe = pd.DataFrame(
        {
            "row_key": [f"row-{index}" for index in range(60)],
            "feature": rng.choice([-2.0, -1.0, 1.0, 2.0], size=60),
            "target": rng.normal(size=60),
        }
    )
    import services.automl.training as training_module

    original_prepare = training_module.prepare_training_data
    fitted_row_keys = []

    def record_fit_rows(fold_dataframe, target_column):
        fitted_row_keys.append(set(fold_dataframe["row_key"]))
        return original_prepare(fold_dataframe, target_column)

    monkeypatch.setattr(training_module, "prepare_training_data", record_fit_rows)
    result = tune_selected_model(
        "LinearRegression",
        LinearRegression(),
        pd.DataFrame(),
        dataframe["target"],
        "Regression",
        raw_training_df=dataframe,
        target_column="target",
    )

    all_row_keys = set(dataframe["row_key"])
    assert result["status"] == "Completed"
    assert len(fitted_row_keys) == 6
    assert all(0 < len(keys) < len(all_row_keys) for keys in fitted_row_keys)
    assert all(keys <= all_row_keys for keys in fitted_row_keys)


def test_binary_classification_pipeline_success():
    df = _build_binary_dataset()

    result = run_automl_pipeline(df, target_column="target", problem_type="Binary Classification")

    assert result["status"] == "success"
    assert result["problem_type"] == "Binary Classification"
    assert len(result["models"]["trained"]) > 0
    assert result["evaluation"]["status"] == "Completed"
    assert result["best_model"]["selection_metric"] == "f1_score"
    assert "f1_score" in result["best_model"]["metrics"]


def test_multiclass_classification_pipeline_success():
    df = _build_multiclass_dataset()

    result = run_automl_pipeline(df, target_column="target", problem_type="Multi-class Classification")

    assert result["status"] == "success"
    assert result["problem_type"] == "Multi-class Classification"
    assert len(result["models"]["trained"]) > 0
    assert result["evaluation"]["status"] == "Completed"
    assert result["best_model"]["selection_metric"] == "f1_score"
    assert "f1_score" in result["best_model"]["metrics"]


def test_missing_target_raises_value_error():
    df = _build_binary_dataset()

    with pytest.raises(ValueError, match="Target column"):
        run_automl_pipeline(df, target_column="missing_target")


def test_unsupported_problem_type_raises_validation_error():
    df = _build_binary_dataset()

    with pytest.raises(ValueError, match="Unsupported problem type"):
        run_automl_pipeline(df, target_column="target", problem_type="Unsupported Task")


def test_all_model_failure_returns_controlled_failure(monkeypatch):
    df = _build_binary_dataset()

    def fake_train_models(*args, **kwargs):
        return {
            "trained_models": {},
            "model_names": [],
            "status": "Failed",
            "training_errors": [{"model_name": "synthetic", "error": "forced failure"}],
        }

    monkeypatch.setattr("services.automl.pipeline.train_models", fake_train_models)

    result = run_automl_pipeline(df, target_column="target", problem_type="Binary Classification")

    assert result["status"] == "failed"
    assert "All candidate models failed" in result["error"]
    assert result["models"]["trained"] == []


def test_pipeline_passes_high_confidence_recommendation_to_supervised_training():
    df = _build_binary_dataset()
    recommendation = {
        "status": "recommended",
        "recommendations": [{"model_family": "RandomForestClassifier"}],
    }

    with patch(
        "services.automl.pipeline.extract_meta_features",
        return_value={"problem_type": "Binary Classification", "row_count": len(df)},
    ), patch(
        "services.automl.pipeline.recommend_models",
        return_value=recommendation,
    ), patch(
        "services.automl.pipeline.train_models",
        wraps=baseline_train_models,
    ) as train_mock:
        result = run_automl_pipeline(
            df,
            target_column="target",
            problem_type="Binary Classification",
        )

    assert result["status"] == "success"
    assert train_mock.call_args.kwargs["recommended_model_order"] == [
        "RandomForestClassifier"
    ]


def test_recommendation_does_not_override_measured_regression_validation(monkeypatch):
    rng = np.random.RandomState(27)
    signal = rng.normal(size=120)
    dataframe = pd.DataFrame(
        {
            "signal": signal,
            "noise_feature": rng.normal(size=120),
            "target": 3.0 * signal + rng.normal(scale=0.1, size=120),
        }
    )
    recommended_model = DummyRegressor(strategy="constant", constant=10_000.0)
    measured_model = LinearRegression()
    captured_validation_results = []
    original_evaluate_models = __import__(
        "services.automl.pipeline", fromlist=["evaluate_models"]
    ).evaluate_models

    def record_evaluation(models, X_test, y_test, problem_type):
        evaluation = original_evaluate_models(models, X_test, y_test, problem_type)
        if len(models) > 1:
            captured_validation_results.extend(evaluation["results"])
        return evaluation

    with patch(
        "services.automl.pipeline.extract_meta_features",
        return_value={"problem_type": "Regression"},
    ), patch(
        "services.automl.pipeline.load_cached_openml_records",
        return_value=[],
    ), patch(
        "services.automl.pipeline.recommend_models",
        return_value={
            "status": "recommended",
            "recommendations": [{"model_family": "RandomForestRegressor"}],
        },
    ), patch(
        "services.automl.pipeline.train_models",
        return_value={
            "trained_models": {
                "RandomForestRegressor": recommended_model,
                "LinearRegression": measured_model,
            },
            "model_names": ["RandomForestRegressor", "LinearRegression"],
            "status": "Completed",
            "training_errors": [],
        },
    ) as train_mock, patch(
        "services.automl.pipeline.tune_selected_model",
        side_effect=lambda model_name, model, *args, **kwargs: {
            "model": model,
            "status": "Skipped",
            "errors": [],
        },
    ), patch(
        "services.automl.pipeline.evaluate_models",
        side_effect=record_evaluation,
    ):
        result = run_automl_pipeline(
            dataframe,
            target_column="target",
            problem_type="Regression",
        )

    assert train_mock.call_args.kwargs["recommended_model_order"] == [
        "RandomForestRegressor"
    ]
    expected = min(
        captured_validation_results,
        key=lambda item: item["metrics"]["rmse"],
    )
    assert result["best_model"]["name"] == expected["model_name"]
    assert result["best_model"]["name"] == "LinearRegression"
    assert result["best_model"]["name"] != "RandomForestRegressor"


def test_pipeline_passes_cached_openml_records_to_recommender(monkeypatch):
    df = _build_binary_dataset()
    openml_records = [{
        "source": "openml",
        "dataset_id": "61",
        "problem_type": "Binary Classification",
        "best_model": "RandomForestClassifier",
        "selection_metric": "f1_score",
        "metric_value": 0.9,
        "meta_features": {"problem_type": "Binary Classification"},
    }]
    recommendation = {
        "status": "recommended",
        "recommendations": [{"model_family": "RandomForestClassifier"}],
    }

    with patch(
        "services.automl.pipeline.load_cached_openml_records",
        return_value=openml_records,
    ) as records_mock, patch(
        "services.automl.pipeline.recommend_models",
        return_value=recommendation,
    ) as recommend_mock:
        result = run_automl_pipeline(
            df,
            target_column="target",
            problem_type="Binary Classification",
        )

    assert result["status"] == "success"
    records_mock.assert_called_once_with()
    assert recommend_mock.call_args.kwargs["openml_records"] == openml_records


def test_pipeline_uses_local_recommendation_when_cached_openml_records_are_empty(monkeypatch):
    df = _build_binary_dataset()
    recommendation = {"status": "low_confidence", "recommendations": []}

    with patch(
        "services.automl.pipeline.load_cached_openml_records",
        return_value=[],
    ), patch(
        "services.automl.pipeline.recommend_models",
        return_value=recommendation,
    ) as recommend_mock:
        result = run_automl_pipeline(
            df,
            target_column="target",
            problem_type="Binary Classification",
        )

    assert result["status"] == "success"
    assert recommend_mock.call_args.kwargs["openml_records"] == []


def test_pipeline_does_not_fetch_openml_during_training(monkeypatch):
    df = _build_binary_dataset()

    with patch(
        "services.automl.pipeline.load_cached_openml_records",
        return_value=[],
    ), patch(
        "services.metalearning.openml_loader.fetch_openml_task",
        side_effect=AssertionError("network fetch must not run during training"),
    ), patch(
        "services.metalearning.openml_loader.fetch_openml_task_runs",
        side_effect=AssertionError("network fetch must not run during training"),
    ), patch(
        "services.metalearning.openml_loader.fetch_openml_task_evaluations",
        side_effect=AssertionError("network fetch must not run during training"),
    ):
        result = run_automl_pipeline(
            df,
            target_column="target",
            problem_type="Binary Classification",
        )

    assert result["status"] == "success"


def test_pipeline_uses_fixed_order_when_recommendation_is_low_confidence():
    df = _build_binary_dataset()

    with patch(
        "services.automl.pipeline.extract_meta_features",
        return_value={"problem_type": "Binary Classification"},
    ), patch(
        "services.automl.pipeline.recommend_models",
        return_value={"status": "low_confidence", "recommendations": []},
    ), patch(
        "services.automl.pipeline.train_models",
        wraps=baseline_train_models,
    ) as train_mock:
        result = run_automl_pipeline(
            df,
            target_column="target",
            problem_type="Binary Classification",
        )

    assert result["status"] == "success"
    assert train_mock.call_args.kwargs["recommended_model_order"] is None
