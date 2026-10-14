import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.automl.persistence as persistence
import services.automl.pipeline as automl_pipeline
from services.automl.pipeline import (
    _build_validation_diagnostics,
    run_automl_pipeline,
)
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


def test_balanced_classification_keeps_weighted_f1_selection():
    selection = select_best_model(
        [
            {
                "model_name": "WeightedWinner",
                "model": object(),
                "metrics": {"f1_score": 0.90, "balanced_accuracy": 0.70},
            },
            {
                "model_name": "BalancedWinner",
                "model": object(),
                "metrics": {"f1_score": 0.82, "balanced_accuracy": 0.95},
            },
        ],
        "Binary Classification",
        training_target=pd.Series([0] * 60 + [1] * 40),
    )

    assert selection["selection_metric"] == "f1_score"
    assert selection["best_model_name"] == "WeightedWinner"


def test_imbalanced_classification_uses_balanced_accuracy_and_rejects_majority_only_winner():
    selection = select_best_model(
        [
            {
                "model_name": "MajorityOnly",
                "model": object(),
                "metrics": {"f1_score": 0.8526, "balanced_accuracy": 0.50},
            },
            {
                "model_name": "MinorityAware",
                "model": object(),
                "metrics": {"f1_score": 0.80, "balanced_accuracy": 0.85},
            },
        ],
        "Binary Classification",
        training_target=pd.Series([0] * 80 + [1] * 40),
    )

    assert selection["selection_metric"] == "balanced_accuracy"
    assert selection["best_model_name"] == "MinorityAware"


def test_imbalanced_selection_ignores_missing_and_non_finite_metrics():
    selection = select_best_model(
        [
            {
                "model_name": "MissingMetric",
                "model": object(),
                "metrics": {"f1_score": 0.99},
            },
            {
                "model_name": "NonFiniteMetric",
                "model": object(),
                "metrics": {"balanced_accuracy": float("nan")},
            },
            {
                "model_name": "ValidMetric",
                "model": object(),
                "metrics": {"balanced_accuracy": 0.65},
            },
        ],
        "Multi-class Classification",
        training_target=pd.Series(["common"] * 90 + ["rare"] * 10),
    )

    assert selection["selection_metric"] == "balanced_accuracy"
    assert selection["best_model_name"] == "ValidMetric"

    failed = select_best_model(
        [{"model_name": "NoValidMetric", "model": object(), "metrics": {"balanced_accuracy": None}}],
        "Binary Classification",
        training_target=pd.Series([0] * 90 + [1] * 10),
    )
    assert failed["status"] == "Failed"
    assert failed["selection_metric"] == "balanced_accuracy"


def test_equal_selection_scores_break_ties_by_model_name():
    selection = select_best_model(
        [
            {"model_name": "Zeta", "model": object(), "metrics": {"balanced_accuracy": 0.75}},
            {"model_name": "Alpha", "model": object(), "metrics": {"balanced_accuracy": 0.75}},
        ],
        "Binary Classification",
        training_target=pd.Series([0] * 80 + [1] * 20),
    )

    assert selection["best_model_name"] == "Alpha"


def test_pipeline_imbalance_detection_uses_training_partition_target_only(monkeypatch):
    training_partition = pd.DataFrame(
        {"feature": range(100), "target": [0] * 80 + [1] * 20}
    )
    validation_partition = pd.DataFrame(
        {"feature": range(100, 120), "target": [0] * 10 + [1] * 10}
    )
    holdout_partition = pd.DataFrame(
        {"feature": range(120, 140), "target": [0] * 2 + [1] * 18}
    )
    dataframe = pd.concat(
        [training_partition, validation_partition, holdout_partition],
        ignore_index=True,
    )
    captured = {}

    class SelectionReached(Exception):
        pass

    def capture_selection(results, problem_type, *, training_target=None):
        captured["problem_type"] = problem_type
        captured["training_target"] = training_target.copy()
        raise SelectionReached

    monkeypatch.setattr(
        automl_pipeline,
        "split_dataset_train_validation_test",
        lambda *args, **kwargs: (
            training_partition,
            validation_partition,
            holdout_partition,
        ),
    )
    monkeypatch.setattr(
        automl_pipeline,
        "_prepare_training_features",
        lambda train_df, *args, **kwargs: {
            "X": train_df[["feature"]].reset_index(drop=True),
            "y": train_df["target"].reset_index(drop=True),
            "feature_names": ["feature"],
            "encoders": {},
            "preprocessing": {"raw_feature_names": ["feature"]},
        },
    )
    monkeypatch.setattr(
        automl_pipeline,
        "train_models",
        lambda *args, **kwargs: {
            "trained_models": {"DecisionTreeClassifier": DecisionTreeClassifier()},
            "training_errors": [],
        },
    )
    monkeypatch.setattr(
        automl_pipeline,
        "tune_selected_model",
        lambda model_name, model, *args, **kwargs: {
            "model": model,
            "status": "Skipped",
            "errors": [],
        },
    )
    monkeypatch.setattr(
        automl_pipeline,
        "evaluate_models",
        lambda *args, **kwargs: {
            "results": [{
                "model_name": "DecisionTreeClassifier",
                "model": DecisionTreeClassifier(),
                "metrics": {"f1_score": 0.8, "balanced_accuracy": 0.7},
            }],
            "errors": [],
        },
    )
    monkeypatch.setattr(automl_pipeline, "extract_meta_features", lambda *args, **kwargs: {})
    monkeypatch.setattr(automl_pipeline, "select_best_model", capture_selection)

    with pytest.raises(SelectionReached):
        run_automl_pipeline(
            dataframe,
            target_column="target",
            problem_type="Binary Classification",
        )

    assert captured["problem_type"] == "Binary Classification"
    assert captured["training_target"].value_counts().to_dict() == {0: 80, 1: 20}
    assert not captured["training_target"].equals(holdout_partition["target"])


def test_regression_and_clustering_selection_ignore_classification_policy():
    regression = select_best_model(
        [
            {"model_name": "HighRMSE", "model": object(), "metrics": {"rmse": 2.0}},
            {"model_name": "LowRMSE", "model": object(), "metrics": {"rmse": 1.0}},
        ],
        "Regression",
        training_target=pd.Series([0] * 90 + [1] * 10),
    )
    clustering = select_best_model(
        [
            {"model_name": "LowSilhouette", "model": object(), "metrics": {"silhouette_score": 0.2}},
            {"model_name": "HighSilhouette", "model": object(), "metrics": {"silhouette_score": 0.8}},
        ],
        "Clustering",
        training_target=pd.Series([0] * 90 + [1] * 10),
    )

    assert regression["selection_metric"] == "rmse"
    assert regression["best_model_name"] == "LowRMSE"
    assert clustering["selection_metric"] == "silhouette_score"
    assert clustering["best_model_name"] == "HighSilhouette"


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


def test_selected_features_define_persisted_and_prediction_schema(tmp_path, monkeypatch):
    dataframe = pd.DataFrame(
        [
            {
                "group": f"g-{group_index}",
                "selected_category": ["a", "b", "c", "d"][row_index % 4],
                "unselected_feature": group_index * 100 + row_index,
                "target": row_index % 2,
            }
            for group_index in range(20)
            for row_index in range(8)
        ]
    )
    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")

    result = run_automl_pipeline(
        dataframe,
        target_column="target",
        problem_type="Binary Classification",
        group_column="group",
        feature_columns=["selected_category", "target", "group"],
    )

    assert result["status"] == "success"
    assert result["preprocessing"]["raw_feature_names"] == ["selected_category"]
    package = load_model_package(result["model"]["model_id"])
    assert package["preprocessing"]["raw_feature_names"] == ["selected_category"]
    prediction = predict_with_model(
        package,
        [{"selected_category": "a", "unselected_feature": 9999}],
    )
    assert prediction["status"] == "success"
    assert prediction["prediction_count"] == 1


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


def test_bayesian_tuning_receives_only_outer_training_rows(monkeypatch, tmp_path):
    dataframe = _build_binary_dataset()
    dataframe["row_marker"] = [f"row-{index}" for index in range(len(dataframe))]
    expected_train, expected_validation, expected_test = split_dataset_train_validation_test(
        dataframe,
        "target",
        problem_type="Binary Classification",
        random_state=42,
    )
    captured = []

    def capture_tuning(model_name, model, X_train, y_train, problem_type, **kwargs):
        captured.append({"raw_training_df": kwargs["raw_training_df"].copy(), "kwargs": kwargs})
        return {
            "model": model,
            "status": "Fallback",
            "strategy": kwargs.get("tuning_strategy", "grid"),
            "errors": [],
        }

    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")
    monkeypatch.setattr("services.automl.pipeline.tune_selected_model", capture_tuning)
    result = run_automl_pipeline(
        dataframe,
        target_column="target",
        problem_type="Binary Classification",
        tuning_strategy="bayesian",
        tuning_trials=2,
    )

    assert result["status"] == "success"
    assert result["tuning"]["strategy"] == "bayesian"
    assert result["tuning"]["trials_per_model"] == 2
    assert len(captured) == 3
    training_markers = set(expected_train["row_marker"])
    validation_markers = set(expected_validation["row_marker"])
    test_markers = set(expected_test["row_marker"])
    for entry in captured:
        received_markers = set(entry["raw_training_df"]["row_marker"])
        assert received_markers == training_markers
        assert received_markers.isdisjoint(validation_markers)
        assert received_markers.isdisjoint(test_markers)
        assert entry["kwargs"]["tuning_strategy"] == "bayesian"
        assert entry["kwargs"]["n_trials"] == 2


def test_auto_policy_uses_only_outer_training_metadata_and_none_keeps_baseline(tmp_path, monkeypatch):
    dataframe = _build_binary_dataset()
    dataframe["row_marker"] = [f"row-{index}" for index in range(len(dataframe))]
    expected_train, _, _ = split_dataset_train_validation_test(
        dataframe,
        "target",
        problem_type="Binary Classification",
        random_state=42,
    )
    import services.automl.pipeline as pipeline_module

    original_policy = pipeline_module.select_auto_optimizer
    policy_inputs = {}

    def capture_policy(**kwargs):
        policy_inputs.update(kwargs)
        return original_policy(**kwargs)

    monkeypatch.setattr(persistence, "DEFAULT_STORAGE_DIR", tmp_path / "models")
    monkeypatch.setattr(pipeline_module, "select_auto_optimizer", capture_policy)
    result = run_automl_pipeline(
        dataframe,
        target_column="target",
        problem_type="Binary Classification",
        tuning_strategy="auto",
        tuning_trials=6,
    )

    assert result["status"] == "success"
    assert policy_inputs["training_rows"] == len(expected_train)
    assert policy_inputs["candidate_models"] == result["models"]["trained"]
    assert "validation_rows" not in policy_inputs
    assert "test_rows" not in policy_inputs
    assert result["tuning"]["requested_mode"] == "auto"
    assert result["tuning"]["selected_strategy"] == "none"
    assert result["tuning"]["trial_budget"] == 0
    assert result["tuning"]["reason"]
    assert all(
        model_result["strategy"] == "none"
        for model_result in result["tuning"]["models"].values()
    )


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
    assert result["validation_diagnostics"][0]["candidate_name"] == "synthetic"
    assert result["validation_diagnostics"][0]["status"] == "training_failed"


def test_validation_diagnostics_cover_candidates_failures_and_missing_metrics():
    diagnostics = _build_validation_diagnostics(
        ["LogisticRegression", "DecisionTreeClassifier", "MissingMetricCandidate"],
        [{"model_name": "RandomForestClassifier", "error": "cannot read C:\\Users\\private\\data.csv\ntraceback"}],
        {
            "LogisticRegression": {"status": "Skipped", "errors": []},
            "DecisionTreeClassifier": {
                "status": "Failed",
                "errors": [{"error": "fit failed at C:\\Users\\private\\model.pkl\ntraceback"}],
            },
        },
        [
            {
                "model_name": "LogisticRegression",
                "metrics": {
                    "balanced_accuracy": 0.75,
                    "f1_score": 0.8,
                    "macro_f1_score": 0.7,
                    "confusion_matrix": [[8, 2], [1, 3]],
                    "confusion_matrix_labels": ["0", "1"],
                    "per_class_metrics": {
                        "0": {"recall": 0.8, "support": 10},
                        "1": {"recall": 0.75, "support": 4},
                    },
                },
            },
            {
                "model_name": "DecisionTreeClassifier",
                "metrics": {"balanced_accuracy": 0.55, "f1_score": 0.6},
            },
            {"model_name": "MissingMetricCandidate", "metrics": {}},
        ],
        [],
        "balanced_accuracy",
        [
            {
                "model_name": "LogisticRegression",
                "metrics": {"f1_score": 0.92, "balanced_accuracy": 0.9},
            },
            {
                "model_name": "DecisionTreeClassifier",
                "metrics": {"f1_score": 0.88, "balanced_accuracy": 0.86},
            },
        ],
        [{
            "model_name": "MissingMetricCandidate",
            "error": "training metric failed at C:\\private\\train.csv\ntraceback",
        }],
    )

    by_name = {item["candidate_name"]: item for item in diagnostics}
    assert by_name["LogisticRegression"]["status"] == "evaluated"
    assert by_name["LogisticRegression"]["selection_value"] == 0.75
    assert by_name["LogisticRegression"]["weighted_f1"] == 0.8
    assert by_name["LogisticRegression"]["macro_f1"] == 0.7
    assert by_name["LogisticRegression"]["confusion_matrix"] == [[8, 2], [1, 3]]
    assert by_name["LogisticRegression"]["training_metrics"]["balanced_accuracy"] == 0.9
    assert by_name["LogisticRegression"]["validation_metrics"]["balanced_accuracy"] == 0.75
    assert by_name["LogisticRegression"]["per_class_metrics"]["1"] == {
        "recall": 0.75,
        "support": 4,
    }
    assert by_name["DecisionTreeClassifier"]["status"] == "evaluated_with_tuning_fallback"
    assert by_name["DecisionTreeClassifier"]["selection_value"] == 0.55
    assert "C:\\Users" not in by_name["DecisionTreeClassifier"]["error_reason"]
    assert "traceback" not in by_name["DecisionTreeClassifier"]["error_reason"]
    assert by_name["MissingMetricCandidate"]["status"] == "missing_metrics"
    assert by_name["MissingMetricCandidate"]["selection_value"] is None
    assert by_name["MissingMetricCandidate"]["training_metrics"] is None
    assert "C:\\private" not in by_name["MissingMetricCandidate"]["training_metrics_error"]
    assert "traceback" not in by_name["MissingMetricCandidate"]["training_metrics_error"]
    assert by_name["RandomForestClassifier"]["status"] == "training_failed"
    assert "C:\\Users" not in by_name["RandomForestClassifier"]["error_reason"]
    assert "traceback" not in by_name["RandomForestClassifier"]["error_reason"]


def test_candidate_validation_diagnostics_use_validation_not_holdout(monkeypatch):
    train = pd.DataFrame({"signal": range(40), "target": [0] * 30 + [1] * 10})
    validation = pd.DataFrame({"signal": range(100, 120), "target": [0, 1] * 10})
    holdout = pd.DataFrame({"signal": range(200, 220), "target": [0, 1] * 10})
    dataframe = pd.concat([train, validation, holdout], ignore_index=True)
    candidates = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": DecisionTreeClassifier(random_state=7),
    }
    monkeypatch.setattr(
        automl_pipeline,
        "split_dataset_train_validation_test",
        lambda *args, **kwargs: (train.copy(), validation.copy(), holdout.copy()),
    )
    monkeypatch.setattr(
        automl_pipeline,
        "train_models",
        lambda *args, **kwargs: {
            "trained_models": candidates,
            "model_names": list(candidates),
            "training_errors": [],
        },
    )
    monkeypatch.setattr(
        automl_pipeline,
        "tune_selected_model",
        lambda name, model, *args, **kwargs: (
            {
                "model": DecisionTreeClassifier(max_depth=1, random_state=42),
                "status": "Completed",
                "metrics": {"f1_score": 0.6},
                "errors": [],
            }
            if name == "DecisionTreeClassifier"
            else {"model": model, "status": "Skipped", "errors": []}
        ),
    )
    evaluation_calls = []
    evaluation_models = []
    evaluation_targets = []

    def evaluate_with_partition_marker(models, features, labels, problem_type):
        evaluation_calls.append(features["signal"].tolist())
        evaluation_models.append(dict(models))
        evaluation_targets.append(labels.tolist())
        if len(models) > 1:
            is_training_partition = features["signal"].iloc[0] < 100
            if is_training_partition:
                score = 0.95
                f1_value = 0.96
            else:
                score = 0.8
                f1_value = 0.7
            return {
                "status": "Completed",
                "errors": ([{
                    "model_name": "RandomForestClassifier",
                    "error": "training score unavailable",
                }] if is_training_partition else []),
                "results": [
                    {
                        "model_name": name,
                        "model": model,
                        "metrics": {
                            "f1_score": f1_value if name == "DecisionTreeClassifier" else f1_value - 0.1,
                            "balanced_accuracy": score if name == "DecisionTreeClassifier" else score - 0.2,
                            "macro_f1_score": f1_value - 0.05,
                            "confusion_matrix": [[7, 3], [2, 8]],
                            "confusion_matrix_labels": ["0", "1"],
                            "per_class_metrics": {
                                "0": {"recall": 0.7, "support": 10},
                                "1": {"recall": 0.8, "support": 10},
                            },
                        },
                    }
                    for name, model in models.items()
                    if not (is_training_partition and name == "RandomForestClassifier")
                ],
            }
        name, model = next(iter(models.items()))
        return {
            "status": "Completed",
            "errors": [],
            "results": [{
                "model_name": name,
                "model": model,
                "metrics": {
                    "f1_score": 0.123,
                    "balanced_accuracy": 0.111,
                    "macro_f1_score": 0.1,
                    "confusion_matrix": [[20, 0], [0, 0]],
                    "confusion_matrix_labels": ["0", "1"],
                    "per_class_metrics": {},
                },
            }],
        }

    monkeypatch.setattr(automl_pipeline, "evaluate_models", evaluate_with_partition_marker)
    monkeypatch.setattr(automl_pipeline, "extract_meta_features", lambda *args, **kwargs: {})
    monkeypatch.setattr(automl_pipeline, "load_cached_openml_records", lambda: [])
    monkeypatch.setattr(
        automl_pipeline,
        "recommend_models",
        lambda *args, **kwargs: {"status": "low_confidence", "recommendations": []},
    )
    monkeypatch.setattr(
        automl_pipeline,
        "save_model_package",
        lambda **kwargs: {
            "model_id": "test-model",
            "model_name": kwargs["model_name"],
            "problem_type": kwargs["problem_type"],
            "target_column": kwargs["target_column"],
            "selection_metric": kwargs["selection_metric"],
            "metrics": kwargs["metrics"],
        },
    )

    result = run_automl_pipeline(dataframe, "target", "Binary Classification")

    assert evaluation_calls == [
        train["signal"].tolist(),
        validation["signal"].tolist(),
        holdout["signal"].tolist(),
    ]
    assert evaluation_targets == [
        train["target"].tolist(),
        validation["target"].tolist(),
        holdout["target"].tolist(),
    ]
    assert "DecisionTreeClassifier__class_weight_balanced" in evaluation_models[1]
    assert evaluation_models[0]["LogisticRegression"] is evaluation_models[1]["LogisticRegression"]
    assert evaluation_models[0]["DecisionTreeClassifier"] is evaluation_models[1]["DecisionTreeClassifier"]
    assert evaluation_models[1]["DecisionTreeClassifier"].max_depth == 1
    diagnostics = {item["candidate_name"]: item for item in result["validation_diagnostics"]}
    assert "DecisionTreeClassifier__class_weight_balanced" in diagnostics
    assert diagnostics["DecisionTreeClassifier"]["selection_value"] == 0.8
    assert diagnostics["DecisionTreeClassifier"]["weighted_f1"] == 0.7
    assert diagnostics["DecisionTreeClassifier"]["training_metrics"]["balanced_accuracy"] == 0.95
    assert diagnostics["DecisionTreeClassifier"]["validation_metrics"]["balanced_accuracy"] == 0.8
    assert diagnostics["RandomForestClassifier"]["training_metrics"] is None
    assert diagnostics["RandomForestClassifier"]["training_metrics_error"] == "training score unavailable"
    assert result["evaluation"]["results"][0]["metrics"]["balanced_accuracy"] == 0.111
    assert all(
        item["balanced_accuracy"] != 0.111
        for item in result["validation_diagnostics"]
    )


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
