import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.automl.training import (
    evaluate_models,
    evaluate_models_cv,
    select_best_model,
    train_models,
    tune_selected_model,
)
from services.automl import optimizer as optimizer_module
from services.automl import training as training_module
from services.automl.persistence import load_model_package, save_model_package
from services.automl.predictor import predict_with_model
from services.automl.trainer import prepare_training_data


class _FakeTrial:
    def __init__(self, number):
        self.number = number
        self.params = {}

    def suggest_float(self, name, low, high, log=False):
        value = float(low if self.number % 2 == 0 else high)
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, step=1, log=False):
        choices = list(range(low, high + 1, step))
        value = choices[self.number % len(choices)]
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[self.number % len(choices)]
        self.params[name] = value
        return value


class _FakeStudy:
    def __init__(self, direction):
        self.direction = direction
        self.trials = []
        self.best_params = None
        self.best_value = None

    def optimize(self, objective, n_trials, show_progress_bar=False):
        for number in range(n_trials):
            trial = _FakeTrial(number)
            value = objective(trial)
            self.trials.append(trial)
            is_better = (
                self.best_value is None
                or (self.direction == "maximize" and value > self.best_value)
                or (self.direction == "minimize" and value < self.best_value)
            )
            if is_better:
                self.best_value = value
                self.best_params = dict(trial.params)


class _FakeOptuna:
    def __init__(self):
        self.studies = []
        self.samplers = SimpleNamespace(
            TPESampler=lambda seed: SimpleNamespace(seed=seed)
        )

    def create_study(self, *, direction, sampler):
        self.last_sampler = sampler
        study = _FakeStudy(direction)
        self.studies.append(study)
        return study


def _use_fake_optuna(monkeypatch):
    optuna_stub = _FakeOptuna()
    monkeypatch.setattr(optimizer_module, "_load_optuna", lambda: optuna_stub)
    return optuna_stub


def test_regression_training_and_selection():
    X, y = make_regression(
        n_samples=200,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )

    training_report = train_models(X_train, y_train, "regression")
    assert training_report["status"] == "Completed"
    assert len(training_report["trained_models"]) >= 2
    assert len(training_report["model_names"]) == len(training_report["trained_models"])
    assert not training_report["training_errors"]

    evaluation = evaluate_models(training_report["trained_models"], X_test, y_test, "regression")
    assert evaluation["status"] == "Completed"
    assert evaluation["results"]
    assert all("metrics" in item for item in evaluation["results"])

    selection = select_best_model(evaluation["results"], "regression")
    assert selection["best_model_name"] is not None
    assert selection["selection_metric"] == "rmse"
    assert selection["status"] == "Selected"


def test_classification_training_and_selection():
    X, y = make_classification(
        n_samples=220,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="label")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
    )

    training_report = train_models(X_train, y_train, "classification")
    assert training_report["status"] == "Completed"
    assert len(training_report["trained_models"]) >= 2

    evaluation = evaluate_models(training_report["trained_models"], X_test, y_test, "classification")
    assert evaluation["status"] == "Completed"
    assert evaluation["results"]
    assert all("metrics" in item for item in evaluation["results"])
    assert all("f1_score" in item["metrics"] for item in evaluation["results"])

    selection = select_best_model(evaluation["results"], "classification")
    assert selection["best_model_name"] is not None
    assert selection["selection_metric"] == "f1_score"
    assert selection["status"] == "Selected"


def test_one_failed_model_does_not_crash_pipeline():
    X, y = make_regression(
        n_samples=150,
        n_features=4,
        noise=0.2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=42
    )

    valid_model = LinearRegression()

    with patch("services.automl.training._build_model_specs", return_value=[
        ("LinearRegression", valid_model),
        ("BrokenModel", object()),
    ]):
        training_report = train_models(X_train, y_train, "regression")

    assert len(training_report["trained_models"]) == 1
    assert training_report["status"] == "Completed with errors"
    assert training_report["training_errors"]

    bad_models = {
        "LinearRegression": training_report["trained_models"]["LinearRegression"],
        "BrokenModel": object(),
    }
    evaluation = evaluate_models(bad_models, X_test, y_test, "regression")
    assert len(evaluation["results"]) == 1
    assert evaluation["errors"]


def test_recommended_order_reorders_known_models_without_dropping_fallbacks():
    X_df = pd.DataFrame({"feature": [0.0, 1.0, 0.5, 1.5]})
    y_series = pd.Series([0, 1, 0, 1], name="target")

    training_report = train_models(
        X_df,
        y_series,
        "classification",
        recommended_model_order=["RandomForestClassifier", "NotARealModel"],
    )

    assert training_report["model_names"] == [
        "RandomForestClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
    ]


def test_model_selection_remains_metric_based_after_reordering():
    results = [
        {"model_name": "RecommendedFirst", "model": object(), "metrics": {"f1_score": 0.4}},
        {"model_name": "MetricWinner", "model": object(), "metrics": {"f1_score": 0.9}},
    ]

    selection = select_best_model(results, "classification")

    assert selection["best_model_name"] == "MetricWinner"


def test_regression_cv_selects_lowest_mean_rmse():
    X, y = make_regression(
        n_samples=120,
        n_features=5,
        n_informative=3,
        noise=0.5,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    trained = train_models(X_df, y_series, "regression")
    cv_eval = evaluate_models_cv(trained["trained_models"], X_df, y_series, "regression")
    selection = select_best_model(cv_eval["results"], "regression")

    assert cv_eval["status"] == "Completed"
    assert selection["best_model_name"] is not None
    assert selection["selection_metric"] == "rmse"
    assert selection["status"] == "Selected"
    assert selection["best_metrics"]["rmse"] == min(
        item["metrics"]["rmse"] for item in cv_eval["results"]
    )


def test_classification_cv_selects_highest_mean_f1():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="label")

    trained = train_models(X_df, y_series, "classification")
    cv_eval = evaluate_models_cv(trained["trained_models"], X_df, y_series, "classification")
    selection = select_best_model(cv_eval["results"], "classification")

    assert cv_eval["status"] == "Completed"
    assert selection["best_model_name"] is not None
    assert selection["selection_metric"] == "f1_score"
    assert selection["status"] == "Selected"
    assert selection["best_metrics"]["f1_score"] == max(
        item["metrics"]["f1_score"] for item in cv_eval["results"]
    )


def test_small_dataset_skips_cv_safely():
    X_df = pd.DataFrame({"feature_0": [0.1, 0.2, 0.3], "feature_1": [0.4, 0.5, 0.6]})
    y_series = pd.Series([0, 1, 0], name="target")

    trained = train_models(X_df, y_series, "classification")
    cv_eval = evaluate_models_cv(trained["trained_models"], X_df, y_series, "classification")

    assert cv_eval["status"] == "Failed"
    assert not cv_eval["results"]
    assert cv_eval["errors"]

    reg_trained = train_models(X_df, pd.Series([0.1, 0.2, 0.3], name="target"), "regression")
    reg_cv = evaluate_models_cv(reg_trained["trained_models"], X_df, pd.Series([0.1, 0.2, 0.3], name="target"), "regression")

    assert reg_cv["status"] == "Completed"
    assert reg_cv["results"]
    assert not reg_cv["errors"]


def test_classification_hpo_optimizes_selected_family_with_cv():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    trained = train_models(X_df, y_series, "classification")
    cv_eval = evaluate_models_cv(trained["trained_models"], X_df, y_series, "classification")
    winner = select_best_model(cv_eval["results"], "classification")

    tuning = tune_selected_model(
        winner["best_model_name"],
        winner["best_model"],
        X_df,
        y_series,
        "classification",
    )

    assert tuning["status"] == "Completed"
    assert "f1_score" in tuning["metrics"]
    assert tuning["parameters"]


def test_imbalanced_grid_tuning_uses_balanced_accuracy_and_class_weights(monkeypatch):
    seen_weights = []

    def evaluate_candidates(models, X, y, problem_type):
        model = next(iter(models.values()))
        class_weight = model.get_params()["class_weight"]
        seen_weights.append(class_weight)
        metrics = (
            {"f1_score": 0.80, "balanced_accuracy": 0.85}
            if class_weight == "balanced"
            else {"f1_score": 0.90, "balanced_accuracy": 0.60}
        )
        return {
            "status": "Completed",
            "errors": [],
            "results": [{"metrics": metrics}],
        }

    monkeypatch.setattr(training_module, "evaluate_models_cv", evaluate_candidates)
    features = pd.DataFrame({"feature": range(100)})
    target = pd.Series([0] * 90 + [1] * 10)

    result = tune_selected_model(
        "DecisionTreeClassifier",
        DecisionTreeClassifier(random_state=42),
        features,
        target,
        "Binary Classification",
    )

    assert result["status"] == "Completed"
    assert result["parameters"]["class_weight"] == "balanced"
    assert result["metrics"]["balanced_accuracy"] == 0.85
    assert set(seen_weights) == {None, "balanced"}


def test_regression_hpo_optimizes_selected_family_with_cv():
    X, y = make_regression(
        n_samples=120,
        n_features=5,
        n_informative=3,
        noise=0.5,
        random_state=42,
    )
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    trained = train_models(X_df, y_series, "regression")
    cv_eval = evaluate_models_cv(trained["trained_models"], X_df, y_series, "regression")
    winner = select_best_model(cv_eval["results"], "regression")

    tuning = tune_selected_model(
        winner["best_model_name"],
        winner["best_model"],
        X_df,
        y_series,
        "regression",
    )

    assert tuning["status"] == "Completed"
    assert "rmse" in tuning["metrics"]
    assert tuning["parameters"]


def test_bayesian_spaces_set_valid_parameters_for_every_supported_estimator():
    fake_optuna = _FakeOptuna()
    trial = _FakeTrial(0)
    available_models = dict(
        training_module._build_model_specs("classification")
        + training_module._build_model_specs("regression")
    )

    assert set(training_module._BAYESIAN_SEARCH_SPACES) == set(available_models)
    for model_name, search_space in training_module._BAYESIAN_SEARCH_SPACES.items():
        parameters = optimizer_module._suggest_parameters(trial, search_space)
        candidate = available_models[model_name].__class__(**available_models[model_name].get_params())
        candidate.set_params(**parameters)
        assert parameters
    assert fake_optuna.studies == []


def test_bayesian_tuning_supports_classification_and_regression_metrics(monkeypatch):
    fake_optuna = _use_fake_optuna(monkeypatch)
    classification_X, classification_y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=12,
    )
    classification_result = tune_selected_model(
        "DecisionTreeClassifier",
        DecisionTreeClassifier(random_state=42),
        pd.DataFrame(classification_X),
        pd.Series(classification_y),
        "Binary Classification",
        tuning_strategy="bayesian",
        n_trials=4,
        random_state=19,
    )

    regression_X, regression_y = make_regression(
        n_samples=90,
        n_features=5,
        noise=2.0,
        random_state=23,
    )
    regression_result = tune_selected_model(
        "DecisionTreeRegressor",
        DecisionTreeRegressor(random_state=42),
        pd.DataFrame(regression_X),
        pd.Series(regression_y),
        "Regression",
        tuning_strategy="bayesian",
        n_trials=3,
        random_state=19,
    )

    assert classification_result["status"] == "Completed"
    assert classification_result["strategy"] == "bayesian"
    assert "f1_score" in classification_result["metrics"]
    assert classification_result["model"].get_params()["max_depth"] in [None, 3, 5, 10, 20, 30]
    assert classification_result["trials_run"] == 4
    assert regression_result["status"] == "Completed"
    assert "rmse" in regression_result["metrics"]
    assert regression_result["model"].get_params()["criterion"] in {
        "squared_error", "absolute_error"
    }
    assert regression_result["trials_run"] == 3
    assert [len(study.trials) for study in fake_optuna.studies] == [4, 3]
    assert all(study.direction in {"maximize", "minimize"} for study in fake_optuna.studies)


def test_bayesian_tuning_runs_with_real_optuna_when_installed():
    pytest.importorskip("optuna")
    X, y = make_regression(n_samples=48, n_features=3, noise=2.0, random_state=13)

    result = tune_selected_model(
        "DecisionTreeRegressor",
        DecisionTreeRegressor(random_state=42),
        pd.DataFrame(X),
        pd.Series(y),
        "Regression",
        tuning_strategy="bayesian",
        n_trials=2,
        random_state=31,
    )

    assert result["status"] == "Completed"
    assert result["strategy"] == "bayesian"
    assert result["trials_run"] == 2
    assert "rmse" in result["metrics"]


def test_bayesian_tuning_is_reproducible_and_uses_seeded_tpe(monkeypatch):
    X, y = make_regression(n_samples=72, n_features=4, noise=3.0, random_state=8)
    frame = pd.DataFrame(X)
    target = pd.Series(y)
    results = []
    samplers = []
    for _ in range(2):
        fake_optuna = _use_fake_optuna(monkeypatch)
        results.append(
            tune_selected_model(
                "DecisionTreeRegressor",
                DecisionTreeRegressor(random_state=42),
                frame,
                target,
                "Regression",
                tuning_strategy="bayesian",
                n_trials=5,
                random_state=77,
            )
        )
        samplers.append(fake_optuna.last_sampler)

    assert results[0]["parameters"] == results[1]["parameters"]
    assert results[0]["metrics"] == results[1]["metrics"]
    assert [sampler.seed for sampler in samplers] == [77, 77]
    assert [result["trials_run"] for result in results] == [5, 5]


def test_bayesian_group_cv_is_disjoint_and_preprocessing_is_fold_local(monkeypatch):
    fake_optuna = _use_fake_optuna(monkeypatch)
    rows = [
        {
            "group": f"group-{group_index}",
            "feature": (group_index % 2) * 4 + row_index / 10,
            "target": group_index % 2,
        }
        for group_index in range(24)
        for row_index in range(3)
    ]
    training_frame = pd.DataFrame(rows)
    preprocessing_rows = []
    checked_splits = []
    original_evaluator = training_module._evaluate_model_cv_with_preprocessing

    def capture_evaluator(model_name, model, raw_df, target_column, problem_type, cv, preparation, group_column=None):
        group_values = raw_df[group_column].reset_index(drop=True)
        labels = raw_df[target_column].reset_index(drop=True)
        for train_indices, validation_indices in cv.split(raw_df, labels, group_values):
            train_groups = set(group_values.iloc[train_indices])
            validation_groups = set(group_values.iloc[validation_indices])
            assert train_groups.isdisjoint(validation_groups)
            checked_splits.append((train_groups, validation_groups))

        def capture_preparation(fold_frame, fold_target):
            preprocessing_rows.append(set(fold_frame["group"]))
            return preparation(fold_frame, fold_target)

        return original_evaluator(
            model_name,
            model,
            raw_df,
            target_column,
            problem_type,
            cv,
            capture_preparation,
            group_column=group_column,
        )

    monkeypatch.setattr(training_module, "_evaluate_model_cv_with_preprocessing", capture_evaluator)
    prepared = prepare_training_data(
        training_frame,
        "target",
        group_column="group",
    )
    result = tune_selected_model(
        "DecisionTreeClassifier",
        DecisionTreeClassifier(random_state=42),
        prepared["X"],
        prepared["y"],
        "Binary Classification",
        raw_training_df=training_frame,
        target_column="target",
        preparation_function=lambda fold_frame, target_column: prepare_training_data(
            fold_frame,
            target_column,
            group_column="group",
        ),
        group_column="group",
        tuning_strategy="bayesian",
        n_trials=2,
    )

    assert result["status"] == "Completed"
    assert checked_splits
    assert len(preprocessing_rows) == len(checked_splits)
    assert all(len(fold_groups) < training_frame["group"].nunique() for fold_groups in preprocessing_rows)
    assert len(fake_optuna.studies[0].trials) == 2


def test_bayesian_optimizer_reports_missing_optuna_and_rejects_excess_trials(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fail_optuna_import(name, *args, **kwargs):
        if name == "optuna":
            raise ModuleNotFoundError("No module named 'optuna'", name="optuna")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_optuna_import)
    with pytest.raises(optimizer_module.OptunaDependencyError, match="pip install.*optuna"):
        optimizer_module.optimize_with_optuna(
            DecisionTreeRegressor(),
            {"max_depth": {"kind": "categorical", "choices": [None, 3]}},
            lambda parameters: 1.0,
            direction="minimize",
            n_trials=2,
        )

    grouped = pd.DataFrame({
        "group": ["first", "first", "second", "second"],
        "feature": [0.0, 1.0, 2.0, 3.0],
        "target": [0, 0, 1, 1],
    })
    with pytest.raises(optimizer_module.OptunaDependencyError, match="select grid search"):
        tune_selected_model(
            "DecisionTreeClassifier",
            DecisionTreeClassifier(),
            grouped[["feature"]],
            grouped["target"],
            "Classification",
            raw_training_df=grouped,
            target_column="target",
            group_column="group",
            tuning_strategy="bayesian",
            n_trials=2,
        )

    with pytest.raises(ValueError, match="between 1 and 30"):
        optimizer_module.optimize_with_optuna(
            DecisionTreeRegressor(),
            {"max_depth": {"kind": "categorical", "choices": [None, 3]}},
            lambda parameters: 1.0,
            direction="minimize",
            n_trials=31,
        )


def test_bayesian_tuned_estimator_persists_and_predicts(tmp_path, monkeypatch):
    _use_fake_optuna(monkeypatch)
    features = pd.DataFrame({"measurement": [float(index) for index in range(60)]})
    target = pd.Series([float(index * 2 + index % 3) for index in range(60)], name="target")
    tuning = tune_selected_model(
        "DecisionTreeRegressor",
        DecisionTreeRegressor(random_state=42),
        features,
        target,
        "Regression",
        tuning_strategy="bayesian",
        n_trials=3,
    )
    fitted_model = tuning["model"].fit(features, target)
    saved = save_model_package(
        model=fitted_model,
        encoders={},
        feature_names=["measurement"],
        target_column="target",
        problem_type="Regression",
        preprocessing={
            "raw_feature_names": ["measurement"],
            "feature_names": ["measurement"],
            "imputed_columns": {"numeric": ["measurement"], "categorical": []},
            "imputation_values": {"numeric": {}, "categorical": {}},
        },
        model_name="DecisionTreeRegressor",
        selection_metric="rmse",
        metrics=tuning["metrics"],
        storage_dir=tmp_path / "models",
    )
    package = load_model_package(saved["model_id"], storage_dir=tmp_path / "models")

    result = predict_with_model(package, [{"measurement": 12.5}])

    assert result["status"] == "success"
    assert result["prediction_count"] == 1
    assert isinstance(result["predictions"][0], float)


def test_auto_optimizer_policy_selects_none_random_and_bayesian_with_work_caps():
    classification_models = [
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
    ]
    spaces = training_module._BAYESIAN_SEARCH_SPACES
    policy = optimizer_module.select_auto_optimizer
    common = {
        "problem_type": "Binary Classification",
        "candidate_models": classification_models,
        "cv_folds": 5,
        "group_count": None,
        "bayesian_models": set(spaces),
        "random_models": set(spaces),
        "search_space_dimensions": {name: len(space) for name, space in spaces.items()},
    }

    small = policy(
        training_rows=100,
        feature_count=10,
        optuna_available=True,
        **common,
    )
    medium = policy(
        training_rows=1000,
        feature_count=10,
        optuna_available=True,
        **common,
    )
    large_config = {**common, "cv_folds": 3}
    large = policy(
        training_rows=5000,
        feature_count=10,
        optuna_available=True,
        **large_config,
    )
    large_without_optuna = policy(
        training_rows=5000,
        feature_count=10,
        optuna_available=False,
        **large_config,
    )

    assert small["selected_strategy"] == "none"
    assert small["trial_budget"] == 0
    assert medium["selected_strategy"] == "random"
    assert medium["trial_budget"] == 7
    assert large["selected_strategy"] == "bayesian"
    assert 2 <= large["trial_budget"] < medium["trial_budget"]
    assert large["workload"]["estimated_total_work_units"] <= 300
    assert large_without_optuna["selected_strategy"] == "random"
    assert all(result["requested_mode"] == "auto" for result in (small, medium, large))
    assert all(result["reason"] for result in (small, medium, large, large_without_optuna))


def test_auto_optimizer_policy_handles_regression_and_rejects_unsupported_models():
    spaces = training_module._BAYESIAN_SEARCH_SPACES
    regression = optimizer_module.select_auto_optimizer(
        training_rows=1000,
        feature_count=8,
        problem_type="Regression",
        candidate_models=[
            "LinearRegression",
            "DecisionTreeRegressor",
            "RandomForestRegressor",
        ],
        cv_folds=3,
        group_count=None,
        optuna_available=False,
        bayesian_models=set(spaces),
        random_models=set(spaces),
        search_space_dimensions={name: len(space) for name, space in spaces.items()},
    )
    unsupported = optimizer_module.select_auto_optimizer(
        training_rows=5000,
        feature_count=10,
        problem_type="Regression",
        candidate_models=["LinearRegression", "UnregisteredRegressor"],
        cv_folds=5,
        group_count=None,
        optuna_available=True,
        bayesian_models=set(spaces),
        random_models=set(spaces),
    )
    invalid_groups = optimizer_module.select_auto_optimizer(
        training_rows=5000,
        feature_count=10,
        problem_type="Binary Classification",
        candidate_models=["DecisionTreeClassifier"],
        cv_folds=2,
        group_count=4,
        optuna_available=True,
        bayesian_models=set(spaces),
        random_models=set(spaces),
    )

    assert regression["selected_strategy"] == "random"
    assert unsupported["selected_strategy"] == "none"
    assert "UnregisteredRegressor" in unsupported["reason"]
    assert invalid_groups["selected_strategy"] == "none"
    assert "folds" in invalid_groups["reason"]


def test_random_search_real_classification_is_reproducible_and_bounded():
    X, y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=29,
    )
    features = pd.DataFrame(X)
    target = pd.Series(y)
    results = [
        tune_selected_model(
            "DecisionTreeClassifier",
            DecisionTreeClassifier(random_state=42),
            features,
            target,
            "Binary Classification",
            tuning_strategy="random",
            n_trials=4,
            random_state=37,
        )
        for _ in range(2)
    ]

    assert all(result["status"] == "Completed" for result in results)
    assert all(result["strategy"] == "random" for result in results)
    assert all(result["trials_run"] == 4 for result in results)
    assert results[0]["parameters"] == results[1]["parameters"]
    assert results[0]["metrics"] == results[1]["metrics"]
    assert "f1_score" in results[0]["metrics"]
    assert results[0]["model"].get_params()["max_depth"] in [None, 3, 5, 10, 20, 30]


def test_random_search_uses_group_aware_fold_local_preprocessing(monkeypatch):
    rows = [
        {
            "group": f"g-{group_index}",
            "feature": float(group_index % 2) + row_index / 10,
            "target": group_index % 2,
        }
        for group_index in range(24)
        for row_index in range(3)
    ]
    training_frame = pd.DataFrame(rows)
    prepared = prepare_training_data(training_frame, "target", group_column="group")
    original = training_module._evaluate_model_cv_with_preprocessing
    checked_splits = []
    fitted_groups = []

    def capture_evaluation(model_name, model, raw_df, target_column, problem_type, cv, preparation, group_column=None):
        labels = raw_df[target_column].reset_index(drop=True)
        groups = raw_df[group_column].reset_index(drop=True)
        for train_indices, valid_indices in cv.split(raw_df, labels, groups):
            train_groups = set(groups.iloc[train_indices])
            valid_groups = set(groups.iloc[valid_indices])
            assert train_groups.isdisjoint(valid_groups)
            checked_splits.append((train_groups, valid_groups))

        def track_preparation(fold_frame, fold_target):
            fitted_groups.append(set(fold_frame[group_column]))
            return preparation(fold_frame, fold_target)

        return original(
            model_name,
            model,
            raw_df,
            target_column,
            problem_type,
            cv,
            track_preparation,
            group_column=group_column,
        )

    monkeypatch.setattr(training_module, "_evaluate_model_cv_with_preprocessing", capture_evaluation)
    result = tune_selected_model(
        "DecisionTreeClassifier",
        DecisionTreeClassifier(random_state=42),
        prepared["X"],
        prepared["y"],
        "Binary Classification",
        raw_training_df=training_frame,
        target_column="target",
        preparation_function=lambda fold_frame, target_column: prepare_training_data(
            fold_frame, target_column, group_column="group"
        ),
        group_column="group",
        tuning_strategy="random",
        n_trials=2,
        random_state=17,
    )

    assert result["status"] == "Completed"
    assert result["trials_run"] == 2
    assert checked_splits
    assert len(checked_splits) == len(fitted_groups)
    assert all(len(group_set) < training_frame["group"].nunique() for group_set in fitted_groups)


def test_no_tuning_returns_the_baseline_estimator_without_search():
    model = DecisionTreeClassifier(random_state=42)
    result = tune_selected_model(
        "DecisionTreeClassifier",
        model,
        pd.DataFrame({"feature": [0, 1, 2, 3]}),
        pd.Series([0, 0, 1, 1]),
        "Binary Classification",
        tuning_strategy="none",
    )

    assert result["model"] is model
    assert result["status"] == "Skipped"
    assert result["strategy"] == "none"
    assert result["errors"] == []
