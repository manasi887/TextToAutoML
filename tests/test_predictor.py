import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.automl.persistence import load_model_package, save_model_package
from services.automl.predictor import predict_with_model


class BinaryPredictModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return [1 if row.sum() >= 2.0 else 0 for row in X]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return [[0.2, 0.8] if row.sum() >= 2.0 else [0.8, 0.2] for row in X]


class RegressionPredictModel:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return [float(row.sum()) for row in X]


class OrderAwareModel:
    def predict(self, X):
        expected = ["Age", "Balance", "country_France", "country_Germany", "gender_Female", "gender_Male"]
        if isinstance(X, pd.DataFrame):
            actual = X.columns.tolist()
            return [1 if actual == expected else 0]
        return [1]


@pytest.fixture
def package_store_dir(tmp_path):
    target = tmp_path / "models"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _make_model_package(package_store_dir, *, model_name="BinaryModel", kind="binary"):
    if kind == "binary":
        country_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        gender_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        country_ohe.fit(np.array(["France", "Germany"]).reshape(-1, 1))
        gender_ohe.fit(np.array(["Female", "Male"]).reshape(-1, 1))
        encoders = {
            "country": {"type": "onehot", "encoder": country_ohe, "feature_names": ["country_France", "country_Germany"]},
            "gender": {"type": "onehot", "encoder": gender_ohe, "feature_names": ["gender_Female", "gender_Male"]},
        }
        feature_names = ["Age", "Balance", "country_France", "country_Germany", "gender_Female", "gender_Male"]
        preprocessing = {
            "removed_columns": {"identifier_columns": [], "constant_columns": []},
            "imputed_columns": {"numeric": ["Age", "Balance"], "categorical": ["country", "gender"]},
            "raw_feature_names": ["Age", "Balance", "country", "gender"],
            "imputation_values": {
                "numeric": {"Age": 35.0, "Balance": 25000.0},
                "categorical": {"country": "France", "gender": "Female"},
            },
        }
        model = BinaryPredictModel()
        target_column = "Exited"
        problem_type = "Binary Classification"
        metric_name = "f1_score"
        metrics = {"f1_score": 0.9}
    else:
        country_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        country_ohe.fit(np.array(["France", "Germany"]).reshape(-1, 1))
        encoders = {"country": {"type": "onehot", "encoder": country_ohe, "feature_names": ["country_France", "country_Germany"]}}
        feature_names = ["Age", "country_France", "country_Germany"]
        preprocessing = {
            "removed_columns": {"identifier_columns": [], "constant_columns": []},
            "imputed_columns": {"numeric": ["Age"], "categorical": ["country"]},
            "raw_feature_names": ["Age", "country"],
            "imputation_values": {"numeric": {"Age": 30.0}, "categorical": {"country": "France"}},
        }
        model = RegressionPredictModel()
        target_column = "target"
        problem_type = "Regression"
        metric_name = "rmse"
        metrics = {"rmse": 0.2}

    saved = save_model_package(
        model=model,
        encoders=encoders,
        feature_names=feature_names,
        target_column=target_column,
        problem_type=problem_type,
        preprocessing=preprocessing,
        model_name=model_name,
        selection_metric=metric_name,
        metrics=metrics,
        storage_dir=package_store_dir,
    )
    return load_model_package(saved["model_id"], storage_dir=package_store_dir)


def test_predict_with_model_binary_classification(package_store_dir):
    package = _make_model_package(package_store_dir)
    result = predict_with_model(package, [{"Age": 30, "Balance": 1000, "country": "France", "gender": "Female"}])
    assert result["predictions"] == [1]
    assert result["prediction_count"] == 1


def test_predict_with_model_regression(package_store_dir):
    package = _make_model_package(package_store_dir, kind="regression")
    result = predict_with_model(package, [{"Age": 30, "country": "France"}])
    assert result["predictions"] == [31.0]


def test_predict_with_model_missing_required_feature(package_store_dir):
    package = _make_model_package(package_store_dir)
    with pytest.raises(ValueError, match="Missing required feature"):
        predict_with_model(package, [{"Age": 30, "Balance": 1000}])


def test_predict_with_model_unknown_categorical_value(package_store_dir):
    package = _make_model_package(package_store_dir)
    result = predict_with_model(package, [{"Age": 30, "Balance": 1000, "country": "Japan", "gender": "Female"}])
    assert result["predictions"] in ([0], [1])


def test_predict_with_model_feature_order_is_preserved(package_store_dir):
    package = _make_model_package(package_store_dir)
    package["model"] = OrderAwareModel()
    result = predict_with_model(package, [{"Age": 30, "Balance": 1000, "country": "France", "gender": "Female"}])
    assert result["predictions"] == [1]


def test_predict_with_model_uses_saved_encoder_without_refitting(package_store_dir):
    package = _make_model_package(package_store_dir)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(np.array(["France", "Germany"]).reshape(-1, 1))
    package["encoders"]["country"] = {"type": "onehot", "encoder": encoder, "feature_names": ["country_France", "country_Germany"]}
    result = predict_with_model(package, [{"Age": 30, "Balance": 1000, "country": "France", "gender": "Female"}])
    assert result["predictions"] == [1]
