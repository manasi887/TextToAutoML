import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.automl.persistence import load_model_package, save_model_package


class DummyModel:
    def predict(self, X):
        return [0] * len(X)


class DemoModel:
    def predict(self, X):
        return [1 for _ in X]


class ModelOne:
    def predict(self, X):
        return [0] * len(X)


class ModelTwo:
    def predict(self, X):
        return [1] * len(X)


class ModelA:
    def predict(self, X):
        return [0] * len(X)


class ModelB:
    def predict(self, X):
        return [1] * len(X)


@pytest.fixture
def model_store_dir(tmp_path):
    target_dir = tmp_path / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _build_dataset() -> pd.DataFrame:
    data = {
        "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "feature_2": [1.2, 0.5, 1.6, 0.7, 1.9, 0.1, 2.2, 0.8, 2.6, 1.1],
        "target": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_save_model_package(model_store_dir):
    model = DummyModel()
    encoders = {"category_col": "dummy_encoder"}
    feature_names = ["feature_1", "feature_2"]
    package_meta = save_model_package(
        model=model,
        encoders=encoders,
        feature_names=feature_names,
        target_column="target",
        problem_type="Binary Classification",
        preprocessing={"removed_columns": {"identifier_columns": [], "constant_columns": []}},
        model_name="DummyModel",
        selection_metric="f1_score",
        metrics={"f1_score": 0.9},
        storage_dir=model_store_dir,
    )

    assert package_meta["model_id"]
    assert package_meta["model_name"] == "DummyModel"
    assert package_meta["problem_type"] == "Binary Classification"
    assert package_meta["target_column"] == "target"
    assert package_meta["feature_count"] == 2
    assert (model_store_dir / package_meta["model_path"]).exists()


def test_load_model_package(model_store_dir):
    model = DummyModel()
    saved = save_model_package(
        model=model,
        encoders={"category_col": "dummy_encoder"},
        feature_names=["feature_1", "feature_2"],
        target_column="target",
        problem_type="Binary Classification",
        preprocessing={"removed_columns": {"identifier_columns": [], "constant_columns": []}},
        model_name="DummyModel",
        selection_metric="f1_score",
        metrics={"f1_score": 0.9},
        storage_dir=model_store_dir,
    )

    loaded = load_model_package(saved["model_id"], storage_dir=model_store_dir)
    assert loaded["model_name"] == "DummyModel"
    assert loaded["target_column"] == "target"
    assert loaded["problem_type"] == "Binary Classification"
    assert loaded["selection_metric"] == "f1_score"
    assert loaded["metrics"]["f1_score"] == 0.9


def test_loaded_model_can_make_predictions(model_store_dir):
    model = DemoModel()
    saved = save_model_package(
        model=model,
        encoders={"feature_1": "dummy_encoder"},
        feature_names=["feature_1", "feature_2"],
        target_column="target",
        problem_type="Binary Classification",
        preprocessing={"removed_columns": {"identifier_columns": [], "constant_columns": []}},
        model_name="DemoModel",
        selection_metric="f1_score",
        metrics={"f1_score": 0.88},
        storage_dir=model_store_dir,
    )

    loaded = load_model_package(saved["model_id"], storage_dir=model_store_dir)
    predictions = loaded["model"].predict([[1, 2], [3, 4]])
    assert predictions == [1, 1]


def test_metadata_is_preserved(model_store_dir):
    saved = save_model_package(
        model=ModelA(),
        encoders={"category_col": "enc"},
        feature_names=["feature_a", "feature_b"],
        target_column="label",
        problem_type="Regression",
        preprocessing={"removed_columns": {"identifier_columns": ["user_id"], "constant_columns": ["constant_col"]}},
        model_name="M",
        selection_metric="rmse",
        metrics={"rmse": 0.123},
        storage_dir=model_store_dir,
    )

    loaded = load_model_package(saved["model_id"], storage_dir=model_store_dir)
    assert loaded["preprocessing"]["removed_columns"]["identifier_columns"] == ["user_id"]
    assert loaded["feature_names"] == ["feature_a", "feature_b"]
    assert loaded["metrics"]["rmse"] == 0.123


def test_unique_model_ids_are_generated(model_store_dir):
    first = save_model_package(
        model=ModelOne(),
        encoders={},
        feature_names=["a"],
        target_column="y",
        problem_type="Binary Classification",
        preprocessing={},
        model_name="M1",
        selection_metric="f1_score",
        metrics={"f1_score": 0.5},
        storage_dir=model_store_dir,
    )
    second = save_model_package(
        model=ModelTwo(),
        encoders={},
        feature_names=["b"],
        target_column="y",
        problem_type="Binary Classification",
        preprocessing={},
        model_name="M2",
        selection_metric="f1_score",
        metrics={"f1_score": 0.6},
        storage_dir=model_store_dir,
    )

    assert first["model_id"] != second["model_id"]
    assert first["model_path"] != second["model_path"]


def test_missing_model_raises_controlled_error(model_store_dir):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_model_package("missing_model_123", storage_dir=model_store_dir)


def test_path_traversal_is_rejected(model_store_dir):
    with pytest.raises(ValueError, match="unsafe"):
        load_model_package("../../secret", storage_dir=model_store_dir)


def test_model_package_contains_preprocessing_information(model_store_dir):
    preprocessing = {
        "removed_columns": {"identifier_columns": ["id"], "constant_columns": ["constant_col"]},
        "imputed_columns": {"numeric": ["age"], "categorical": ["country"]},
    }
    saved = save_model_package(
        model=ModelB(),
        encoders={"country": "encoder"},
        feature_names=["age", "country"],
        target_column="label",
        problem_type="Binary Classification",
        preprocessing=preprocessing,
        model_name="M",
        selection_metric="f1_score",
        metrics={"f1_score": 0.8},
        storage_dir=model_store_dir,
    )

    loaded = load_model_package(saved["model_id"], storage_dir=model_store_dir)
    assert loaded["preprocessing"]["removed_columns"]["identifier_columns"] == ["id"]
    assert loaded["preprocessing"]["imputed_columns"]["numeric"] == ["age"]
    assert loaded["encoders"]["country"] == "encoder"


def test_multiple_saves_do_not_overwrite_each_other(model_store_dir):
    first = save_model_package(
        model=ModelOne(),
        encoders={},
        feature_names=["a"],
        target_column="y",
        problem_type="Regression",
        preprocessing={},
        model_name="M1",
        selection_metric="rmse",
        metrics={"rmse": 1.0},
        storage_dir=model_store_dir,
    )
    second = save_model_package(
        model=ModelTwo(),
        encoders={},
        feature_names=["b"],
        target_column="y",
        problem_type="Regression",
        preprocessing={},
        model_name="M2",
        selection_metric="rmse",
        metrics={"rmse": 2.0},
        storage_dir=model_store_dir,
    )

    assert (model_store_dir / first["model_path"]).exists()
    assert (model_store_dir / second["model_path"]).exists()
    assert first["model_path"] != second["model_path"]
