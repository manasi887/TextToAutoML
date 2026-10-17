import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.automl.pipeline import run_automl_pipeline
from services.automl.trainer import split_dataset_train_validation_test
from services.automl.training import evaluate_models


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, features):
        return np.asarray(self.predictions)


def test_multiclass_metrics_report_weighted_and_macro_averages():
    target = pd.Series(["a", "a", "a", "b", "b", "c"])
    predictions = ["a", "a", "b", "b", "b", "a"]

    result = evaluate_models(
        {"fixed": FixedPredictionModel(predictions)},
        pd.DataFrame({"feature": range(len(target))}),
        target,
        "Multi-class Classification",
    )["results"][0]["metrics"]

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["precision"] == pytest.approx(5 / 9)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(0.6)
    assert result["macro_precision"] == pytest.approx(4 / 9)
    assert result["macro_recall"] == pytest.approx(5 / 9)
    assert result["macro_f1_score"] == pytest.approx(22 / 45)
    assert result["confusion_matrix_labels"] == ["a", "b", "c"]
    assert result["confusion_matrix"] == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_regression_metrics_match_hand_calculation_with_zero_targets():
    result = evaluate_models(
        {"fixed": FixedPredictionModel([1.0, 3.0, -1.0])},
        pd.DataFrame({"feature": [0, 1, 2]}),
        pd.Series([0.0, 2.0, -2.0]),
        "Regression",
    )["results"][0]["metrics"]

    assert result == {"mae": 1.0, "rmse": 1.0, "r2": 0.625}
    assert "mape" not in result
    assert "accuracy" not in result


@pytest.mark.parametrize(
    ("problem_type", "target", "expected_error"),
    [
        ("Regression", [1.0, np.nan], "finite numeric values"),
        ("Regression", [1.0, np.inf], "finite numeric values"),
        ("Classification", ["valid", None], "missing labels"),
    ],
)
def test_evaluation_rejects_invalid_targets_without_reporting_metrics(
    problem_type, target, expected_error
):
    evaluation = evaluate_models(
        {"fixed": FixedPredictionModel([1, 1])},
        pd.DataFrame({"feature": [0, 1]}),
        pd.Series(target),
        problem_type,
    )

    assert evaluation["status"] == "Failed"
    assert evaluation["results"] == []
    assert expected_error in evaluation["errors"][0]["error"]


@pytest.mark.parametrize(
    ("target", "expected_error"),
    [
        ([1.0, 2.0, "bad", 4.0], "must contain numeric values"),
        ([1.0, 2.0, np.inf, 4.0], "must contain finite numeric values"),
        ([np.nan] * 4, "at least one numeric value"),
    ],
)
def test_regression_pipeline_rejects_invalid_target_values(target, expected_error):
    dataframe = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0],
            "target": target,
        }
    )

    with pytest.raises(ValueError, match=expected_error):
        run_automl_pipeline(dataframe, "target", "Regression")


def test_train_validation_test_split_drops_missing_targets_before_partitioning():
    dataframe = pd.DataFrame(
        {
            "row_id": range(30),
            "feature": np.arange(30, dtype=float),
            "target": [float(value) if value % 7 else np.nan for value in range(30)],
        }
    )

    partitions = split_dataset_train_validation_test(
        dataframe,
        "target",
        problem_type="Regression",
        random_state=21,
    )
    row_ids = [set(partition["row_id"]) for partition in partitions]

    assert all(partition["target"].notna().all() for partition in partitions)
    assert row_ids[0].isdisjoint(row_ids[1])
    assert row_ids[0].isdisjoint(row_ids[2])
    assert row_ids[1].isdisjoint(row_ids[2])
    assert set.union(*row_ids) == set(dataframe.loc[dataframe["target"].notna(), "row_id"])
