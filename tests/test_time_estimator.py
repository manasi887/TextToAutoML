import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.automl.time_estimator import estimate_training_time


def test_estimate_training_time_is_deterministic_and_returns_range():
    result = estimate_training_time(10000, 18, "Binary Classification", 3)

    assert result == estimate_training_time(10000, 18, "Binary Classification", 3)
    assert set(result) == {"estimated_seconds", "min_seconds", "max_seconds"}
    assert result["min_seconds"] <= result["estimated_seconds"] <= result["max_seconds"]


def test_estimate_increases_with_dataset_size_and_model_count():
    small = estimate_training_time(1000, 5, "Regression", 1)
    large = estimate_training_time(10000, 10, "Regression", 1)
    more_models = estimate_training_time(1000, 5, "Regression", 3)

    assert large["estimated_seconds"] > small["estimated_seconds"]
    assert more_models["estimated_seconds"] > small["estimated_seconds"]


def test_problem_type_multiplier_affects_estimate():
    regression = estimate_training_time(1000, 5, "Regression", 2)
    multiclass = estimate_training_time(1000, 5, "Multi-class Classification", 2)
    clustering = estimate_training_time(1000, 5, "Clustering", 2)

    assert multiclass["estimated_seconds"] > regression["estimated_seconds"]
    assert clustering["estimated_seconds"] < regression["estimated_seconds"]


@pytest.mark.parametrize("field", ["rows", "features", "model_count"])
def test_estimate_rejects_negative_inputs(field):
    values = {"rows": 1, "features": 1, "problem_type": "Regression", "model_count": 1}
    values[field] = -1

    with pytest.raises(ValueError, match="non-negative"):
        estimate_training_time(**values)
