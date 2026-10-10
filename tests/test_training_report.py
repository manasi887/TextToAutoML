import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.reporting.training_report import generate_training_report


def test_training_report_includes_model_id_and_failed_model_count():
    report = generate_training_report(
        {
            "target_column": "Exited",
            "problem_type": "Binary Classification",
        },
        {
            "problem_type": "Binary Classification",
            "data": {
                "original_rows": 100,
                "feature_count": 6,
                "training_rows": 80,
                "test_rows": 20,
            },
            "preprocessing": {
                "raw_feature_names": ["age", "country", "balance"],
            },
            "models": {
                "trained": ["RandomForestClassifier"],
                "failed": [{"model_name": "LogisticRegression", "error": "failed"}],
            },
            "best_model": {
                "name": "RandomForestClassifier",
                "selection_metric": "f1_score",
                "metrics": {"f1_score": 0.9},
            },
            "validation_diagnostics": [{
                "candidate_name": "RandomForestClassifier",
                "training_metrics": {"f1_score": 0.98, "balanced_accuracy": 0.97},
                "validation_metrics": {"f1_score": 0.9, "balanced_accuracy": 0.89},
            }],
            "model": {"model_id": "automl_20260910_232625_aaf2da"},
        },
    )

    assert report["model_id"] == "automl_20260910_232625_aaf2da"
    assert report["dataset"]["features"] == 3
    assert report["warnings"] == {"failed_model_count": 1}
    assert report["models"] == {"trained": 1, "failed": 1}
    assert report["best_model"]["metrics"] == {"f1_score": 0.9}
    assert report["best_model"]["training_metrics"]["f1_score"] == 0.98
    assert report["best_model"]["validation_metrics"]["f1_score"] == 0.9
    json.dumps(report)


def test_training_report_remains_compatible_without_saved_model_metadata():
    report = generate_training_report(
        {"target_column": "target", "problem_type": "Regression"},
        {
            "data": {},
            "models": {"trained": [], "failed": []},
            "best_model": {},
        },
    )

    assert report["model_id"] is None
    assert report["warnings"] == {"failed_model_count": 0}
    assert report["task"] == {"target": "target", "problem_type": "Regression"}