"""Focused tests for NLP to AutoML integration."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.automl.nlp_integration import integrate_nlp_with_automl


class NlpIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame({"Exited": [0, 1, 0], "feature": [1, 2, 3]})
        self.resolution = {
            "target_column": "Exited",
            "problem_type": "Binary Classification",
            "confidence": 0.92,
            "candidates": [],
            "needs_clarification": False,
            "reason": "Resolved",
        }

    def test_blocks_when_not_ready_without_training(self) -> None:
        nlp_result = {
            "ready_for_training": False,
            "dataset_resolution": {
                **self.resolution,
                "needs_clarification": True,
            },
        }
        with patch("services.automl.nlp_integration.run_automl_pipeline") as run_mock:
            result = integrate_nlp_with_automl(self.dataframe, nlp_result)

        run_mock.assert_not_called()
        self.assertIsNone(result["automl_training"])
        self.assertTrue(result["needs_clarification"])
        self.assertFalse(result["ready_for_training"])

    def test_runs_automl_with_resolved_target_and_problem_type(self) -> None:
        nlp_result = {
            "ready_for_training": True,
            "dataset_resolution": self.resolution,
        }
        training_output = {"status": "Completed", "score": 0.91}
        with patch(
            "services.automl.nlp_integration.run_automl_pipeline",
            return_value=training_output,
        ) as run_mock:
            result = integrate_nlp_with_automl(self.dataframe, nlp_result)

        run_mock.assert_called_once_with(
            self.dataframe, "Exited", "Binary Classification"
        )
        self.assertEqual(result["automl_training"], training_output)
        self.assertEqual(result["nlp_resolution"], self.resolution)
        self.assertFalse(result["needs_clarification"])

    def test_rejects_missing_resolution(self) -> None:
        with self.assertRaises(ValueError):
            integrate_nlp_with_automl(
                self.dataframe,
                {"ready_for_training": True},
            )

    def test_requires_resolved_target_when_ready(self) -> None:
        nlp_result = {
            "ready_for_training": True,
            "dataset_resolution": {
                **self.resolution,
                "target_column": None,
            },
        }
        with self.assertRaises(ValueError):
            integrate_nlp_with_automl(self.dataframe, nlp_result)

    def test_serializes_actual_sklearn_internal_model(self) -> None:
        nlp_result = {
            "ready_for_training": True,
            "dataset_resolution": self.resolution,
        }
        training_output = {
            "status": "success",
            "_internal_best_model": LogisticRegression(),
            "metrics": {"score": 0.91},
        }
        with patch(
            "services.automl.nlp_integration.run_automl_pipeline",
            return_value=training_output,
        ):
            result = integrate_nlp_with_automl(self.dataframe, nlp_result)

        self.assertNotIn("_internal_best_model", result["automl_training"])
        self.assertEqual(result["automl_training"]["metrics"]["score"], 0.91)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
