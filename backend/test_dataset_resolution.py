"""Focused tests for dataset context resolution."""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.nlp.dataset_resolution import resolve_dataset_context


class DatasetResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame({"target": [0, 1, 0], "feature": [1, 2, 3]})

    def test_uses_clear_nlp_target_and_data_grounded_type(self) -> None:
        nlp_result = {
            "intent": {"intent": "classification"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "classification"},
            "target": {
                "matched_column": "target",
                "confidence": 0.95,
                "needs_clarification": False,
            },
            "needs_clarification": True,
        }
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value={"target_candidates": []},
        ):
            result = resolve_dataset_context(nlp_result, self.dataframe)

        self.assertEqual(result["target_column"], "target")
        self.assertEqual(result["problem_type"], "Binary Classification")
        self.assertFalse(result["needs_clarification"])

    def test_auto_selects_one_dominant_compatible_candidate(self) -> None:
        regression_dataframe = pd.DataFrame(
            {"target": [0.1, 1.5, 2.8], "feature": [1, 2, 3]}
        )
        nlp_result = {
            "intent": {"intent": "regression"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "regression"},
            "target": {"matched_column": None, "needs_clarification": True},
            "needs_clarification": True,
        }
        target_report = {
            "target_candidates": [
                {
                    "column": "target",
                    "score": 8.0,
                    "confidence": 0.91,
                    "inferred_problem_type": "Regression",
                    "reasons": ["target_like_name"],
                },
                {
                    "column": "feature",
                    "score": 1.0,
                    "confidence": 0.51,
                    "inferred_problem_type": "Regression",
                    "reasons": [],
                },
            ],
            "requires_user_confirmation": False,
        }
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value=target_report,
        ):
            result = resolve_dataset_context(nlp_result, regression_dataframe)

        self.assertEqual(result["target_column"], "target")
        self.assertEqual(result["problem_type"], "Regression")
        self.assertFalse(result["needs_clarification"])

    def test_ambiguous_candidates_require_clarification(self) -> None:
        nlp_result = {
            "intent": {"intent": "classification"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "classification"},
            "target": {"matched_column": None, "needs_clarification": True},
            "needs_clarification": True,
        }
        target_report = {
            "target_candidates": [
                {"column": "target", "score": 5.0, "confidence": 0.8, "inferred_problem_type": "Binary Classification"},
                {"column": "feature", "score": 4.5, "confidence": 0.78, "inferred_problem_type": "Regression"},
            ],
            "requires_user_confirmation": True,
        }
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value=target_report,
        ):
            result = resolve_dataset_context(nlp_result, self.dataframe)

        self.assertIsNone(result["target_column"])
        self.assertTrue(result["needs_clarification"])
        self.assertIn("ambiguous", result["reason"])

    def test_no_candidate_requires_clarification(self) -> None:
        nlp_result = {
            "intent": {"intent": "regression"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "regression"},
            "target": {"matched_column": None, "needs_clarification": True},
            "needs_clarification": True,
        }
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value={"target_candidates": [], "requires_user_confirmation": False},
        ):
            result = resolve_dataset_context(nlp_result, self.dataframe)

        self.assertIsNone(result["target_column"])
        self.assertTrue(result["needs_clarification"])
        self.assertIn("No dataset target", result["reason"])

    def test_strong_explicit_target_overrides_conflicting_nlp_type(self) -> None:
        nlp_result = {
            "intent": {"intent": "regression"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "regression"},
            "target": {
                "matched_column": "target",
                "confidence": 0.95,
                "needs_clarification": False,
            },
            "needs_clarification": True,
        }
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value={"target_candidates": []},
        ):
            result = resolve_dataset_context(nlp_result, self.dataframe)

        self.assertEqual(result["target_column"], "target")
        self.assertEqual(result["problem_type"], "Binary Classification")
        self.assertFalse(result["needs_clarification"])

    def test_clear_intent_uses_dominant_candidate_when_target_unmatched(self) -> None:
        nlp_result = {
            "intent": {"intent": "classification"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "classification"},
            "target": {"matched_column": None, "needs_clarification": True},
            "needs_clarification": True,
        }
        target_report = {
            "target_candidates": [
                {
                    "column": "target",
                    "score": 8.0,
                    "confidence": 0.91,
                    "inferred_problem_type": "Binary Classification",
                },
                {
                    "column": "feature",
                    "score": 1.0,
                    "confidence": 0.51,
                    "inferred_problem_type": "Regression",
                },
            ],
            "requires_user_confirmation": False,
        }
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value=target_report,
        ):
            result = resolve_dataset_context(nlp_result, self.dataframe)

        self.assertEqual(result["target_column"], "target")
        self.assertEqual(result["problem_type"], "Binary Classification")
        self.assertFalse(result["needs_clarification"])

    def test_natural_language_leave_resolves_exited_from_actual_type(self) -> None:
        nlp_result = {
            "intent": {"intent": "classification"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "classification"},
            "target": {
                "target_reference": "whether customers will leave",
                "matched_column": None,
                "needs_clarification": True,
            },
            "needs_clarification": True,
        }
        target_report = {
            "target_candidates": [
                {
                    "column": "Exited",
                    "score": 8.0,
                    "confidence": 0.91,
                    "inferred_problem_type": "Regression",
                },
                {
                    "column": "feature",
                    "score": 1.0,
                    "confidence": 0.51,
                    "inferred_problem_type": "Regression",
                },
            ],
            "requires_user_confirmation": False,
        }
        dataframe = pd.DataFrame(
            {"Exited": [0, 1, 0], "feature": [1, 2, 3]}
        )
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value=target_report,
        ):
            result = resolve_dataset_context(nlp_result, dataframe)

        self.assertEqual(result["target_column"], "Exited")
        self.assertEqual(result["problem_type"], "Binary Classification")
        self.assertFalse(result["needs_clarification"])

    def test_dominant_candidate_without_confirmation_flag_resolves(self) -> None:
        nlp_result = {
            "intent": {"intent": "classification"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "classification"},
            "target": {"matched_column": None, "needs_clarification": True},
        }
        target_report = {
            "target_candidates": [
                {"column": "Exited", "score": 8.0, "confidence": 0.91},
                {"column": "feature", "score": 1.0, "confidence": 0.51},
            ]
        }
        dataframe = pd.DataFrame(
            {"Exited": [0, 1, 0], "feature": [1, 2, 3]}
        )
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value=target_report,
        ):
            result = resolve_dataset_context(nlp_result, dataframe)

        self.assertEqual(result["target_column"], "Exited")
        self.assertEqual(result["problem_type"], "Binary Classification")
        self.assertFalse(result["needs_clarification"])

    def test_clustering_does_not_require_target(self) -> None:
        nlp_result = {
            "intent": {"intent": "clustering"},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "clustering"},
            "target": {"matched_column": None, "needs_clarification": False},
            "needs_clarification": False,
        }
        with patch(
            "services.nlp.dataset_resolution.detect_target_candidates",
            return_value={"target_candidates": []},
        ), patch(
            "services.nlp.dataset_resolution.detect_problem_type"
        ) as detect_problem_type_mock:
            result = resolve_dataset_context(nlp_result, self.dataframe)

        self.assertIsNone(result["target_column"])
        self.assertEqual(result["problem_type"], "Clustering")
        detect_problem_type_mock.assert_not_called()
        self.assertFalse(result["needs_clarification"])


if __name__ == "__main__":
    unittest.main()
