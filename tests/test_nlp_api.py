import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app


class NlpApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.upload_dir = (
            Path(__file__).resolve().parents[1] / "storage" / "uploads"
        )
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.created_files = []

    def tearDown(self):
        for file_path in self.created_files:
            file_path.unlink(missing_ok=True)

    def _write_dataset(self, filename, data):
        file_path = self.upload_dir / filename
        pd.DataFrame(data).to_csv(file_path, index=False)
        self.created_files.append(file_path)
        return file_path

    def test_nlp_train_processes_and_integrates(self):
        dataset_name = "nlp_train_test.csv"
        self._write_dataset(
            dataset_name,
            {"target": [0, 1, 0], "feature": [1, 2, 3]},
        )
        nlp_result = {
            "ready_for_training": True,
            "dataset_resolution": {
                "target_column": "target",
                "problem_type": "Binary Classification",
            },
        }
        integration_result = {
            "nlp_resolution": nlp_result["dataset_resolution"],
            "automl_training": {"status": "success"},
            "ready_for_training": True,
        }
        report = {"summary": "Classification training completed."}

        with patch(
            "api.nlp.process_nlp_request", return_value=nlp_result
        ) as process_mock, patch(
            "api.nlp.integrate_nlp_with_automl", return_value=integration_result
        ) as integrate_mock, patch(
            "api.nlp.generate_training_report", return_value=report
        ) as report_mock:
            response = self.client.post(
                "/nlp/train",
                json={"filename": dataset_name, "text": "Predict the target"},
            )

        self.assertEqual(response.status_code, 200)
        expected_response = {**integration_result, "report": report}
        self.assertEqual(response.json(), expected_response)
        process_mock.assert_called_once()
        integrate_mock.assert_called_once()
        report_mock.assert_called_once_with(
            integration_result["nlp_resolution"],
            integration_result["automl_training"],
        )

    def test_upload_then_nlp_analyze_uses_shared_upload_directory(self):
        dataset_name = "upload_nlp_regression.csv"
        dataset_content = "feature,target\n1,0\n2,1\n3,0\n"

        upload_response = self.client.post(
            "/upload/",
            files={"file": (dataset_name, dataset_content, "text/csv")},
        )

        self.assertEqual(upload_response.status_code, 200)
        self.assertEqual(upload_response.json()["filename"], dataset_name)

        saved_file = self.upload_dir / dataset_name
        self.assertTrue(saved_file.exists())
        self.created_files.append(saved_file)

        nlp_result = {
            "ready_for_training": False,
            "needs_clarification": False,
            "dataset_resolution": {"target_column": "target", "problem_type": "Binary Classification"},
        }

        with patch("api.nlp.process_nlp_request", return_value=nlp_result) as process_mock:
            analysis_response = self.client.post(
                "/nlp/analyze",
                json={"filename": dataset_name, "text": "Predict the target"},
            )

        self.assertEqual(analysis_response.status_code, 200)
        self.assertEqual(analysis_response.json(), nlp_result)
        process_mock.assert_called_once()

    def test_nlp_train_missing_dataset_returns_404(self):
        response = self.client.post(
            "/nlp/train",
            json={"filename": "missing_nlp_train.csv", "text": "Predict target"},
        )

        self.assertEqual(response.status_code, 404)

    def test_nlp_train_expected_processing_errors_return_400(self):
        dataset_name = "nlp_train_error.csv"
        self._write_dataset(
            dataset_name,
            {"target": [0, 1], "feature": [1, 2]},
        )

        with patch(
            "api.nlp.process_nlp_request",
            side_effect=ValueError("invalid NLP request"),
        ):
            response = self.client.post(
                "/nlp/train",
                json={"filename": dataset_name, "text": "Predict target"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid NLP request", response.json()["detail"])

    def test_clarification_loop_blocks_then_trains_after_clarification(self):
        dataset_name = "nlp_clarification_loop.csv"
        self._write_dataset(
            dataset_name,
            {"Exited": [0, 1, 0], "feature": [1, 2, 3]},
        )
        ambiguous_result = {
            "user_text": "Analyze this dataset",
            "intent": {"intent": "unknown", "confidence": 0.86},
            "intent_analysis": {
                "needs_clarification": True,
                "clarification_reason": "Please clarify what you would like to predict.",
            },
            "task": {"problem_type": None, "clarification_required": True},
            "target": {"matched_column": None, "needs_clarification": True},
            "dataset_resolution": {
                "target_column": None,
                "problem_type": "Clustering",
                "needs_clarification": True,
                "reason": "The NLP result requires clarification before selecting a target.",
            },
            "needs_clarification": True,
            "ready_for_training": False,
            "training_time_estimate": None,
        }
        clarified_result = {
            "user_text": "Predict whether customers will leave",
            "intent": {"intent": "classification", "confidence": 0.91},
            "intent_analysis": {"needs_clarification": False},
            "task": {"problem_type": "classification", "clarification_required": False},
            "target": {
                "matched_column": "Exited",
                "confidence": 0.95,
                "needs_clarification": False,
            },
            "dataset_resolution": {
                "target_column": "Exited",
                "problem_type": "Binary Classification",
                "needs_clarification": False,
                "confidence": 0.92,
                "candidates": [],
                "reason": "Resolved target and problem type.",
            },
            "needs_clarification": False,
            "ready_for_training": True,
            "training_time_estimate": {
                "estimated_seconds": 3.0,
                "min_seconds": 2.4,
                "max_seconds": 3.75,
            },
        }
        training_output = {
            "status": "success",
            "model": {"model_id": "test_model"},
            "training_time_seconds": 0.25,
        }
        report = {"summary": "Classification training completed."}

        with patch(
            "api.nlp.process_nlp_request",
            side_effect=[ambiguous_result, ambiguous_result, clarified_result],
        ), patch(
            "services.automl.nlp_integration.run_automl_pipeline",
            return_value=training_output,
        ) as automl_mock, patch(
            "api.nlp.generate_training_report",
            return_value=report,
        ):
            analyze_response = self.client.post(
                "/nlp/analyze",
                json={"filename": dataset_name, "text": "Analyze this dataset"},
            )
            blocked_train_response = self.client.post(
                "/nlp/train",
                json={"filename": dataset_name, "text": "Analyze this dataset"},
            )
            clarified_train_response = self.client.post(
                "/nlp/train",
                json={
                    "filename": dataset_name,
                    "text": "Predict whether customers will leave",
                },
            )

        self.assertEqual(analyze_response.status_code, 200)
        self.assertTrue(analyze_response.json()["needs_clarification"])
        self.assertFalse(analyze_response.json()["ready_for_training"])
        self.assertIsNone(analyze_response.json()["training_time_estimate"])
        self.assertIn(
            "Please clarify",
            analyze_response.json()["intent_analysis"]["clarification_reason"],
        )

        self.assertEqual(blocked_train_response.status_code, 200)
        blocked_body = blocked_train_response.json()
        self.assertTrue(blocked_body["needs_clarification"])
        self.assertFalse(blocked_body["ready_for_training"])
        self.assertIsNone(blocked_body["automl_training"])

        self.assertEqual(clarified_train_response.status_code, 200)
        clarified_body = clarified_train_response.json()
        self.assertFalse(clarified_body["needs_clarification"])
        self.assertTrue(clarified_body["ready_for_training"])
        self.assertEqual(
            clarified_body["nlp_resolution"]["target_column"], "Exited"
        )
        self.assertEqual(
            clarified_body["nlp_resolution"]["problem_type"],
            "Binary Classification",
        )
        self.assertEqual(
            clarified_body["automl_training"]["status"], training_output["status"]
        )
        self.assertEqual(
            clarified_body["automl_training"]["model"], training_output["model"]
        )
        self.assertGreaterEqual(
            clarified_body["automl_training"]["training_time_seconds"], 0
        )
        self.assertEqual(clarified_body["report"], report)
        automl_mock.assert_called_once()
        automl_args = automl_mock.call_args.args
        self.assertTrue(automl_args[0].equals(pd.read_csv(self.upload_dir / dataset_name)))
        self.assertEqual(automl_args[1:], ("Exited", "Binary Classification"))

    def test_nonsense_requests_require_clarification_without_training(self):
        dataset_name = "nlp_nonsense.csv"
        self._write_dataset(
            dataset_name,
            {"Exited": [0, 1, 0], "feature": [1, 2, 3]},
        )
        unknown_result = {
            "intent": {"intent": "unknown", "confidence": 0.8},
            "intent_analysis": {
                "needs_clarification": True,
                "clarification_reason": "The request is unclear. Please clarify the task.",
            },
            "task": {"problem_type": None, "clarification_required": True},
            "target": {"matched_column": None, "needs_clarification": True},
            "dataset_resolution": {
                "target_column": None,
                "problem_type": "Clustering",
                "needs_clarification": True,
                "reason": "The NLP result requires clarification before selecting a target.",
            },
            "needs_clarification": True,
            "ready_for_training": False,
        }

        with patch(
            "api.nlp.process_nlp_request", return_value=unknown_result
        ), patch(
            "services.automl.nlp_integration.run_automl_pipeline"
        ) as automl_mock:
            for text in ("sfiwnga", "???", "qzxv blorpt mivka"):
                analyze_response = self.client.post(
                    "/nlp/analyze",
                    json={"filename": dataset_name, "text": text},
                )
                self.assertEqual(analyze_response.status_code, 200)
                analyze_body = analyze_response.json()
                self.assertTrue(analyze_body["needs_clarification"])
                self.assertFalse(analyze_body["ready_for_training"])
                self.assertIn(
                    "clarify",
                    analyze_body["intent_analysis"]["clarification_reason"].lower(),
                )

                response = self.client.post(
                    "/nlp/train",
                    json={"filename": dataset_name, "text": text},
                )

                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertTrue(body["needs_clarification"])
                self.assertFalse(body["ready_for_training"])
                self.assertIsNone(body["automl_training"])

        automl_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()