import sys
import tempfile
import unittest
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app
from services.automl.trainer import split_dataset_train_validation_test
from services.nlp.intent_detection import IntentModelUnavailableError
from services.nlp.target_matching import ModelDependencyError


def _uses_real_intent_model(test_method):
    @wraps(test_method)
    def run_with_real_model(*args, **kwargs):
        from services.nlp import intent_detection

        original_state = (
            intent_detection.MODEL_DIR,
            intent_detection._tokenizer,
            intent_detection._model,
            intent_detection._label_mapping,
            intent_detection._temperature,
        )
        intent_detection.MODEL_DIR = (
            Path(__file__).resolve().parents[1]
            / "backend"
            / "storage"
            / "nlp_models"
            / "intent_classifier"
        )
        intent_detection._tokenizer = None
        intent_detection._model = None
        intent_detection._label_mapping = None
        intent_detection._temperature = None
        try:
            return test_method(*args, **kwargs)
        finally:
            (
                intent_detection.MODEL_DIR,
                intent_detection._tokenizer,
                intent_detection._model,
                intent_detection._label_mapping,
                intent_detection._temperature,
            ) = original_state

    return run_with_real_model


class NlpApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.upload_dir = (
            Path(__file__).resolve().parents[1] / "storage" / "uploads"
        )
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.created_files = []
        self.explanation_patcher = patch(
            "api.nlp.generate_result_explanation",
            return_value="Explanation based only on the supplied AutoML result.",
        )
        self.explanation_mock = self.explanation_patcher.start()
        self.addCleanup(self.explanation_patcher.stop)

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
            "automl_training": {
                "status": "success",
                "problem_type": "Binary Classification",
                "target_column": "target",
                "best_model": {
                    "name": "RandomForestClassifier",
                    "selection_metric": "accuracy",
                    "metrics": {"accuracy": 0.853},
                },
            },
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
        expected_response = {
            **integration_result,
            "report": report,
            "explanation": "Explanation based only on the supplied AutoML result.",
        }
        self.assertEqual(response.json(), expected_response)
        process_mock.assert_called_once()
        integrate_mock.assert_called_once()
        report_mock.assert_called_once_with(
            integration_result["nlp_resolution"],
            integration_result["automl_training"],
        )
        self.explanation_mock.assert_called_once_with(
            integration_result["automl_training"]
        )
        self.assertEqual(
            response.json()["automl_training"], integration_result["automl_training"]
        )
        self.assertEqual(
            response.json()["automl_training"]["best_model"]["metrics"]["accuracy"],
            0.853,
        )

    def test_nlp_train_forwards_selected_feature_columns(self):
        dataset_name = "nlp_train_features.csv"
        self._write_dataset(
            dataset_name,
            {"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]},
        )
        nlp_result = {
            "ready_for_training": True,
            "dataset_resolution": {
                "target_column": "target",
                "problem_type": "Binary Classification",
            },
        }
        integration_result = {
            "ready_for_training": True,
            "automl_training": {"status": "failed"},
        }

        with patch("api.nlp.process_nlp_request", return_value=nlp_result), patch(
            "api.nlp.integrate_nlp_with_automl",
            return_value=integration_result,
        ) as integrate_mock:
            response = self.client.post(
                "/nlp/train",
                json={
                    "filename": dataset_name,
                    "text": "Predict target",
                    "feature_columns": ["feature"],
                    "tuning_strategy": "bayesian",
                    "tuning_trials": 4,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            integrate_mock.call_args.kwargs["feature_columns"],
            ["feature"],
        )
        self.assertEqual(integrate_mock.call_args.kwargs["tuning_strategy"], "bayesian")
        self.assertEqual(integrate_mock.call_args.kwargs["tuning_trials"], 4)

    def test_nlp_train_forwards_auto_optimizer_mode(self):
        dataset_name = "nlp_train_auto_optimizer.csv"
        self._write_dataset(
            dataset_name,
            {"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]},
        )
        nlp_result = {
            "ready_for_training": True,
            "dataset_resolution": {
                "target_column": "target",
                "problem_type": "Binary Classification",
            },
        }
        with patch("api.nlp.process_nlp_request", return_value=nlp_result), patch(
            "api.nlp.integrate_nlp_with_automl",
            return_value={"ready_for_training": True, "automl_training": {"status": "failed"}},
        ) as integrate_mock:
            response = self.client.post(
                "/nlp/train",
                json={
                    "filename": dataset_name,
                    "text": "Predict target",
                    "tuning_strategy": "auto",
                    "tuning_trials": 6,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(integrate_mock.call_args.kwargs["tuning_strategy"], "auto")
        self.assertEqual(integrate_mock.call_args.kwargs["tuning_trials"], 6)

    def test_nlp_training_rejects_invalid_tuning_settings(self):
        response = self.client.post(
            "/nlp/train",
            json={
                "filename": "unused.csv",
                "text": "Predict target",
                "tuning_strategy": "random",
                "tuning_trials": 31,
            },
        )

        self.assertEqual(response.status_code, 422)

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

    def test_proxy_diagnostic_uses_only_training_rows_and_returns_warnings(self):
        dataset_name = "nlp_proxy_diagnostic.csv"
        dataframe = pd.DataFrame(
            [
                {
                    "group": f"g-{group_index}",
                    "feature": row_index,
                    "target": row_index % 2,
                    "target_copy": row_index % 2,
                }
                for group_index in range(12)
                for row_index in range(8)
            ]
        )
        self._write_dataset(dataset_name, dataframe)
        nlp_result = {
            "ready_for_training": True,
            "needs_clarification": False,
            "dataset_resolution": {
                "target_column": "target",
                "problem_type": "Binary Classification",
            },
        }
        captured = {}
        warning = {
            "feature": "target_copy",
            "detection_type": "exact_target_copy",
            "evidence": {"rows_checked": 0},
            "message": "Review feature provenance and prediction-time availability.",
        }
        split_partitions = {}

        def capture_diagnostic(training_df, target_column, group_column=None, problem_type=None):
            captured["training_df"] = training_df.copy()
            captured["target_column"] = target_column
            captured["group_column"] = group_column
            captured["problem_type"] = problem_type
            return {"target_column": target_column, "warnings": [warning]}

        def capture_split(*args, **kwargs):
            partitions = split_dataset_train_validation_test(*args, **kwargs)
            split_partitions["partitions"] = partitions
            split_partitions["kwargs"] = kwargs
            return partitions

        with patch("api.nlp.process_nlp_request", return_value=nlp_result), patch(
            "api.nlp.diagnose_target_proxy_features",
            side_effect=capture_diagnostic,
        ) as diagnostic_mock, patch(
            "api.nlp.split_dataset_train_validation_test",
            side_effect=capture_split,
        ) as split_mock:
            response = self.client.post(
                "/nlp/diagnose-target-proxies",
                json={
                    "filename": dataset_name,
                    "text": "Classify target",
                    "group_column": "group",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["warnings"], [warning])
        self.assertEqual(body["training_rows_checked"], len(captured["training_df"]))
        self.assertEqual(captured["target_column"], "target")
        self.assertEqual(captured["group_column"], "group")
        self.assertEqual(captured["problem_type"], "Binary Classification")
        self.assertEqual(split_partitions["kwargs"]["group_column"], "group")

        train_df, validation_df, test_df = split_partitions["partitions"]
        self.assertTrue(captured["training_df"].equals(train_df))
        self.assertLess(len(train_df), len(dataframe))
        group_sets = [set(partition["group"]) for partition in (train_df, validation_df, test_df)]
        self.assertTrue(group_sets[0].isdisjoint(group_sets[1]))
        self.assertTrue(group_sets[0].isdisjoint(group_sets[2]))
        self.assertTrue(group_sets[1].isdisjoint(group_sets[2]))
        diagnostic_mock.assert_called_once()

    def test_proxy_diagnostic_skips_clustering_and_unresolved_requests(self):
        dataset_name = "nlp_proxy_diagnostic_invalid.csv"
        self._write_dataset(dataset_name, {"target": [0, 1], "feature": [1, 2]})
        nlp_results = [
            {
                "ready_for_training": True,
                "dataset_resolution": {
                    "target_column": None,
                    "problem_type": "Clustering",
                },
            },
            {
                "ready_for_training": False,
                "needs_clarification": True,
                "dataset_resolution": {
                    "target_column": None,
                    "problem_type": "Classification",
                },
            },
        ]

        with patch("api.nlp.process_nlp_request", side_effect=nlp_results), patch(
            "api.nlp.diagnose_target_proxy_features"
        ) as diagnostic_mock, patch(
            "api.nlp.split_dataset_train_validation_test"
        ) as split_mock:
            responses = [
                self.client.post(
                    "/nlp/diagnose-target-proxies",
                    json={"filename": dataset_name, "text": "Analyze this dataset"},
                )
                for _ in nlp_results
            ]

        self.assertEqual([response.status_code for response in responses], [400, 400])
        self.assertEqual(
            [response.json()["detail"]["code"] for response in responses],
            ["unsupported_diagnostic_task", "clarification_required"],
        )
        diagnostic_mock.assert_not_called()
        split_mock.assert_not_called()

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

    def test_model_dependency_failures_return_specific_503_for_every_nlp_flow(self):
        dataset_name = "nlp_model_dependency_errors.csv"
        self._write_dataset(
            dataset_name,
            {"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]},
        )
        endpoints = (
            "/nlp/analyze",
            "/nlp/train",
            "/nlp/diagnose-target-proxies",
        )
        failures = (
            (
                IntentModelUnavailableError("missing at C:\\private\\intent"),
                "intent_classifier",
                "intent_model_unavailable",
            ),
            (
                ModelDependencyError("missing local embedding and reranker"),
                "target_matching",
                "target_matching_models_unavailable",
            ),
        )

        for error, capability, code in failures:
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint, capability=capability), patch(
                    "api.nlp.process_nlp_request", side_effect=error
                ):
                    response = self.client.post(
                        endpoint,
                        json={"filename": dataset_name, "text": "Predict target"},
                    )

                self.assertEqual(response.status_code, 503)
                detail = response.json()["detail"]
                self.assertEqual(detail["code"], code)
                self.assertEqual(detail["capability"], capability)
                self.assertNotIn("C:\\private", response.text)

    def test_invalid_nlp_input_returns_400_for_every_nlp_flow(self):
        dataset_name = "nlp_invalid_input_errors.csv"
        self._write_dataset(
            dataset_name,
            {"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]},
        )

        for endpoint in (
            "/nlp/analyze",
            "/nlp/train",
            "/nlp/diagnose-target-proxies",
        ):
            with self.subTest(endpoint=endpoint), patch(
                "api.nlp.process_nlp_request",
                side_effect=ValueError("invalid NLP request"),
            ):
                response = self.client.post(
                    endpoint,
                    json={"filename": dataset_name, "text": "Predict target"},
                )

            self.assertEqual(response.status_code, 400)

    def test_unexpected_nlp_failures_return_safe_500_and_are_logged(self):
        dataset_name = "nlp_internal_error.csv"
        self._write_dataset(
            dataset_name,
            {"target": [0, 1, 0, 1], "feature": [1, 2, 3, 4]},
        )
        client = TestClient(app, raise_server_exceptions=False)

        for endpoint in (
            "/nlp/analyze",
            "/nlp/train",
            "/nlp/diagnose-target-proxies",
        ):
            with self.subTest(endpoint=endpoint), self.assertLogs(
                "api.nlp", level="ERROR"
            ), patch(
                "api.nlp.process_nlp_request",
                side_effect=RuntimeError("internal failure at C:\\private\\trace"),
            ):
                response = client.post(
                    endpoint,
                    json={"filename": dataset_name, "text": "Predict target"},
                )

            self.assertEqual(response.status_code, 500)
            self.assertEqual(
                response.json()["detail"], "The NLP request could not be completed."
            )
            self.assertNotIn("C:\\private", response.text)

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

    @_uses_real_intent_model
    def test_automatic_workflow_uploads_trains_persists_and_predicts(self):
        row_count = 600
        rows = range(row_count)
        dataframe = pd.DataFrame(
            {
                "feature_a": [index % 11 for index in rows],
                "feature_b": [index % 17 for index in rows],
                "segment": ["north", "south", "west"] * (row_count // 3),
                "Exited": ["yes" if index % 7 > 2 else "no" for index in rows],
            }
        )
        task_text = "Predict whether a customer will churn using the Exited column."

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            model_dir = temporary_path / "models"
            history_path = temporary_path / "training_history.json"
            dataset_name = f"automatic_e2e_{temporary_path.name}.csv"
            self.created_files.append(self.upload_dir / dataset_name)

            with patch(
                "services.automl.persistence.DEFAULT_STORAGE_DIR", model_dir
            ), patch(
                "services.automl.run_history.HISTORY_PATH", history_path
            ), patch(
                "services.metalearning.recommender.HISTORY_PATH", history_path
            ):
                upload_response = self.client.post(
                    "/upload/",
                    files={
                        "file": (
                            dataset_name,
                            dataframe.to_csv(index=False).encode("utf-8"),
                            "text/csv",
                        )
                    },
                )
                self.assertEqual(upload_response.status_code, 200)
                upload_body = upload_response.json()
                self.assertTrue(upload_body["validation"]["valid"])
                self.assertEqual(upload_body["analysis"]["dataset_info"]["rows"], row_count)

                analysis_response = self.client.post(
                    "/nlp/analyze",
                    json={"filename": dataset_name, "text": task_text},
                )
                self.assertEqual(analysis_response.status_code, 200)
                analysis = analysis_response.json()
                self.assertEqual(analysis["intent"]["intent"], "classification")
                self.assertEqual(analysis["dataset_resolution"]["target_column"], "Exited")
                self.assertEqual(
                    analysis["dataset_resolution"]["problem_type"],
                    "Binary Classification",
                )
                self.assertTrue(analysis["ready_for_training"])

                training_response = self.client.post(
                    "/nlp/train",
                    json={
                        "filename": dataset_name,
                        "text": task_text,
                        "tuning_strategy": "auto",
                    },
                )
                self.assertEqual(training_response.status_code, 200)
                training_body = training_response.json()
                self.assertTrue(training_body["ready_for_training"])
                training = training_body["automl_training"]
                self.assertEqual(training["status"], "success")
                self.assertEqual(training["target_column"], "Exited")
                self.assertEqual(training["problem_type"], "Binary Classification")
                self.assertTrue(training["models"]["trained"])

                decision = training["tuning"]
                self.assertEqual(decision["requested_mode"], "auto")
                self.assertEqual(decision["selected_strategy"], "random")
                self.assertEqual(decision["trial_budget"], 8)
                self.assertEqual(decision["budget_unit"], "trials_per_model")
                self.assertTrue(decision["reason"])
                self.assertEqual(
                    training["best_model"]["selection_metric"], "f1_score"
                )
                selected_result = next(
                    result
                    for result in training["evaluation"]["results"]
                    if result["model_name"] == training["best_model"]["name"]
                )
                self.assertEqual(
                    selected_result["metrics"], training["best_model"]["metrics"]
                )

                model_id = training["model"]["model_id"]
                self.assertTrue(model_id)
                from services.automl.persistence import load_model_package

                package = load_model_package(model_id, storage_dir=model_dir)
                self.assertEqual(package["target_column"], "Exited")
                self.assertEqual(
                    package["preprocessing"]["raw_feature_names"],
                    ["feature_a", "feature_b", "segment"],
                )
                self.assertNotIn("Exited", package["preprocessing"]["raw_feature_names"])
                self.assertTrue(history_path.is_file())

                prediction_response = self.client.post(
                    "/predict/",
                    json={
                        "model_id": model_id,
                        "data": [
                            {
                                "feature_a": 12,
                                "feature_b": 12,
                                "segment": "north",
                            }
                        ],
                    },
                )
                self.assertEqual(prediction_response.status_code, 200)
                self.assertEqual(prediction_response.json()["prediction_count"], 1)

                target_included_response = self.client.post(
                    "/predict/",
                    json={
                        "model_id": model_id,
                        "data": [
                            {
                                "feature_a": 12,
                                "feature_b": 12,
                                "segment": "north",
                                "Exited": "yes",
                            }
                        ],
                    },
                )
                self.assertEqual(target_included_response.status_code, 400)
                self.assertIn("must not be provided", target_included_response.json()["detail"])

    @_uses_real_intent_model
    def test_predict_body_performance_class_resolves_real_categorical_target(self):
        dataset_path = self.upload_dir / "bodyPerformance.csv"
        self.assertTrue(dataset_path.is_file())

        response = self.client.post(
            "/nlp/analyze",
            json={
                "filename": dataset_path.name,
                "text": "Predict body performance class",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["dataset_resolution"]["problem_type"],
            "Multi-class Classification",
        )
        self.assertEqual(body["dataset_resolution"]["target_column"], "class")
        self.assertFalse(body["needs_clarification"])
        self.assertTrue(body["ready_for_training"])

    def test_predict_body_performance_category_cue_works_with_regression_classifier(self):
        dataset_path = self.upload_dir / "bodyPerformance.csv"
        probabilities = {
            "classification": 0.01,
            "regression": 0.98,
            "clustering": 0.005,
            "unknown": 0.005,
        }
        regression_prediction = {
            "intent": "regression",
            "confidence": probabilities["regression"],
            "probabilities": probabilities,
        }

        with patch(
            "services.nlp.pipeline.detect_user_intent",
            return_value=regression_prediction,
        ):
            response = self.client.post(
                "/nlp/analyze",
                json={
                    "filename": dataset_path.name,
                    "text": "Predict body performance category",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"]["intent"], "regression")
        self.assertEqual(body["task"]["problem_type"], "classification")
        self.assertEqual(
            body["dataset_resolution"]["problem_type"],
            "Multi-class Classification",
        )
        self.assertEqual(body["dataset_resolution"]["target_column"], "class")
        self.assertFalse(body["needs_clarification"])
        self.assertTrue(body["ready_for_training"])

    @_uses_real_intent_model
    def test_natural_churn_request_reaches_real_automl_training(self):
        from services.automl import persistence, run_history
        from services.automl.nlp_integration import integrate_nlp_with_automl
        from services.metalearning import recommender
        from services.nlp.pipeline import process_nlp_request

        indexes = range(120)
        dataframe = pd.DataFrame(
            {
                "Balance": [(index * 37) % 5000 for index in indexes],
                "Age": [18 + (index * 7) % 60 for index in indexes],
                "Exited": [index % 2 for index in indexes],
            }
        )
        nlp_result = process_nlp_request("Predict customer churn.", dataframe)

        self.assertEqual(nlp_result["intent"]["intent"], "classification")
        self.assertEqual(nlp_result["target"]["matched_column"], "Exited")
        self.assertEqual(
            nlp_result["dataset_resolution"]["problem_type"],
            "Binary Classification",
        )
        self.assertFalse(nlp_result["needs_clarification"])
        self.assertTrue(nlp_result["ready_for_training"])

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            history_path = temporary_path / "training_history.json"
            with patch.object(
                persistence, "DEFAULT_STORAGE_DIR", temporary_path / "models"
            ), patch.object(run_history, "HISTORY_PATH", history_path), patch.object(
                recommender, "HISTORY_PATH", history_path
            ):
                integration = integrate_nlp_with_automl(
                    dataframe,
                    nlp_result,
                    dataset_name="natural_churn.csv",
                )

            training = integration["automl_training"]
            self.assertEqual(training["status"], "success")
            self.assertEqual(training["target_column"], "Exited")
            self.assertEqual(training["problem_type"], "Binary Classification")
            self.assertEqual(training["best_model"]["selection_metric"], "f1_score")
            self.assertTrue(training["model"]["model_id"])

    @_uses_real_intent_model
    def test_classification_does_not_accept_a_continuous_sales_target(self):
        from services.nlp.pipeline import process_nlp_request

        indexes = range(40)
        dataframe = pd.DataFrame(
            {
                "Sales": [10.0 + index * 13.5 for index in indexes],
                "Exited": [index % 2 for index in indexes],
                "Age": [18 + (index * 7) % 50 for index in indexes],
            }
        )

        result = process_nlp_request("Classify sales.", dataframe)

        self.assertEqual(result["task"]["problem_type"], "classification")
        self.assertEqual(result["target"]["matched_column"], "Sales")
        self.assertIsNone(result["dataset_resolution"]["target_column"])
        self.assertTrue(result["needs_clarification"])
        self.assertFalse(result["ready_for_training"])

    @_uses_real_intent_model
    def test_real_multiclass_and_regression_requests_reach_automl(self):
        from services.automl import persistence, run_history
        from services.automl.nlp_integration import integrate_nlp_with_automl
        from services.metalearning import recommender
        from services.nlp.pipeline import process_nlp_request

        indexes = range(120)
        cases = (
            (
                "multiclass",
                pd.DataFrame(
                    {
                        "visits": [index % 13 for index in indexes],
                        "spend": [(index * 19) % 700 for index in indexes],
                        "category": [["A", "B", "C"][index % 3] for index in indexes],
                    }
                ),
                "Classify customers into their different categories using the category column.",
                "category",
                "Multi-class Classification",
                "f1_score",
            ),
            (
                "regression",
                pd.DataFrame(
                    {
                        "units": [index % 19 for index in indexes],
                        "discount": [(index % 5) / 10 for index in indexes],
                        "Sales": [100 + index * 2.5 + index % 7 for index in indexes],
                    }
                ),
                "Predict the sales value using Sales as the target.",
                "Sales",
                "Regression",
                "rmse",
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for name, dataframe, text, target, problem_type, metric in cases:
                nlp_result = process_nlp_request(text, dataframe)
                self.assertTrue(nlp_result["ready_for_training"], name)
                self.assertEqual(nlp_result["dataset_resolution"]["target_column"], target)
                self.assertEqual(
                    nlp_result["dataset_resolution"]["problem_type"], problem_type
                )

                case_dir = root / name
                history_path = case_dir / "training_history.json"
                with patch.object(
                    persistence, "DEFAULT_STORAGE_DIR", case_dir / "models"
                ), patch.object(run_history, "HISTORY_PATH", history_path), patch.object(
                    recommender, "HISTORY_PATH", history_path
                ):
                    integration = integrate_nlp_with_automl(
                        dataframe,
                        nlp_result,
                        dataset_name=f"{name}.csv",
                    )

                training = integration["automl_training"]
                self.assertEqual(training["status"], "success", name)
                self.assertEqual(training["target_column"], target)
                self.assertEqual(training["problem_type"], problem_type)
                self.assertEqual(training["best_model"]["selection_metric"], metric)
                self.assertTrue(training["model"]["model_id"])

    @_uses_real_intent_model
    def test_real_explicit_target_clustering_and_clarification_routes(self):
        from services.automl import persistence, run_history
        from services.automl.nlp_integration import integrate_nlp_with_automl
        from services.metalearning import recommender
        from services.nlp.pipeline import process_nlp_request

        indexes = range(120)
        binary_frame = pd.DataFrame(
            {
                "Balance": [(index * 37) % 5000 for index in indexes],
                "Age": [18 + (index * 7) % 60 for index in indexes],
                "Exited": [index % 2 for index in indexes],
            }
        )
        explicit = process_nlp_request(
            "Predict Exited using Exited as the target.", binary_frame
        )
        self.assertEqual(explicit["target"]["target_reference"], "exited")
        self.assertTrue(explicit["target"]["explicit_target_column"])
        self.assertEqual(explicit["dataset_resolution"]["target_column"], "Exited")
        self.assertEqual(
            explicit["dataset_resolution"]["problem_type"], "Binary Classification"
        )
        self.assertTrue(explicit["ready_for_training"])

        clustering_frame = pd.DataFrame(
            {
                "x": [index % 11 for index in indexes],
                "y": [(index * 7) % 17 for index in indexes],
                "z": [(index * 13) % 23 for index in indexes],
            }
        )
        clustering = process_nlp_request(
            "Find natural groups in this dataset.", clustering_frame
        )
        self.assertEqual(clustering["intent"]["intent"], "clustering")
        self.assertIsNone(clustering["dataset_resolution"]["target_column"])
        self.assertEqual(clustering["dataset_resolution"]["problem_type"], "Clustering")
        self.assertTrue(clustering["ready_for_training"])

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            history_path = root / "training_history.json"
            with patch.object(
                persistence, "DEFAULT_STORAGE_DIR", root / "models"
            ), patch.object(run_history, "HISTORY_PATH", history_path), patch.object(
                recommender, "HISTORY_PATH", history_path
            ):
                integration = integrate_nlp_with_automl(
                    clustering_frame,
                    clustering,
                    dataset_name="clustering.csv",
                )
            training = integration["automl_training"]
            self.assertEqual(training["status"], "success")
            self.assertEqual(training["problem_type"], "Clustering")
            self.assertIsNone(training["target_column"])
            self.assertEqual(training["best_model"]["selection_metric"], "silhouette_score")

        for text in ("Analyze this dataset.", "Tell me what you find."):
            result = process_nlp_request(text, binary_frame)
            self.assertEqual(result["intent"]["intent"], "unknown")
            self.assertTrue(result["intent_analysis"]["needs_clarification"])
            self.assertTrue(result["needs_clarification"])
            self.assertFalse(result["ready_for_training"])
            blocked = integrate_nlp_with_automl(binary_frame, result)
            self.assertTrue(blocked["needs_clarification"])
            self.assertIsNone(blocked["automl_training"])


if __name__ == "__main__":
    unittest.main()