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
            Path(__file__).resolve().parents[1] / "backend" / "storage" / "uploads"
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
            "automl_training": {"status": "Completed"},
            "ready_for_training": True,
        }

        with patch(
            "api.nlp.process_nlp_request", return_value=nlp_result
        ) as process_mock, patch(
            "api.nlp.integrate_nlp_with_automl", return_value=integration_result
        ) as integrate_mock:
            response = self.client.post(
                "/nlp/train",
                json={"filename": dataset_name, "text": "Predict the target"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), integration_result)
        process_mock.assert_called_once()
        integrate_mock.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()