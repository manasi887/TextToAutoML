import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.nlp.intent_detection as intent_detection


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1]])}


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        return SimpleNamespace(logits=torch.tensor([[0.1, 2.0, 0.2, -0.5]]))


def test_prediction_loads_local_artifacts_without_network(tmp_path):
    (tmp_path / "label_mapping.json").write_text(
        json.dumps(
            {
                "id_to_label": {
                    "0": "classification",
                    "1": "regression",
                    "2": "clustering",
                    "3": "unknown",
                }
            }
        ),
        encoding="utf-8",
    )
    intent_detection.MODEL_DIR = tmp_path
    intent_detection._tokenizer = None
    intent_detection._model = None
    intent_detection._label_mapping = None

    with patch.object(
        intent_detection.AutoTokenizer,
        "from_pretrained",
        return_value=FakeTokenizer(),
    ) as tokenizer_loader, patch.object(
        intent_detection.AutoModelForSequenceClassification,
        "from_pretrained",
        return_value=FakeModel(),
    ) as model_loader:
        result = intent_detection.detect_user_intent("Predict house prices")

    tokenizer_loader.assert_called_once_with(tmp_path)
    model_loader.assert_called_once_with(tmp_path)
    assert result["intent"] == "regression"
    assert result["confidence"] > 0.0
    assert set(result["probabilities"]) == {
        "classification",
        "regression",
        "clustering",
        "unknown",
    }


def test_missing_local_model_directory_fails_before_prediction():
    intent_detection.MODEL_DIR = Path("missing-intent-model-directory")
    intent_detection._tokenizer = None
    intent_detection._model = None
    intent_detection._label_mapping = None

    with patch.object(intent_detection.AutoTokenizer, "from_pretrained") as loader:
        try:
            intent_detection.detect_user_intent("Predict house prices")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing local model directory should fail")

    loader.assert_not_called()