import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.nlp.intent_detection as intent_detection
from services.nlp.train_intent_model import calibrate_temperature, save_model, save_temperature


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


def test_prediction_uses_persisted_temperature_for_calibration(tmp_path):
    class TemperatureFakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            return SimpleNamespace(logits=torch.tensor([[2.0, 0.0, 0.0, 0.0]]))

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
    (tmp_path / "temperature.json").write_text(
        json.dumps({"temperature": 2.0}),
        encoding="utf-8",
    )
    intent_detection.MODEL_DIR = tmp_path
    intent_detection._tokenizer = None
    intent_detection._model = None
    intent_detection._label_mapping = None
    intent_detection._temperature = None

    with patch.object(
        intent_detection.AutoTokenizer,
        "from_pretrained",
        return_value=FakeTokenizer(),
    ), patch.object(
        intent_detection.AutoModelForSequenceClassification,
        "from_pretrained",
        return_value=TemperatureFakeModel(),
    ):
        result = intent_detection.detect_user_intent("Predict house prices")

    assert result["intent"] == "classification"
    assert intent_detection._temperature == 2.0
    expected_confidence = torch.softmax(torch.tensor([[2.0, 0.0, 0.0, 0.0]]) / 2.0, dim=-1)[0, 0]
    unscaled_confidence = torch.softmax(torch.tensor([[2.0, 0.0, 0.0, 0.0]]), dim=-1)[0, 0]
    assert result["confidence"] == pytest.approx(float(expected_confidence))
    assert result["confidence"] < float(unscaled_confidence)

    (tmp_path / "temperature.json").write_text(
        json.dumps({"temperature": 0.5}),
        encoding="utf-8",
    )
    intent_detection._temperature = None
    sharper_result = intent_detection.detect_user_intent("Predict house prices")
    assert intent_detection._temperature == 0.5
    assert sharper_result["confidence"] > result["confidence"]


@pytest.mark.parametrize(
    "artifact_text",
    [
        None,
        "not json",
        "[]",
        "{}",
        '{"temperature": "invalid"}',
        '{"temperature": 0}',
        '{"temperature": -1}',
        '{"temperature": NaN}',
        '{"temperature": Infinity}',
        '{"temperature": true}',
    ],
)
def test_invalid_or_missing_temperature_falls_back_to_one(tmp_path, artifact_text):
    intent_detection.MODEL_DIR = tmp_path
    intent_detection._temperature = None
    temperature_path = tmp_path / "temperature.json"
    if artifact_text is not None:
        temperature_path.write_text(artifact_text, encoding="utf-8")

    assert intent_detection._load_temperature() == 1.0


def test_save_temperature_persists_the_learned_value(tmp_path):
    class FakeTrainer:
        def save_model(self, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    class FakeModelTokenizer:
        def save_pretrained(self, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    learned_temperature = 1.7

    save_model(
        FakeTrainer(),
        FakeModelTokenizer(),
        tmp_path,
        temperature=learned_temperature,
    )

    assert json.loads((tmp_path / "temperature.json").read_text(encoding="utf-8")) == {
        "temperature": learned_temperature
    }


def test_calibration_learns_a_positive_temperature():
    class FixedLogitModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            return SimpleNamespace(
                logits=torch.tensor(
                    [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]],
                    device=inputs["input_ids"].device,
                )
            )

    validation_batch = {
        "input_ids": torch.tensor([[1], [2]]),
        "labels": torch.tensor([0, 1]),
    }
    learned_temperature = calibrate_temperature(
        FixedLogitModel(),
        [validation_batch],
        device=torch.device("cpu"),
        max_iter=5,
    )

    assert math.isfinite(learned_temperature)
    assert learned_temperature > 0