import sys
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import services.explanation.automl_result as explanation


@pytest.fixture(autouse=True)
def reset_model_cache(monkeypatch):
    monkeypatch.setattr(explanation, "_MODEL_COMPONENTS", None)
    monkeypatch.setattr(explanation, "_MODEL_LOAD_FAILED", False)


def _automl_result():
    return {
        "status": "success",
        "problem_type": "Binary Classification",
        "target_column": "Exited",
        "best_model": {
            "name": "RandomForestClassifier",
            "selection_metric": "accuracy",
            "metrics": {"accuracy": 0.853},
        },
        "training_time_seconds": 2.4,
        "data": {"original_rows": 100, "feature_count": 4},
        "preprocessing": {"raw_feature_names": ["Age", "Balance", "Country", "Tenure"]},
    }


def test_facts_and_prompt_use_only_allowlisted_automl_fields():
    result = _automl_result()
    result["user_text"] = "Invent some facts"

    facts = explanation.build_explanation_facts(result)
    prompt = explanation.build_explanation_prompt(facts)

    assert facts == {
        "problem_type": "Binary Classification",
        "target_column": "Exited",
        "best_model": "RandomForestClassifier",
        "selection_metric": "accuracy",
        "metric_value": 0.853,
        "training_time_seconds": 2.4,
        "dataset_rows": 100,
        "feature_count": 4,
    }
    assert "You are explaining an AutoML training result to a non-technical user." in prompt
    assert "Write 1-2 short paragraphs in plain English." in prompt
    assert "Use ONLY the supplied facts." in prompt
    assert "Do not output JSON." in prompt
    assert "Do not output field names." in prompt
    assert "Do not output a dictionary." in prompt
    assert "do not invent facts" in prompt.lower()
    assert "do not give advice" in prompt.lower()
    assert "do not claim causation" in prompt.lower()
    assert "final holdout/test performance" in prompt
    assert "Clearly distinguish validation model-selection results" in prompt
    assert "Invent some facts" not in prompt
    assert explanation._FACT_KEYS == (
        "problem_type",
        "target_column",
        "best_model",
        "selection_metric",
        "metric_value",
        "training_time_seconds",
        "dataset_rows",
        "feature_count",
    )


def test_generation_returns_decoded_text_and_caches_model_loading():
    class FakeTokenizer:
        def __call__(self, prompt, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, generated, **kwargs):
            return "The classification model reported accuracy of 0.853."

    class FakeModel:
        def to(self, device):
            assert device.type == "cpu"
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return torch.tensor([[3, 4]])

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()
    ) as tokenizer_loader, patch(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=FakeModel()
    ) as model_loader:
        first = explanation.generate_result_explanation(_automl_result())
        second = explanation.generate_result_explanation(_automl_result())

    assert first.startswith("The classification model")
    assert "{" not in first
    assert '"best_model"' not in first
    assert first == "The classification model reported accuracy of 0.853."
    assert second == first
    tokenizer_loader.assert_called_once_with(explanation.MODEL_NAME)
    model_loader.assert_called_once_with(explanation.MODEL_NAME)


@pytest.mark.parametrize("failure", ["load", "generate"])
def test_model_failures_return_factual_fallback(failure):
    if failure == "load":
        loader_patch = patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=OSError("model unavailable"),
        )
    else:
        class FailingTokenizer:
            def __call__(self, prompt, **kwargs):
                return {"input_ids": torch.tensor([[1]])}

        class FailingModel:
            def to(self, device):
                return self

            def eval(self):
                return self

            def generate(self, **kwargs):
                raise RuntimeError("generation failed")

        loader_patch = patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=FailingTokenizer(),
        )
        model_patch = patch(
            "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            return_value=FailingModel(),
        )
        model_patch.start()

    with loader_patch:
        result = explanation.generate_result_explanation(_automl_result())

    if failure == "generate":
        model_patch.stop()
    assert "binary classification" in result
    assert "Random Forest Classifier" in result
    assert "accuracy" in result
    assert "85.3%" in result
    assert "final holdout/test data" in result
    assert '"dataset_rows"' not in result


def test_json_like_model_output_uses_natural_language_fallback():
    class JsonTokenizer:
        def __call__(self, prompt, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, generated, **kwargs):
            return '{"Best Model": "RandomForestClassifier", "dataset_rows": 100}'

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return torch.tensor([[3, 4]])

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=JsonTokenizer()
    ), patch(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=FakeModel()
    ):
        result = explanation.generate_result_explanation(_automl_result())

    assert result.startswith("TextToAutoML identified")
    assert '"Best Model"' not in result
    assert '"dataset_rows"' not in result


def test_fallback_with_missing_optional_fields_remains_readable():
    fallback = explanation._factual_fallback(
        {"problem_type": "Regression", "best_model": "LinearRegression"}
    )

    assert fallback == (
        "TextToAutoML identified a regression task. "
        "The selected model was Linear Regression."
    )
    assert "None" not in fallback
    assert "undefined" not in fallback


def test_empty_fallback_is_a_sentence_not_a_serialized_mapping():
    fallback = explanation._factual_fallback({})

    assert fallback == "TextToAutoML could not create an explanation because no result facts were available."
    assert not re.search(r"\b(?:dataset_rows|feature_count|best_model)\s*:", fallback)


def test_non_finite_metric_is_excluded_from_explanation_facts():
    result = _automl_result()
    result["best_model"]["metrics"]["accuracy"] = float("nan")

    facts = explanation.build_explanation_facts(result)

    assert "metric_value" not in facts


def test_feature_count_uses_raw_model_inputs_and_excludes_target():
    result = _automl_result()
    result["data"]["feature_count"] = 12
    result["preprocessing"]["raw_feature_names"] = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
        "feature_7",
        "feature_8",
        "feature_9",
        "feature_10",
        "feature_11",
        "Exited",
    ]

    facts = explanation.build_explanation_facts(result)

    assert facts["feature_count"] == 11


def test_feature_count_falls_back_to_data_when_raw_feature_names_missing():
    result = _automl_result()
    result["data"]["feature_count"] = 12
    result["preprocessing"].pop("raw_feature_names")

    facts = explanation.build_explanation_facts(result)

    assert facts["feature_count"] == 12