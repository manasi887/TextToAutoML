import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app
from services.nlp.modality_router import NlpRoutingRequest, route_nlp_request
from services.nlp.train_intent_model import LABEL_NAMES, NUM_LABELS


client = TestClient(app)


@pytest.mark.parametrize(
    ("text", "context", "intent", "ready", "required_context"),
    [
        (
            "Train a CNN on these images",
            {"asset_type": "image_dataset"},
            "IMAGE_TRAINING_REQUEST",
            True,
            [],
        ),
        (
            "Train a classifier using this image dataset",
            {"asset_type": "image_dataset"},
            "IMAGE_TRAINING_REQUEST",
            True,
            [],
        ),
        (
            "Classify this image",
            {"asset_type": "image", "image_model_id": "img_test"},
            "IMAGE_PREDICTION",
            True,
            [],
        ),
        (
            "Classify this image",
            {"asset_type": "image"},
            "IMAGE_PREDICTION",
            False,
            ["image_model_id"],
        ),
        (
            "Show my image training report",
            {"image_training_result_available": True},
            "IMAGE_TRAINING_REPORT",
            True,
            [],
        ),
        (
            "Show my image training report",
            {},
            "IMAGE_TRAINING_REPORT",
            False,
            ["image_training_result"],
        ),
        (
            "Analyze this image dataset",
            {"asset_type": "image_dataset"},
            "IMAGE_DATASET_ANALYSIS",
            True,
            [],
        ),
        (
            "Tell me about this image dataset",
            {"asset_type": "image_dataset"},
            "IMAGE_DATASET_ANALYSIS",
            True,
            [],
        ),
        (
            "Explain this prediction",
            {"asset_type": "image"},
            "IMAGE_MODEL_EXPLANATION",
            False,
            [],
        ),
        (
            "Analyze this",
            {},
            "UNKNOWN_AMBIGUOUS",
            False,
            [],
        ),
        (
            "Do something with these images",
            {"asset_type": "image_dataset"},
            "UNKNOWN_AMBIGUOUS",
            False,
            [],
        ),
        (
            "Predict customer churn",
            {"asset_type": "tabular"},
            "TABULAR_REQUEST",
            True,
            [],
        ),
        (
            "Classify these records",
            {"asset_type": "tabular"},
            "TABULAR_REQUEST",
            True,
            [],
        ),
        (
            "Train a regression model for Sales",
            {"asset_type": "tabular"},
            "TABULAR_REQUEST",
            True,
            [],
        ),
        (
            "Train a CNN on these images",
            {"asset_type": "tabular"},
            "UNKNOWN_AMBIGUOUS",
            False,
            [],
        ),
        (
            "Train a CNN on these images",
            {"asset_type": "image_dataset", "image_model_id": "stale-model"},
            "IMAGE_TRAINING_REQUEST",
            True,
            [],
        ),
    ],
)
def test_routes_requests_using_intent_and_context(
    text, context, intent, ready, required_context
):
    decision = route_nlp_request(NlpRoutingRequest(text=text, **context))

    assert decision.intent == intent
    assert decision.ready is ready
    assert decision.required_context == required_context
    assert decision.clarification_required is (intent == "UNKNOWN_AMBIGUOUS")


def test_prediction_requires_both_single_image_and_model_when_missing():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text="Classify this image",
            image_available=False,
            image_model_id=None,
        )
    )

    assert decision.intent == "IMAGE_PREDICTION"
    assert decision.ready is False
    assert decision.required_context == ["image", "image_model_id"]


def test_existing_model_prediction_with_negative_training_language_routes_to_prediction():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text=(
                "Use my existing trained image classification model with ID "
                "img_20261011_231604_5c125d to predict whether this image contains a cat or a dog. "
                "Do not train a new model."
            ),
            asset_type="image",
            image_available=True,
            image_model_id="img_20261011_231604_5c125d",
        )
    )

    assert decision.intent == "IMAGE_PREDICTION"
    assert decision.ready is True
    assert decision.required_context == []


def test_negative_training_phrase_without_another_action_requires_clarification():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text="Do not train a new model.",
            asset_type="image_dataset",
            image_dataset_available=True,
        )
    )

    assert decision.intent == "UNKNOWN_AMBIGUOUS"
    assert decision.ready is False
    assert decision.clarification_required is True


def test_classify_this_image_is_prediction_even_when_training_data_is_available():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text="Classify this image using the existing model.",
            asset_type="image_dataset",
            image_dataset_available=True,
            image_available=True,
            image_model_id="img_existing",
        )
    )

    assert decision.intent == "IMAGE_PREDICTION"


def test_explicit_image_request_without_asset_context_routes_but_requires_dataset():
    decision = route_nlp_request(NlpRoutingRequest(text="Train a CNN on these images"))

    assert decision.intent == "IMAGE_TRAINING_REQUEST"
    assert decision.ready is False
    assert decision.required_context == ["image_dataset"]


def test_training_language_with_single_image_does_not_become_prediction_when_zip_is_missing():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text="Train a new model to classify this image.",
            asset_type="image",
            image_available=True,
        )
    )

    assert decision.intent == "IMAGE_TRAINING_REQUEST"
    assert decision.ready is False
    assert decision.required_context == ["image_dataset"]


def test_image_dataset_availability_flag_provides_image_context():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text="Train a model to classify cats and dogs",
            image_dataset_available=True,
        )
    )

    assert decision.intent == "IMAGE_TRAINING_REQUEST"
    assert decision.ready is True


def test_explicit_image_training_with_both_assets_remains_training_despite_prediction_language():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text="Train a new model using this ZIP to classify this image and predict its class.",
            asset_type="image_dataset",
            image_dataset_available=True,
            image_available=True,
            image_model_id="img_existing",
        )
    )

    assert decision.intent == "IMAGE_TRAINING_REQUEST"
    assert decision.ready is True


def test_stale_image_training_result_does_not_override_tabular_report_request():
    decision = route_nlp_request(
        NlpRoutingRequest(
            text="Show the training report for customer churn",
            asset_type="tabular",
            image_training_result_available=True,
        )
    )

    assert decision.intent == "TABULAR_REQUEST"
    assert decision.modality == "tabular"


def test_route_api_returns_structured_decision():
    response = client.post(
        "/nlp/route",
        json={"text": "Train a CNN on these images", "asset_type": "image_dataset"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "modality": "image",
        "intent": "IMAGE_TRAINING_REQUEST",
        "ready": True,
        "clarification_required": False,
        "reason": "Ready to route to image training.",
        "required_context": [],
    }


def test_existing_distilbert_label_contract_is_unchanged():
    mapping_path = (
        Path(__file__).resolve().parents[1]
        / "backend"
        / "storage"
        / "nlp_models"
        / "intent_classifier"
        / "label_mapping.json"
    )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    assert NUM_LABELS == 4
    assert LABEL_NAMES == ("classification", "regression", "clustering", "unknown")
    assert set(mapping["id_to_label"].values()) == set(LABEL_NAMES)


@pytest.mark.parametrize("text", ["", "   "])
def test_route_api_rejects_empty_text(text):
    response = client.post("/nlp/route", json={"text": text})

    assert response.status_code == 422