import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_router_without_external_dependencies():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "backend"
        / "services"
        / "nlp"
        / "modality_router.py"
    )
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **values):
            for name, value in values.items():
                setattr(self, name, value)

    def field(default=None, **kwargs):
        return default

    def field_validator(*fields, **kwargs):
        def decorate(function):
            return function

        return decorate

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = field
    pydantic_stub.field_validator = field_validator

    previous_pydantic = sys.modules.get("pydantic")
    sys.modules["pydantic"] = pydantic_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "nlp_modality_router_under_test",
            source_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load NLP router from {source_path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_pydantic is None:
            del sys.modules["pydantic"]
        else:
            sys.modules["pydantic"] = previous_pydantic
    return module


router = _load_router_without_external_dependencies()


def _request(text: str, **context):
    request_context = {
        "asset_type": None,
        "image_dataset_available": False,
        "image_available": False,
        "image_model_id": None,
        "image_training_result_available": False,
    }
    request_context.update(context)
    return SimpleNamespace(
        text=text,
        **request_context,
    )


INTENT_MATRIX = [
    pytest.param(
        "IMAGE_DESCRIPTION",
        "Describe this image",
        {"asset_type": "image", "image_available": True},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Clarify",
        id="image-description-blocked",
    ),
    pytest.param(
        "IMAGE_CLASSIFICATION",
        "Classify this image",
        {"asset_type": "image", "image_available": True, "image_model_id": "img_saved"},
        "IMAGE_PREDICTION",
        True,
        False,
        "Ready",
        id="image-classification-uses-existing-model",
    ),
    pytest.param(
        "IMAGE_OBJECT_ANALYSIS",
        "Identify all objects in this image",
        {"asset_type": "image", "image_available": True, "image_model_id": "img_saved"},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Object-level image analysis is not supported",
        id="object-analysis-blocked-not-image-classification",
    ),
    pytest.param(
        "IMAGE_QUALITY_ANALYSIS",
        "Assess the quality of this image",
        {"asset_type": "image", "image_available": True},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Clarify",
        id="image-quality-analysis-blocked",
    ),
    pytest.param(
        "IMAGE_ENHANCEMENT",
        "Enhance this image",
        {"asset_type": "image", "image_available": True},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Clarify",
        id="image-enhancement-blocked",
    ),
    pytest.param(
        "IMAGE_EDITING",
        "Remove the background from this image",
        {"asset_type": "image", "image_available": True},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Clarify",
        id="image-editing-blocked",
    ),
    pytest.param(
        "IMAGE_COMPARISON",
        "Compare these two images",
        {"asset_type": "image", "image_available": True},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Clarify",
        id="image-comparison-blocked",
    ),
    pytest.param(
        "DATASET_ANALYSIS",
        "Analyze this image dataset",
        {"asset_type": "image_dataset", "image_dataset_available": True},
        "IMAGE_DATASET_ANALYSIS",
        True,
        False,
        "Ready to analyze",
        id="image-dataset-analysis-supported",
    ),
    pytest.param(
        "DATASET_QUALITY_ANALYSIS",
        "Analyze this image dataset for quality issues and duplicates",
        {"asset_type": "image_dataset", "image_dataset_available": True},
        "IMAGE_DATASET_ANALYSIS",
        True,
        False,
        "Ready to analyze",
        id="image-dataset-quality-analysis-supported",
    ),
    pytest.param(
        "DATASET_RECOMMENDATION",
        "Recommend a dataset for this project",
        {},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "recommendations are not routed",
        id="dataset-recommendation-blocked",
    ),
    pytest.param(
        "MODEL_RECOMMENDATION",
        "Recommend a model for these customer records",
        {"asset_type": "tabular"},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "recommendations are not routed",
        id="standalone-model-recommendation-blocked",
    ),
    pytest.param(
        "MODEL_EXPLANATION",
        "Explain this image model",
        {"asset_type": "image", "image_available": True},
        "IMAGE_MODEL_EXPLANATION",
        False,
        False,
        "not implemented",
        id="image-model-explanation-explicitly-blocked",
    ),
    pytest.param(
        "PREDICTION",
        "Predict whether this customer will churn from these records",
        {"asset_type": "tabular"},
        "TABULAR_REQUEST",
        True,
        False,
        "tabular NLP path",
        id="tabular-prediction-dispatch",
    ),
    pytest.param(
        "PREDICTION_EXPLANATION",
        "Explain why this image was predicted to be a cat",
        {"asset_type": "image", "image_available": True, "image_model_id": "img_saved"},
        "IMAGE_MODEL_EXPLANATION",
        False,
        False,
        "not implemented",
        id="prediction-explanation-blocked",
    ),
    pytest.param(
        "CONFIDENCE_QUERY",
        "How confident is the existing model about this image?",
        {"asset_type": "image", "image_available": True, "image_model_id": "img_saved"},
        "IMAGE_PREDICTION",
        True,
        False,
        "Ready",
        id="confidence-query-runs-image-prediction",
    ),
    pytest.param(
        "TOP_K_PREDICTION",
        "Show the top three likely classes for this image",
        {"asset_type": "image", "image_available": True, "image_model_id": "img_saved"},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Top-K predictions are not supported",
        id="top-k-prediction-blocked",
    ),
    pytest.param(
        "IMAGE_PREPROCESSING",
        "Resize and normalize this image for the model",
        {"asset_type": "image", "image_available": True},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Clarify",
        id="standalone-image-preprocessing-blocked",
    ),
    pytest.param(
        "TRAINING_REQUEST",
        "Train a CNN using this image dataset ZIP",
        {"asset_type": "image_dataset", "image_dataset_available": True},
        "IMAGE_TRAINING_REQUEST",
        True,
        False,
        "Ready to route to image training",
        id="image-training-supported",
    ),
    pytest.param(
        "TRAINING_REPORT",
        "Show my image training report",
        {"image_training_result_available": True},
        "IMAGE_TRAINING_REPORT",
        True,
        False,
        "context is available",
        id="training-report-supported-with-result",
    ),
    pytest.param(
        "MODEL_REPORT",
        "Show the report for this trained image model",
        {"image_training_result_available": True},
        "IMAGE_TRAINING_REPORT",
        True,
        False,
        "context is available",
        id="model-report-uses-training-report",
    ),
    pytest.param(
        "UNKNOWN/AMBIGUOUS",
        "Do something useful",
        {},
        "UNKNOWN_AMBIGUOUS",
        False,
        True,
        "Clarify",
        id="unknown-ambiguous-clarifies",
    ),
]


@pytest.mark.parametrize(
    (
        "category",
        "prompt",
        "context",
        "expected_intent",
        "expected_ready",
        "expected_clarification",
        "reason_fragment",
    ),
    INTENT_MATRIX,
)
def test_requested_intent_matrix(
    category,
    prompt,
    context,
    expected_intent,
    expected_ready,
    expected_clarification,
    reason_fragment,
):
    decision = router.route_nlp_request(_request(prompt, **context))

    assert decision.intent == expected_intent, category
    assert decision.ready is expected_ready, category
    assert decision.clarification_required is expected_clarification, category
    assert reason_fragment.lower() in decision.reason.lower(), category


def test_current_trained_classifier_and_artifact_mapping_still_have_four_labels():
    repository_root = Path(__file__).resolve().parents[1]
    labels_path = (
        repository_root / "backend" / "services" / "nlp" / "data" / "intent_dataset.csv"
    )
    mapping_path = (
        repository_root
        / "backend"
        / "storage"
        / "nlp_models"
        / "intent_classifier"
        / "label_mapping.json"
    )
    with labels_path.open(encoding="utf-8", newline="") as dataset_file:
        labels = {row["intent"] for row in csv.DictReader(dataset_file)}
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    artifact_labels = set(mapping["id_to_label"].values())

    assert labels == {"classification", "regression", "clustering", "unknown"}
    assert artifact_labels == labels
    assert len(labels) == 4


def test_requested_intent_matrix_covers_all_21_categories():
    assert len(INTENT_MATRIX) == 21


def test_image_prediction_requires_image_and_model_and_rejects_invalid_model_id():
    missing_both = router.route_nlp_request(
        _request("Classify this image")
    )
    invalid_model_id = router.route_nlp_request(
        _request(
            "Predict this image",
            asset_type="image",
            image_available=True,
            image_model_id="../outside",
        )
    )

    assert missing_both.intent == "IMAGE_PREDICTION"
    assert missing_both.ready is False
    assert missing_both.required_context == ["image", "image_model_id"]
    assert invalid_model_id.intent == "IMAGE_PREDICTION"
    assert invalid_model_id.ready is False
    assert invalid_model_id.required_context == ["valid_image_model_id"]


def test_analysis_training_and_report_routes_require_their_input_assets():
    analysis = router.route_nlp_request(_request("Analyze this image dataset"))
    training = router.route_nlp_request(_request("Train a CNN on these images"))
    report = router.route_nlp_request(_request("Show my image training report"))

    assert analysis.intent == "IMAGE_DATASET_ANALYSIS"
    assert analysis.ready is False
    assert analysis.required_context == ["image_dataset"]
    assert training.intent == "IMAGE_TRAINING_REQUEST"
    assert training.ready is False
    assert training.required_context == ["image_dataset"]
    assert report.intent == "IMAGE_TRAINING_REPORT"
    assert report.ready is False
    assert report.required_context == ["image_training_result"]


def test_conflicting_image_and_tabular_instructions_request_clarification():
    decision = router.route_nlp_request(
        _request(
            "Train an image classifier and predict customer churn from these records",
            asset_type="image_dataset",
            image_dataset_available=True,
        )
    )

    assert decision.intent == "UNKNOWN_AMBIGUOUS"
    assert decision.ready is False
    assert decision.clarification_required is True
