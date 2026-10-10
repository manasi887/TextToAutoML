import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


AssetType = Literal["tabular", "image_dataset", "image"]
RouteModality = Literal["image", "tabular", "unknown"]
RouteIntent = Literal[
    "TABULAR_REQUEST",
    "IMAGE_TRAINING_REQUEST",
    "IMAGE_PREDICTION",
    "IMAGE_TRAINING_REPORT",
    "IMAGE_DATASET_ANALYSIS",
    "IMAGE_MODEL_EXPLANATION",
    "UNKNOWN_AMBIGUOUS",
]


class NlpRoutingRequest(BaseModel):
    text: str = Field(..., min_length=1)
    asset_type: AssetType | None = None
    image_dataset_available: bool = False
    image_available: bool = False
    image_model_id: str | None = None
    image_training_result_available: bool = False

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        return cleaned

    @field_validator("image_model_id")
    @classmethod
    def normalize_image_model_id(cls, value: str | None) -> str | None:
        cleaned = value.strip() if value is not None else None
        return cleaned or None


class NlpRoutingDecision(BaseModel):
    modality: RouteModality
    intent: RouteIntent
    ready: bool = Field(description="Whether this routing decision has its required context.")
    clarification_required: bool
    reason: str
    required_context: list[str] = Field(default_factory=list)


_IMAGE_REFERENCE = re.compile(r"\b(?:images?|photos?|pictures?|cnn|visual)\b", re.IGNORECASE)
_IMAGE_DATASET_REFERENCE = re.compile(
    r"\b(?:image|images|photo|photos|picture|pictures)\s+datasets?\b",
    re.IGNORECASE,
)
_TRAIN_ACTION = re.compile(r"\b(?:train|build|create|fit)\b", re.IGNORECASE)
_TRAINING_OBJECTIVE = re.compile(
    r"\b(?:classifier|classification|classify|model|cnn)\b", re.IGNORECASE
)
_PREDICTION_ACTION = re.compile(
    r"\b(?:classify|predict|recognize|identify)\b", re.IGNORECASE
)
_REPORT_ACTION = re.compile(
    r"\b(?:report|accuracy|accurate|metrics|results?)\b", re.IGNORECASE
)
_ANALYSIS_ACTION = re.compile(
    r"\b(?:analy[sz]e|analysis|inspect|explore)\b|\btell\s+me\s+about\b",
    re.IGNORECASE,
)
_EXPLANATION_ACTION = re.compile(r"\b(?:explain|why|reason)\b", re.IGNORECASE)
_TABULAR_REFERENCE = re.compile(
    r"\b(?:customers?|churn|sales|records?|rows?|columns?|spreadsheet|csv|xlsx|regression|clustering)\b",
    re.IGNORECASE,
)
_TABULAR_ACTION = re.compile(
    r"\b(?:predict|classify|train|forecast|estimate|cluster|group|segment)\b",
    re.IGNORECASE,
)


def _decision(
    modality: RouteModality,
    intent: RouteIntent,
    *,
    ready: bool,
    clarification_required: bool = False,
    reason: str,
    required_context: list[str] | None = None,
) -> NlpRoutingDecision:
    return NlpRoutingDecision(
        modality=modality,
        intent=intent,
        ready=ready,
        clarification_required=clarification_required,
        reason=reason,
        required_context=required_context or [],
    )


def _unknown(reason: str) -> NlpRoutingDecision:
    return _decision(
        "unknown",
        "UNKNOWN_AMBIGUOUS",
        ready=False,
        clarification_required=True,
        reason=reason,
    )


def route_nlp_request(request: NlpRoutingRequest) -> NlpRoutingDecision:
    text = request.text
    explicit_image = bool(_IMAGE_REFERENCE.search(text))
    explicit_dataset = bool(_IMAGE_DATASET_REFERENCE.search(text))
    explicit_tabular = bool(_TABULAR_REFERENCE.search(text))
    tabular_asset = request.asset_type == "tabular"
    image_dataset_asset = request.asset_type == "image_dataset"
    image_asset = request.asset_type == "image"
    has_image_context = (
        image_dataset_asset
        or image_asset
        or request.image_dataset_available
        or request.image_available
    )

    if explicit_image and explicit_tabular:
        return _unknown(
            "The request refers to both image and tabular data. Clarify which asset to use."
        )
    if explicit_image and tabular_asset:
        return _unknown(
            "The request refers to image data, but the active asset is tabular. Select an image asset or revise the request."
        )
    if has_image_context and explicit_tabular and not explicit_image:
        return _unknown(
            "The request sounds tabular, but the active asset is an image. Clarify which asset to use."
        )
    if tabular_asset:
        return _decision(
            "tabular",
            "TABULAR_REQUEST",
            ready=True,
            reason=(
                "Route through the existing tabular NLP path; its analysis determines task readiness."
            ),
        )

    image_dataset_available = (
        request.image_dataset_available or image_dataset_asset
    )
    image_available = request.image_available or image_asset
    image_request = explicit_image or has_image_context

    if (
        image_request
        and _TRAIN_ACTION.search(text)
        and (
            explicit_image
            or image_dataset_asset
            or request.image_dataset_available
        )
        and (explicit_image or _TRAINING_OBJECTIVE.search(text))
    ):
        required_context = [] if image_dataset_available else ["image_dataset"]
        return _decision(
            "image",
            "IMAGE_TRAINING_REQUEST",
            ready=not required_context,
            reason=(
                "Ready to route to image training."
                if not required_context
                else "An image dataset is required for image training."
            ),
            required_context=required_context,
        )

    report_requested = bool(_REPORT_ACTION.search(text)) and bool(
        re.search(r"\b(?:training|train|image|cnn|model)\b", text, re.IGNORECASE)
    )
    if report_requested or (
        request.image_training_result_available and _REPORT_ACTION.search(text)
    ):
        ready = request.image_training_result_available
        return _decision(
            "image",
            "IMAGE_TRAINING_REPORT",
            ready=ready,
            reason=(
                "Image training-result context is available."
                if ready
                else "No image training result is currently available."
            ),
            required_context=[] if ready else ["image_training_result"],
        )

    if (
        _EXPLANATION_ACTION.search(text)
        and (image_request or request.image_training_result_available)
        and re.search(r"\b(?:prediction|predicted|cnn|model|image)\b", text, re.IGNORECASE)
    ):
        return _decision(
            "image",
            "IMAGE_MODEL_EXPLANATION",
            ready=False,
            reason="Image model explainability is not implemented yet.",
        )

    dataset_analysis = (
        _ANALYSIS_ACTION.search(text)
        and (
            explicit_dataset
            or (
                image_dataset_asset
                and re.search(r"\bdataset\b", text, re.IGNORECASE)
            )
        )
    )
    if dataset_analysis:
        return _decision(
            "image",
            "IMAGE_DATASET_ANALYSIS",
            ready=False,
            reason="Standalone image dataset analysis is not implemented yet.",
        )

    if image_request and _PREDICTION_ACTION.search(text):
        required_context: list[str] = []
        if not image_available:
            required_context.append("image")
        if not request.image_model_id:
            required_context.append("image_model_id")
        return _decision(
            "image",
            "IMAGE_PREDICTION",
            ready=not required_context,
            reason=(
                "Ready to route to image prediction."
                if not required_context
                else (
                    "A single image and a trained image model are required for image prediction."
                    if "image" in required_context
                    else "A trained image model is required for image prediction."
                )
            ),
            required_context=required_context,
        )

    if tabular_asset or explicit_tabular or _TABULAR_ACTION.search(text):
        return _decision(
            "tabular",
            "TABULAR_REQUEST",
            ready=True,
            reason=(
                "Route through the existing tabular NLP path; its analysis determines task readiness."
            ),
        )

    return _unknown(
        "Clarify whether you want image training, image prediction, a training report, or another supported task."
    )