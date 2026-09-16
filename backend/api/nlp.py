from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config import UPLOAD_DIR

from services.dataset.loader import load_dataset
from services.automl.nlp_integration import integrate_nlp_with_automl
from services.reporting.training_report import generate_training_report
from services.nlp.pipeline import process_nlp_request


router = APIRouter(
    prefix="/nlp",
    tags=["Natural Language Processing"],
)


class NLPRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)

    @field_validator("filename", "text")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned


def _safe_upload_path(filename: str) -> Path:
    candidate = (UPLOAD_DIR / filename).resolve()
    if not candidate.is_relative_to(UPLOAD_DIR):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Requested dataset path is outside the configured upload directory.",
        )
    return candidate


@router.post("/analyze")
async def analyze_nlp_request(request: NLPRequest):
    """Analyze a natural-language request against an uploaded dataset."""
    filename = request.filename
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only the uploaded filename itself may be used; directory traversal is not allowed.",
        )

    file_path = _safe_upload_path(filename)
    if file_path.suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and Excel files are allowed for NLP analysis.",
        )
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{filename}' was not found in the uploads directory.",
        )

    try:
        df = load_dataset(str(file_path))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to load dataset '{filename}': {exc}",
        ) from exc

    try:
        return process_nlp_request(request.text, df)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid NLP request: {exc}",
        ) from exc


@router.post("/train")
async def train_from_nlp_request(request: NLPRequest):
    """Resolve a natural-language request and run AutoML when ready."""
    filename = request.filename
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only the uploaded filename itself may be used; directory traversal is not allowed.",
        )

    file_path = _safe_upload_path(filename)
    if file_path.suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and Excel files are allowed for NLP training.",
        )
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{filename}' was not found in the uploads directory.",
        )

    try:
        df = load_dataset(str(file_path))
        nlp_result = process_nlp_request(request.text, df)
        integration_result = integrate_nlp_with_automl(
            df,
            nlp_result,
            dataset_name=filename,
        )
        automl_training = integration_result.get("automl_training")
        if (
            integration_result.get("ready_for_training") is True
            and isinstance(automl_training, dict)
            and automl_training.get("status") == "success"
        ):
            integration_result["report"] = generate_training_report(
                integration_result["nlp_resolution"],
                automl_training,
            )
        return integration_result
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to process NLP training request: {exc}",
        ) from exc
