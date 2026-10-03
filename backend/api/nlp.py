from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config import UPLOAD_DIR

from services.dataset.loader import load_dataset
from services.automl.nlp_integration import integrate_nlp_with_automl
from services.automl.target_proxy_diagnostic import diagnose_target_proxy_features
from services.automl.trainer import split_dataset_train_validation_test
from services.reporting.training_report import generate_training_report
from services.nlp.pipeline import process_nlp_request


router = APIRouter(
    prefix="/nlp",
    tags=["Natural Language Processing"],
)


class NLPRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    group_column: str | None = None

    @field_validator("filename", "text")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("group_column")
    @classmethod
    def normalize_group_column(cls, value: str | None) -> str | None:
        cleaned = value.strip() if value is not None else None
        return cleaned or None


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


@router.post("/diagnose-target-proxies")
async def diagnose_nlp_target_proxies(request: NLPRequest):
    """Diagnose exact target relationships using only supervised training rows."""
    filename = request.filename
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_filename",
                "message": "Only the uploaded filename itself may be used; directory traversal is not allowed.",
            },
        )

    file_path = _safe_upload_path(filename)
    if file_path.suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_file_type",
                "message": "Only CSV and Excel files are allowed for NLP diagnostics.",
            },
        )
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "dataset_not_found", "message": f"Dataset '{filename}' was not found."},
        )

    try:
        dataframe = load_dataset(str(file_path))
        nlp_result = process_nlp_request(request.text, dataframe)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_nlp_request", "message": str(exc)},
        ) from exc

    if not nlp_result.get("ready_for_training"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "clarification_required",
                "message": "Resolve the task and target before checking target-proxy features.",
            },
        )

    resolution = nlp_result.get("dataset_resolution")
    if not isinstance(resolution, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_resolution",
                "message": "NLP analysis did not return a dataset resolution.",
            },
        )

    target_column = resolution.get("target_column")
    problem_type = str(resolution.get("problem_type") or "")
    normalized_problem_type = problem_type.strip().lower()
    if "clustering" in normalized_problem_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_diagnostic_task",
                "message": "Target-proxy diagnostics apply only to supervised classification or regression.",
            },
        )
    if not ("classification" in normalized_problem_type or "regression" in normalized_problem_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_diagnostic_task",
                "message": "A supervised classification or regression task is required.",
            },
        )
    if not isinstance(target_column, str) or target_column not in dataframe.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "target_not_resolved",
                "message": "A valid supervised target column was not resolved.",
            },
        )

    try:
        train_df, _, _ = split_dataset_train_validation_test(
            dataframe,
            target_column,
            problem_type=problem_type,
            group_column=request.group_column,
        )
        diagnostic = diagnose_target_proxy_features(
            train_df,
            target_column,
            group_column=request.group_column,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_diagnostic_split", "message": str(exc)},
        ) from exc

    return {
        "status": "success",
        "target_column": target_column,
        "problem_type": problem_type,
        "group_column": request.group_column,
        "training_rows_checked": len(train_df),
        "warnings": diagnostic["warnings"],
    }


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
            group_column=request.group_column,
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
