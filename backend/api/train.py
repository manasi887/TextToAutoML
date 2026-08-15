from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from services.automl.pipeline import run_automl_pipeline
from services.dataset.loader import load_dataset
from services.dataset.validator import validate_dataset_df

router = APIRouter(
    prefix="/train",
    tags=["Model Training"],
)

UPLOAD_DIR = (Path(__file__).resolve().parents[1] / "storage" / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class TrainRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    problem_type: str | None = None
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Filename must not be empty.")
        if Path(cleaned).name != cleaned:
            raise ValueError("Only the uploaded filename itself may be used; directory traversal is not allowed.")
        return cleaned


def _safe_upload_path(filename: str) -> Path:
    candidate = (UPLOAD_DIR / filename).resolve()
    if not candidate.is_relative_to(UPLOAD_DIR):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Requested dataset path is outside the configured upload directory.",
        )
    return candidate


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith("sklearn"):
        return str(value.__class__.__name__)
    return str(value)


def _serialize_training_result(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": _to_json_safe(payload.get("status")),
        "filename": _to_json_safe(payload.get("filename")),
        "target_column": _to_json_safe(payload.get("target_column")),
        "problem_type": _to_json_safe(payload.get("problem_type")),
        "data": _to_json_safe(payload.get("data", {})),
        "preprocessing": _to_json_safe(payload.get("preprocessing", {})),
        "models": _to_json_safe(payload.get("models", {})),
        "evaluation": _to_json_safe(payload.get("evaluation", {})),
        "best_model": _to_json_safe(payload.get("best_model", {})),
    }

    if isinstance(result["preprocessing"], dict):
        result["preprocessing"] = {
            key: value
            for key, value in result["preprocessing"].items()
            if key not in {"encoders"}
        }

    if isinstance(result["best_model"], dict):
        result["best_model"] = {
            "name": result["best_model"].get("name"),
            "selection_metric": result["best_model"].get("selection_metric"),
            "metrics": result["best_model"].get("metrics", {}),
            "reason": result["best_model"].get("reason"),
        }

    if isinstance(result["evaluation"], dict):
        evaluation = result["evaluation"]
        safe_results = []
        for item in evaluation.get("results", []):
            if not isinstance(item, dict):
                continue
            safe_results.append({
                "model_name": item.get("model_name"),
                "metrics": item.get("metrics", {}),
                "status": item.get("status"),
            })
        result["evaluation"] = {
            "status": evaluation.get("status"),
            "results": safe_results,
            "errors": evaluation.get("errors", []),
        }

    return result


@router.post("/")
async def train_dataset(request: TrainRequest):
    """Train a previously uploaded dataset through the frozen AutoML orchestration pipeline."""

    filename = request.filename
    target_column = request.target_column.strip()
    problem_type = request.problem_type.strip() if request.problem_type else None

    file_path = _safe_upload_path(filename)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{filename}' was not found in the uploads directory.",
        )

    extension = file_path.suffix.lower()
    allowed_extensions = {".csv", ".xlsx"}
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and Excel files are allowed for training.",
        )

    try:
        df = load_dataset(str(file_path))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported dataset file: {exc}",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for loader issues
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load the requested dataset.",
        ) from exc

    validation = validate_dataset_df(df)
    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Dataset validation failed before training.",
                "validation": validation,
            },
        )

    if target_column == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A target column must be explicitly supplied.",
        )
    if target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' does not exist in dataset '{filename}'.",
        )

    try:
        pipeline_result = run_automl_pipeline(
            df,
            target_column=target_column,
            problem_type=problem_type,
            test_size=request.test_size,
            random_state=request.random_state,
        )
    except ValueError as exc:
        message = str(exc)
        if "Unsupported problem type" in message or "No supervised problem type" in message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message,
            ) from exc
        if "Target column" in message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for pipeline issues
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during training.",
        ) from exc

    response = _serialize_training_result({
        **pipeline_result,
        "filename": filename,
    })
    return response
