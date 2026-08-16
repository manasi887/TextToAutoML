from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.automl.persistence import load_model_package
from services.automl.predictor import predict_with_model

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


class PredictionRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    data: list[dict[str, Any]] = Field(...)


@router.post("/")
async def predict_dataset(request: PredictionRequest):
    """Load a saved model package and generate predictions for raw feature records."""
    if not request.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prediction data cannot be empty.",
        )

    try:
        model_package = load_model_package(request.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_id}' was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for corrupted package load
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model package could not be loaded or is corrupted.",
        ) from exc

    try:
        return predict_with_model(model_package, request.data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for inference failures
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed while applying the saved preprocessing and model logic.",
        ) from exc
