from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from services.automl.persistence import load_model_package
from services.automl.predictor import _resolve_raw_feature_names, predict_with_model

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


@router.post("/file")
async def predict_uploaded_dataset(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Predict uploaded CSV/XLSX rows using only the persisted model package."""
    if not model_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A non-empty model ID is required.",
        )
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A CSV or Excel file is required.",
        )

    extension = Path(file.filename.replace("\\", "/")).suffix.lower()
    if extension not in {".csv", ".xlsx"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and Excel (.xlsx) files are supported for prediction.",
        )

    try:
        model_package = load_model_package(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id.strip()}' was not found.",
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
        contents = await file.read()
        if not contents:
            raise ValueError("The uploaded prediction file is empty.")
        if extension == ".csv":
            try:
                dataframe = pd.read_csv(BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                dataframe = pd.read_csv(BytesIO(contents), encoding="latin1")
        else:
            dataframe = pd.read_excel(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read the uploaded CSV or Excel prediction file.",
        ) from exc

    if dataframe.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded prediction file contains no data rows.",
        )
    if not dataframe.columns.is_unique:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded prediction file contains duplicate column names.",
        )

    required_columns = _resolve_raw_feature_names(model_package)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file is missing required feature column(s): {', '.join(missing_columns)}.",
        )

    try:
        return predict_with_model(model_package, dataframe.to_dict(orient="records"))
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
