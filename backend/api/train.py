from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.automl.pipeline import run_automl_pipeline
from services.dataset.loader import load_dataset
from services.dataset.validator import validate_dataset_df

router = APIRouter(
    prefix="/train",
    tags=["Model Training"]
)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def train_dataset(file: UploadFile = File(...)):
    """
    Accept an uploaded dataset and execute the AutoML training pipeline.
    """

    allowed_extensions = [".csv", ".xlsx"]
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files are allowed for training."
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = load_dataset(str(file_path))

        validation = validate_dataset_df(df)
        if not validation["valid"]:
            return {
                "status": "Failed",
                "message": "Dataset validation failed.",
                "validation": validation,
            }

        pipeline_result = run_automl_pipeline(df)

        return {
            "status": "Completed",
            "problem_type": pipeline_result.get("problem_type"),
            "best_model": pipeline_result.get("best_model"),
            "evaluation": pipeline_result.get("evaluation"),
            "saved_model": pipeline_result.get("saved_model"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {
            "status": "Failed",
            "message": "An error occurred while running the AutoML pipeline.",
            "error": str(exc),
        }
