from pathlib import Path
from services.dataset.analyze import analyze_dataset_df
from services.dataset.loader import load_dataset
from services.dataset.preprocess import preprocess_dataset
from services.dataset.validator import validate_dataset_df
from services.automl.problem_detection import generate_automl_recommendation
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(
    prefix="/upload",
    tags=["Dataset Upload"]
)

# Folder where uploaded datasets will be stored
UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel dataset.
    """

    # Allowed file types
    allowed_extensions = [".csv", ".xlsx"]

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files are allowed."
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: Load dataset once and reuse for validation, analysis, and preprocessing
    df = load_dataset(str(file_path))

    # Step 2: Validate
    validation = validate_dataset_df(df)

    if not validation["valid"]:
        return {
            "message": "Dataset validation failed.",
            "validation": validation
        }

    # Step 3: Analyze
    analysis = analyze_dataset_df(df)

    # Step 4: Preprocess
    processed_df, preprocessing = preprocess_dataset(df)

    # Step 5 and 6: Dataset Intelligence and AutoML recommendation engine
    automl_recommendation = generate_automl_recommendation(
        processed_df,
        generated_columns=preprocessing.get("generated_columns", []),
    )
    dataset_intelligence = automl_recommendation["dataset_intelligence"]

    return {
        "message": "Dataset uploaded successfully.",
        "filename": file.filename,
        "saved_to": str(file_path),
        "validation": validation,
        "analysis": analysis,
        "preprocessing": preprocessing,
        "dataset_intelligence": dataset_intelligence,
        "automl_recommendation": automl_recommendation,
    }
