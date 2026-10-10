import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from config import PROJECT_ROOT
from services.image_classification.dataset import (
    analyze_extracted_dataset,
    extract_zip_dataset,
    split_dataset,
)
from services.image_classification.persistence import save_image_model_package
from services.image_classification.predict import predict_image_file
from services.image_classification.training import train_image_classifier

router = APIRouter(
    prefix="/image",
    tags=["Image Classification"],
)


@router.post("/analyze")
async def analyze_image_dataset(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image dataset ZIP file is required.",
        )

    safe_filename = Path(file.filename.replace("\\", "/")).name
    if Path(safe_filename).suffix.lower() != ".zip":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only ZIP image datasets are supported for analysis.",
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="image_analyze_"))
    extracted_root: Path | None = None
    try:
        zip_path = temp_dir / safe_filename
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        archive_size_bytes = zip_path.stat().st_size
        extracted_root, summary = extract_zip_dataset(zip_path)
        return analyze_extracted_dataset(
            extracted_root,
            summary,
            archive_size_bytes=archive_size_bytes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        if extracted_root is not None:
            shutil.rmtree(extracted_root, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/predict")
async def predict_image_dataset(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    if not model_id or not model_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A valid model_id is required.")
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is required.")

    safe_filename = Path(file.filename.replace("\\", "/")).name
    extension = Path(safe_filename).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image format for prediction.")

    temp_dir = Path(tempfile.mkdtemp(prefix="image_predict_"))
    image_path = temp_dir / safe_filename
    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        prediction = predict_image_file(
            image_path,
            model_id,
            storage_dir=PROJECT_ROOT / "storage" / "image_models",
        )
        return {
            "task": "Image Classification",
            "model_id": prediction["model_id"],
            "model": prediction["architecture"],
            "prediction": {
                "class": prediction["predicted_class"],
                "confidence": float(prediction["confidence"]),
            },
            "probabilities": prediction["probabilities"],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/train")
async def train_image_dataset(
    file: UploadFile = File(...),
    epochs: int = Form(5),
    batch_size: int = Form(16),
    learning_rate: float = Form(0.001),
    seed: int = Form(42),
    train_ratio: float = Form(0.70),
    validation_ratio: float = Form(0.15),
    test_ratio: float = Form(0.15),
    pretrained: bool = Form(True),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image dataset file is required.")

    safe_filename = Path(file.filename.replace("\\", "/")).name
    extension = Path(safe_filename).suffix.lower()
    if extension != ".zip":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only ZIP image datasets are supported for this milestone.")

    if not (0.0 < train_ratio < 1.0 and 0.0 < validation_ratio < 1.0 and 0.0 < test_ratio < 1.0):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Split ratios must all be strictly between 0 and 1.")

    if not (abs((train_ratio + validation_ratio + test_ratio) - 1.0) < 1e-9):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Training, validation, and test ratios must sum to 1.0.")

    temp_dir = Path(tempfile.mkdtemp(prefix="image_api_"))
    extracted_root: Path | None = None

    try:
        zip_path = temp_dir / safe_filename
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_root, summary = extract_zip_dataset(zip_path)
        split = summary.get("predefined_split")
        if split is None:
            split = split_dataset(
                summary,
                seed=seed,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
            )
        training_result = train_image_classifier(
            extracted_root,
            class_to_index=summary["class_to_index"],
            split=split,
            image_labels=summary["image_labels"],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            pretrained=pretrained,
        )

        package = save_image_model_package(
            training_result["model"],
            {
                "model_type": "Image Classification",
                "architecture": "MobileNetV2",
                "class_names": summary["class_names"],
                "class_to_index": summary["class_to_index"],
                "image_size": 224,
                "channels": 3,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "model_metrics": {"test_accuracy": training_result["test_accuracy"]},
                "training_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "pretrained_requested": pretrained,
                    "pretrained_weights_loaded": training_result["pretrained_weights_loaded"],
                },
                "pretrained_weights_loaded": training_result["pretrained_weights_loaded"],
                "package_version": "1.0.0",
                "training_samples": training_result["training_samples"],
                "validation_samples": training_result["validation_samples"],
                "test_samples": training_result["test_samples"],
            },
            storage_dir=PROJECT_ROOT / "storage" / "image_models",
        )

        response = {
            "task": "Image Classification",
            "model": "MobileNetV2",
            "classes": summary["class_names"],
            "metrics": {
                "accuracy": float(training_result["test_accuracy"]),
            },
            "samples": {
                "train": int(training_result["training_samples"]),
                "validation": int(training_result["validation_samples"]),
                "test": int(training_result["test_samples"]),
            },
            "training_time_seconds": float(training_result["training_time_seconds"]),
            "model_id": package["model_id"],
            "model_type": "Image Classification",
            "class_to_index": summary["class_to_index"],
        }
        return response
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    finally:
        if extracted_root is not None:
            shutil.rmtree(extracted_root, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
