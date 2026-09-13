"""Secure FastAPI endpoints for CNN image-classification training and prediction.

This module provides a dedicated router for uploading a ZIP archive containing an
image dataset, training a TensorFlow/Keras CNN model, and predicting the class of
an uploaded image with a trained model. It intentionally keeps all client-supplied
paths outside the trusted storage area and validates all archive and image inputs
before any extraction or inference is performed.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from services.cnn.dataset import validate_dataset_structure
from services.cnn.predict_cnn_model import predict_image
from services.cnn.train_cnn_model import train_cnn_model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cnn",
    tags=["CNN Image Classification"],
)

STORAGE_ROOT = Path(__file__).resolve().parents[1] / "storage"
CNN_DATASET_ROOT = STORAGE_ROOT / "cnn_datasets"
MODEL_ROOT = STORAGE_ROOT / "models" / "cnn"
PREDICTION_UPLOAD_DIR = STORAGE_ROOT / "cnn_prediction_uploads"

CNN_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
PREDICTION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_DATASET_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_PREDICTION_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_SIZE_BYTES = 500 * 1024 * 1024
MODEL_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


async def _save_upload_with_size_limit(
    upload_file: UploadFile,
    destination_path: Path,
    max_bytes: int,
    field_name: str,
) -> None:
    """Persist an uploaded file in 1 MB chunks while enforcing a byte-size limit."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    chunk_size = 1024 * 1024

    try:
        with destination_path.open("wb") as destination_handle:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(
                        f"Uploaded {field_name} exceeds the maximum allowed size of {max_bytes} bytes."
                    )
                destination_handle.write(chunk)
    except ValueError:
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise
    except Exception:
        if destination_path.exists():
            destination_path.unlink(missing_ok=True)
        raise


def _safe_storage_path(root: Path, *parts: str) -> Path:
    """Build and validate a path under a trusted base directory."""
    target = (root.joinpath(*parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resolved path escapes the trusted storage directory.",
        )
    return target


def _validate_archived_image_name(member_name: str) -> None:
    """Reject archive entries that attempt traversal or unsafe paths."""
    normalized = member_name.replace("\\", "/")
    pure_path = PurePosixPath(normalized)

    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(
            "Archive member paths must be relative and cannot use absolute paths or '..'."
        )

    if pure_path.name:
        suffix = PurePosixPath(pure_path.name).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS and not member_name.endswith("/"):
            raise ValueError(
                "Archive entries may only be directories or image files with extensions: "
                ".jpg, .jpeg, .png, .bmp"
            )


def _validate_zip_archive(file_obj: UploadFile, zip_path: Path) -> tuple[Path, int]:
    """Validate the uploaded ZIP file without allowing path traversal or oversized payloads."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Uploaded archive was not saved to disk: {zip_path}")

    total_uncompressed_size = 0
    entry_count = 0

    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                entry_count += 1
                if entry_count > MAX_ARCHIVE_ENTRIES:
                    raise ValueError(
                        "Archive contains too many entries; maximum allowed is 10,000."
                    )

                if info.flag_bits & 0x1:
                    raise ValueError("Encrypted ZIP entries are not allowed.")

                if info.filename:
                    _validate_archived_image_name(info.filename)

                if info.is_dir():
                    continue

                total_uncompressed_size += info.file_size
                if total_uncompressed_size > MAX_ARCHIVE_SIZE_BYTES:
                    raise ValueError(
                        "Archive exceeds the maximum allowed uncompressed size of 500 MB."
                    )

                if info.filename.endswith("/"):
                    continue

                if not info.filename:
                    raise ValueError("Archive contains an empty file entry.")
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("Uploaded file is not a valid ZIP archive.") from exc

    return zip_path, entry_count


def _resolve_extracted_dataset_root(extract_root: Path) -> Path:
    """Resolve the dataset root after extracting a ZIP archive safely."""
    required_splits = ("train", "validation", "test")

    if all((extract_root / split).is_dir() for split in required_splits):
        return extract_root

    child_dirs = [child for child in extract_root.iterdir() if child.is_dir()]
    if len(child_dirs) == 1:
        wrapper = child_dirs[0]
        if all((wrapper / split).is_dir() for split in required_splits):
            return wrapper

    raise ValueError(
        "Archive must contain train/, validation/, and test/ at the root or under a single wrapper folder."
    )


def _extract_safe_zip(zip_path: Path, destination_root: Path) -> Path:
    """Extract an uploaded dataset archive while preventing path traversal."""
    destination_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if not info.filename:
                continue

            normalized_name = info.filename.replace("\\", "/")
            pure_path = PurePosixPath(normalized_name)
            if pure_path.is_absolute() or ".." in pure_path.parts:
                raise ValueError(
                    "Archive member paths must be relative and cannot use absolute paths or parent traversal."
                )

            safe_path = destination_root.joinpath(*pure_path.parts)
            if not safe_path.is_relative_to(destination_root.resolve()):
                raise ValueError("Archive member path resolves outside the extraction directory.")

            if info.is_dir():
                safe_path.mkdir(parents=True, exist_ok=True)
                continue

            suffix = PurePosixPath(pure_path.name).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                raise ValueError(
                    "Archive entries may only be directories or image files with extensions: "
                    ".jpg, .jpeg, .png, .bmp"
                )

            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, safe_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    dataset_root = _resolve_extracted_dataset_root(destination_root)
    validate_dataset_structure(dataset_root)
    return dataset_root


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON-native values to JSON-safe Python primitives."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


async def _train_cnn_model_from_archive(
    dataset_zip: UploadFile,
    image_height: int,
    image_width: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    dropout_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Validate, extract, and train a CNN model from a ZIP archive upload."""
    if not dataset_zip.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded dataset archive is missing a filename.",
        )

    archive_name = Path(dataset_zip.filename).name
    if not archive_name or Path(archive_name).suffix.lower() != ".zip":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip archive files are allowed for CNN dataset uploads.",
        )

    upload_dir = Path(tempfile.mkdtemp(prefix="cnn_dataset_", dir=str(CNN_DATASET_ROOT)))
    zip_path = upload_dir / f"{uuid.uuid4().hex}.zip"
    extracted_root = upload_dir / "extracted"

    try:
        await _save_upload_with_size_limit(
            upload_file=dataset_zip,
            destination_path=zip_path,
            max_bytes=MAX_DATASET_UPLOAD_BYTES,
            field_name="dataset ZIP",
        )

        _validate_zip_archive(dataset_zip, zip_path)
        dataset_root = _extract_safe_zip(zip_path, extracted_root)

        model_id = uuid.uuid4().hex
        model_output_dir = MODEL_ROOT / model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)

        training_result = await run_in_threadpool(
            train_cnn_model,
            str(dataset_root),
            str(model_output_dir),
            image_size=(image_height, image_width),
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            seed=seed,
        )

        response = {
            "status": "success",
            "model_id": model_id,
            "model_path": str((model_output_dir / "best_cnn_model.keras").resolve()),
            "metadata_path": str((model_output_dir / "cnn_metadata.json").resolve()),
            "class_names": _json_safe(training_result.get("class_names", [])),
            "input_shape": _json_safe(training_result.get("input_shape", [])),
            "test_metrics": _json_safe(training_result.get("test_metrics", {})),
            "training_history": _json_safe(training_result.get("training_history", [])),
        }
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for training failure paths.
        logger.exception("Unexpected CNN training error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while training the CNN model.",
        ) from exc
    finally:
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)


@router.post("/train")
async def train_cnn_dataset(
    dataset_zip: UploadFile = File(...),
    image_height: int = Form(224),
    image_width: int = Form(224),
    batch_size: int = Form(32),
    epochs: int = Form(10),
    learning_rate: float = Form(0.001),
    dropout_rate: float = Form(0.30),
    seed: int = Form(42),
) -> dict[str, Any]:
    """Train a CNN image-classification model from a securely validated ZIP dataset upload."""
    try:
        return await _train_cnn_model_from_archive(
            dataset_zip=dataset_zip,
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            seed=seed,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/predict/{model_id}")
async def predict_cnn_model(
    model_id: str,
    image: UploadFile = File(...),
) -> dict[str, Any]:
    """Predict the class for a single uploaded image using a saved CNN model."""
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_id must be a 32-character lowercase hexadecimal UUID value.",
        )

    if image.filename is None or not image.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is missing a valid filename.",
        )

    suffix = Path(image.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .jpg, .jpeg, .png, and .bmp image uploads are allowed.",
        )

    model_dir = _safe_storage_path(MODEL_ROOT, model_id)
    model_path = _safe_storage_path(model_dir, "best_cnn_model.keras")
    metadata_path = _safe_storage_path(model_dir, "cnn_metadata.json")

    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved CNN model not found for model_id '{model_id}'.",
        )
    if not metadata_path.exists() or not metadata_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved CNN metadata not found for model_id '{model_id}'.",
        )

    temp_path = PREDICTION_UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        await _save_upload_with_size_limit(
            upload_file=image,
            destination_path=temp_path,
            max_bytes=MAX_PREDICTION_IMAGE_UPLOAD_BYTES,
            field_name="prediction image",
        )

        try:
            prediction = await run_in_threadpool(
                predict_image,
                str(model_path),
                str(temp_path),
                str(metadata_path),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        response = {"model_id": model_id, **_json_safe(prediction)}
        return response
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard for prediction failures.
        logger.exception("Unexpected CNN prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while predicting with the CNN model.",
        ) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()


__all__ = ["router"]
