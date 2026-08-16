from __future__ import annotations

import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


DEFAULT_STORAGE_DIR = (Path(__file__).resolve().parents[2] / "storage" / "models").resolve()
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def generate_model_id(prefix: str = "automl") -> str:
    """Create a unique, file-safe model identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{prefix}_{timestamp}_{suffix}"


def save_model_package(
    model: Any,
    encoders: dict[str, Any],
    feature_names: list[str],
    target_column: str,
    problem_type: str,
    preprocessing: dict[str, Any],
    model_name: str,
    selection_metric: str,
    metrics: dict[str, Any],
    *,
    storage_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the selected model and all metadata needed for future feature reconstruction."""
    target_storage_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_STORAGE_DIR
    safe_storage_dir = Path(target_storage_dir).resolve()
    safe_storage_dir.mkdir(parents=True, exist_ok=True)

    model_id = generate_model_id()
    model_path = safe_storage_dir / f"{model_id}.joblib"

    package = {
        "model_id": model_id,
        "model": model,
        "encoders": encoders,
        "feature_names": feature_names,
        "target_column": target_column,
        "problem_type": problem_type,
        "preprocessing": preprocessing,
        "model_name": model_name,
        "selection_metric": selection_metric,
        "metrics": metrics,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    joblib.dump(package, model_path)

    model_file_name = f"{model_id}.joblib"
    return {
        "model_id": model_id,
        "model_path": model_file_name,
        "model_name": model_name,
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_count": len(feature_names or []),
        "selection_metric": selection_metric,
        "metrics": metrics,
        "created_at": package["created_at"],
    }


def load_model_package(model_id: str, *, storage_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a persisted model package from the model storage directory."""
    if model_id is None or not isinstance(model_id, str):
        raise ValueError("Model ID must be a non-empty string.")
    cleaned_id = model_id.strip()
    if cleaned_id == "":
        raise ValueError("Model ID must not be empty.")
    if not MODEL_ID_PATTERN.fullmatch(cleaned_id):
        raise ValueError("unsafe model_id value; only letters, numbers, and underscores are allowed.")

    target_storage_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_STORAGE_DIR
    safe_storage_dir = Path(target_storage_dir).resolve()
    model_path = (safe_storage_dir / f"{cleaned_id}.joblib").resolve()

    if not model_path.is_relative_to(safe_storage_dir):
        raise ValueError("Unsafe model_id path; it resolves outside the allowed storage directory.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{cleaned_id}' does not exist.")

    package = joblib.load(model_path)
    if not isinstance(package, dict):
        raise ValueError("The persisted model package is malformed.")
    return package
