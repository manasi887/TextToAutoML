from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import build_mobilenet_v2_classifier

DEFAULT_IMAGE_STORAGE_DIR = (Path(__file__).resolve().parents[3] / "storage" / "image_models").resolve()


def generate_image_model_id(prefix: str = "img") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{prefix}_{timestamp}_{suffix}"


def save_image_model_package(model: nn.Module, manifest: dict[str, Any], *, storage_dir: str | Path | None = None) -> dict[str, Any]:
    target_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_IMAGE_STORAGE_DIR
    safe_dir = Path(target_dir).resolve()
    safe_dir.mkdir(parents=True, exist_ok=True)

    model_id = generate_image_model_id()
    package_dir = safe_dir / model_id
    package_dir.mkdir(parents=True, exist_ok=True)

    model_path = package_dir / "model.pt"
    manifest_path = package_dir / "manifest.json"
    torch.save(model.state_dict(), model_path)
    manifest_payload = {
        "model_type": "Image Classification",
        "architecture": "MobileNetV2",
        "model_id": model_id,
        "package_version": manifest.get("package_version", "1.0.0"),
        **manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest_payload, file, indent=2, sort_keys=True)

    return {
        "model_id": model_id,
        "package_dir": str(package_dir),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
    }


def load_image_model_package(model_id: str, *, storage_dir: str | Path | None = None) -> dict[str, Any]:
    if not model_id or not isinstance(model_id, str):
        raise ValueError("Model ID must be a non-empty string.")

    target_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_IMAGE_STORAGE_DIR
    safe_dir = Path(target_dir).resolve()
    package_dir = (safe_dir / model_id).resolve()
    if not package_dir.is_relative_to(safe_dir):
        raise ValueError("Unsafe model package path detected.")
    manifest_path = package_dir / "manifest.json"
    model_path = package_dir / "model.pt"
    if not manifest_path.exists():
        raise FileNotFoundError("Image model manifest not found.")
    if not model_path.exists():
        raise FileNotFoundError("Image model weights not found.")

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest.get("class_to_index"), dict) or not manifest.get("class_names"):
        raise ValueError("Image model manifest is missing a valid class mapping.")
    if manifest.get("architecture") not in {"MobileNetV2"}:
        raise ValueError("Unsupported image model architecture in manifest.")

    model = build_mobilenet_v2_classifier(num_classes=len(manifest["class_names"]), pretrained=False)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model_id": model_id,
        "model": model,
        "manifest": manifest,
    }
