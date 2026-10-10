from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.nn.functional import softmax
from torchvision import transforms

from .persistence import load_image_model_package

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _validate_image_file(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image '{path.name}' does not exist.")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix or 'unknown'}")

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            return rgb_image
    except Exception as exc:  # pragma: no cover - defensive guard around image decoding
        raise ValueError(f"Corrupt or unreadable image file: {path.name}") from exc


def predict_image_file(image_path: str | Path, model_id: str, *, storage_dir: str | Path | None = None) -> dict[str, object]:
    if model_id is None or not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("A valid model_id is required for image prediction.")

    package = load_image_model_package(model_id, storage_dir=storage_dir)
    manifest = package["manifest"]
    model = package["model"]
    architecture = manifest.get("architecture")
    if architecture != "MobileNetV2":
        raise ValueError(f"Unsupported image model architecture: {architecture}")

    class_names = manifest.get("class_names")
    class_to_index = manifest.get("class_to_index")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Image model manifest is missing a valid class list.")
    if not isinstance(class_to_index, dict) or not class_to_index:
        raise ValueError("Image model manifest is missing a valid class mapping.")

    image = _validate_image_file(image_path)
    image_size = int(manifest.get("image_size", 224))
    mean = manifest.get("normalization_mean", [0.485, 0.456, 0.406])
    std = manifest.get("normalization_std", [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    tensor = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = softmax(logits, dim=1).squeeze(0)

    ordered_class_names = list(class_names)
    if len(ordered_class_names) != len(probabilities):
        raise ValueError("Image model class count does not match the loaded architecture output dimension.")

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = ordered_class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())
    probability_map = {ordered_class_names[index]: float(probabilities[index].item()) for index in range(len(ordered_class_names))}

    return {
        "model_id": model_id,
        "architecture": architecture,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_map,
    }
