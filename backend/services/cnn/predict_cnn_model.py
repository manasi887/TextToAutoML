"""Prediction helper for a saved TensorFlow/Keras CNN image classifier.

This module loads a trained model and metadata, validates the expected image
configuration, and predicts the class for a single image without executing any
training logic or automatic code on import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def _validate_file_path(path_value: str | Path, field_name: str) -> Path:
    """Validate that a path exists and is a file."""
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{field_name} is not a file: {path}")
    return path


def _to_python_float(value: Any) -> float:
    """Convert TensorFlow/NumPy scalar values to a standard Python float."""
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def predict_image(
    model_path: str | Path,
    image_path: str | Path,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict the class of a single image using a saved CNN model.

    Args:
        model_path: Path to the saved Keras model file.
        image_path: Path to the input image file.
        metadata_path: Optional path to the metadata JSON file. If omitted, a
            file named cnn_metadata.json in the same directory as model_path is used.

    Returns:
        A dictionary containing the predicted class, index, confidence, and class
        probabilities.

    Raises:
        FileNotFoundError: If required model, image, or metadata files are missing.
        ValueError: If metadata is invalid, the model is incompatible, or the image
            cannot be loaded or processed.
    """
    model_file = _validate_file_path(model_path, "model_path")
    image_file = _validate_file_path(image_path, "image_path")

    if metadata_path is None:
        metadata_file = model_file.parent / "cnn_metadata.json"
    else:
        metadata_file = _validate_file_path(metadata_path, "metadata_path")

    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata_path does not exist: {metadata_file}")
    if not metadata_file.is_file():
        raise ValueError(f"metadata_path is not a file: {metadata_file}")

    try:
        with metadata_file.open("r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata_path is not valid JSON: {metadata_file}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata JSON must be an object/dictionary: {metadata_file}")

    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(
            "metadata['class_names'] must be a non-empty list of strings; "
            f"received: {class_names!r}"
        )
    if any(not isinstance(class_name, str) or not class_name for class_name in class_names):
        raise ValueError(
            "metadata['class_names'] must contain only non-empty strings; "
            f"received: {class_names!r}"
        )

    image_size = metadata.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError(
            "metadata['image_size'] must be a list containing exactly two positive integers; "
            f"received: {image_size!r}"
        )
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in image_size):
        raise ValueError(
            "metadata['image_size'] values must be positive integers; "
            f"received: {image_size!r}"
        )

    image_height, image_width = [int(value) for value in image_size]

    try:
        model = tf.keras.models.load_model(str(model_file), compile=False)
    except (OSError, ValueError, IOError) as exc:
        raise ValueError(f"Could not load the Keras model from '{model_file}': {exc}") from exc

    model_input_shape = model.input_shape
    if model_input_shape is None:
        raise ValueError("Loaded model has no defined input shape and cannot be validated.")

    expected_input_shape = (None, image_height, image_width, 3)
    if len(model_input_shape) != 4:
        raise ValueError(
            "Model input shape must be rank 4 with format (batch, height, width, channels); "
            f"received: {model_input_shape!r}"
        )

    actual_height = model_input_shape[1]
    actual_width = model_input_shape[2]
    actual_channels = model_input_shape[3]
    if actual_height != image_height or actual_width != image_width or actual_channels != 3:
        raise ValueError(
            "Model input shape is incompatible with metadata image_size and expected RGB channels. "
            f"Expected (height={image_height}, width={image_width}, channels=3); "
            f"received model input shape: {model_input_shape!r}."
        )

    model_output_shape = model.output_shape
    if model_output_shape is None:
        raise ValueError("Loaded model has no defined output shape and cannot be validated.")
    if len(model_output_shape) != 2:
        raise ValueError(
            "Model output shape must be rank 2 with format (batch_size, num_classes); "
            f"received: {model_output_shape!r}"
        )
    expected_num_classes = len(class_names)
    actual_num_classes = model_output_shape[-1]
    if actual_num_classes != expected_num_classes:
        raise ValueError(
            "Model output classes do not match metadata class_names length. "
            f"Expected {expected_num_classes} classes; model output has {actual_num_classes}."
        )

    try:
        image = tf.keras.utils.load_img(
            str(image_file),
            target_size=(image_height, image_width),
            color_mode="rgb",
        )
    except Exception as exc:
        raise ValueError(f"Could not load image as RGB: {image_file}") from exc

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    try:
        probabilities = model.predict(image_batch, verbose=0)
    except Exception as exc:
        raise ValueError(f"Model prediction failed for image '{image_file}': {exc}") from exc

    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probabilities.shape[0] != len(class_names):
        raise ValueError(
            "Prediction output length does not match class_names length. "
            f"Expected {len(class_names)} probabilities, got {probabilities.shape[0]}."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class_name = class_names[predicted_index]
    confidence = _to_python_float(probabilities[predicted_index])

    probability_map = {
        class_name: _to_python_float(probabilities[index])
        for index, class_name in enumerate(class_names)
    }

    logger.info(
        "Predicted class '%s' with confidence %.6f for image '%s'",
        predicted_class_name,
        confidence,
        image_file,
    )

    return {
        "image_path": str(image_file),
        "predicted_class": predicted_class_name,
        "predicted_class_index": predicted_index,
        "confidence": confidence,
        "probabilities": probability_map,
    }


__all__ = ["predict_image"]
