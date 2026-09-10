"""Training entry point for the TensorFlow/Keras CNN image-classification module.

This module loads image datasets, builds a CNN model, trains it with validation,
and saves the best model plus metadata. It is intentionally standalone and does
not start training when the file is imported.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

from .dataset import load_image_datasets
from .model import build_cnn_model

logger = logging.getLogger(__name__)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; received: {value!r}")
    return value


def _validate_image_size(image_size: Sequence[int]) -> tuple[int, int]:
    """Validate the image size tuple and convert it to integers."""
    if not isinstance(image_size, (tuple, list)):
        raise ValueError(
            "image_size must be a tuple of two positive integers; "
            f"received: {image_size!r}"
        )
    if len(image_size) != 2:
        raise ValueError(
            "image_size must contain exactly two values: (height, width); "
            f"received: {image_size!r}"
        )

    height, width = image_size
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError(
            "image_size height must be a positive integer; "
            f"received: {height!r}"
        )
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError(
            "image_size width must be a positive integer; "
            f"received: {width!r}"
        )

    return int(height), int(width)


def _as_python_value(value: Any) -> Any:
    """Convert TensorFlow or NumPy values to plain Python values for JSON output."""
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_python_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _as_python_value(item) for key, item in value.items()}
    return value


def _read_csv_history(csv_path: Path) -> list[dict[str, Any]]:
    """Load the CSV history file into a JSON-friendly list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append({key: _as_python_value(value) for key, value in row.items()})
    return rows


def train_cnn_model(
    dataset_root: str | Path,
    model_output_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.30,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a CNN image-classification model and save outputs to disk.

    Args:
        dataset_root: Root directory containing train, validation, and test folders.
        model_output_dir: Directory where model and metadata files will be stored.
        image_size: Target image size as (height, width).
        batch_size: Batch size for the training and validation datasets.
        epochs: Number of training epochs.
        learning_rate: Learning rate passed to the CNN optimizer.
        dropout_rate: Dropout probability used in the CNN model.
        seed: Random seed used for reproducible training.

    Returns:
        A dictionary containing the model path, metadata path, class names,
        input shape, test metrics, and training history.

    Raises:
        ValueError: When validation parameters are invalid.
        FileNotFoundError: When the dataset root or required dataset folders are invalid.
    """
    if not isinstance(dataset_root, (str, Path)):
        raise ValueError(
            "dataset_root must be a path-like value pointing to the dataset root; "
            f"received: {dataset_root!r}"
        )
    if not isinstance(model_output_dir, (str, Path)):
        raise ValueError(
            "model_output_dir must be a path-like value; "
            f"received: {model_output_dir!r}"
        )

    validated_batch_size = _validate_positive_int(batch_size, "batch_size")
    validated_epochs = _validate_positive_int(epochs, "epochs")
    validated_image_size = _validate_image_size(image_size)

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer; received: {seed!r}")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError(
            "learning_rate must be a positive number; "
            f"received: {learning_rate!r}"
        )
    if not isinstance(dropout_rate, (int, float)) or not 0 <= dropout_rate < 1:
        raise ValueError(
            "dropout_rate must be greater than or equal to 0 and less than 1; "
            f"received: {dropout_rate!r}"
        )

    output_dir = Path(model_output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Could not create model_output_dir '{output_dir}': {exc}") from exc

    tf.keras.utils.set_random_seed(seed)

    logger.info(
        "Loading image datasets from '%s' with image_size=%s and batch_size=%s",
        dataset_root,
        validated_image_size,
        validated_batch_size,
    )
    train_dataset, validation_dataset, test_dataset, class_names = load_image_datasets(
        dataset_root=dataset_root,
        image_size=validated_image_size,
        batch_size=validated_batch_size,
        seed=seed,
    )

    input_shape = (validated_image_size[0], validated_image_size[1], 3)
    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=len(class_names),
        learning_rate=float(learning_rate),
        dropout_rate=float(dropout_rate),
    )

    best_model_path = output_dir / "best_cnn_model.keras"
    history_path = output_dir / "training_history.csv"
    metadata_path = output_dir / "cnn_metadata.json"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(history_path),
            append=False,
        ),
    ]

    logger.info("Starting model training for %s epochs", validated_epochs)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=validated_epochs,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(best_model_path)
    test_metrics = best_model.evaluate(test_dataset, return_dict=True)
    test_metrics = _as_python_value(test_metrics)

    metadata = {
        "class_names": list(class_names),
        "image_size": [validated_image_size[0], validated_image_size[1]],
        "batch_size": validated_batch_size,
        "epochs": validated_epochs,
        "learning_rate": float(learning_rate),
        "dropout_rate": float(dropout_rate),
        "test_metrics": test_metrics,
        "model_path": str(best_model_path),
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    training_history = _read_csv_history(history_path)

    result = {
        "model_path": str(best_model_path),
        "metadata_path": str(metadata_path),
        "class_names": list(class_names),
        "input_shape": [input_shape[0], input_shape[1], input_shape[2]],
        "test_metrics": test_metrics,
        "training_history": training_history,
    }

    logger.info("Training completed successfully. Model saved to: %s", best_model_path)
    return result


__all__ = ["train_cnn_model"]
