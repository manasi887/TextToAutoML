"""Utilities for loading and validating image datasets for CNN training.

This module is intentionally independent from the NLP pipeline and only handles
image classification datasets stored in a standard train/validation/test layout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import tensorflow as tf

logger = logging.getLogger(__name__)


def _get_class_names(split_dir: Path) -> list[str]:
    """Return sorted class directory names for a dataset split."""
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def validate_dataset_structure(dataset_root: str | Path) -> None:
    """Validate the required image dataset directory layout.

    The dataset is expected to have three top-level folders: train, validation,
    and test. Each split must contain at least one class folder, and all splits
    must share the same class-folder names in the same order.

    Args:
        dataset_root: Root directory containing the train, validation, and test
            folders.

    Raises:
        FileNotFoundError: If the root directory or any expected split directory is
            missing.
        ValueError: If the split layout is invalid or class names do not match.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    required_splits = ("train", "validation", "test")
    split_dirs: dict[str, Path] = {}

    for split_name in required_splits:
        split_dir = root / split_name
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split '{split_name}' not found at: {split_dir}"
            )
        if not split_dir.is_dir():
            raise NotADirectoryError(
                f"Dataset split '{split_name}' is not a directory: {split_dir}"
            )
        split_dirs[split_name] = split_dir

    class_names_by_split: dict[str, list[str]] = {}
    for split_name, split_dir in split_dirs.items():
        class_names = _get_class_names(split_dir)
        if not class_names:
            raise ValueError(
                f"Dataset split '{split_name}' does not contain any class folders: {split_dir}"
            )
        class_names_by_split[split_name] = class_names

    training_names = class_names_by_split["train"]
    for split_name in ("validation", "test"):
        split_names = class_names_by_split[split_name]
        if split_names != training_names:
            raise ValueError(
                "Class folders are inconsistent across dataset splits. "
                f"Training classes: {training_names}; "
                f"{split_name} classes: {split_names}."
            )

    logger.info(
        "Dataset structure validated successfully for root '%s' with classes: %s",
        root,
        training_names,
    )


def _normalize_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Convert image pixels from 0-255 uint8 to floating point values in [0, 1]."""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_image_datasets(
    dataset_root: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load train, validation, and test image datasets from a directory layout.

    Args:
        dataset_root: Root directory that contains train, validation, and test
            folders.
        image_size: Width and height to resize all images to.
        batch_size: Number of images in each dataset batch.
        seed: Random seed to use for dataset shuffling and deterministic loading.

    Returns:
        A tuple containing the train dataset, validation dataset, test dataset, and
        the ordered class names from the training split.

    Raises:
        FileNotFoundError: If the dataset root or required split folders are missing.
        ValueError: If the dataset structure is invalid.
    """
    root = Path(dataset_root)
    validate_dataset_structure(root)

    train_dir = root / "train"
    validation_dir = root / "validation"
    test_dir = root / "test"

    class_names = _get_class_names(train_dir)
    if not class_names:
        raise ValueError(f"Training dataset does not contain any classes: {train_dir}")

    logger.info(
        "Loading image datasets from '%s' with image_size=%s, batch_size=%s",
        root,
        image_size,
        batch_size,
    )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(validation_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    train_dataset = train_dataset.map(
        _normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.map(
        _normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(
        _normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    logger.info("Finished loading datasets. Class names: %s", class_names)
    return train_dataset, validation_dataset, test_dataset, class_names


__all__ = [
    "validate_dataset_structure",
    "load_image_datasets",
]
