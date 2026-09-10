from pathlib import Path
import sys

import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from services.cnn.dataset import load_image_datasets, validate_dataset_structure


def _create_valid_dataset(root: Path) -> Path:
    for split in ("train", "validation", "test"):
        for class_name in ("class_a", "class_b"):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image = tf.random.uniform((8, 8, 3), minval=0, maxval=255, dtype=tf.float32)
            encoded = tf.image.encode_png(tf.cast(image, tf.uint8))
            tf.io.write_file(str(class_dir / "sample.png"), encoded)
    return root


def test_load_image_datasets_returns_expected_dataset_properties(tmp_path):
    dataset_root = _create_valid_dataset(tmp_path / "images")

    train_ds, validation_ds, test_ds, class_names = load_image_datasets(
        dataset_root,
        image_size=(16, 16),
        batch_size=2,
        seed=42,
    )

    assert class_names == ["class_a", "class_b"]

    images, labels = next(iter(train_ds))
    assert images.dtype == tf.float32
    assert images.shape[1:] == (16, 16, 3)
    assert tf.reduce_all(images >= 0.0)
    assert tf.reduce_all(images <= 1.0)
    assert validation_ds is not None
    assert test_ds is not None
    assert labels.dtype in {tf.int32, tf.int64}


def test_validate_dataset_structure_raises_file_not_found_for_missing_split(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "train" / "class_a").mkdir(parents=True)
    (dataset_root / "validation" / "class_a").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        validate_dataset_structure(dataset_root)


def test_validate_dataset_structure_raises_value_error_when_class_folders_differ(tmp_path):
    dataset_root = tmp_path / "dataset"
    for split, classes in {
        "train": ("class_a", "class_b"),
        "validation": ("class_a", "class_c"),
        "test": ("class_a", "class_b"),
    }.items():
        for class_name in classes:
            (dataset_root / split / class_name).mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        validate_dataset_structure(dataset_root)
