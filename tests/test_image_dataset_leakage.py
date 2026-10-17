import io
import importlib.util
import shutil
import zipfile
from pathlib import Path

import pytest
from PIL import Image

_DATASET_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "backend"
    / "services"
    / "image_classification"
    / "dataset.py"
)
_DATASET_SPEC = importlib.util.spec_from_file_location(
    "image_dataset_under_test",
    _DATASET_MODULE_PATH,
)
if _DATASET_SPEC is None or _DATASET_SPEC.loader is None:
    raise ImportError(f"Could not load image dataset service from {_DATASET_MODULE_PATH}.")
_DATASET_MODULE = importlib.util.module_from_spec(_DATASET_SPEC)
_DATASET_SPEC.loader.exec_module(_DATASET_MODULE)

extract_zip_dataset = _DATASET_MODULE.extract_zip_dataset
split_dataset = _DATASET_MODULE.split_dataset
validate_split_for_leakage = _DATASET_MODULE.validate_split_for_leakage


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 12), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_pre_split_zip(path: Path, *, duplicate_across_splits: bool) -> None:
    duplicate = _image_bytes((10, 20, 30))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split_name, split_color in (
            ("train", (40, 50, 60)),
            ("validation", (70, 80, 90)),
            ("test", (100, 110, 120)),
        ):
            for class_name, class_offset in (("cats", 0), ("dogs", 1)):
                contents = (
                    (
                        duplicate
                        if split_name in {"train", "test"}
                        else _image_bytes((100, 110, 120))
                    )
                    if duplicate_across_splits and class_name == "cats"
                    else _image_bytes(
                        tuple(channel + class_offset for channel in split_color)
                    )
                )
                archive.writestr(
                    f"{split_name}/{class_name}/image.png",
                    contents,
                )


def _write_images(root: Path, class_name: str, images: list[bytes]) -> list[str]:
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, contents in enumerate(images):
        image_path = class_dir / f"{index}.png"
        image_path.write_bytes(contents)
        paths.append(str(image_path))
    return paths


def _digest_to_partition(split: dict[str, list[str]]) -> dict[bytes, str]:
    return {
        Path(image_path).read_bytes(): split_name
        for split_name, paths in split.items()
        for image_path in paths
    }


def test_duplicate_files_within_one_partition_are_allowed(tmp_path):
    duplicate = _image_bytes((10, 20, 30))
    split = {
        "train": [str(tmp_path / "first.png"), str(tmp_path / "copy.png")],
        "validation": [str(tmp_path / "validation.png")],
        "test": [str(tmp_path / "test.png")],
    }
    labels = {}
    for image_path, contents in zip(
        split["train"] + split["validation"] + split["test"],
        [duplicate, duplicate, _image_bytes((40, 50, 60)), _image_bytes((70, 80, 90))],
    ):
        Path(image_path).write_bytes(contents)
        labels[image_path] = "cats"

    validate_split_for_leakage(split, labels)


def test_byte_identical_content_across_partitions_is_rejected(tmp_path):
    duplicate = _image_bytes((10, 20, 30))
    train_path = tmp_path / "train.png"
    test_path = tmp_path / "test.png"
    train_path.write_bytes(duplicate)
    test_path.write_bytes(duplicate)
    split = {
        "train": [str(train_path)],
        "validation": [str(tmp_path / "validation.png")],
        "test": [str(test_path)],
    }
    Path(split["validation"][0]).write_bytes(_image_bytes((40, 50, 60)))
    labels = {path: "cats" for paths in split.values() for path in paths}

    with pytest.raises(ValueError, match="crosses the test, train partitions.*contaminated"):
        validate_split_for_leakage(split, labels)


def test_automatic_split_keeps_all_identical_content_in_one_partition(tmp_path):
    duplicate = _image_bytes((10, 20, 30))
    images_by_class = {
        "cats": _write_images(
            tmp_path,
            "cats",
            [duplicate, duplicate]
            + [_image_bytes((index * 20, 30, 40)) for index in range(1, 4)],
        ),
        "dogs": _write_images(
            tmp_path,
            "dogs",
            [_image_bytes((20, index * 30, 50)) for index in range(4)],
        ),
    }
    summary = {"class_names": ["cats", "dogs"], "images_by_class": images_by_class}

    split = split_dataset(summary, seed=42)
    validate_split_for_leakage(
        split,
        {path: class_name for class_name, paths in images_by_class.items() for path in paths},
    )

    partition_by_bytes = _digest_to_partition(split)
    assert partition_by_bytes[duplicate] in {"train", "validation", "test"}
    assert sum(path.read_bytes() == duplicate for paths in split.values() for path in map(Path, paths)) == 1
    assert sum(len(paths) for paths in split.values()) == 8


def test_pre_split_folders_with_cross_partition_duplicates_are_rejected(tmp_path):
    archive_path = tmp_path / "pre_split_duplicates.zip"
    _write_pre_split_zip(archive_path, duplicate_across_splits=True)
    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        split = summary["predefined_split"]
        with pytest.raises(ValueError, match="crosses the test, train partitions"):
            validate_split_for_leakage(split, summary["image_labels"])
        assert all(
            Path(image_path).relative_to(extracted_root).parts[0] == split_name
            for split_name, paths in split.items()
            for image_path in paths
        )
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_pre_split_folders_without_cross_partition_duplicates_are_accepted(tmp_path):
    archive_path = tmp_path / "pre_split_unique.zip"
    _write_pre_split_zip(archive_path, duplicate_across_splits=False)
    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        validate_split_for_leakage(
            summary["predefined_split"],
            summary["image_labels"],
        )
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_automatic_split_rejects_too_few_unique_contents_per_class(tmp_path):
    duplicate = _image_bytes((10, 20, 30))
    images_by_class = {
        "cats": _write_images(
            tmp_path,
            "cats",
            [duplicate, duplicate, _image_bytes((40, 50, 60))],
        ),
        "dogs": _write_images(
            tmp_path,
            "dogs",
            [_image_bytes((index * 20, 30, 40)) for index in range(9)],
        ),
    }

    with pytest.raises(ValueError, match="Class 'cats' needs at least 3 unique image contents"):
        split_dataset({"class_names": ["cats", "dogs"], "images_by_class": images_by_class})


def test_identical_image_content_with_conflicting_labels_is_rejected(tmp_path):
    duplicate = _image_bytes((10, 20, 30))
    images_by_class = {
        "cats": _write_images(
            tmp_path,
            "cats",
            [duplicate, _image_bytes((40, 50, 60)), _image_bytes((70, 80, 90))],
        ),
        "dogs": _write_images(
            tmp_path,
            "dogs",
            [duplicate, _image_bytes((100, 110, 120)), _image_bytes((130, 140, 150))],
        ),
    }

    with pytest.raises(ValueError, match="conflicting class labels"):
        split_dataset({"class_names": ["cats", "dogs"], "images_by_class": images_by_class})
