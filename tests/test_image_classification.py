import io
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app
from services.image_classification.dataset import extract_zip_dataset, split_dataset, validate_zip_dataset
from services.image_classification.model import build_mobilenet_v2_classifier
from services.image_classification.persistence import load_image_model_package, save_image_model_package
from services.image_classification.predict import predict_image_file
from services.image_classification.training import ImagePathDataset, train_image_classifier
from services.image_classification import model as image_model
from api import image_classification as image_api


def _make_rgb_image(size=(32, 32), color=(255, 0, 0)):
    image = Image.new("RGB", size, color=color)
    return image


def _write_image_file(path: Path, color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    _make_rgb_image(color=color).save(path)


def _make_dataset_zip(path: Path, classes: dict[str, int], *, include_unsupported: bool = False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for class_name, count in classes.items():
            for index in range(count):
                image_name = f"sample_{index}.jpg"
                archive.writestr(f"dataset/{class_name}/{image_name}", b"")
        if include_unsupported:
            archive.writestr("dataset/cat/notes.txt", "not an image")


def _write_real_images_to_zip(path: Path, class_map: dict[str, int]):
    buffer = io.BytesIO()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for class_index, (class_name, count) in enumerate(class_map.items()):
            for index in range(count):
                image_bytes = io.BytesIO()
                Image.new(
                    "RGB",
                    (32, 32),
                    color=(
                        (index * 20 + class_index * 80) % 255,
                        (index * 30 + class_index * 55) % 255,
                        (50 + class_index * 70) % 255,
                    ),
                ).save(image_bytes, format="JPEG")
                archive.writestr(f"dataset/{class_name}/image_{index}.jpg", image_bytes.getvalue())


def _write_custom_images_to_zip(path: Path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, mode, size, image_format in entries:
            image_bytes = io.BytesIO()
            if mode == "L":
                color = 128
            elif mode == "RGBA":
                color = (40, 80, 120, 255)
            else:
                color = (40, 80, 120)
            Image.new(mode, size, color=color).save(image_bytes, format=image_format)
            archive.writestr(member_name, image_bytes.getvalue())


def _encoded_image_bytes(size=(16, 16), color=(10, 20, 30), image_format="JPEG"):
    image_bytes = io.BytesIO()
    Image.new("RGB", size, color=color).save(image_bytes, format=image_format)
    return image_bytes.getvalue()


def _write_raw_image_entries(path: Path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, image_bytes in entries:
            archive.writestr(member_name, image_bytes)


def _write_segmentation_layout_zip(path: Path, *, conflicting_training_label: bool = False):
    entries = []
    train_images = {}
    for class_index, class_name in enumerate(("buildings", "forest")):
        for index in range(3):
            contents = _encoded_image_bytes(
                color=(20 + class_index * 100 + index * 15, 30 + index * 20, 40)
            )
            train_images[(class_name, index)] = contents
            image_name = (
                "17378.jpg"
                if class_name == "forest" and index == 0
                else f"{index}.jpg"
            )
            entries.append((
                f"seg_train/seg_train/{class_name}/{image_name}",
                contents,
            ))
    if conflicting_training_label:
        entries.append((
            "seg_train/seg_train/forest/conflicting.jpg",
            train_images[("buildings", 0)],
        ))
    entries.extend([
        ("seg_test/seg_test/buildings/20057.jpg", _encoded_image_bytes(color=(210, 20, 30))),
        ("seg_pred/seg_pred/10225.jpg", train_images[("forest", 0)]),
    ])
    _write_raw_image_entries(path, entries)


def _write_presplit_duplicate_zip(path: Path, *, duplicate_across_splits: bool):
    split_names = ("train", "validation", "test")
    classes = ("cats", "dogs")
    encoded = {}
    for split_index, split_name in enumerate(split_names):
        for class_index, class_name in enumerate(classes):
            color = (
                15 + split_index * 60 + class_index * 20,
                20 + split_index * 30 + class_index * 40,
                25 + split_index * 20 + class_index * 50,
            )
            encoded[(split_name, class_name)] = _encoded_image_bytes(color=color)
    if duplicate_across_splits:
        encoded[("test", "cats")] = encoded[("train", "cats")]
    _write_raw_image_entries(
        path,
        [
            (f"{split_name}/{class_name}/image.jpg", encoded[(split_name, class_name)])
            for split_name in split_names
            for class_name in classes
        ],
    )


def _post_image_analysis(client, archive_path: Path, filename: str = "dataset.zip"):
    with archive_path.open("rb") as dataset_file:
        return client.post(
            "/image/analyze",
            files={"file": (filename, dataset_file, "application/zip")},
        )


def _write_pre_split_images_to_zip(
    path: Path,
    *,
    wrapper: str | None = None,
    missing_class=None,
    missing_split: str | None = None,
):
    class_counts = {
        "train": {"cats": 2, "dogs": 1},
        "validation": {"cats": 1, "dogs": 2},
        "test": {"cats": 2, "dogs": 2},
    }
    if missing_class:
        split_name, class_name = missing_class
        class_counts[split_name].pop(class_name)
    if missing_split:
        class_counts.pop(missing_split)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split_index, (split_name, classes) in enumerate(class_counts.items()):
            for class_index, (class_name, count) in enumerate(classes.items()):
                for index in range(count):
                    image_bytes = io.BytesIO()
                    Image.new(
                        "RGB",
                        (16, 16),
                        color=(
                            20 + split_index * 60 + class_index * 25 + index * 7,
                            25 + split_index * 50 + class_index * 30 + index * 9,
                            30 + split_index * 40 + class_index * 35 + index * 11,
                        ),
                    ).save(
                        image_bytes, format="JPEG"
                    )
                    prefix = f"{wrapper}/" if wrapper else ""
                    archive.writestr(
                        f"{prefix}{split_name}/{class_name}/{index}.jpg",
                        image_bytes.getvalue(),
                    )


def test_valid_zip_dataset_is_accepted(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    _write_real_images_to_zip(archive_path, {"cat": 4, "dog": 4})

    summary = validate_zip_dataset(archive_path)

    assert summary["class_names"] == ["cat", "dog"]
    assert summary["class_to_index"] == {"cat": 0, "dog": 1}
    assert summary["image_count"] == 8
    assert summary["valid_image_count"] == 8
    assert summary["per_class_counts"] == {"cat": 4, "dog": 4}
    assert summary["unique_per_class_counts"] == {"cat": 4, "dog": 4}
    assert summary["predefined_split"] is None


def test_multiple_class_folders_are_discovered_and_mapping_is_deterministic(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    _write_real_images_to_zip(archive_path, {"zebra": 3, "cat": 3, "dog": 3})

    summary = validate_zip_dataset(archive_path)

    assert summary["class_names"] == ["cat", "dog", "zebra"]
    assert summary["class_to_index"] == {"cat": 0, "dog": 1, "zebra": 2}


def test_unsupported_and_corrupt_images_are_reported_as_skipped(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for class_name in ("cat", "dog"):
            for index in range(3):
                archive.writestr(
                    f"dataset/{class_name}/{index}.jpg",
                    _encoded_image_bytes(color=(index * 30 + (class_name == "dog") * 100, 20, 40)),
                )
        archive.writestr("dataset/cat/unsupported.gif", b"not-real-gif")
        archive.writestr("dataset/dog/corrupt.jpg", b"not-a-jpeg")

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        report = image_api.analyze_extracted_dataset(
            extracted_root,
            summary,
            archive_size_bytes=archive_path.stat().st_size,
        )

        assert summary["total_file_count"] == 8
        assert summary["valid_image_count"] == 6
        assert summary["skipped_file_count"] == 2
        assert {item["path"] for item in summary["skipped_files"]} == {
            "dataset/cat/unsupported.gif",
            "dataset/dog/corrupt.jpg",
        }
        assert report["quality"]["validation"] == "passed"
        assert report["dataset"]["skipped_file_count"] == 2
        assert any("unsupported file type" in item["reason"] for item in report["dataset"]["skipped_files"])
        assert any("corrupt or unreadable image" in item["reason"] for item in report["dataset"]["skipped_files"])
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_same_class_duplicate_images_are_deduplicated_for_training(tmp_path):
    archive_path = tmp_path / "same_class_duplicates.zip"
    image_a = _encoded_image_bytes(color=(10, 20, 30))
    image_b = _encoded_image_bytes(color=(40, 50, 60))
    image_c = _encoded_image_bytes(color=(70, 80, 90))
    dog_images = [
        _encoded_image_bytes(color=(110, 20, 30)),
        _encoded_image_bytes(color=(140, 50, 60)),
        _encoded_image_bytes(color=(170, 80, 90)),
    ]
    entries = [
        ("dataset/cats/a.jpg", image_a),
        ("dataset/cats/a-copy.jpg", image_a),
        ("dataset/cats/b.jpg", image_b),
        ("dataset/cats/c.jpg", image_c),
        *[(f"dataset/dogs/{index}.jpg", contents) for index, contents in enumerate(dog_images)],
    ]
    _write_raw_image_entries(archive_path, entries)
    original_bytes = archive_path.read_bytes()

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        assert summary["per_class_counts"] == {"cats": 4, "dogs": 3}
        assert summary["unique_per_class_counts"] == {"cats": 3, "dogs": 3}
        assert len(summary["images_by_class"]["cats"]) == 3
        assert len(summary["source_images_by_class"]["cats"]) == 4
        assert summary["duplicate_group_count"] == 1
        split = split_dataset(summary, seed=42)
        assert all(len(paths) == 2 for paths in split.values())
        assert len(set().union(*map(set, split.values()))) == 6
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)

    assert archive_path.read_bytes() == original_bytes


def test_same_class_duplicates_within_provided_split_are_deduplicated(tmp_path):
    archive_path = tmp_path / "provided_split_duplicates.zip"
    entries = []
    source_image = None
    for split_index, split_name in enumerate(("train", "validation", "test")):
        for class_index, class_name in enumerate(("cats", "dogs")):
            contents = _encoded_image_bytes(
                color=(20 + split_index * 60 + class_index * 30, 30, 40)
            )
            entries.append((f"{split_name}/{class_name}/image.jpg", contents))
            if split_name == "train" and class_name == "cats":
                source_image = contents
    assert source_image is not None
    entries.append(("train/cats/copy.jpg", source_image))
    _write_raw_image_entries(archive_path, entries)

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        assert len(summary["predefined_split"]["train"]) == 2
        assert len(summary["predefined_split"]["validation"]) == 2
        assert len(summary["predefined_split"]["test"]) == 2
        assert summary["duplicate_group_count"] == 1
        report = image_api.analyze_extracted_dataset(
            extracted_root,
            summary,
            archive_size_bytes=archive_path.stat().st_size,
        )
        assert report["quality"]["validation"] == "passed"
        assert report["splits"]["per_class_counts"]["cats"] == {
            "train": 1,
            "validation": 1,
            "test": 1,
        }
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_segmentation_layout_excludes_unlabeled_folders_before_duplicate_checks(tmp_path):
    archive_path = tmp_path / "segmentation_layout.zip"
    _write_segmentation_layout_zip(archive_path)

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        assert summary["class_names"] == ["buildings", "forest"]
        assert summary["image_count"] == 6
        assert summary["valid_image_count"] == 8
        assert summary["excluded_image_counts"] == {"seg_test": 1, "seg_pred": 1}
        assert summary["duplicate_group_count"] == 0
        forest_training_image = (
            extracted_root / "seg_train" / "seg_train" / "forest" / "17378.jpg"
        )
        prediction_image = extracted_root / "seg_pred" / "seg_pred" / "10225.jpg"
        assert forest_training_image.read_bytes() == prediction_image.read_bytes()
        assert all(
            Path(image_path).relative_to(extracted_root).parts[:2]
            == ("seg_train", "seg_train")
            for paths in summary["images_by_class"].values()
            for image_path in paths
        )
        assert all(
            Path(image_path).relative_to(extracted_root).parts[0]
            not in {"seg_test", "seg_pred"}
            for image_path in summary["image_labels"]
        )
        split = split_dataset(summary)
        assert all(
            Path(image_path).relative_to(extracted_root).parts[0] == "seg_train"
            for paths in split.values()
            for image_path in paths
        )
        report = image_api.analyze_extracted_dataset(
            extracted_root,
            summary,
            archive_size_bytes=archive_path.stat().st_size,
        )
        assert report["quality"]["validation"] == "passed"
        assert report["dataset"]["excluded_image_counts"] == {
            "seg_test": 1,
            "seg_pred": 1,
        }
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_segmentation_layout_still_rejects_conflicting_labeled_classes(tmp_path):
    archive_path = tmp_path / "segmentation_training_conflict.zip"
    _write_segmentation_layout_zip(
        archive_path,
        conflicting_training_label=True,
    )

    with pytest.raises(ValueError, match="conflicting class labels") as error:
        extract_zip_dataset(archive_path)

    message = str(error.value)
    assert "seg_train/seg_train/buildings/0.jpg" in message
    assert "seg_train/seg_train/forest/conflicting.jpg" in message


def test_byte_identical_images_with_conflicting_class_labels_report_both_files(tmp_path):
    archive_path = tmp_path / "conflicting_labels.zip"
    duplicate = _encoded_image_bytes(color=(10, 20, 30))
    _write_raw_image_entries(
        archive_path,
        [
            ("dataset/cats/cat-copy.jpg", duplicate),
            ("dataset/dogs/dog-copy.jpg", duplicate),
        ],
    )

    with pytest.raises(ValueError) as error:
        extract_zip_dataset(archive_path)

    message = str(error.value)
    assert "conflicting class labels" in message
    assert "cats: dataset/cats/cat-copy.jpg" in message
    assert "dogs: dataset/dogs/dog-copy.jpg" in message


def test_empty_class_directory_is_reported_and_blocks_training(tmp_path):
    archive_path = tmp_path / "empty_class.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dataset/empty_class/", "")
        for class_name, offset in (("cats", 0), ("dogs", 100)):
            for index in range(3):
                archive.writestr(
                    f"dataset/{class_name}/{index}.jpg",
                    _encoded_image_bytes(color=(offset + index * 20, 30, 40)),
                )

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        report = image_api.analyze_extracted_dataset(
            extracted_root,
            summary,
            archive_size_bytes=archive_path.stat().st_size,
        )
        assert summary["per_class_counts"]["empty_class"] == 0
        assert summary["unique_per_class_counts"]["empty_class"] == 0
        assert report["quality"]["validation"] == "failed"
        assert any("empty_class" in finding for finding in report["quality"]["findings"])
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_class_with_fewer_than_three_unique_images_fails_preflight_and_split(tmp_path):
    archive_path = tmp_path / "insufficient_unique_images.zip"
    _write_raw_image_entries(
        archive_path,
        [
            ("dataset/cats/a.jpg", _encoded_image_bytes(color=(10, 20, 30))),
            ("dataset/cats/a-copy.jpg", _encoded_image_bytes(color=(10, 20, 30))),
            ("dataset/cats/b.jpg", _encoded_image_bytes(color=(40, 50, 60))),
            *[
                (
                    f"dataset/dogs/{index}.jpg",
                    _encoded_image_bytes(color=(100 + index * 30, 50, 60)),
                )
                for index in range(3)
            ],
        ],
    )

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        report = image_api.analyze_extracted_dataset(
            extracted_root,
            summary,
            archive_size_bytes=archive_path.stat().st_size,
        )
        assert report["quality"]["validation"] == "failed"
        assert any("cats (2)" in finding for finding in report["quality"]["findings"])
        with pytest.raises(ValueError, match="Class 'cats' needs at least 3 unique image contents"):
            split_dataset(summary)
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_zip_path_traversal_is_rejected(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../../outside.jpg", b"not allowed")

    with pytest.raises(ValueError, match="unsafe|traversal|path"):
        validate_zip_dataset(archive_path)


def test_empty_class_folder_and_fewer_than_two_classes_are_rejected(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dataset/cat/.keep", "")

    with pytest.raises(ValueError, match="No valid supported image files|at least 2|class"):
        validate_zip_dataset(archive_path)

    single_class_path = tmp_path / "single.zip"
    _write_real_images_to_zip(single_class_path, {"cat": 3})
    with pytest.raises(ValueError, match="at least 2|class"):
        validate_zip_dataset(single_class_path)


def test_split_is_deterministic_and_non_overlapping(tmp_path):
    dataset_root = tmp_path / "images"
    dataset_root.mkdir()
    for class_index, (class_name, count) in enumerate({"cat": 8, "dog": 8}.items()):
        class_dir = dataset_root / class_name
        class_dir.mkdir()
        for idx in range(count):
            _write_image_file(
                class_dir / f"{idx}.jpg",
                color=(class_index * 120 + idx * 12, 30 + idx * 15, 80 + class_index * 40),
            )

    summary = {
        "class_names": ["cat", "dog"],
        "class_to_index": {"cat": 0, "dog": 1},
        "images_by_class": {
            "cat": [str(dataset_root / "cat" / f"{idx}.jpg") for idx in range(8)],
            "dog": [str(dataset_root / "dog" / f"{idx}.jpg") for idx in range(8)],
        },
    }

    first = split_dataset(summary, seed=42)
    second = split_dataset(summary, seed=42)

    assert first == second
    merged = set(first["train"]) | set(first["validation"]) | set(first["test"])
    assert len(merged) == 16
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])


def test_small_split_keeps_every_class_in_each_partition(tmp_path):
    dataset_root = tmp_path / "small_images"
    images_by_class = {}
    for class_index, class_name in enumerate(("cat", "dog")):
        class_dir = dataset_root / class_name
        paths = []
        for index in range(3):
            image_path = class_dir / f"{index}.jpg"
            _write_image_file(
                image_path,
                color=(class_index * 120 + index * 35, 40 + index * 45, 70 + class_index * 60),
            )
            paths.append(str(image_path))
        images_by_class[class_name] = paths
    summary = {"class_names": ["cat", "dog"], "images_by_class": images_by_class}

    split = split_dataset(summary, seed=42)

    for partition in ("train", "validation", "test"):
        assert len(split[partition]) == 2
        assert {Path(path).parent.name for path in split[partition]} == {"cat", "dog"}
    assert split == split_dataset(summary, seed=42)
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_split_rejects_class_too_small_for_all_partitions(tmp_path):
    images_by_class = {"cat": [], "dog": []}
    for class_index, (class_name, count) in enumerate((("cat", 2), ("dog", 3))):
        for index in range(count):
            image_path = tmp_path / class_name / f"{index}.jpg"
            _write_image_file(
                image_path,
                color=(class_index * 120 + index * 35, 40 + index * 45, 70 + class_index * 60),
            )
            images_by_class[class_name].append(str(image_path))

    with pytest.raises(ValueError, match="Class 'cat' needs at least 3 unique image contents"):
        split_dataset({"class_names": ["cat", "dog"], "images_by_class": images_by_class})


def test_nested_image_paths_keep_validator_assigned_class(tmp_path):
    archive_path = tmp_path / "nested_dataset.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for class_index, class_name in enumerate(("cat", "dog")):
            for index in range(3):
                image_bytes = io.BytesIO()
                Image.new(
                    "RGB",
                    (16, 16),
                    color=(index * 20 + class_index * 100, 30 + class_index * 60, 40),
                ).save(image_bytes, format="JPEG")
                archive.writestr(
                    f"dataset/{class_name}/nested/subfolder/{index}.jpg",
                    image_bytes.getvalue(),
                )

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        paths = [path for class_paths in summary["images_by_class"].values() for path in class_paths]
        image_dataset = ImagePathDataset(
            paths,
            summary["class_to_index"],
            summary["image_labels"],
        )
        labels_by_class = {}
        for image_path, label in zip(paths, [image_dataset[index][1].item() for index in range(len(paths))]):
            labels_by_class.setdefault(summary["image_labels"][image_path], set()).add(label)

        assert summary["class_names"] == ["cat", "dog"]
        assert labels_by_class == {"cat": {0}, "dog": {1}}
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_pre_split_layout_uses_nested_class_directories_and_original_partitions(tmp_path):
    archive_path = tmp_path / "pre_split.zip"
    _write_pre_split_images_to_zip(archive_path)

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        assert summary["class_names"] == ["cats", "dogs"]
        assert summary["class_to_index"] == {"cats": 0, "dogs": 1}
        assert {
            split_name: len(paths)
            for split_name, paths in summary["predefined_split"].items()
        } == {"train": 3, "validation": 3, "test": 4}
        for split_name, paths in summary["predefined_split"].items():
            assert all(
                Path(path).relative_to(extracted_root).parts[0] == split_name
                for path in paths
            )
            assert {summary["image_labels"][path] for path in paths} == {"cats", "dogs"}
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_pre_split_layout_with_common_wrapper_is_recognized(tmp_path):
    archive_path = tmp_path / "wrapped_pre_split.zip"
    _write_pre_split_images_to_zip(archive_path, wrapper="dataset")

    extracted_root, summary = extract_zip_dataset(archive_path)
    try:
        assert summary["class_names"] == ["cats", "dogs"]
        assert {
            split_name: len(paths)
            for split_name, paths in summary["predefined_split"].items()
        } == {"train": 3, "validation": 3, "test": 4}
        assert all(
            Path(path).relative_to(extracted_root).parts[:2] == ("dataset", split_name)
            for split_name, paths in summary["predefined_split"].items()
            for path in paths
        )
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def test_pre_split_layout_rejects_inconsistent_class_sets(tmp_path):
    archive_path = tmp_path / "inconsistent_pre_split.zip"
    _write_pre_split_images_to_zip(
        archive_path,
        missing_class=("validation", "dogs"),
    )

    with pytest.raises(ValueError, match="same class directories"):
        validate_zip_dataset(archive_path)


def test_pre_split_layout_requires_all_expected_split_directories(tmp_path):
    archive_path = tmp_path / "missing_test_split.zip"
    _write_pre_split_images_to_zip(archive_path, missing_split="test")

    with pytest.raises(ValueError, match="must contain train, validation, and test"):
        validate_zip_dataset(archive_path)


def test_image_api_preserves_pre_split_partitions_without_random_split(tmp_path, monkeypatch):
    archive_path = tmp_path / "pre_split_api.zip"
    _write_pre_split_images_to_zip(archive_path)
    observed = {}

    def unexpected_random_split(*args, **kwargs):
        raise AssertionError("random splitter must not be called for a pre-split dataset")

    def capture_training(dataset_path, *, class_to_index, split, image_labels, **kwargs):
        observed["split"] = split
        observed["class_to_index"] = class_to_index
        observed["image_labels"] = image_labels
        return {
            "model": object(),
            "test_accuracy": 0.75,
            "training_samples": len(split["train"]),
            "validation_samples": len(split["validation"]),
            "test_samples": len(split["test"]),
            "training_time_seconds": 1.0,
            "pretrained_weights_loaded": False,
        }

    def capture_package(model, manifest, **kwargs):
        observed["manifest"] = manifest
        return {"model_id": "img_presplit_test"}

    monkeypatch.setattr(image_api, "split_dataset", unexpected_random_split)
    monkeypatch.setattr(image_api, "train_image_classifier", capture_training)
    monkeypatch.setattr(image_api, "save_image_model_package", capture_package)

    with archive_path.open("rb") as dataset_file:
        response = TestClient(app).post(
            "/image/train",
            files={"file": (archive_path.name, dataset_file, "application/zip")},
            data={"pretrained": "false"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["classes"] == ["cats", "dogs"]
    assert payload["class_to_index"] == {"cats": 0, "dogs": 1}
    assert payload["samples"] == {"train": 3, "validation": 3, "test": 4}
    assert observed["manifest"]["class_names"] == ["cats", "dogs"]
    assert observed["manifest"]["class_to_index"] == {"cats": 0, "dogs": 1}
    assert observed["manifest"]["training_samples"] == 3
    assert observed["manifest"]["validation_samples"] == 3
    assert observed["manifest"]["test_samples"] == 4
    for split_name, paths in observed["split"].items():
        assert len(paths) == payload["samples"][split_name]
        assert {observed["image_labels"][path] for path in paths} == {"cats", "dogs"}


def test_image_api_rejects_cross_partition_duplicate_content(tmp_path):
    archive_path = tmp_path / "pre_split_leakage.zip"
    _write_presplit_duplicate_zip(archive_path, duplicate_across_splits=True)

    with archive_path.open("rb") as dataset_file:
        response = TestClient(app).post(
            "/image/train",
            files={"file": (archive_path.name, dataset_file, "application/zip")},
            data={"pretrained": "false"},
        )

    assert response.status_code == 400
    assert "Exact duplicate image content crosses the test, train partitions" in response.json()["detail"]
    assert "held-out evaluation would be contaminated" in response.json()["detail"]


def test_extraction_temp_directory_is_removed_when_validation_fails(tmp_path, monkeypatch):
    extracted_root = tmp_path / "validation_extract"
    extracted_root.mkdir()
    monkeypatch.setattr(
        "services.image_classification.dataset._extract_zip_to_tempdir",
        lambda _: extracted_root,
    )
    monkeypatch.setattr(
        "services.image_classification.dataset._validate_extracted_dataset",
        lambda _: (_ for _ in ()).throw(ValueError("invalid dataset")),
    )

    with pytest.raises(ValueError, match="invalid dataset"):
        extract_zip_dataset(tmp_path / "unused.zip")

    assert not extracted_root.exists()


def test_image_analysis_reports_balanced_unsplit_dataset_and_never_trains(
    tmp_path, monkeypatch
):
    archive_path = tmp_path / "balanced.zip"
    _write_real_images_to_zip(archive_path, {"cats": 3, "dogs": 3})
    for function_name in (
        "train_image_classifier",
        "save_image_model_package",
        "predict_image_file",
    ):
        monkeypatch.setattr(
            image_api,
            function_name,
            lambda *args, **kwargs: pytest.fail(f"{function_name} must not be called"),
        )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dataset"]["image_count"] == 6
    assert payload["dataset"]["class_count"] == 2
    assert payload["dataset"]["archive_size_bytes"] == archive_path.stat().st_size
    with zipfile.ZipFile(archive_path) as archive:
        expected_extracted_size = sum(member.file_size for member in archive.infolist() if not member.is_dir())
    assert payload["dataset"]["extracted_size_bytes"] == expected_extracted_size
    assert payload["classes"]["names"] == ["cats", "dogs"]
    assert payload["classes"]["images_per_class"] == {"cats": 3, "dogs": 3}
    assert payload["classes"]["distribution"] == {
        "cats": {"count": 3, "percentage": 50.0},
        "dogs": {"count": 3, "percentage": 50.0},
    }
    assert payload["splits"] == {
        "source": "not_provided",
        "counts": None,
        "per_class_counts": None,
    }
    assert payload["quality"]["validation"] == "passed"
    assert payload["quality"]["findings"] == []
    assert payload["quality"]["metrics"]["class_balance"]["imbalance_ratio"] == 1.0
    assert payload["cnn_suitability"] == {"status": "not_assessed", "reasons": []}


def test_image_analysis_reports_imbalanced_distribution_without_suitability_score(tmp_path):
    archive_path = tmp_path / "imbalanced.zip"
    _write_real_images_to_zip(archive_path, {"cats": 3, "dogs": 1})

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["classes"]["distribution"] == {
        "cats": {"count": 3, "percentage": 75.0},
        "dogs": {"count": 1, "percentage": 25.0},
    }
    assert payload["cnn_suitability"] == {"status": "not_assessed", "reasons": []}
    assert payload["quality"]["metrics"]["class_balance"] == {
        "largest_class_count": 3,
        "smallest_class_count": 1,
        "imbalance_ratio": 3.0,
    }
    assert payload["warnings"] == []


def test_image_analysis_reports_pre_split_totals_and_per_class_counts_without_resplitting(
    tmp_path, monkeypatch
):
    archive_path = tmp_path / "pre_split.zip"
    _write_pre_split_images_to_zip(archive_path)
    monkeypatch.setattr(
        image_api,
        "split_dataset",
        lambda *args, **kwargs: pytest.fail("analysis must not generate a split"),
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["splits"] == {
        "source": "provided",
        "counts": {"train": 3, "validation": 3, "test": 4},
        "per_class_counts": {
            "cats": {"train": 2, "validation": 1, "test": 2},
            "dogs": {"train": 1, "validation": 2, "test": 2},
        },
    }


def test_image_analysis_reports_dimensions_modes_channels_and_decoded_formats(tmp_path):
    archive_path = tmp_path / "metadata.zip"
    _write_custom_images_to_zip(
        archive_path,
        [
            ("dataset/cats/wide.jpg", "RGB", (20, 10), "JPEG"),
            ("dataset/cats/gray.png", "L", (10, 10), "PNG"),
            ("dataset/dogs/alpha.png", "RGBA", (5, 10), "PNG"),
        ],
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    properties = response.json()["image_properties"]
    assert properties["extensions"] == {".jpg": 1, ".png": 2}
    assert properties["formats"] == {"JPEG": 1, "PNG": 2}
    assert properties["width"] == {"min": 5, "max": 20, "average": 11.67}
    assert properties["height"] == {"min": 10, "max": 10, "average": 10.0}
    assert properties["aspect_ratio"] == {"min": 0.5, "max": 2.0, "average": 1.1667}
    assert properties["modes"] == {"L": 1, "RGB": 1, "RGBA": 1}
    assert properties["channels"] == {"1": 1, "3": 1, "4": 1}
    assert "training converts images to RGB" in properties["channel_note"]
    assert response.json()["quality"]["metrics"]["dimension_consistency"] == {
        "consistent": False,
        "unique_dimensions": 3,
    }
    assert "Images have inconsistent dimensions." in response.json()["quality"]["findings"]


def test_image_analysis_reports_consistent_dimensions_and_resolution_statistics(tmp_path):
    archive_path = tmp_path / "resolution.zip"
    _write_custom_images_to_zip(
        archive_path,
        [
            ("dataset/cats/a.jpg", "RGB", (2, 2), "JPEG"),
            ("dataset/cats/b.png", "L", (2, 3), "PNG"),
            ("dataset/dogs/c.png", "RGBA", (2, 4), "PNG"),
            ("dataset/dogs/d.jpg", "RGB", (3, 4), "JPEG"),
        ],
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    metrics = response.json()["quality"]["metrics"]
    assert metrics["dimension_consistency"] == {
        "consistent": False,
        "unique_dimensions": 4,
    }
    assert metrics["resolution"] == {
        "min_pixel_area": 4,
        "max_pixel_area": 12,
        "average_pixel_area": 7.5,
        "median_pixel_area": 7.0,
        "median_method": "exact_disk_backed_order_statistic",
    }


def test_image_analysis_marks_identical_dimensions_consistent(tmp_path):
    archive_path = tmp_path / "consistent_dimensions.zip"
    _write_custom_images_to_zip(
        archive_path,
        [
            ("dataset/cats/a.jpg", "RGB", (12, 8), "JPEG"),
            ("dataset/dogs/b.png", "L", (12, 8), "PNG"),
        ],
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    assert response.json()["quality"]["metrics"]["dimension_consistency"] == {
        "consistent": True,
        "unique_dimensions": 1,
    }
    assert "Images have inconsistent dimensions." not in response.json()["quality"]["findings"]


def test_image_analysis_reports_minimum_class_size_and_existing_split_rule(tmp_path):
    archive_path = tmp_path / "class_size.zip"
    _write_real_images_to_zip(archive_path, {"cats": 4, "dogs": 3})

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    metrics = response.json()["quality"]["metrics"]
    assert metrics["class_size"] == {
        "minimum_images_per_class": 3,
        "minimum_unique_images_per_class": 3,
    }
    assert metrics["split_feasibility"] == {
        "minimum_images_per_class": 3,
        "minimum_unique_images_per_class": 3,
        "supports_existing_minimum_3_per_class_rule": True,
    }


def test_image_analysis_distinguishes_existing_split_rule_from_general_suitability(tmp_path):
    archive_path = tmp_path / "below_split_rule.zip"
    _write_real_images_to_zip(archive_path, {"cats": 2, "dogs": 1})

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    metrics = response.json()["quality"]["metrics"]
    assert metrics["class_size"]["minimum_images_per_class"] == 1
    assert metrics["split_feasibility"]["supports_existing_minimum_3_per_class_rule"] is False
    assert response.json()["cnn_suitability"] == {"status": "not_assessed", "reasons": []}
    assert response.json()["warnings"] == []


def test_image_analysis_reports_exact_byte_duplicate_groups_and_relative_paths(tmp_path):
    archive_path = tmp_path / "duplicates.zip"
    duplicate_bytes = _encoded_image_bytes()
    _write_raw_image_entries(
        archive_path,
        [
            ("dataset/cats/first.jpg", duplicate_bytes),
            ("dataset/cats/copy.jpg", duplicate_bytes),
            ("dataset/dogs/dog.jpg", _encoded_image_bytes(color=(200, 10, 50))),
        ],
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    metrics = response.json()["quality"]["metrics"]
    duplicates = metrics["duplicates"]
    assert duplicates["method"] == "sha256_file_bytes"
    assert duplicates["duplicate_file_count"] == 2
    assert duplicates["duplicate_group_count"] == 1
    assert duplicates["groups"] == [
        {
            "hash": duplicates["groups"][0]["hash"],
            "file_count": 2,
            "files": ["dataset/cats/copy.jpg", "dataset/cats/first.jpg"],
            "files_truncated": False,
        }
    ]
    assert all(not Path(file_path).is_absolute() for file_path in duplicates["groups"][0]["files"])
    assert "Exact duplicate image files were detected." in response.json()["quality"]["findings"]


def test_image_analysis_does_not_call_different_encodings_near_duplicates(tmp_path):
    archive_path = tmp_path / "different_encodings.zip"
    same_pixels = Image.new("RGB", (16, 16), color=(30, 60, 90))
    jpeg_bytes = io.BytesIO()
    png_bytes = io.BytesIO()
    same_pixels.save(jpeg_bytes, format="JPEG")
    same_pixels.save(png_bytes, format="PNG")
    _write_raw_image_entries(
        archive_path,
        [
            ("dataset/cats/image.jpg", jpeg_bytes.getvalue()),
            ("dataset/dogs/image.png", png_bytes.getvalue()),
        ],
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    duplicates = response.json()["quality"]["metrics"]["duplicates"]
    assert duplicates["duplicate_file_count"] == 0
    assert duplicates["duplicate_group_count"] == 0
    assert duplicates["groups"] == []


def test_image_analysis_returns_bounded_duplicate_group_samples(tmp_path):
    archive_path = tmp_path / "large_duplicate_group.zip"
    duplicate_bytes = _encoded_image_bytes()
    entries = [
        (f"dataset/cats/copy_{index:03}.jpg", duplicate_bytes)
        for index in range(25)
    ]
    entries.append(("dataset/dogs/other.jpg", _encoded_image_bytes(color=(1, 2, 3))))
    _write_raw_image_entries(archive_path, entries)

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    duplicates = response.json()["quality"]["metrics"]["duplicates"]
    assert duplicates["duplicate_file_count"] == 25
    assert duplicates["duplicate_group_count"] == 1
    assert len(duplicates["groups"]) == 1
    assert duplicates["groups"][0]["file_count"] == 25
    assert len(duplicates["groups"][0]["files"]) == 10
    assert duplicates["groups"][0]["files_truncated"] is True


def test_image_analysis_caps_returned_duplicate_groups_without_losing_counts(tmp_path):
    archive_path = tmp_path / "many_duplicate_groups.zip"
    entries = []
    for group_index in range(101):
        duplicate_bytes = _encoded_image_bytes(
            size=(2, 2),
            color=(group_index + 1, (group_index * 3) % 255, (group_index * 7) % 255),
            image_format="PNG",
        )
        entries.extend(
            [
                (f"dataset/cats/group_{group_index:03}_a.png", duplicate_bytes),
                (f"dataset/cats/group_{group_index:03}_b.png", duplicate_bytes),
            ]
        )
    entries.append(("dataset/dogs/unique.png", _encoded_image_bytes(color=(250, 1, 2), image_format="PNG")))
    _write_raw_image_entries(archive_path, entries)

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    duplicates = response.json()["quality"]["metrics"]["duplicates"]
    assert duplicates["duplicate_file_count"] == 202
    assert duplicates["duplicate_group_count"] == 101
    assert len(duplicates["groups"]) == 100
    assert duplicates["groups_truncated"] is True


@pytest.mark.parametrize("duplicate_across_splits", [True, False])
def test_image_analysis_checks_exact_duplicates_across_provided_splits(
    tmp_path, duplicate_across_splits
):
    archive_path = tmp_path / f"split_duplicates_{duplicate_across_splits}.zip"
    _write_presplit_duplicate_zip(
        archive_path,
        duplicate_across_splits=duplicate_across_splits,
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    metrics = response.json()["quality"]["metrics"]
    cross_split = metrics["cross_split_duplicates"]
    assert cross_split["checked"] is True
    if duplicate_across_splits:
        assert cross_split["match_count"] == 1
        assert cross_split["matches"][0]["splits"] == ["test", "train"]
        assert "Possible exact-duplicate split leakage was detected." in response.json()["quality"]["findings"]
    else:
        assert cross_split["match_count"] == 0
        assert cross_split["matches"] == []


def test_image_analysis_reports_cross_split_duplicates_unchecked_for_unsplit_data(tmp_path):
    archive_path = tmp_path / "unsplit_duplicates.zip"
    _write_real_images_to_zip(archive_path, {"cats": 2, "dogs": 2})

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    cross_split = response.json()["quality"]["metrics"]["cross_split_duplicates"]
    assert cross_split == {
        "checked": False,
        "reason": "No predefined train/validation/test splits were provided.",
    }


def test_image_analysis_quality_metrics_are_deterministic(tmp_path):
    archive_path = tmp_path / "deterministic_metrics.zip"
    duplicate_bytes = _encoded_image_bytes()
    _write_raw_image_entries(
        archive_path,
        [
            ("dataset/cats/first.jpg", duplicate_bytes),
            ("dataset/cats/copy.jpg", duplicate_bytes),
            ("dataset/dogs/other.jpg", _encoded_image_bytes(color=(200, 10, 50))),
        ],
    )

    first = _post_image_analysis(TestClient(app), archive_path)
    second = _post_image_analysis(TestClient(app), archive_path)

    assert first.status_code == second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["quality"]["metrics"] == second_payload["quality"]["metrics"]
    assert first_payload["quality"]["findings"] == second_payload["quality"]["findings"]


@pytest.mark.parametrize(
    ("filename", "contents", "detail"),
    [
        ("dataset.csv", b"not a zip", "Only ZIP"),
        ("dataset.zip", b"not a zip", "not a valid ZIP"),
    ],
)
def test_image_analysis_rejects_non_zip_and_invalid_zip(tmp_path, filename, contents, detail):
    client = TestClient(app)
    response = client.post(
        "/image/analyze",
        files={"file": (filename, contents, "application/octet-stream")},
    )

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_image_analysis_reports_corrupt_image_without_training(tmp_path, monkeypatch):
    archive_path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        image_bytes = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image_bytes, format="JPEG")
        archive.writestr("dataset/cats/valid.jpg", image_bytes.getvalue())
        archive.writestr("dataset/cats/corrupt.jpg", b"not a jpeg")
        dog_image_bytes = io.BytesIO()
        Image.new("RGB", (8, 8), color=(20, 80, 120)).save(dog_image_bytes, format="JPEG")
        archive.writestr("dataset/dogs/valid.jpg", dog_image_bytes.getvalue())
    monkeypatch.setattr(
        image_api,
        "train_image_classifier",
        lambda *args, **kwargs: pytest.fail("corrupt dataset must not train"),
    )

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dataset"]["valid_image_count"] == 2
    assert payload["dataset"]["skipped_file_count"] == 1
    assert payload["dataset"]["skipped_files"][0]["path"] == "dataset/cats/corrupt.jpg"
    assert "corrupt or unreadable image" in payload["dataset"]["skipped_files"][0]["reason"]


def test_image_analysis_reports_unsupported_files(tmp_path):
    archive_path = tmp_path / "unsupported.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        image_bytes = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image_bytes, format="JPEG")
        archive.writestr("dataset/cats/cat.jpg", image_bytes.getvalue())
        archive.writestr("dataset/cats/notes.gif", b"unsupported")
        dog_image_bytes = io.BytesIO()
        Image.new("RGB", (8, 8), color=(20, 80, 120)).save(dog_image_bytes, format="JPEG")
        archive.writestr("dataset/dogs/dog.jpg", dog_image_bytes.getvalue())

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dataset"]["skipped_file_count"] == 1
    assert payload["dataset"]["skipped_files"][0]["path"] == "dataset/cats/notes.gif"
    assert "unsupported file type" in payload["dataset"]["skipped_files"][0]["reason"]


def test_image_analysis_rejects_unsafe_zip_paths(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../../outside.jpg", b"unsafe")

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 400
    assert "unsafe" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    ("filename", "archive_options", "detail"),
    [
        ("missing_split.zip", {"missing_split": "validation"}, "missing: validation"),
        (
            "mismatched_classes.zip",
            {"missing_class": ("validation", "dogs")},
            "same class directories",
        ),
    ],
)
def test_image_analysis_rejects_invalid_pre_split_layouts(
    tmp_path, filename, archive_options, detail
):
    archive_path = tmp_path / filename
    _write_pre_split_images_to_zip(archive_path, **archive_options)

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_image_analysis_rejects_fewer_than_two_classes(tmp_path):
    archive_path = tmp_path / "one_class.zip"
    _write_real_images_to_zip(archive_path, {"cats": 2})

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 400
    assert "at least 2 classes" in response.json()["detail"]


def test_image_analysis_cleans_upload_and_extraction_directories_after_success(
    tmp_path, monkeypatch
):
    archive_path = tmp_path / "cleanup.zip"
    _write_real_images_to_zip(archive_path, {"cats": 1, "dogs": 1})
    temporary_directories = []

    def tracked_mkdtemp(prefix):
        directory = tmp_path / f"{prefix}{len(temporary_directories)}"
        directory.mkdir()
        temporary_directories.append(directory)
        return str(directory)

    monkeypatch.setattr(image_api.tempfile, "mkdtemp", tracked_mkdtemp)

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 200, response.text
    assert len(temporary_directories) == 2
    assert all(not directory.exists() for directory in temporary_directories)


def test_image_analysis_cleans_upload_and_extraction_directories_after_validation_failure(
    tmp_path, monkeypatch
):
    archive_path = tmp_path / "validation_failure.zip"
    _write_real_images_to_zip(archive_path, {"cats": 3})
    temporary_directories = []

    def tracked_mkdtemp(prefix):
        directory = tmp_path / f"{prefix}{len(temporary_directories)}"
        directory.mkdir()
        temporary_directories.append(directory)
        return str(directory)

    monkeypatch.setattr(image_api.tempfile, "mkdtemp", tracked_mkdtemp)

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 400
    assert len(temporary_directories) == 2
    assert all(not directory.exists() for directory in temporary_directories)


def test_image_analysis_cleans_upload_and_extraction_directories_after_analysis_failure(
    tmp_path, monkeypatch
):
    archive_path = tmp_path / "analysis_failure.zip"
    _write_real_images_to_zip(archive_path, {"cats": 1, "dogs": 1})
    temporary_directories = []

    def tracked_mkdtemp(prefix):
        directory = tmp_path / f"{prefix}{len(temporary_directories)}"
        directory.mkdir()
        temporary_directories.append(directory)
        return str(directory)

    def fail_analysis(*args, **kwargs):
        raise ValueError("simulated analysis failure")

    monkeypatch.setattr(image_api.tempfile, "mkdtemp", tracked_mkdtemp)
    monkeypatch.setattr(image_api, "analyze_extracted_dataset", fail_analysis)

    response = _post_image_analysis(TestClient(app), archive_path)

    assert response.status_code == 400
    assert response.json()["detail"] == "simulated analysis failure"
    assert len(temporary_directories) == 2
    assert all(not directory.exists() for directory in temporary_directories)


def test_model_output_dimension_matches_number_of_classes():
    model = build_mobilenet_v2_classifier(num_classes=3, pretrained=False)

    assert model.classifier[1].out_features == 3


class _FakeMobileNetV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 1000))


def test_pretrained_status_true_when_imagenet_weights_load(monkeypatch):
    calls = []

    def fake_mobilenet_v2(*, weights):
        calls.append(weights)
        return _FakeMobileNetV2()

    monkeypatch.setattr(image_model.models, "mobilenet_v2", fake_mobilenet_v2)

    model = build_mobilenet_v2_classifier(num_classes=2, pretrained=True)

    assert calls == [image_model.MobileNet_V2_Weights.IMAGENET1K_V1]
    assert model.pretrained_weights_loaded is True
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_pretrained_status_false_when_imagenet_loading_falls_back(monkeypatch):
    calls = []

    def fake_mobilenet_v2(*, weights):
        calls.append(weights)
        if weights is not None:
            raise OSError("weights unavailable")
        return _FakeMobileNetV2()

    monkeypatch.setattr(image_model.models, "mobilenet_v2", fake_mobilenet_v2)

    model = build_mobilenet_v2_classifier(num_classes=2, pretrained=True)

    assert calls == [image_model.MobileNet_V2_Weights.IMAGENET1K_V1, None]
    assert model.pretrained_weights_loaded is False


def test_training_function_works_with_tiny_test_model_dataset(tmp_path):
    dataset_root = tmp_path / "images"
    dataset_root.mkdir()
    for class_index, (class_name, count) in enumerate({"cat": 4, "dog": 4}.items()):
        class_dir = dataset_root / class_name
        class_dir.mkdir()
        for idx in range(count):
            _write_image_file(
                class_dir / f"{idx}.jpg",
                color=(class_index * 120 + idx * 35, 80 + class_index * 50, 100 + idx * 25),
            )

    class_to_index = {"cat": 0, "dog": 1}
    image_labels = {
        str(dataset_root / class_name / f"{index}.jpg"): class_name
        for class_name in class_to_index
        for index in range(4)
    }
    split = {
        "train": [str(dataset_root / "cat" / "0.jpg"), str(dataset_root / "cat" / "1.jpg"), str(dataset_root / "dog" / "0.jpg"), str(dataset_root / "dog" / "1.jpg")],
        "validation": [str(dataset_root / "cat" / "2.jpg"), str(dataset_root / "dog" / "2.jpg")],
        "test": [str(dataset_root / "cat" / "3.jpg"), str(dataset_root / "dog" / "3.jpg")],
    }

    result = train_image_classifier(
        dataset_path=dataset_root,
        class_to_index=class_to_index,
        split=split,
        image_labels=image_labels,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        seed=42,
        pretrained=False,
    )

    assert result["model_name"] == "MobileNetV2"
    assert result["num_classes"] == 2
    assert result["training_samples"] == 4
    assert result["validation_samples"] == 2
    assert result["test_samples"] == 2
    assert result["training_time_seconds"] >= 0
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["pretrained_weights_loaded"] is False


def test_image_model_package_can_be_saved_and_loaded(tmp_path):
    model = build_mobilenet_v2_classifier(num_classes=2, pretrained=False)
    metadata = {
        "model_type": "Image Classification",
        "architecture": "MobileNetV2",
        "class_names": ["cat", "dog"],
        "class_to_index": {"cat": 0, "dog": 1},
        "image_size": 224,
        "channels": 3,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "model_metrics": {"test_accuracy": 0.91},
        "pretrained_weights_loaded": False,
        "training_config": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
        "package_version": "1.0.0",
    }

    package = save_image_model_package(model, metadata, storage_dir=tmp_path)
    loaded = load_image_model_package(package["model_id"], storage_dir=tmp_path)

    assert loaded["manifest"]["class_names"] == ["cat", "dog"]
    assert loaded["manifest"]["pretrained_weights_loaded"] is False
    assert loaded["model"].classifier[1].out_features == 2


def test_api_training_contract_works(tmp_path):
    archive_path = tmp_path / "image_dataset.zip"
    _write_real_images_to_zip(archive_path, {"cat": 4, "dog": 4})

    client = TestClient(app)
    with archive_path.open("rb") as dataset_file:
        response = client.post(
            "/image/train",
            files={"file": (archive_path.name, dataset_file, "application/zip")},
            data={"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "seed": 42, "train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15, "pretrained": False},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task"] == "Image Classification"
    assert payload["model"] == "MobileNetV2"
    assert payload["classes"] == ["cat", "dog"]
    assert payload["samples"]["train"] > 0
    assert payload["training_time_seconds"] >= 0
    assert payload["model_id"]
    storage_dir = Path(__file__).resolve().parents[1] / "storage" / "image_models"
    try:
        loaded = load_image_model_package(payload["model_id"], storage_dir=storage_dir)
        assert loaded["manifest"]["pretrained_weights_loaded"] is False
        assert loaded["manifest"]["training_config"]["pretrained_requested"] is False
    finally:
        shutil.rmtree(storage_dir / payload["model_id"], ignore_errors=True)


@pytest.mark.parametrize("training_fails", [False, True])
def test_image_training_cleans_temps_and_uses_uploaded_basename(
    tmp_path, monkeypatch, training_fails
):
    temp_dirs = []
    extracted_root = tmp_path / "extracted_images"
    permanent_models = tmp_path / "permanent_models"
    permanent_models.mkdir()

    def make_temp_dir(prefix):
        path = tmp_path / f"{prefix}{len(temp_dirs)}"
        path.mkdir()
        temp_dirs.append(path)
        return str(path)

    def extract(zip_path):
        assert zip_path.parent == temp_dirs[0]
        assert zip_path.name == "dataset.zip"
        assert zip_path.exists()
        extracted_root.mkdir()
        return extracted_root, {
            "class_names": ["cat", "dog"],
            "class_to_index": {"cat": 0, "dog": 1},
            "image_labels": {},
        }

    def train(*args, **kwargs):
        if training_fails:
            raise RuntimeError("simulated training failure")
        return {
            "model": object(),
            "test_accuracy": 0.5,
            "training_samples": 4,
            "validation_samples": 2,
            "test_samples": 2,
            "training_time_seconds": 1.0,
            "pretrained_weights_loaded": False,
        }

    monkeypatch.setattr(image_api.tempfile, "mkdtemp", make_temp_dir)
    monkeypatch.setattr(image_api, "extract_zip_dataset", extract)
    monkeypatch.setattr(image_api, "split_dataset", lambda *args, **kwargs: {
        "train": ["train"], "validation": ["validation"], "test": ["test"]
    })
    monkeypatch.setattr(image_api, "train_image_classifier", train)
    monkeypatch.setattr(
        image_api,
        "save_image_model_package",
        lambda *args, **kwargs: {"model_id": "img_test"},
    )

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/image/train",
        files={"file": ("../../dataset.zip", b"zip", "application/zip")},
    )

    assert response.status_code == (500 if training_fails else 200)
    assert not extracted_root.exists()
    assert all(not directory.exists() for directory in temp_dirs)
    assert permanent_models.is_dir()


def test_image_prediction_uses_uploaded_basename_inside_temp_directory(tmp_path, monkeypatch):
    temp_dir = tmp_path / "image_predict_tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(image_api.tempfile, "mkdtemp", lambda **kwargs: str(temp_dir))

    def predict(image_path, model_id, **kwargs):
        assert image_path.parent == temp_dir
        assert image_path.name == "cat.jpg"
        assert image_path.read_bytes() == b"image bytes"
        return {
            "model_id": model_id,
            "architecture": "MobileNetV2",
            "predicted_class": "cat",
            "confidence": 0.9,
            "probabilities": {"cat": 0.9, "dog": 0.1},
        }

    monkeypatch.setattr(image_api, "predict_image_file", predict)
    response = TestClient(app).post(
        "/image/predict",
        files={"file": ("../../cat.jpg", b"image bytes", "image/jpeg")},
        data={"model_id": "img_test"},
    )

    assert response.status_code == 200
    assert response.json()["prediction"]["class"] == "cat"
    assert not temp_dir.exists()


def test_valid_image_prediction_uses_manifest_class_order(tmp_path):
    model = build_mobilenet_v2_classifier(num_classes=2, pretrained=False)
    with torch.no_grad():
        model.classifier[1].weight.zero_()
        model.classifier[1].bias.copy_(torch.tensor([0.0, 8.0]))

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), color=(255, 255, 255)).save(image_path)

    package = save_image_model_package(
        model,
        {
            "model_type": "Image Classification",
            "architecture": "MobileNetV2",
            "class_names": ["cat", "dog"],
            "class_to_index": {"cat": 0, "dog": 1},
            "image_size": 224,
            "channels": 3,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "model_metrics": {"test_accuracy": 0.91},
            "training_config": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
            "package_version": "1.0.0",
        },
        storage_dir=tmp_path,
    )

    result = predict_image_file(image_path, package["model_id"], storage_dir=tmp_path)

    assert result["predicted_class"] == "dog"
    assert result["model_id"] == package["model_id"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert list(result["probabilities"].keys()) == ["cat", "dog"]
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_prediction_rejects_missing_model_invalid_manifest_and_unsupported_inputs(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(image_path)

    with pytest.raises(ValueError, match="valid model_id"):
        predict_image_file(image_path, "", storage_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        predict_image_file(image_path, "missing_model", storage_dir=tmp_path)

    model = build_mobilenet_v2_classifier(num_classes=2, pretrained=False)
    package = save_image_model_package(
        model,
        {
            "model_type": "Image Classification",
            "architecture": "MobileNetV2",
            "class_names": ["cat", "dog"],
            "class_to_index": {"cat": 0, "dog": 1},
            "image_size": 224,
            "channels": 3,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "package_version": "1.0.0",
        },
        storage_dir=tmp_path,
    )

    invalid_manifest_path = tmp_path / package["model_id"] / "manifest.json"
    invalid_manifest_path.write_text('{"architecture": "UnsupportedNet", "class_names": ["cat", "dog"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported image model architecture|valid class mapping|class"):
        predict_image_file(image_path, package["model_id"], storage_dir=tmp_path)

    invalid_manifest_path.write_text(
        '{"model_type": "Image Classification", "architecture": "MobileNetV2", "class_names": ["cat", "dog"], "class_to_index": {"cat": 0, "dog": 1}, "image_size": 224, "channels": 3, "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225], "package_version": "1.0.0"}',
        encoding="utf-8",
    )

    bad_image = tmp_path / "bad.jpg"
    bad_image.write_bytes(b"not-a-real-image")
    with pytest.raises(ValueError, match="Corrupt|unsupported|image"):
        predict_image_file(bad_image, package["model_id"], storage_dir=tmp_path)

    unsupported_image = tmp_path / "bad.gif"
    unsupported_image.write_bytes(b"not-real-gif")
    with pytest.raises(ValueError, match="unsupported|format"):
        predict_image_file(unsupported_image, package["model_id"], storage_dir=tmp_path)


def test_api_prediction_contract_works(tmp_path):
    model = build_mobilenet_v2_classifier(num_classes=2, pretrained=False)
    with torch.no_grad():
        model.classifier[1].weight.zero_()
        model.classifier[1].bias.copy_(torch.tensor([0.0, 12.0]))

    image_path = tmp_path / "prediction_input.jpg"
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(image_path)

    package = save_image_model_package(
        model,
        {
            "model_type": "Image Classification",
            "architecture": "MobileNetV2",
            "class_names": ["cat", "dog"],
            "class_to_index": {"cat": 0, "dog": 1},
            "image_size": 224,
            "channels": 3,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "package_version": "1.0.0",
        },
        storage_dir=Path(__file__).resolve().parents[1] / "storage" / "image_models",
    )

    client = TestClient(app)
    with image_path.open("rb") as uploaded_image:
        response = client.post(
            "/image/predict",
            files={"file": (image_path.name, uploaded_image, "image/jpeg")},
            data={"model_id": package["model_id"]},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task"] == "Image Classification"
    assert payload["model"] == "MobileNetV2"
    assert payload["prediction"]["class"] == "dog"
    assert 0.0 <= payload["prediction"]["confidence"] <= 1.0
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-6
