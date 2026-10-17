from __future__ import annotations

import hashlib
import sqlite3
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_SPLITS = ("train", "validation", "test")
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_DUPLICATE_GROUPS = 100
_MAX_DUPLICATE_PATHS = 10
_MAX_DUPLICATE_PATH_LENGTH = 512


def _safe_member_name(member_name: str) -> str:
    if not member_name:
        raise ValueError("archive contains an empty member name.")
    normalized = member_name.replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("archive contains an empty member path.")
    if normalized.startswith("/") or normalized.startswith("../") or normalized == "..":
        raise ValueError(f"unsafe archive member path: {member_name!r}")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe zip path traversal detected: {member_name!r}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe archive member path: {member_name!r}")
    return normalized


def _detect_wrapper_and_strip(paths: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    if not paths:
        return []
    first_segments = {parts[0] for parts in paths if parts}
    if first_segments <= set(DATASET_SPLITS):
        return paths
    if len(first_segments) == 1 and all(len(parts) >= 2 for parts in paths):
        return [parts[1:] for parts in paths]
    return paths


def _is_segmentation_dataset_layout(
    image_paths: list[tuple[str, ...]],
    directory_paths: list[tuple[str, ...]],
) -> bool:
    train_tree_present = any(
        parts[:2] == ("seg_train", "seg_train") and len(parts) >= 4
        for parts in image_paths
    ) or any(
        parts[:2] == ("seg_train", "seg_train") and len(parts) >= 3
        for parts in directory_paths
    )
    test_tree_present = any(
        parts[:2] == ("seg_test", "seg_test") and len(parts) >= 3
        for parts in image_paths
    ) or any(
        parts[:2] == ("seg_test", "seg_test") and len(parts) >= 2
        for parts in directory_paths
    )
    prediction_tree_present = any(
        parts[:2] == ("seg_pred", "seg_pred") and len(parts) >= 3
        for parts in image_paths
    ) or any(
        parts[:2] == ("seg_pred", "seg_pred") and len(parts) >= 2
        for parts in directory_paths
    )
    return train_tree_present and test_tree_present and prediction_tree_present


def _extract_zip_to_tempdir(zip_path: str | Path) -> Path:
    source_zip = Path(zip_path)
    if not source_zip.exists():
        raise FileNotFoundError(f"ZIP dataset '{source_zip}' does not exist.")
    if not zipfile.is_zipfile(source_zip):
        raise ValueError(f"'{source_zip.name}' is not a valid ZIP archive.")

    temp_root = Path(tempfile.mkdtemp(prefix="image_dataset_"))
    try:
        with zipfile.ZipFile(source_zip, "r") as archive:
            for member in archive.infolist():
                safe_name = _safe_member_name(member.filename)
                target_path = temp_root / safe_name
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    if not target_path.resolve().is_relative_to(temp_root.resolve()):
                        raise ValueError(f"unsafe extraction path detected: {member.filename!r}")
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.parent.is_relative_to(temp_root):
                    raise ValueError(f"unsafe extraction path detected: {member.filename!r}")
                with archive.open(member, "r") as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return temp_root


def _collect_image_entries(
    extracted_root: Path,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], list[dict[str, str]], int]:
    gathered: list[tuple[str, ...]] = []
    directories: list[tuple[str, ...]] = []
    skipped_files: list[dict[str, str]] = []
    total_files = 0
    for path in sorted(extracted_root.rglob("*")):
        if path.is_dir():
            directories.append(tuple(path.relative_to(extracted_root).parts))
            continue
        total_files += 1
        suffix = path.suffix.lower()
        rel = path.relative_to(extracted_root)
        parts = rel.parts
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            skipped_files.append({
                "path": rel.as_posix(),
                "reason": f"unsupported file type '{suffix or 'no extension'}'",
            })
            continue
        try:
            _validate_image_file(path)
        except ValueError as exc:
            skipped_files.append({"path": rel.as_posix(), "reason": str(exc)})
            continue
        gathered.append(tuple(rel.parts))
    return gathered, directories, skipped_files, total_files


def _validate_image_file(path: Path) -> tuple[int, int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            channels = len(image.getbands())
            return width, height, channels
    except Exception as exc:  # pragma: no cover - defensive runtime check
        raise ValueError(f"corrupt or unreadable image file: {path.name}") from exc


def _validate_extracted_dataset(extracted_root: Path) -> dict[str, Any]:
    image_rel_paths, directory_paths, skipped_files, total_files = _collect_image_entries(extracted_root)
    if not image_rel_paths:
        if skipped_files:
            skipped_details = "; ".join(
                f"{entry['path']}: {entry['reason']}" for entry in skipped_files[:10]
            )
            if len(skipped_files) > 10:
                skipped_details += f"; and {len(skipped_files) - 10} more skipped files"
            raise ValueError(
                f"No valid supported image files were found ({total_files} files inspected). "
                f"Skipped files: {skipped_details}."
            )
        raise ValueError("no valid image files were found in the uploaded ZIP dataset.")

    valid_image_count = len(image_rel_paths)
    split_directories = set(DATASET_SPLITS)
    excluded_image_counts: dict[str, int] = {}
    if _is_segmentation_dataset_layout(image_rel_paths, directory_paths):
        training_paths = [
            parts for parts in image_rel_paths
            if parts[:2] == ("seg_train", "seg_train") and len(parts) >= 4
        ]
        malformed_training_paths = [
            parts for parts in image_rel_paths
            if parts[0] == "seg_train"
            and not (parts[:2] == ("seg_train", "seg_train") and len(parts) >= 4)
        ]
        if malformed_training_paths:
            raise ValueError(
                "Images in the seg_train folder must be under "
                "seg_train/seg_train/<class>/."
            )
        unexpected_paths = [
            parts for parts in image_rel_paths
            if parts[0] not in {"seg_train", "seg_test", "seg_pred"}
        ]
        if unexpected_paths:
            raise ValueError(
                "The recognized segmentation dataset layout contains supported images "
                "outside seg_train, seg_test, and seg_pred folders."
            )

        excluded_image_counts = {
            folder: sum(parts[0] == folder for parts in image_rel_paths)
            for folder in ("seg_test", "seg_pred")
        }
        image_rel_paths = training_paths
        normalized_image_paths = [parts[2:] for parts in training_paths]
        normalized_directory_paths = [
            parts[2:]
            for parts in directory_paths
            if len(parts) >= 3 and parts[:2] == ("seg_train", "seg_train")
        ]
        is_pre_split = False
        common_wrapper = None
    else:
        normalized_image_paths = _detect_wrapper_and_strip(image_rel_paths)
        first_segments = {parts[0] for parts in normalized_image_paths if parts}
        is_pre_split = bool(first_segments & split_directories)
        if is_pre_split and first_segments != split_directories:
            missing_splits = sorted(split_directories - first_segments)
            raise ValueError(
                "Pre-split image dataset must contain train, validation, and test directories; "
                f"missing: {', '.join(missing_splits)}."
            )
        original_first_segments = {parts[0] for parts in image_rel_paths if parts}
        common_wrapper = None
        if (
            len(original_first_segments) == 1
            and next(iter(original_first_segments)) not in split_directories
            and all(len(parts) >= 2 for parts in image_rel_paths)
        ):
            common_wrapper = next(iter(original_first_segments))
        normalized_directory_paths = [
            parts[1:]
            if common_wrapper and parts[0] == common_wrapper
            else parts
            for parts in directory_paths
        ]

    class_to_files: dict[str, list[str]] = {}
    source_images_by_class: dict[str, list[str]] = {}
    image_labels: dict[str, str] = {}
    predefined_split = {split_name: [] for split_name in DATASET_SPLITS}
    split_classes = {split_name: set() for split_name in DATASET_SPLITS}

    for normalized_parts in normalized_directory_paths:
        if not normalized_parts:
            continue
        if is_pre_split and normalized_parts[0] in split_directories:
            if len(normalized_parts) >= 2:
                split_classes[normalized_parts[0]].add(normalized_parts[1])
                class_to_files.setdefault(normalized_parts[1], [])
                source_images_by_class.setdefault(normalized_parts[1], [])
        elif not is_pre_split:
            class_to_files.setdefault(normalized_parts[0], [])
            source_images_by_class.setdefault(normalized_parts[0], [])

    for rel_parts, normalized_parts in zip(image_rel_paths, normalized_image_paths):
        minimum_parts = 3 if is_pre_split else 2
        if len(normalized_parts) < minimum_parts:
            raise ValueError("each image must reside inside a class folder.")
        actual_path = extracted_root.joinpath(*rel_parts)
        if is_pre_split:
            split_name, class_name = normalized_parts[:2]
            predefined_split[split_name].append(str(actual_path.resolve()))
            split_classes[split_name].add(class_name)
        else:
            class_name = normalized_parts[0]
        resolved_path = str(actual_path.resolve())
        class_to_files.setdefault(class_name, []).append(resolved_path)
        source_images_by_class.setdefault(class_name, []).append(resolved_path)
        image_labels[resolved_path] = class_name

    if len(class_to_files) < 2:
        raise ValueError("image dataset must contain at least 2 classes.")
    if is_pre_split and any(
        split_classes[split_name] != split_classes["train"]
        for split_name in DATASET_SPLITS[1:]
    ):
        raise ValueError(
            "Pre-split image dataset must contain the same class directories "
            "in train, validation, and test."
        )

    content_groups: dict[str, list[tuple[str, str]]] = {}
    for class_name, file_paths in source_images_by_class.items():
        for file_path in file_paths:
            content_groups.setdefault(_sha256_file(file_path), []).append((class_name, file_path))

    duplicate_groups: list[dict[str, Any]] = []
    for digest, entries in content_groups.items():
        labels = sorted({class_name for class_name, _ in entries})
        if len(labels) > 1:
            conflicting_files = ", ".join(
                f"{class_name}: {Path(file_path).relative_to(extracted_root).as_posix()}"
                for class_name, file_path in entries
            )
            raise ValueError(
                "Byte-identical image content has conflicting class labels "
                f"({conflicting_files}); correct the labels before training."
            )
        if len(entries) > 1:
            duplicate_groups.append({
                "hash": digest,
                "class_name": labels[0],
                "files": [
                    Path(file_path).relative_to(extracted_root).as_posix()
                    for _, file_path in entries
                ],
            })

    if is_pre_split:
        for split_name, paths in predefined_split.items():
            seen_digests: set[str] = set()
            deduplicated_paths: list[str] = []
            for file_path in paths:
                digest = _sha256_file(file_path)
                if digest not in seen_digests:
                    seen_digests.add(digest)
                    deduplicated_paths.append(file_path)
            predefined_split[split_name] = deduplicated_paths
    else:
        for class_name, file_paths in source_images_by_class.items():
            seen_digests: set[str] = set()
            class_to_files[class_name] = []
            for file_path in file_paths:
                digest = _sha256_file(file_path)
                if digest not in seen_digests:
                    seen_digests.add(digest)
                    class_to_files[class_name].append(file_path)

    class_names = sorted(class_to_files)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    per_class_counts = {name: len(class_to_files[name]) for name in class_names}
    source_per_class_counts = {
        name: len(source_images_by_class.get(name, [])) for name in class_names
    }
    unique_per_class_counts = {
        name: len({
            _sha256_file(path)
            for path in source_images_by_class.get(name, [])
        })
        for name in class_names
    }
    image_count = sum(source_per_class_counts.values())
    image_extensions = sorted({
        Path(file_path).suffix.lower()
        for files in source_images_by_class.values()
        for file_path in files
    })

    summary: dict[str, Any] = {
        "class_names": class_names,
        "class_to_index": class_to_index,
        "image_count": image_count,
        "valid_image_count": valid_image_count,
        "excluded_image_counts": excluded_image_counts,
        "total_file_count": total_files,
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files,
        "per_class_counts": source_per_class_counts,
        "unique_per_class_counts": unique_per_class_counts,
        "duplicate_groups": duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": sum(len(group["files"]) for group in duplicate_groups),
        "image_extensions": image_extensions,
        "images_by_class": {name: sorted(class_to_files[name]) for name in class_names},
        "source_images_by_class": {
            name: sorted(source_images_by_class.get(name, [])) for name in class_names
        },
        "image_labels": image_labels,
        "predefined_split": (
            {name: sorted(paths) for name, paths in predefined_split.items()}
            if is_pre_split
            else None
        ),
        "extract_dir": str(extracted_root),
    }
    return summary


def validate_zip_dataset(zip_path: str | Path) -> dict[str, Any]:
    """Validate a ZIP image dataset and return a deterministic summary."""
    extracted_root = _extract_zip_to_tempdir(zip_path)
    return _validate_extracted_dataset(extracted_root)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(
            f"Could not hash image '{Path(path).name}' for duplicate-leakage validation."
        ) from error
    return digest.hexdigest()


def validate_split_for_leakage(
    split: dict[str, list[str]],
    image_labels: dict[str, str],
) -> None:
    """Reject byte-identical image content that crosses partitions or class labels."""
    content_partitions: dict[str, set[str]] = {}
    content_labels: dict[str, set[str]] = {}
    content_paths: dict[str, str] = {}

    for split_name in DATASET_SPLITS:
        for image_path in split.get(split_name, []):
            label = image_labels.get(image_path)
            if label is None:
                raise ValueError(
                    f"Image '{Path(image_path).name}' has no class label for "
                    "duplicate-leakage validation."
                )
            digest = _sha256_file(image_path)
            content_partitions.setdefault(digest, set()).add(split_name)
            content_labels.setdefault(digest, set()).add(label)
            content_paths.setdefault(digest, Path(image_path).name)

    for digest, labels in content_labels.items():
        if len(labels) > 1:
            raise ValueError(
                "Byte-identical image content has conflicting class labels "
                f"('{content_paths[digest]}'); correct the labels before training."
            )

    for digest, partitions in content_partitions.items():
        if len(partitions) > 1:
            names = ", ".join(sorted(partitions))
            raise ValueError(
                "Exact duplicate image content crosses the "
                f"{names} partitions; held-out evaluation would be contaminated. "
                "Remove the duplicates or place all copies in one partition."
            )


def split_dataset(summary: dict[str, Any], *, seed: int = 42, train_ratio: float = 0.70, validation_ratio: float = 0.15, test_ratio: float = 0.15) -> dict[str, list[str]]:
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
    if (
        not np.all(np.isfinite(ratios))
        or np.any(ratios <= 0.0)
        or not np.isclose(ratios.sum(), 1.0)
    ):
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.0.")

    class_names = list(summary.get("class_names", []))
    images_by_class = summary.get("images_by_class", {})
    if not class_names or len(class_names) < 2:
        raise ValueError("A valid image split requires at least 2 classes.")

    train_files: list[str] = []
    validation_files: list[str] = []
    test_files: list[str] = []
    rng = np.random.default_rng(seed)
    content_labels: dict[str, str] = {}
    unique_image_count = 0

    for class_name in class_names:
        file_paths = list(images_by_class.get(class_name, []))
        groups_by_digest: dict[str, list[str]] = {}
        for file_path in file_paths:
            digest = _sha256_file(file_path)
            prior_label = content_labels.setdefault(digest, class_name)
            if prior_label != class_name:
                raise ValueError(
                    "Byte-identical image content has conflicting class labels "
                    f"('{Path(file_path).name}'); correct the labels before training."
                )
            groups_by_digest.setdefault(digest, []).append(file_path)

        content_groups = [[paths[0]] for paths in groups_by_digest.values()]
        total = len(content_groups)
        if total < 3:
            raise ValueError(
                f"Class '{class_name}' needs at least 3 unique image contents "
                "to include it in training, validation, and test splits."
            )
        unique_image_count += total
        rng.shuffle(content_groups)

        ideal_counts = total * ratios
        counts = np.floor(ideal_counts).astype(int)
        remainder = total - int(counts.sum())
        for split_index in np.argsort(-(ideal_counts - counts))[:remainder]:
            counts[split_index] += 1

        for split_index in np.flatnonzero(counts == 0):
            donors = [index for index, count in enumerate(counts) if count > 1]
            if not donors:
                raise ValueError(
                    f"Class '{class_name}' is too small for non-empty training, "
                    "validation, and test splits."
                )
            donor = max(donors, key=lambda index: counts[index] - ideal_counts[index])
            counts[donor] -= 1
            counts[split_index] += 1

        train_count, validation_count, test_count = (int(count) for count in counts)

        for group in content_groups[:train_count]:
            train_files.extend(group)
        for group in content_groups[train_count:train_count + validation_count]:
            validation_files.extend(group)
        for group in content_groups[train_count + validation_count:train_count + validation_count + test_count]:
            test_files.extend(group)

    combined = set(train_files) | set(validation_files) | set(test_files)
    if len(combined) != unique_image_count:
        raise ValueError("Split did not preserve every unique image content exactly once.")

    return {
        "train": sorted(train_files),
        "validation": sorted(validation_files),
        "test": sorted(test_files),
    }


def extract_zip_dataset(zip_path: str | Path) -> tuple[Path, dict[str, Any]]:
    extracted_root = _extract_zip_to_tempdir(zip_path)
    try:
        summary = _validate_extracted_dataset(extracted_root)
    except BaseException:
        shutil.rmtree(extracted_root, ignore_errors=True)
        raise
    summary["extract_dir"] = str(extracted_root)
    return extracted_root, summary


def _duplicate_metrics(connection: sqlite3.Connection) -> dict[str, Any]:
    duplicate_group_count, duplicate_file_count = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(group_size), 0)
        FROM (
            SELECT COUNT(*) AS group_size
            FROM image_records
            GROUP BY digest
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    group_rows = connection.execute(
        """
        SELECT digest, COUNT(*) AS file_count
        FROM image_records
        GROUP BY digest
        HAVING COUNT(*) > 1
        ORDER BY digest
        LIMIT ?
        """,
        (_MAX_DUPLICATE_GROUPS,),
    ).fetchall()

    groups = []
    for digest, file_count in group_rows:
        paths = connection.execute(
            """
            SELECT relative_path
            FROM image_records
            WHERE digest = ?
            ORDER BY relative_path
            LIMIT ?
            """,
            (digest, _MAX_DUPLICATE_PATHS),
        ).fetchall()
        displayed_paths = []
        path_text_truncated = False
        for (relative_path,) in paths:
            if len(relative_path) > _MAX_DUPLICATE_PATH_LENGTH:
                relative_path = relative_path[:_MAX_DUPLICATE_PATH_LENGTH - 3] + "..."
                path_text_truncated = True
            displayed_paths.append(relative_path)
        groups.append(
            {
                "hash": digest,
                "file_count": file_count,
                "files": displayed_paths,
                "files_truncated": (
                    file_count > len(displayed_paths) or path_text_truncated
                ),
            }
        )

    return {
        "method": "sha256_file_bytes",
        "note": "Detects byte-identical files only; re-encoded copies may have different hashes.",
        "duplicate_file_count": int(duplicate_file_count),
        "duplicate_group_count": int(duplicate_group_count),
        "groups": groups,
        "groups_truncated": duplicate_group_count > len(groups),
    }


def _cross_split_duplicate_metrics(
    connection: sqlite3.Connection,
    *,
    splits_provided: bool,
) -> dict[str, Any]:
    if not splits_provided:
        return {
            "checked": False,
            "reason": "No predefined train/validation/test splits were provided.",
        }

    match_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT digest
            FROM image_records
            GROUP BY digest
            HAVING COUNT(DISTINCT split_name) > 1
        )
        """
    ).fetchone()[0]
    match_rows = connection.execute(
        """
        SELECT digest, COUNT(*) AS file_count
        FROM image_records
        GROUP BY digest
        HAVING COUNT(DISTINCT split_name) > 1
        ORDER BY digest
        LIMIT ?
        """,
        (_MAX_DUPLICATE_GROUPS,),
    ).fetchall()

    matches = []
    for digest, file_count in match_rows:
        split_names = [
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT split_name
                FROM image_records
                WHERE digest = ? AND split_name IS NOT NULL
                ORDER BY split_name
                """,
                (digest,),
            )
        ]
        matches.append(
            {"hash": digest, "file_count": file_count, "splits": split_names}
        )

    return {
        "checked": True,
        "match_count": int(match_count),
        "matches": matches,
        "matches_truncated": match_count > len(matches),
    }


def analyze_extracted_dataset(
    extracted_root: str | Path,
    summary: dict[str, Any],
    *,
    archive_size_bytes: int,
) -> dict[str, Any]:
    """Return deterministic facts for an already validated image dataset."""
    extracted_path = Path(extracted_root)
    class_names = list(summary["class_names"])
    per_class_counts = dict(summary["per_class_counts"])
    image_count = int(summary["image_count"])

    distribution = {
        class_name: {
            "count": per_class_counts[class_name],
            "percentage": round(per_class_counts[class_name] * 100 / image_count, 2),
        }
        for class_name in class_names
    }
    extracted_size_bytes = sum(
        path.stat().st_size
        for path in extracted_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    )

    extensions: dict[str, int] = {}
    formats: dict[str, int] = {}
    modes: dict[str, int] = {}
    channels: dict[str, int] = {}
    width_min: int | None = None
    width_max: int | None = None
    height_min: int | None = None
    height_max: int | None = None
    aspect_ratio_min: float | None = None
    aspect_ratio_max: float | None = None
    width_total = 0
    height_total = 0
    aspect_ratio_total = 0.0
    metadata_count = 0
    predefined_split = summary.get("predefined_split")
    if predefined_split is None:
        split_source = "not_provided"
        split_counts = None
        per_class_split_counts = None
    else:
        split_source = "provided"
        split_counts = {
            split_name: len(predefined_split[split_name])
            for split_name in DATASET_SPLITS
        }
        per_class_split_counts = {
            class_name: {split_name: 0 for split_name in DATASET_SPLITS}
            for class_name in class_names
        }
        for split_name in DATASET_SPLITS:
            for image_path in predefined_split[split_name]:
                class_name = summary["image_labels"][image_path]
                per_class_split_counts[class_name][split_name] += 1

    with tempfile.NamedTemporaryFile(
        prefix=".image_quality_",
        suffix=".sqlite3",
        dir=extracted_path,
        delete=False,
    ) as temporary_index:
        index_path = Path(temporary_index.name)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(index_path)
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("PRAGMA cache_size = -2048")
        connection.executescript(
            """
            CREATE TABLE image_records (
                digest TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                pixel_area INTEGER NOT NULL,
                split_name TEXT
            );
            CREATE TABLE dimensions (
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                PRIMARY KEY (width, height)
            );
            CREATE TABLE provided_splits (
                path TEXT PRIMARY KEY,
                split_name TEXT NOT NULL
            );
            """
        )
        if predefined_split is not None:
            connection.executemany(
                "INSERT INTO provided_splits(path, split_name) VALUES (?, ?)",
                (
                    (image_path, split_name)
                    for split_name in DATASET_SPLITS
                    for image_path in predefined_split[split_name]
                ),
            )

        for class_name in class_names:
            for image_path in summary.get(
                "source_images_by_class", summary["images_by_class"]
            )[class_name]:
                path = Path(image_path)
                try:
                    with Image.open(path) as image:
                        width, height = image.size
                        image_mode = image.mode
                        image_format = image.format or "unknown"
                        channel_count = len(image.getbands())
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"Could not inspect validated image '{path.name}'."
                    ) from error

                if width <= 0 or height <= 0:
                    raise ValueError(f"Image '{path.name}' has invalid dimensions.")

                extension = path.suffix.lower()
                extensions[extension] = extensions.get(extension, 0) + 1
                formats[image_format] = formats.get(image_format, 0) + 1
                modes[image_mode] = modes.get(image_mode, 0) + 1
                channel_key = str(channel_count)
                channels[channel_key] = channels.get(channel_key, 0) + 1

                aspect_ratio = width / height
                width_min = width if width_min is None else min(width_min, width)
                width_max = width if width_max is None else max(width_max, width)
                height_min = height if height_min is None else min(height_min, height)
                height_max = height if height_max is None else max(height_max, height)
                aspect_ratio_min = (
                    aspect_ratio
                    if aspect_ratio_min is None
                    else min(aspect_ratio_min, aspect_ratio)
                )
                aspect_ratio_max = (
                    aspect_ratio
                    if aspect_ratio_max is None
                    else max(aspect_ratio_max, aspect_ratio)
                )
                width_total += width
                height_total += height
                aspect_ratio_total += aspect_ratio
                metadata_count += 1

                relative_path = path.relative_to(extracted_path).as_posix()
                split_row = connection.execute(
                    "SELECT split_name FROM provided_splits WHERE path = ?",
                    (str(path.resolve()),),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO image_records(
                        digest, relative_path, class_name, pixel_area, split_name
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        _sha256_file(path),
                        relative_path,
                        class_name,
                        width * height,
                        split_row[0] if split_row else None,
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO dimensions(width, height) VALUES (?, ?)",
                    (width, height),
                )

        if metadata_count != image_count:
            raise ValueError(
                "Validated image count does not match the image metadata scan."
            )

        connection.commit()
        connection.execute(
            "CREATE INDEX image_records_digest ON image_records(digest)"
        )
        connection.execute(
            "CREATE INDEX image_records_pixel_area ON image_records(pixel_area, relative_path)"
        )
        dimension_count = connection.execute(
            "SELECT COUNT(*) FROM dimensions"
        ).fetchone()[0]
        pixel_area_summary = connection.execute(
            "SELECT MIN(pixel_area), MAX(pixel_area), AVG(pixel_area) FROM image_records"
        ).fetchone()
        middle_offsets = ((image_count - 1) // 2, image_count // 2)
        middle_areas = [
            connection.execute(
                """
                SELECT pixel_area
                FROM image_records
                ORDER BY pixel_area, relative_path
                LIMIT 1 OFFSET ?
                """,
                (offset,),
            ).fetchone()[0]
            for offset in middle_offsets
        ]
        median_pixel_area = round(sum(middle_areas) / 2, 2)
        duplicate_metrics = _duplicate_metrics(connection)
        cross_split_metrics = _cross_split_duplicate_metrics(
            connection,
            splits_provided=predefined_split is not None,
        )
    finally:
        if connection is not None:
            connection.close()
        index_path.unlink(missing_ok=True)

    minimum_images_per_class = min(per_class_counts.values())
    largest_class_count = max(per_class_counts.values())
    smallest_class_count = min(per_class_counts.values())
    unique_per_class_counts = dict(summary["unique_per_class_counts"])
    findings = []
    validation_blockers = []
    if dimension_count > 1:
        findings.append("Images have inconsistent dimensions.")
    if duplicate_metrics["duplicate_group_count"] > 0:
        findings.append("Exact duplicate image files were detected.")
    if summary["skipped_files"]:
        findings.append(
            f"{len(summary['skipped_files'])} unsupported or unreadable file(s) were skipped."
        )
    empty_classes = [
        name for name, count in per_class_counts.items() if count == 0
    ]
    if empty_classes:
        validation_blockers.append(
            f"Empty classes have no readable supported images: {', '.join(empty_classes)}."
        )
    if predefined_split is None:
        insufficient_classes = [
            name for name, count in unique_per_class_counts.items() if count < 3
        ]
        if insufficient_classes:
            validation_blockers.append(
                "Each class needs at least 3 unique image contents for train, "
                "validation, and test splits: "
                + ", ".join(
                    f"{name} ({unique_per_class_counts[name]})"
                    for name in insufficient_classes
                )
                + "."
            )
    elif per_class_split_counts is not None:
        insufficient_splits = [
            f"{class_name} ({split_name})"
            for class_name, counts in per_class_split_counts.items()
            for split_name, count in counts.items()
            if count == 0
        ]
        if insufficient_splits:
            validation_blockers.append(
                "Every class must have at least one image in each provided split: "
                + ", ".join(insufficient_splits)
                + "."
            )
    if cross_split_metrics.get("match_count", 0) > 0:
        findings.append("Possible exact-duplicate split leakage was detected.")
        validation_blockers.append(
            "Byte-identical images cross provided train/validation/test partitions."
        )
    findings.extend(validation_blockers)

    return {
        "dataset": {
            "image_count": image_count,
            "total_file_count": int(summary["total_file_count"]),
            "valid_image_count": int(summary["valid_image_count"]),
            "excluded_image_counts": dict(summary.get("excluded_image_counts", {})),
            "skipped_file_count": int(summary["skipped_file_count"]),
            "skipped_files": summary["skipped_files"],
            "class_count": len(class_names),
            "archive_size_bytes": int(archive_size_bytes),
            "extracted_size_bytes": int(extracted_size_bytes),
        },
        "classes": {
            "names": class_names,
            "images_per_class": per_class_counts,
            "unique_images_per_class": unique_per_class_counts,
            "distribution": distribution,
        },
        "splits": {
            "source": split_source,
            "counts": split_counts,
            "per_class_counts": per_class_split_counts,
        },
        "image_properties": {
            "extensions": dict(sorted(extensions.items())),
            "formats": dict(sorted(formats.items())),
            "width": {
                "min": width_min,
                "max": width_max,
                "average": round(width_total / metadata_count, 2),
            },
            "height": {
                "min": height_min,
                "max": height_max,
                "average": round(height_total / metadata_count, 2),
            },
            "aspect_ratio": {
                "min": round(aspect_ratio_min, 4),
                "max": round(aspect_ratio_max, 4),
                "average": round(aspect_ratio_total / metadata_count, 4),
            },
            "modes": dict(sorted(modes.items())),
            "channels": dict(sorted(channels.items())),
            "channel_note": "Source image bands; training converts images to RGB.",
        },
        "quality": {
            "validation": "failed" if validation_blockers else "passed",
            "findings": findings,
            "metrics": {
                "dimension_consistency": {
                    "consistent": dimension_count == 1,
                    "unique_dimensions": int(dimension_count),
                },
                "resolution": {
                    "min_pixel_area": int(pixel_area_summary[0]),
                    "max_pixel_area": int(pixel_area_summary[1]),
                    "average_pixel_area": round(pixel_area_summary[2], 2),
                    "median_pixel_area": median_pixel_area,
                    "median_method": "exact_disk_backed_order_statistic",
                },
                "class_size": {
                    "minimum_images_per_class": minimum_images_per_class,
                    "minimum_unique_images_per_class": min(
                        unique_per_class_counts.values()
                    ),
                },
                "split_feasibility": {
                    "minimum_images_per_class": minimum_images_per_class,
                    "minimum_unique_images_per_class": min(
                        unique_per_class_counts.values()
                    ),
                    "supports_existing_minimum_3_per_class_rule": (
                        all(count >= 3 for count in unique_per_class_counts.values())
                        if predefined_split is None
                        else not validation_blockers
                    ),
                },
                "class_balance": {
                    "largest_class_count": largest_class_count,
                    "smallest_class_count": smallest_class_count,
                    "imbalance_ratio": (
                        largest_class_count / smallest_class_count
                        if smallest_class_count
                        else None
                    ),
                },
                "duplicates": duplicate_metrics,
                "cross_split_duplicates": cross_split_metrics,
            },
        },
        "warnings": [],
        "cnn_suitability": {"status": "not_assessed", "reasons": []},
    }
