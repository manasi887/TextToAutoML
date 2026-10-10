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
    if len(first_segments) == 1 and all(len(parts) >= 2 for parts in paths):
        return [parts[1:] for parts in paths]
    return paths


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
                if member.is_dir():
                    continue
                safe_name = _safe_member_name(member.filename)
                target_path = temp_root / safe_name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.parent.is_relative_to(temp_root):
                    raise ValueError(f"unsafe extraction path detected: {member.filename!r}")
                with archive.open(member, "r") as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return temp_root


def _collect_image_entries(extracted_root: Path) -> list[tuple[str, ...]]:
    gathered: list[tuple[str, ...]] = []
    class_dirs: set[str] = set()
    for path in sorted(extracted_root.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        rel = path.relative_to(extracted_root)
        parts = rel.parts
        if len(parts) >= 2:
            class_dirs.add(parts[0] if len(parts) > 1 else "")
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            if path.name.lower().startswith(".") or path.name.lower() in {"readme.md", "readme.txt", "thumbs.db", "desktop.ini"}:
                continue
            if len(parts) >= 2:
                raise ValueError(f"unsupported file type encountered in dataset: {path.name}")
            continue
        gathered.append(tuple(rel.parts))
    if not gathered:
        if class_dirs:
            first_class = sorted(class_dirs)[0]
            raise ValueError(f"class directory is empty: {first_class}")
        raise ValueError("no valid image files were found in the uploaded ZIP dataset.")
    return gathered


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
    image_rel_paths = _collect_image_entries(extracted_root)
    normalized_image_paths = _detect_wrapper_and_strip(image_rel_paths)
    first_segments = {parts[0] for parts in normalized_image_paths if parts}
    split_directories = set(DATASET_SPLITS)
    is_pre_split = bool(first_segments & split_directories)
    if is_pre_split and first_segments != split_directories:
        missing_splits = sorted(split_directories - first_segments)
        raise ValueError(
            "Pre-split image dataset must contain train, validation, and test directories; "
            f"missing: {', '.join(missing_splits)}."
        )

    class_to_files: dict[str, list[str]] = {}
    image_labels: dict[str, str] = {}
    predefined_split = {split_name: [] for split_name in DATASET_SPLITS}
    split_classes = {split_name: set() for split_name in DATASET_SPLITS}

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

    for class_name, file_paths in class_to_files.items():
        if not file_paths:
            raise ValueError(f"class directory is empty: {class_name}")
        for file_path in file_paths:
            path = Path(file_path)
            _validate_image_file(path)

    class_names = sorted(class_to_files)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    per_class_counts = {name: len(class_to_files[name]) for name in class_names}
    image_count = sum(per_class_counts.values())
    image_extensions = sorted({Path(file_path).suffix.lower() for files in class_to_files.values() for file_path in files})

    summary: dict[str, Any] = {
        "class_names": class_names,
        "class_to_index": class_to_index,
        "image_count": image_count,
        "per_class_counts": per_class_counts,
        "image_extensions": image_extensions,
        "images_by_class": {name: sorted(class_to_files[name]) for name in class_names},
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

    for class_name in class_names:
        file_paths = list(images_by_class.get(class_name, []))
        total = len(file_paths)
        if total < 3:
            raise ValueError(
                f"Class '{class_name}' needs at least 3 images to include it in "
                "training, validation, and test splits."
            )
        rng.shuffle(file_paths)

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

        train_files.extend(file_paths[:train_count])
        validation_files.extend(file_paths[train_count:train_count + validation_count])
        test_files.extend(file_paths[train_count + validation_count:train_count + validation_count + test_count])

    combined = set(train_files) | set(validation_files) | set(test_files)
    if len(combined) != sum(len(images_by_class[class_name]) for class_name in class_names):
        raise ValueError("Split did not preserve all images exactly once.")

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
            for image_path in summary["images_by_class"][class_name]:
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

                digest = hashlib.sha256()
                try:
                    with path.open("rb") as image_file:
                        for chunk in iter(
                            lambda: image_file.read(_HASH_CHUNK_SIZE), b""
                        ):
                            digest.update(chunk)
                except OSError as error:
                    raise ValueError(
                        f"Could not hash validated image '{path.name}'."
                    ) from error

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
                        digest.hexdigest(),
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
    findings = []
    if dimension_count > 1:
        findings.append("Images have inconsistent dimensions.")
    if duplicate_metrics["duplicate_group_count"] > 0:
        findings.append("Exact duplicate image files were detected.")
    if cross_split_metrics.get("match_count", 0) > 0:
        findings.append("Possible exact-duplicate split leakage was detected.")

    return {
        "dataset": {
            "image_count": image_count,
            "class_count": len(class_names),
            "archive_size_bytes": int(archive_size_bytes),
            "extracted_size_bytes": int(extracted_size_bytes),
        },
        "classes": {
            "names": class_names,
            "images_per_class": per_class_counts,
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
            "validation": "passed",
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
                },
                "split_feasibility": {
                    "minimum_images_per_class": minimum_images_per_class,
                    "supports_existing_minimum_3_per_class_rule": (
                        minimum_images_per_class >= 3
                    ),
                },
                "class_balance": {
                    "largest_class_count": largest_class_count,
                    "smallest_class_count": smallest_class_count,
                    "imbalance_ratio": largest_class_count / smallest_class_count,
                },
                "duplicates": duplicate_metrics,
                "cross_split_duplicates": cross_split_metrics,
            },
        },
        "warnings": [],
        "cnn_suitability": {"status": "not_assessed", "reasons": []},
    }
