"""Load OpenML datasets and normalize them for meta-learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from services.metalearning.meta_features import extract_meta_features


DEFAULT_OPENML_CACHE_DIR = (
	Path(__file__).resolve().parents[2] / "storage" / "openml_cache"
).resolve()
DEFAULT_OPENML_BENCHMARK_RECORDS_PATH = DEFAULT_OPENML_CACHE_DIR / "benchmark_records.json"
OPENML_API_BASE = "https://www.openml.org/api/v1/json"

SUPPORTED_MODEL_FAMILIES = {
	"LogisticRegression",
	"DecisionTreeClassifier",
	"RandomForestClassifier",
	"LinearRegression",
	"DecisionTreeRegressor",
	"RandomForestRegressor",
}


class OpenMLLoadError(RuntimeError):
	"""Raised when an OpenML dataset cannot be downloaded or normalized."""


def load_cached_openml_records(
	records_path: str | Path | None = None,
) -> list[dict[str, Any]]:
	"""Read optional local OpenML benchmark records without network access."""
	path = Path(records_path) if records_path is not None else DEFAULT_OPENML_BENCHMARK_RECORDS_PATH
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError, TypeError):
		return []
	if not isinstance(payload, list):
		return []
	return [
		record for record in payload
		if isinstance(record, dict) and record.get("source") == "openml"
	]


def _json_safe(value: Any) -> Any:
	if value is None or isinstance(value, (str, bool, int, float)):
		return value
	if isinstance(value, Mapping):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_json_safe(item) for item in value]
	if hasattr(value, "item"):
		return _json_safe(value.item())
	return str(value)


def _fetch_openml_json(path: str, *, timeout: float = 10.0, session: Any = None) -> dict[str, Any]:
	"""Fetch one public OpenML JSON resource without writing to project history."""
	try:
		import requests

		client = session or requests
		response = client.get(f"{OPENML_API_BASE}/{path.lstrip('/')}", timeout=timeout)
		response.raise_for_status()
		payload = response.json()
	except Exception as exc:
		raise OpenMLLoadError(f"Unable to fetch OpenML resource '{path}': {exc}") from exc

	if not isinstance(payload, dict):
		raise OpenMLLoadError(f"OpenML resource '{path}' returned malformed JSON")
	return payload


def fetch_openml_task(task_id: int | str, *, timeout: float = 10.0, session: Any = None) -> dict[str, Any]:
	"""Fetch public task metadata."""
	return _fetch_openml_json(f"task/{task_id}", timeout=timeout, session=session)


def fetch_openml_run(run_id: int | str, *, timeout: float = 10.0, session: Any = None) -> dict[str, Any]:
	"""Fetch public run metadata."""
	return _fetch_openml_json(f"run/{run_id}", timeout=timeout, session=session)


def fetch_openml_task_runs(task_id: int | str, *, timeout: float = 10.0, session: Any = None) -> dict[str, Any]:
	"""Fetch public runs associated with a task."""
	return _fetch_openml_json(f"run/list/task/{task_id}", timeout=timeout, session=session)


def fetch_openml_task_evaluations(
	task_id: int | str,
	*,
	timeout: float = 10.0,
	session: Any = None,
) -> dict[str, Any]:
	"""Fetch public evaluation results associated with a task."""
	return _fetch_openml_json(f"evaluation/list/task/{task_id}", timeout=timeout, session=session)


def _records_from_payload(payload: Mapping[str, Any], *keys: str) -> list[dict[str, Any]]:
	for key in keys:
		value = payload.get(key)
		if isinstance(value, list):
			return [item for item in value if isinstance(item, dict)]
		if isinstance(value, dict):
			for nested_key in (key, "run", "evaluation", "task"):
				nested = value.get(nested_key)
				if isinstance(nested, list):
					return [item for item in nested if isinstance(item, dict)]
	return []


def _first_value(payload: Mapping[str, Any], *keys: str) -> Any:
	for key in keys:
		if payload.get(key) is not None:
			return payload[key]
	return None


def _normalize_model_family(run: Mapping[str, Any]) -> str | None:
	text = " ".join(
		str(run.get(key, ""))
		for key in ("model", "model_name", "flow_name", "implementation", "name")
	).lower()
	for family in SUPPORTED_MODEL_FAMILIES:
		if family.lower() in text:
			return family
	return None


def _normalize_metric(metric: Any, problem_type: str | None) -> str | None:
	text = str(metric or "").strip().lower().replace("-", "_")
	if "classification" in str(problem_type or "").lower():
		return "f1_score" if "f1" in text or "f_measure" in text else None
	if "regression" in str(problem_type or "").lower():
		return "rmse" if "rmse" in text or "root_mean_squared_error" in text else None
	return None


def load_openml_dataset(
	dataset_id: int | str,
	version: int | str | None = None,
	*,
	cache_dir: str | Path | None = None,
) -> dict[str, Any]:
	"""Download an OpenML dataset and return a normalized DataFrame contract."""
	if dataset_id is None or str(dataset_id).strip() == "":
		raise ValueError("dataset_id is required")

	try:
		from sklearn.datasets import fetch_openml
	except ImportError as exc:  # pragma: no cover - dependency is required by backend
		raise OpenMLLoadError("scikit-learn is required to load OpenML datasets") from exc

	target_cache = Path(cache_dir) if cache_dir is not None else DEFAULT_OPENML_CACHE_DIR
	target_cache = target_cache.resolve()
	target_cache.mkdir(parents=True, exist_ok=True)

	fetch_kwargs: dict[str, Any] = {
		"data_id": dataset_id,
		"as_frame": True,
		"data_home": str(target_cache),
	}
	if version is not None:
		fetch_kwargs["version"] = version

	try:
		dataset = fetch_openml(**fetch_kwargs)
		frame = dataset.frame.copy() if isinstance(dataset.frame, pd.DataFrame) else None
		target = dataset.target
	except Exception as exc:
		raise OpenMLLoadError(
			f"Unable to load OpenML dataset {dataset_id}"
			f"{f' version {version}' if version is not None else ''}: {exc}"
		) from exc

	if frame is None:
		raise OpenMLLoadError("OpenML response did not contain a pandas DataFrame")

	target_column = str(getattr(target, "name", None) or "target")
	if target_column not in frame.columns:
		target_values = target.reset_index(drop=True) if hasattr(target, "reset_index") else target
		frame[target_column] = target_values

	return {
		"dataframe": frame.reset_index(drop=True),
		"target_column": target_column,
		"dataset_id": str(dataset_id),
		"dataset_version": None if version is None else str(version),
		"cache_dir": str(target_cache),
	}


def build_openml_meta_record(
	dataframe: pd.DataFrame,
	target_column: str,
	dataset_id: int | str,
	dataset_version: int | str | None = None,
	*,
	problem_type: str | None = None,
	benchmark_outcome: Mapping[str, Any] | None = None,
	task_id: int | str | None = None,
	run_id: int | str | None = None,
) -> dict[str, Any]:
	"""Build a JSON-safe OpenML meta-record without inventing outcomes."""
	if not isinstance(dataframe, pd.DataFrame):
		raise TypeError("dataframe must be a pandas DataFrame")
	if target_column not in dataframe.columns:
		raise ValueError(f"target column '{target_column}' is not present in the DataFrame")

	outcome = benchmark_outcome or {}
	meta_features = extract_meta_features(
		dataframe,
		target_column=target_column,
		problem_type=problem_type,
	)
	record = {
		"source": "openml",
		"dataset_id": str(dataset_id),
		"dataset_version": None if dataset_version is None else str(dataset_version),
		"task_id": None if task_id is None else str(task_id),
		"run_id": None if run_id is None else str(run_id),
		"problem_type": problem_type or meta_features.get("problem_type"),
		"meta_features": meta_features,
		"best_model": outcome.get("best_model"),
		"selection_metric": outcome.get("selection_metric"),
		"metric_value": outcome.get("metric_value"),
	}
	return _json_safe(record)


def import_openml_benchmark_records(
	dataframe: pd.DataFrame,
	target_column: str,
	dataset_id: int | str,
	task_id: int | str,
	dataset_version: int | str | None = None,
	*,
	problem_type: str | None = None,
	timeout: float = 10.0,
	session: Any = None,
) -> list[dict[str, Any]]:
	"""Import compatible public benchmark outcomes without modifying local history."""
	task_payload = fetch_openml_task(task_id, timeout=timeout, session=session)
	task_data = task_payload.get("task", task_payload)
	resolved_problem_type = problem_type or _first_value(task_data, "problem_type", "task_type")
	runs_payload = fetch_openml_task_runs(task_id, timeout=timeout, session=session)
	evaluations_payload = fetch_openml_task_evaluations(task_id, timeout=timeout, session=session)
	runs = _records_from_payload(runs_payload, "runs", "run")
	evaluations = _records_from_payload(evaluations_payload, "evaluations", "evaluation")
	if not evaluations:
		return []

	runs_by_id = {str(_first_value(run, "run_id", "id")): run for run in runs}
	records: list[dict[str, Any]] = []
	for evaluation in evaluations:
		run_id = _first_value(evaluation, "run_id", "run")
		run = runs_by_id.get(str(run_id), {})
		model_family = _normalize_model_family(run)
		metric_name = _normalize_metric(
			_first_value(evaluation, "measure", "measure_name", "evaluation_measure", "metric"),
			resolved_problem_type,
		)
		metric_value = _first_value(evaluation, "value", "score", "result")
		if model_family is None or metric_name is None or not isinstance(metric_value, (int, float)):
			continue
		records.append(
			build_openml_meta_record(
				dataframe,
				target_column,
				dataset_id,
				dataset_version,
				problem_type=resolved_problem_type,
				benchmark_outcome={
					"best_model": model_family,
					"selection_metric": metric_name,
					"metric_value": metric_value,
				},
				task_id=task_id,
				run_id=run_id,
			)
		)
	return records
