import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.metalearning.openml_loader import (
    OpenMLLoadError,
    build_openml_meta_record,
    import_openml_benchmark_records,
    load_cached_openml_records,
    load_openml_dataset,
)


def test_load_openml_dataset_normalizes_frame_target_and_cache(tmp_path):
    dataset = SimpleNamespace(
        frame=pd.DataFrame({"feature": [1, 2], "class": ["a", "b"]}),
        target=pd.Series(["a", "b"], name="class"),
    )

    with patch("sklearn.datasets.fetch_openml", return_value=dataset) as fetch:
        result = load_openml_dataset(61, version=2, cache_dir=tmp_path / "openml")

    fetch.assert_called_once_with(
        data_id=61,
        version=2,
        as_frame=True,
        data_home=str((tmp_path / "openml").resolve()),
    )
    assert result["target_column"] == "class"
    assert result["dataframe"].equals(dataset.frame)
    assert result["dataset_id"] == "61"
    assert result["dataset_version"] == "2"
    assert result["cache_dir"] == str((tmp_path / "openml").resolve())


def test_load_openml_dataset_adds_target_when_frame_excludes_it(tmp_path):
    dataset = SimpleNamespace(
        frame=pd.DataFrame({"feature": [1, 2]}),
        target=pd.Series([0, 1], name="label"),
    )

    with patch("sklearn.datasets.fetch_openml", return_value=dataset):
        result = load_openml_dataset("abc", cache_dir=tmp_path)

    assert result["dataframe"]["label"].tolist() == [0, 1]
    assert result["target_column"] == "label"


def test_load_openml_dataset_wraps_network_errors(tmp_path):
    with patch("sklearn.datasets.fetch_openml", side_effect=OSError("network unavailable")):
        with pytest.raises(OpenMLLoadError, match="Unable to load OpenML dataset"):
            load_openml_dataset(61, cache_dir=tmp_path)


def test_build_openml_meta_record_is_json_safe_without_inventing_outcomes():
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    record = build_openml_meta_record(
        dataframe,
        "target",
        61,
        dataset_version=1,
        problem_type="Binary Classification",
    )

    json.dumps(record, allow_nan=False)
    assert record["source"] == "openml"
    assert record["dataset_id"] == "61"
    assert record["dataset_version"] == "1"
    assert record["problem_type"] == "Binary Classification"
    assert record["meta_features"]["feature_count"] == 1
    assert record["best_model"] is None
    assert record["selection_metric"] is None
    assert record["metric_value"] is None


def test_build_openml_meta_record_preserves_supplied_benchmark_outcome():
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    record = build_openml_meta_record(
        dataframe,
        "target",
        61,
        benchmark_outcome={
            "best_model": "LogisticRegression",
            "selection_metric": "f1_score",
            "metric_value": 0.9,
        },
    )

    assert record["best_model"] == "LogisticRegression"
    assert record["selection_metric"] == "f1_score"
    assert record["metric_value"] == 0.9


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return _Response(self.responses.pop(0))


def test_import_openml_benchmark_records_filters_and_preserves_provenance():
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    session = _Session([
        {"task": {"task_id": 7, "task_type": "Binary Classification"}},
        {"runs": {"run": [
            {"run_id": 11, "flow_name": "sklearn.linear_model.LogisticRegression"},
            {"run_id": 12, "flow_name": "UnsupportedModel"},
        ]}},
        {"evaluations": {"evaluation": [
            {"run_id": 11, "measure": "predictive_f1", "value": 0.91},
            {"run_id": 12, "measure": "predictive_f1", "value": 0.99},
            {"run_id": 11, "measure": "predictive_accuracy", "value": 0.95},
        ]}},
    ])

    records = import_openml_benchmark_records(
        dataframe,
        "target",
        61,
        7,
        dataset_version=2,
        session=session,
    )

    assert len(records) == 1
    assert records[0]["best_model"] == "LogisticRegression"
    assert records[0]["selection_metric"] == "f1_score"
    assert records[0]["metric_value"] == 0.91
    assert records[0]["dataset_id"] == "61"
    assert records[0]["dataset_version"] == "2"
    assert records[0]["task_id"] == "7"
    assert records[0]["run_id"] == "11"
    assert len(session.calls) == 3
    assert all(timeout == 10.0 for _, timeout in session.calls)


def test_import_openml_benchmark_records_returns_empty_for_missing_evaluations():
    dataframe = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    session = _Session([
        {"task": {"task_type": "Regression"}},
        {"runs": {"run": []}},
        {"evaluations": {"evaluation": []}},
    ])

    assert import_openml_benchmark_records(dataframe, "target", 61, 7, session=session) == []


def test_fetch_openml_json_wraps_http_and_malformed_errors():
    class BadResponse(_Response):
        def raise_for_status(self):
            raise RuntimeError("503 service unavailable")

    class BadSession:
        def __init__(self, response):
            self.response = response

        def get(self, url, timeout):
            return self.response

    from services.metalearning.openml_loader import _fetch_openml_json

    with pytest.raises(OpenMLLoadError, match="Unable to fetch OpenML resource"):
        _fetch_openml_json("task/7", session=BadSession(BadResponse({})))

    with pytest.raises(OpenMLLoadError, match="OpenML resource.*malformed JSON"):
        _fetch_openml_json("task/7", session=BadSession(_Response(None)))


def test_load_cached_openml_records_reads_only_openml_records(tmp_path):
    path = tmp_path / "benchmark_records.json"
    path.write_text(
        json.dumps([
            {"source": "openml", "dataset_id": "61"},
            {"source": "local", "dataset_id": "local"},
            {"dataset_id": "missing-source"},
        ]),
        encoding="utf-8",
    )

    assert load_cached_openml_records(path) == [{"source": "openml", "dataset_id": "61"}]


def test_load_cached_openml_records_missing_or_malformed_is_empty(tmp_path):
    missing = load_cached_openml_records(tmp_path / "missing.json")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("not json", encoding="utf-8")

    assert missing == []
    assert load_cached_openml_records(malformed_path) == []