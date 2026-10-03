import inspect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.automl.target_proxy_diagnostic import diagnose_target_proxy_features


def _warnings_by_feature(result):
    return {warning["feature"]: warning for warning in result["warnings"]}


def test_flags_an_exact_target_copy():
    dataframe = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "copied_target": [0, 1, 0, 1],
        "feature": [2, 4, 6, 8],
    })

    warnings = _warnings_by_feature(
        diagnose_target_proxy_features(dataframe, "target")
    )

    assert warnings["copied_target"]["detection_type"] == "exact_target_copy"
    assert warnings["copied_target"]["evidence"]["matching_non_missing_rows"] == 4
    assert "feature" not in warnings


def test_flags_an_unambiguous_one_to_one_encoding_separately():
    dataframe = pd.DataFrame({
        "target": ["low", "high", "low", "high"],
        "encoded_target": [10, 20, 10, 20],
    })

    warnings = _warnings_by_feature(
        diagnose_target_proxy_features(dataframe, "target")
    )

    assert warnings["encoded_target"]["detection_type"] == "one_to_one_target_encoding"
    assert warnings["encoded_target"]["evidence"]["mapping_is_bijective"] is True


def test_does_not_flag_a_non_unique_feature_to_target_mapping():
    dataframe = pd.DataFrame({
        "target": [0, 0, 1, 1],
        "feature": ["a", "b", "c", "c"],
    })

    assert diagnose_target_proxy_features(dataframe, "target")["warnings"] == []


def test_excludes_target_and_group_columns_from_candidates():
    dataframe = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "group": [0, 1, 0, 1],
        "feature": ["a", "b", "a", "b"],
    })

    warnings = diagnose_target_proxy_features(
        dataframe,
        "target",
        group_column="group",
    )["warnings"]

    assert {warning["feature"] for warning in warnings} == {"feature"}


def test_legitimate_feature_is_not_modified_or_removed():
    dataframe = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "legitimate_feature": [10, 10, 20, 30],
    })
    original = dataframe.copy(deep=True)

    result = diagnose_target_proxy_features(dataframe, "target")

    assert result["warnings"] == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_missing_and_mixed_values_are_handled_without_false_mapping_warning():
    dataframe = pd.DataFrame({
        "target": pd.Series([0, 1, 0, 1, None], dtype="object"),
        "exact_copy": pd.Series([0, 1, 0, 1, None], dtype="object"),
        "mixed_non_bijective": pd.Series([1, "one", 1, 2, None], dtype="object"),
        "incomplete_encoding": ["a", None, "a", "b", None],
    })

    warnings = _warnings_by_feature(
        diagnose_target_proxy_features(dataframe, "target")
    )

    assert warnings["exact_copy"]["detection_type"] == "exact_target_copy"
    assert "mixed_non_bijective" not in warnings
    assert "incomplete_encoding" not in warnings


def test_all_missing_target_does_not_create_a_vacuous_copy_warning():
    dataframe = pd.DataFrame({
        "target": [None, None],
        "all_missing_feature": [None, None],
    })

    assert diagnose_target_proxy_features(dataframe, "target")["warnings"] == []


def test_diagnostic_accepts_only_one_training_frame():
    parameters = inspect.signature(diagnose_target_proxy_features).parameters

    assert list(parameters) == ["training_df", "target_column", "group_column"]