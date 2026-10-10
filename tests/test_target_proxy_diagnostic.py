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

    assert list(parameters) == [
        "training_df",
        "target_column",
        "group_column",
        "problem_type",
    ]
    assert "validation_df" not in parameters
    assert "test_df" not in parameters


def test_classification_association_includes_effect_size_and_adjusted_significance():
    rows = 160
    target = ["high" if index % 2 else "low" for index in range(rows)]
    dataframe = pd.DataFrame({
        "target": target,
        "available_score": [
            (100 if value == "high" else 0) + index % 17
            for index, value in enumerate(target)
        ],
    })

    warnings = _warnings_by_feature(
        diagnose_target_proxy_features(
            dataframe,
            "target",
            problem_type="Binary Classification",
        )
    )
    warning = warnings["available_score"]

    assert warning["detection_type"] == "statistical_association"
    assert warning["evidence"]["method"] == "Kruskal-Wallis H test"
    assert warning["evidence"]["effect_size_name"] == "epsilon_squared"
    assert warning["evidence"]["effect_size"] > 0
    assert warning["evidence"]["p_value"] < 0.05
    assert warning["evidence"]["q_value"] <= warning["evidence"]["false_discovery_rate_alpha"]


def test_categorical_classification_uses_chi_square_and_fisher_when_sparse():
    target = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    categories = ["a"] * 5 + ["b"] + ["a"] + ["b"] * 5
    pearson_frame = pd.DataFrame({"target": target * 10, "category": categories * 10})
    fisher_frame = pd.DataFrame({
        "target": [0] * 7 + [1] * 7,
        "category": ["a"] * 6 + ["b"] + ["a"] + ["b"] * 6,
    })

    pearson_warning = _warnings_by_feature(
        diagnose_target_proxy_features(
            pearson_frame,
            "target",
            problem_type="Classification",
        )
    )["category"]
    fisher_result = diagnose_target_proxy_features(
        fisher_frame,
        "target",
        problem_type="Classification",
    )
    fisher_warnings = [
        warning for warning in fisher_result["warnings"]
        if warning["detection_type"] == "statistical_association"
    ]

    assert pearson_warning["evidence"]["method"] == "Pearson chi-square test"
    assert pearson_warning["evidence"]["effect_size_name"] == "cramers_v"
    assert fisher_warnings[0]["evidence"]["method"] == "Fisher exact test"
    assert fisher_warnings[0]["evidence"]["q_value"] <= 0.05


def test_regression_numeric_and_categorical_associations_are_supported():
    rows = 120
    dataframe = pd.DataFrame({
        "target": [float(index) + (index % 5) / 10 for index in range(rows)],
        "numeric_signal": [float(index) for index in range(rows)],
        "category_signal": ["later" if index >= rows // 2 else "earlier" for index in range(rows)],
    })

    warnings = _warnings_by_feature(
        diagnose_target_proxy_features(dataframe, "target", problem_type="Regression")
    )

    assert warnings["numeric_signal"]["evidence"]["method"] == "Spearman rank correlation test"
    assert warnings["numeric_signal"]["evidence"]["effect_size_name"] == "absolute_spearman_rho"
    assert warnings["category_signal"]["evidence"]["method"] == "Kruskal-Wallis H test"
    assert warnings["category_signal"]["evidence"]["effect_size_name"] == "epsilon_squared"


def test_legitimate_informative_predictor_is_not_labelled_as_leakage_or_removed():
    rows = 120
    target = [index % 2 for index in range(rows)]
    dataframe = pd.DataFrame({
        "target": target,
        "available_measurement": [value * 50 + index % 19 for index, value in enumerate(target)],
    })
    original = dataframe.copy(deep=True)
    selected_features = ["available_measurement"]

    warning = _warnings_by_feature(
        diagnose_target_proxy_features(dataframe, "target", problem_type="Classification")
    )["available_measurement"]

    assert warning["detection_type"] == "statistical_association"
    assert "leakage" not in warning["detection_type"].lower()
    assert "not proof of leakage" in warning["message"]
    assert selected_features == ["available_measurement"]
    pd.testing.assert_frame_equal(dataframe, original)


def test_missing_values_constants_small_samples_and_mixed_categories_are_safe():
    dataframe = pd.DataFrame({
        "target": pd.Series([0, 1] * 15 + [None], dtype="object"),
        "constant": [1] * 31,
        "high_cardinality": [f"record-{index}" for index in range(31)],
        "mixed_category": [1 if index % 2 else "one" for index in range(31)],
        "numeric_with_missing": [None if index % 7 == 0 else index for index in range(31)],
    })
    small_frame = pd.DataFrame({"target": [0, 1], "feature": [3, 9]})

    result = diagnose_target_proxy_features(
        dataframe,
        "target",
        problem_type="Classification",
    )
    small_result = diagnose_target_proxy_features(
        small_frame,
        "target",
        problem_type="Classification",
    )

    assert "constant" not in _warnings_by_feature(result)
    assert result["statistical_tests_skipped_for_sparse_categories"] >= 1
    assert all(
        warning["detection_type"] != "statistical_association"
        for warning in small_result["warnings"]
    )


def test_grouped_association_uses_one_summary_per_group_and_discloses_limits():
    dataframe = pd.DataFrame(
        [
            {
                "group": f"entity-{group_index}",
                "target": group_index % 2,
                "measurement": (group_index % 2) * 40 + group_index + row_index / 10,
            }
            for group_index in range(40)
            for row_index in range(5)
        ]
    )

    warnings = diagnose_target_proxy_features(
        dataframe,
        "target",
        group_column="group",
        problem_type="Classification",
    )["warnings"]
    warning = next(
        item for item in warnings
        if item["feature"] == "measurement" and item["detection_type"] == "statistical_association"
    )

    assert warning["evidence"]["rows_unit"] == "groups"
    assert warning["evidence"]["rows_used"] == 40
    assert "not cluster-robust" in warning["evidence"]["group_inference_note"]
    assert "independent groups" in warning["message"]