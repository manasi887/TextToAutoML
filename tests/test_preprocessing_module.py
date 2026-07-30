import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the backend package is importable from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from services.dataset.preprocess import (
    calculate_delivery_time,
    convert_date_columns,
    extract_date_features,
    handle_missing_values,
    preprocess_dataset,
    remove_duplicates,
)


def test_remove_duplicates_removes_duplicate_rows():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "b": ["x", "x", "y"],
        }
    )

    cleaned_df, report = remove_duplicates(df)

    assert len(cleaned_df) == 2
    assert report["status"] == "Completed"
    assert report["details"]["duplicates_removed"] == 1
    assert report["details"]["rows_before"] == 3
    assert report["details"]["rows_after"] == 2


def test_handle_missing_values_fills_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "numeric": [1.0, None, 3.0],
            "category": ["a", None, "a"],
        }
    )

    cleaned_df, report = handle_missing_values(df)

    assert cleaned_df.loc[1, "numeric"] == pytest.approx(2.0)
    assert cleaned_df.loc[1, "category"] == "a"
    assert report["status"] == "Completed"
    assert report["details"]["missing_before"] == 2
    assert report["details"]["missing_after"] == 0
    assert report["details"]["missing_values_filled"] == 2


def test_convert_date_columns_skips_non_datetime_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "remarks": ["foo", "bar"],
        }
    )

    cleaned_df, report = convert_date_columns(df)

    assert report["status"] == "Skipped"
    assert report["reason"] == "No date columns found."
    assert cleaned_df.equals(df)


def test_convert_date_columns_parses_valid_date_strings():
    df = pd.DataFrame(
        {
            "Order Date": ["2023-01-01", "2023-02-01"],
            "value": [10, 20],
        }
    )

    cleaned_df, report = convert_date_columns(df)

    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["Order Date"])
    assert report["status"] == "Completed"
    assert report["details"]["date_columns_detected"] == ["Order Date"]
    assert report["details"]["date_columns_converted"] == ["Order Date"]
    assert report["details"]["date_columns_nat_count"] == 0


def test_convert_date_columns_ignores_malformed_date_strings():
    df = pd.DataFrame(
        {
            "Order Date": ["2023-01-01", "2023-02-01"],
            "Bad Date": ["not a date", "also bad"],
        }
    )

    cleaned_df, report = convert_date_columns(df)

    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["Order Date"])
    assert cleaned_df["Bad Date"].dtype == object
    assert report["status"] == "Completed"
    assert report["details"]["date_columns_detected"] == ["Order Date"]
    assert report["details"]["date_columns_converted"] == ["Order Date"]


def test_extract_date_features_skips_when_no_datetime_columns():
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": ["x", "y"],
        }
    )

    updated_df, report = extract_date_features(df)

    assert report["status"] == "Skipped"
    assert report["reason"] == "No datetime columns found."
    assert updated_df.equals(df)


def test_extract_date_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "Ship Date": pd.to_datetime(["2023-01-05", "2023-01-06"]),
        }
    )

    updated_df, report = extract_date_features(df)

    expected_columns = [
        "Order Year",
        "Order Month",
        "Order Day",
        "Order Weekday",
        "Ship Year",
        "Ship Month",
        "Ship Day",
        "Ship Weekday",
    ]

    assert all(column in updated_df.columns for column in expected_columns)
    assert report["status"] == "Completed"
    assert report["details"]["generated_column_count"] == 8
    assert list(updated_df.columns[-8:]) == expected_columns


def test_calculate_delivery_time_returns_day_difference():
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2023-01-01", "2023-01-05"]),
            "Ship Date": pd.to_datetime(["2023-01-04", "2023-01-06"]),
        }
    )

    updated_df, report = calculate_delivery_time(df)

    assert "Delivery Time (Days)" in updated_df.columns
    assert updated_df["Delivery Time (Days)"].tolist() == [3, 1]
    assert report["status"] == "Completed"
    assert report["details"]["generated_column"] == "Delivery Time (Days)"
    assert report["details"]["negative_delivery_days"] == 0


def test_calculate_delivery_time_counts_negative_delivery_days():
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2023-01-05", "2023-01-01"]),
            "Ship Date": pd.to_datetime(["2023-01-04", "2023-01-06"]),
        }
    )

    updated_df, report = calculate_delivery_time(df)

    assert updated_df["Delivery Time (Days)"].tolist() == [-1, 5]
    assert report["status"] == "Completed"
    assert report["details"]["negative_delivery_days"] == 1


def test_calculate_delivery_time_skips_missing_order_date():
    df = pd.DataFrame(
        {
            "Ship Date": pd.to_datetime(["2023-01-04", "2023-01-06"]),
        }
    )

    updated_df, report = calculate_delivery_time(df)

    assert report["status"] == "Skipped"
    assert "Delivery Time (Days)" not in updated_df.columns


def test_calculate_delivery_time_skips_missing_ship_date():
    df = pd.DataFrame(
        {
            "Order Date": pd.to_datetime(["2023-01-01", "2023-01-05"]),
        }
    )

    updated_df, report = calculate_delivery_time(df)

    assert report["status"] == "Skipped"
    assert "Delivery Time (Days)" not in updated_df.columns


def test_preprocess_dataset_integration():
    df = pd.DataFrame(
        {
            "Order Date": ["2023-01-01", "2023-01-01", "2023-01-03"],
            "Ship Date": ["2023-01-04", "2023-01-05", "2023-01-07"],
            "category": ["a", None, "b"],
            "numeric": [1.0, None, 3.0],
        }
    )

    cleaned_df, report = preprocess_dataset(df)

    assert report["duplicates"]["details"]["duplicates_removed"] == 1
    assert report["missing_values"]["details"]["missing_values_filled"] == 2
    assert report["date_conversion"]["status"] == "Completed"
    assert report["date_features"]["status"] == "Completed"
    assert report["delivery_time"]["status"] == "Completed"
    assert "Order Year" in cleaned_df.columns
    assert "Delivery Time (Days)" in cleaned_df.columns
    assert cleaned_df.shape[0] == 2
