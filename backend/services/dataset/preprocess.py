import re
import warnings

import pandas as pd

# ==========================================================
# DATA CLEANING FUNCTIONS
# ==========================================================

def remove_duplicates(df: pd.DataFrame, copy_df: bool = True):
    """
    Remove duplicate rows from the dataset.
    """

    if copy_df:
        df = df.copy()

    rows_before = len(df)

    cleaned_df = df.drop_duplicates()

    rows_after = len(cleaned_df)

    preprocessing_info = {
        "status": "Completed",
        "details": {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_removed": rows_before - rows_after
        }
    }

    return cleaned_df, preprocessing_info


def normalize_missing_markers(df: pd.DataFrame, copy_df: bool = True):
    """Convert common string sentinel values to proper pandas missing values."""

    cleaned_df = df.copy() if copy_df else df
    missing_markers = {
        "", " ", "nan", "NaN", "N/A", "n/a", "NA", "na",
        "null", "NULL", "None", "none", "unknown", "Unknown",
        "?"
    }

    normalized_count = 0

    for column in cleaned_df.columns:
        if not pd.api.types.is_object_dtype(cleaned_df[column]) and not pd.api.types.is_string_dtype(cleaned_df[column]):
            continue

        original_values = cleaned_df[column].copy()
        cleaned_series = cleaned_df[column].map(
            lambda value: pd.NA if _is_missing_marker(value) else value
        )
        normalized_count += int((cleaned_series.isna() & original_values.notna()).sum())
        cleaned_df[column] = cleaned_series

    return cleaned_df, {
        "status": "Completed",
        "details": {
            "values_normalized": normalized_count,
            "markers_used": ["", " ", "?", "NA", "N/A", "na", "null", "unknown"],
        },
    }


def _is_missing_marker(value):
    """Return True for common missing-value sentinels while being conservative."""

    if pd.isna(value):
        return True

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return True
        return cleaned.lower() in {
            "na", "n/a", "null", "none", "unknown", "?"
        }

    return False


def handle_missing_values(df: pd.DataFrame, copy_df: bool = True):
    """
    Fill missing values.

    Numerical columns -> Mean
    Categorical columns -> Mode
    """

    cleaned_df = df.copy() if copy_df else df

    missing_before = int(cleaned_df.isnull().sum().sum())

    numerical_columns = cleaned_df.select_dtypes(include=["number"]).columns

    for column in numerical_columns:
        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].mean()
        )

    categorical_columns = cleaned_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        if not cleaned_df[column].mode().empty:
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].mode()[0]
            )

    missing_after = int(cleaned_df.isnull().sum().sum())

    preprocessing_info = {
        "status": "Completed",
        "details": {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "missing_values_filled": missing_before - missing_after
        }
    }

    return cleaned_df, preprocessing_info


# ==========================================================
# DATE PROCESSING FUNCTIONS
# ==========================================================

def _get_datetime_columns(df: pd.DataFrame):
    """
    Return columns that are datetime-like in a DataFrame.

    Detection strategy:
    - Immediately include columns that are already datetime dtype.
    - For object/string columns, sample up to the first 100 non-null values and
      attempt to parse them with pandas.to_datetime. The sampler allows a small
      number of malformed values. If most sampled values are parseable, the
      column is treated as datetime-like.

    This approach is more tolerant to mixed or noisy date strings while still
    avoiding false positives on arbitrary text columns.
    """

    datetime_columns = []
    sample_limit = 100
    parse_threshold = 0.8  # require at least 80% of sampled values to parse

    for column in df.columns:
        series = df[column]

        # Already a datetime dtype -> include
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        # Only consider object/string columns for sampling-based detection
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        non_null_values = series.dropna()
        if non_null_values.empty:
            continue

        # Sample up to the first N non-null values to limit cost and keep behavior deterministic
        sample = non_null_values.iloc[:sample_limit]

        # Try parsing without forcing dayfirst; let pandas infer formats
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually"
            )
            parsed_default = pd.to_datetime(sample, errors="coerce")

        parsed_count_default = int(parsed_default.notna().sum())
        parsed_ratio_default = parsed_count_default / float(len(sample)) if len(sample) else 0.0

        # If default parsing already parses most samples, accept column
        if parsed_ratio_default >= parse_threshold:
            datetime_columns.append(column)
            continue

        # Otherwise, try the alternative dayfirst=True parsing as a fallback
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually"
            )
            parsed_dayfirst = pd.to_datetime(sample, errors="coerce", dayfirst=True)

        parsed_count_dayfirst = int(parsed_dayfirst.notna().sum())
        parsed_ratio_dayfirst = parsed_count_dayfirst / float(len(sample)) if len(sample) else 0.0

        # Accept if either strategy yields a sufficiently high parsing ratio
        if max(parsed_ratio_default, parsed_ratio_dayfirst) >= parse_threshold:
            datetime_columns.append(column)

    return datetime_columns


def convert_date_columns(df: pd.DataFrame, copy_df: bool = True):
    """
    Convert date-like columns to datetime dtype.
    """

    cleaned_df = df.copy() if copy_df else df

    date_columns = _get_datetime_columns(cleaned_df)
    converted_columns = []
    failed_columns = []

    total_nat_count = 0

    for column in date_columns:
        try:
            # First, attempt natural inference without forcing dayfirst
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format, so each element will be parsed individually"
                )
                parsed_default = pd.to_datetime(
                                    cleaned_df[column],
                                    errors="coerce",
                                )

            # If default parsing yields meaningful non-null values, accept it
            if parsed_default.notna().any():
                cleaned_df[column] = parsed_default
                nat_count = int(cleaned_df[column].isna().sum())
                total_nat_count += nat_count
                converted_columns.append(column)
                continue

            # Fallback: try dayfirst=True if default parsing parsed almost nothing
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format, so each element will be parsed individually"
                )
                parsed_dayfirst = pd.to_datetime(
                                    cleaned_df[column],
                                    errors="coerce",
                                    dayfirst=True
                                )

            if parsed_dayfirst.notna().any():
                cleaned_df[column] = parsed_dayfirst
                nat_count = int(cleaned_df[column].isna().sum())
                total_nat_count += nat_count
                converted_columns.append(column)
            else:
                # Track NaT count even when conversion fails
                nat_count = int(parsed_dayfirst.isna().sum())
                total_nat_count += nat_count
                failed_columns.append(
                    {
                        "column": column,
                        "reason": "Conversion produced only NaT values."
                    }
                )
        except (ValueError, TypeError, OverflowError) as exc:
            failed_columns.append(
                {
                    "column": column,
                    "reason": str(exc)
                }
            )

    if date_columns:
        details = {
            "date_columns_detected": date_columns,
            "date_columns_converted": converted_columns,
            "date_columns_nat_count": total_nat_count,
        }

        if failed_columns:
            details["date_columns_failed"] = failed_columns
            details["date_columns_failed_count"] = len(failed_columns)

        preprocessing_info = {
            "status": "Completed",
            "details": details
        }
    else:
        preprocessing_info = {
            "status": "Skipped",
            "reason": "No date columns found."
        }

    return cleaned_df, preprocessing_info


def extract_date_features(df: pd.DataFrame, copy_df: bool = True):
    """
    Extract date-based features from datetime columns.

    The function detects datetime columns using Pandas data types and
    creates new columns for the year, month, day, and weekday of each
    datetime field. The original datetime columns are preserved.
    """

    updated_df = df.copy() if copy_df else df

    datetime_columns = _get_datetime_columns(updated_df)

    if not datetime_columns:
        return updated_df, {
            "status": "Skipped",
            "reason": "No datetime columns found."
        }

    generated_columns = []

    for column in datetime_columns:
        base_name = re.sub(r"(?i)\bdate\b", "", column).strip()
        if not base_name:
            base_name = column

        year_col = f"{base_name} Year"
        month_col = f"{base_name} Month"
        day_col = f"{base_name} Day"
        weekday_col = f"{base_name} Weekday"

        date_series = updated_df[column]
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format, so each element will be parsed individually"
                )
                # Let pandas infer formats naturally first
                parsed_default = pd.to_datetime(
                                    date_series,
                                    errors="coerce",
                                )

            if parsed_default.notna().any():
                date_series = parsed_default
            else:
                # Fallback to dayfirst=True only if default parsing yielded almost nothing
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually"
                    )
                    parsed_dayfirst = pd.to_datetime(
                                            date_series,
                                            errors="coerce",
                                            dayfirst=True,
                                        )

                if parsed_dayfirst.notna().any():
                    date_series = parsed_dayfirst
                else:
                    # Keep the default parsed series (likely all NaT) to preserve behavior
                    date_series = parsed_default

        updated_df[year_col] = date_series.dt.year
        updated_df[month_col] = date_series.dt.month
        updated_df[day_col] = date_series.dt.day
        updated_df[weekday_col] = date_series.dt.weekday

        generated_columns.extend([
            year_col,
            month_col,
            day_col,
            weekday_col
        ])

    preprocessing_report = {
        "status": "Completed",
        "details": {
            "datetime_columns": datetime_columns,
            "generated_columns": generated_columns,
            "generated_column_count": len(generated_columns)
        }
    }

    if not datetime_columns:
        return updated_df, {
            "status": "Skipped",
            "reason": "No datetime columns found."
        }

    if not generated_columns and datetime_columns:
        preprocessing_report["details"]["generated_columns"] = []

    return updated_df, preprocessing_report

    preprocessing_info = {
        "status": "Completed",
        "details": {
            "datetime_columns": datetime_columns,
            "generated_columns": generated_columns,
            "generated_column_count": len(generated_columns)
        }
    }

    preprocessing_report = {
        "status": "Completed",
        "details": {
            "datetime_columns": datetime_columns,
            "generated_columns": generated_columns,
            "generated_column_count": len(generated_columns)
        }
    }

    return updated_df, preprocessing_info


def calculate_delivery_time(df: pd.DataFrame, copy_df: bool = True):
    """
    Calculate delivery time in days from order and ship dates.

    If both "Order Date" and "Ship Date" exist as datetime columns,
    this function creates a new column named
    "Delivery Time (Days)" containing the difference in days.
    """

    updated_df = df.copy() if copy_df else df

    order_column = "Order Date"
    ship_column = "Ship Date"
    delivery_column = "Delivery Time (Days)"

    datetime_columns = _get_datetime_columns(updated_df)
    if order_column not in datetime_columns or ship_column not in datetime_columns:
        return updated_df, {
            "status": "Skipped",
            "reason": (
                "Order Date and Ship Date must both be present and "
                "datetime columns."
            )
        }

    order_series = updated_df[order_column]
    ship_series = updated_df[ship_column]

    if not pd.api.types.is_datetime64_any_dtype(order_series):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually"
            )
            order_series = pd.to_datetime(
                order_series,
                errors="coerce",
                dayfirst=True
            )

    if not pd.api.types.is_datetime64_any_dtype(ship_series):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually"
            )
            ship_series = pd.to_datetime(
                ship_series,
                errors="coerce",
                dayfirst=True
            )

    delivery_series = (ship_series - order_series).dt.days
    negative_delivery_days = int(delivery_series.lt(0).sum())

    updated_df[delivery_column] = delivery_series.astype("Int64")

    preprocessing_info = {
        "status": "Completed",
        "details": {
            "order_column": order_column,
            "ship_column": ship_column,
            "generated_column": delivery_column,
            "negative_delivery_days": negative_delivery_days
        }
    }

    if not isinstance(preprocessing_report := preprocessing_report if 'preprocessing_report' in globals() else None, dict):
        preprocessing_report = {"generated_columns": []}
    preprocessing_report["generated_columns"] = list(dict.fromkeys(preprocessing_report.get("generated_columns", []) + [delivery_column]))

    return updated_df, preprocessing_info


# ==========================================================
# FEATURE ENGINEERING FUNCTIONS
# ==========================================================

# (We'll add these in the next step)


# ==========================================================
# MAIN PREPROCESSING PIPELINE
# ==========================================================

def preprocess_dataset(df: pd.DataFrame):
    """
    Run the complete preprocessing pipeline.
    """

    preprocessing_report = {
        "generated_columns": []
    }

    working_df = df.copy()

    working_df, duplicate_report = remove_duplicates(working_df, copy_df=False)
    preprocessing_report["duplicates"] = duplicate_report

    working_df, missing_normalization_report = normalize_missing_markers(working_df, copy_df=False)
    preprocessing_report["missing_value_normalization"] = missing_normalization_report

    working_df, missing_report = handle_missing_values(working_df, copy_df=False)
    preprocessing_report["missing_values"] = missing_report

    working_df, date_report = convert_date_columns(working_df, copy_df=False)
    preprocessing_report["date_conversion"] = date_report

    working_df, date_features_report = extract_date_features(
        working_df,
        copy_df=False
    )
    preprocessing_report["date_features"] = date_features_report

    working_df, delivery_time_report = calculate_delivery_time(
        working_df,
        copy_df=False
    )
    preprocessing_report["delivery_time"] = delivery_time_report

    return working_df, preprocessing_report