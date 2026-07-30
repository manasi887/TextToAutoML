from services.dataset.loader import load_dataset
import pandas as pd


def validate_dataset(file_path: str):
    """
    Validate a dataset before preprocessing or training.
    Returns validation status, errors, and warnings.
    """

    try:
        df = load_dataset(file_path)

    except ValueError as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": []
        }

    return validate_dataset_df(df)


def validate_dataset_df(df: pd.DataFrame):
    """
    Validate a DataFrame and return status, errors, and warnings.
    """

    errors = []
    warnings = []

    # Rule 1: Dataset should not be empty
    if df.empty:
        errors.append("Dataset is empty.")

    # Rule 2: At least 2 columns are required
    if len(df.columns) < 2:
        errors.append(
            "Dataset must contain at least two columns (features and target)."
        )

    # Rule 3: Warn if dataset has very few rows
    if len(df) < 50:
        warnings.append(
            "Dataset has fewer than 50 rows. Model performance may be poor."
        )

    # Rule 4: Check for completely empty columns
    empty_columns = [
        column
        for column in df.columns
        if df[column].isnull().all()
    ]

    if empty_columns:
        warnings.append(
            f"Completely empty columns found: {', '.join(empty_columns)}"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
