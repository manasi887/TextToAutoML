from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _resolve_raw_feature_names(model_package: dict[str, Any]) -> list[str]:
    preprocessing = model_package.get("preprocessing", {}) or {}
    raw_names = preprocessing.get("raw_feature_names")
    if isinstance(raw_names, list) and raw_names:
        return [str(name) for name in raw_names]

    imputed_columns = preprocessing.get("imputed_columns", {}) or {}
    raw_columns: list[str] = []
    for section in (imputed_columns.get("numeric", []), imputed_columns.get("categorical", [])):
        for value in section:
            raw_columns.append(str(value))
    for column in model_package.get("encoders", {}):
        raw_columns.append(str(column))

    deduplicated: list[str] = []
    for column in raw_columns:
        if column not in deduplicated:
            deduplicated.append(column)
    return deduplicated


def _validate_prediction_input(model_package: dict[str, Any], input_data: Any) -> pd.DataFrame:
    if not isinstance(input_data, list):
        raise ValueError("Prediction `data` must be a list of records.")
    if not input_data:
        raise ValueError("Prediction `data` cannot be empty.")

    for index, record in enumerate(input_data):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} must be an object with feature names and values.")

    raw_df = pd.DataFrame(input_data)
    if raw_df.empty:
        raise ValueError("Prediction `data` cannot be empty.")

    expected_raw_columns = _resolve_raw_feature_names(model_package)
    required_columns = [str(column) for column in expected_raw_columns]
    if not required_columns:
        raise ValueError("The saved model package does not include the required raw feature metadata.")

    target_column = str(model_package.get("target_column", "")).strip()
    if target_column and target_column in raw_df.columns:
        raise ValueError(f"The target column '{target_column}' must not be provided during prediction.")

    missing_columns = [column for column in required_columns if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature(s): {', '.join(missing_columns)}")

    return raw_df[required_columns].copy()


def _apply_missing_value_strategy(df: pd.DataFrame, model_package: dict[str, Any]) -> pd.DataFrame:
    preprocessing = model_package.get("preprocessing", {}) or {}
    imputation_values = preprocessing.get("imputation_values", {})
    if not isinstance(imputation_values, dict):
        imputation_values = {}

    numeric_imputation = imputation_values.get("numeric", {}) or {}
    categorical_imputation = imputation_values.get("categorical", {}) or {}

    for column in df.columns:
        if column in numeric_imputation:
            df[column] = df[column].fillna(numeric_imputation[column])
        elif pd.api.types.is_numeric_dtype(df[column]):
            if df[column].isna().any():
                raise ValueError(
                    "The saved model package is missing training-time numeric imputation values for feature "
                    f"'{column}'. This is a training-serving compatibility gap."
                )

        if column in categorical_imputation:
            df[column] = df[column].fillna(categorical_imputation[column])
        elif df[column].dtype == object or pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            if df[column].isna().any():
                raise ValueError(
                    "The saved model package is missing training-time categorical imputation values for feature "
                    f"'{column}'. This is a training-serving compatibility gap."
                )

    return df


def _apply_saved_encoders(df: pd.DataFrame, model_package: dict[str, Any]) -> pd.DataFrame:
    encoders = model_package.get("encoders", {}) or {}
    if not isinstance(encoders, dict):
        raise ValueError("The saved model package is missing encoder metadata.")

    encoded_parts: list[pd.DataFrame] = []
    for column in df.columns:
        if column not in encoders:
            encoded_parts.append(df[[column]])
            continue

        encoder_info = encoders[column]
        encoder = encoder_info.get("encoder")
        if encoder is None:
            encoded_parts.append(df[[column]])
            continue

        values = df[column].astype(str).fillna("<missing>")
        transformed = encoder.transform(values.to_numpy().reshape(-1, 1))
        if encoder_info.get("type") == "onehot":
            feature_names = encoder_info.get("feature_names") or encoder.get_feature_names_out([column]).tolist()
            dense_values = np.asarray(transformed)
            if hasattr(dense_values, "toarray"):
                dense_values = dense_values.toarray()
            encoded_df = pd.DataFrame(dense_values, columns=feature_names, index=df.index)
        else:
            dense_values = np.asarray(transformed).reshape(-1)
            encoded_df = pd.DataFrame({column: dense_values}, index=df.index)

        encoded_parts.append(encoded_df)

    combined = pd.concat(encoded_parts, axis=1)

    expected_feature_order = list(model_package.get("feature_names", combined.columns.tolist()))
    missing_expected = [feature for feature in expected_feature_order if feature not in combined.columns]
    if missing_expected:
        raise ValueError(
            "The saved model package is incompatible with the input data: missing encoded feature(s): "
            f"{', '.join(missing_expected)}"
        )

    return combined[expected_feature_order].copy()


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalar and array values to native Python types for JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_python_scalar(item) for item in value]
    if isinstance(value, tuple):
        return [_to_python_scalar(item) for item in value]
    return value


def predict_with_model(model_package: dict[str, Any], input_data: Any) -> dict[str, Any]:
    """Apply saved preprocessing and encoders to raw input data and return predictions."""
    if not isinstance(model_package, dict) or not model_package.get("model"):
        raise ValueError("A valid persisted model package is required for prediction.")

    raw_df = _validate_prediction_input(model_package, input_data)
    raw_df = _apply_missing_value_strategy(raw_df, model_package)
    encoded_df = _apply_saved_encoders(raw_df, model_package)

    model = model_package["model"]
    predictions = model.predict(encoded_df)
    predictions = _to_python_scalar(predictions)
    if isinstance(predictions, list):
        prediction_list = predictions
    elif isinstance(predictions, (int, float, bool)):
        prediction_list = [predictions]
    else:
        prediction_list = list(predictions)

    result: dict[str, Any] = {
        "status": "success",
        "model_id": model_package.get("model_id"),
        "model_name": model_package.get("model_name"),
        "problem_type": model_package.get("problem_type"),
        "target_column": model_package.get("target_column"),
        "predictions": prediction_list,
        "prediction_count": len(prediction_list),
    }

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(encoded_df)
            result["probabilities"] = _to_python_scalar(probabilities)
        except Exception:
            pass

    return result
