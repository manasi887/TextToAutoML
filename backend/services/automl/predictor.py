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

    expected_raw_columns = _resolve_raw_feature_names(model_package)
    required_columns = [str(column) for column in expected_raw_columns]
    if not required_columns:
        raise ValueError("The saved model package does not include the required raw feature metadata.")

    target_column = str(model_package.get("target_column", "")).strip()
    if target_column and target_column in raw_df.columns:
        raise ValueError(f"The target column '{target_column}' must not be provided during prediction.")

    return raw_df.reindex(columns=required_columns).copy()


def _display_class_labels(model_package: dict[str, Any]) -> list[str]:
    preprocessing = model_package.get("preprocessing", {}) or {}
    classes = preprocessing.get("target_classes", [])
    if not isinstance(classes, list) or not classes:
        target_encoder = preprocessing.get("target_encoder")
        classes = getattr(target_encoder, "classes_", [])
    if not isinstance(classes, list) and not isinstance(classes, np.ndarray):
        classes = []
    classes = [value.item() if isinstance(value, np.generic) else value for value in classes]
    if not isinstance(classes, list) or not classes:
        model = model_package.get("model")
        classes = getattr(model, "classes_", [])
        if isinstance(classes, np.ndarray):
            classes = classes.tolist()
    if not classes and str(model_package.get("problem_type", "")).lower().find("classification") >= 0:
        classes = [0, 1]

    if len(classes) == 2 and set(classes) == {0, 1}:
        target_column = str(model_package.get("target_column", "target")).strip()
        normalized_target = target_column.replace("_", " ").strip()
        if normalized_target.lower() == "churn":
            positive_label = "Churned"
        else:
            positive_label = normalized_target.title() or "Positive"
        return [f"Not {positive_label}", positive_label]

    return [str(value) for value in classes]


def _apply_missing_value_strategy(df: pd.DataFrame, model_package: dict[str, Any]) -> pd.DataFrame:
    preprocessing = model_package.get("preprocessing", {}) or {}
    imputation_values = preprocessing.get("imputation_values", {})
    if not isinstance(imputation_values, dict):
        imputation_values = {}

    numeric_imputation = imputation_values.get("numeric", {}) or {}
    categorical_imputation = imputation_values.get("categorical", {}) or {}
    encoders = model_package.get("encoders", {}) or {}
    imputed_columns = preprocessing.get("imputed_columns", {}) or {}
    categorical_columns = set(imputed_columns.get("categorical", []) or [])

    for column in df.columns:
        if (column in encoders or column in categorical_columns) and df[column].isna().any() and column not in categorical_imputation:
            continue
        if column in numeric_imputation:
            df[column] = df[column].fillna(numeric_imputation[column])
        elif column not in categorical_columns and pd.api.types.is_numeric_dtype(df[column]):
            if df[column].isna().any():
                raise ValueError(
                    "The saved model package is missing training-time numeric imputation values for feature "
                    f"'{column}'. This is a training-serving compatibility gap."
                )

        if column in categorical_imputation:
            df[column] = df[column].fillna(categorical_imputation[column])
        elif column in encoders or column in categorical_columns:
            # Older packages may not persist a categorical fill value; the saved
            # encoder remains responsible for handling an unseen missing marker.
            continue
        elif df[column].dtype == object or pd.api.types.is_string_dtype(df[column]) or isinstance(df[column].dtype, pd.CategoricalDtype):
            if df[column].isna().any():
                raise ValueError(
                    "The saved model package is missing training-time categorical imputation values for feature "
                    f"'{column}'. This is a training-serving compatibility gap."
                )

    return df


def _validate_numeric_features(df: pd.DataFrame, model_package: dict[str, Any]) -> pd.DataFrame:
    preprocessing = model_package.get("preprocessing", {}) or {}
    imputed_columns = preprocessing.get("imputed_columns", {}) or {}
    numeric_columns = set(imputed_columns.get("numeric", []) or [])
    numeric_columns.update(
        (preprocessing.get("imputation_values", {}).get("numeric", {}) or {}).keys()
    )

    for column in numeric_columns.intersection(df.columns):
        original = df[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_values = original.notna() & converted.isna()
        if invalid_values.any():
            raise ValueError(f"Feature '{column}' must contain numeric values.")
        finite_values = converted.dropna().to_numpy(dtype=float)
        if finite_values.size and not np.isfinite(finite_values).all():
            raise ValueError(f"Feature '{column}' must contain finite numeric values.")
        df[column] = converted
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
    raw_df = _validate_numeric_features(raw_df, model_package)
    raw_df = _apply_missing_value_strategy(raw_df, model_package)
    encoded_df = _apply_saved_encoders(raw_df, model_package)

    model = model_package["model"]
    predictions = np.asarray(model.predict(encoded_df))
    if predictions.ndim == 0:
        predictions = predictions.reshape(1)
    if predictions.shape[0] != len(encoded_df):
        raise ValueError("The model returned a different number of predictions than input records.")
    if str(model_package.get("problem_type", "")).lower().find("regression") >= 0:
        try:
            predictions = predictions.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("The regression model returned non-numeric predictions.") from exc
    if np.issubdtype(predictions.dtype, np.number) and not np.isfinite(predictions).all():
        raise ValueError("The model returned non-finite predictions.")
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

    class_labels = _display_class_labels(model_package)
    if class_labels and len(class_labels) == len(set(class_labels)):
        result["prediction_labels"] = [
            class_labels[int(value)] if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(class_labels) else str(value)
            for value in prediction_list
        ]
        if "probabilities" in result:
            result["probability_labels"] = class_labels

    return result
