from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd


def load_model(model_path: str) -> Dict[str, object]:
    """
    Load a saved model from disk using joblib.

    Returns a structured dictionary with the loaded model and path.
    """

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(path)
    return {
        "status": "Completed",
        "model_path": str(path),
        "model": model,
    }


def predict(model: Any, input_data: Union[pd.DataFrame, pd.Series, np.ndarray, List, Dict]) -> Dict[str, object]:
    """
    Generate predictions for new input data using a loaded model.

    Accepts pandas DataFrames, Series, NumPy arrays, lists, or dicts. Returns
    predictions in a structured dictionary to keep the API reusable.
    """

    if not hasattr(model, "predict"):
        raise AttributeError("The provided model must implement a predict() method.")

    if isinstance(input_data, pd.DataFrame):
        data = input_data.to_numpy()
    elif isinstance(input_data, pd.Series):
        data = input_data.to_numpy().reshape(-1, 1)
    elif isinstance(input_data, np.ndarray):
        data = input_data
    elif isinstance(input_data, dict):
        data = pd.DataFrame([input_data]).to_numpy()
    elif isinstance(input_data, list):
        data = np.asarray(input_data)
    else:
        raise TypeError(
            "input_data must be a pandas DataFrame, Series, numpy array, list, or dict."
        )

    predictions = model.predict(data)
    predictions_array = np.asarray(predictions)
    return {
        "status": "Completed",
        "predictions": predictions_array.tolist(),
        "count": int(predictions_array.shape[0]) if predictions_array.ndim > 0 else 1,
    }
