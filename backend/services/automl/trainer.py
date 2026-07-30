from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class _LabelEncoder:
    """Minimal label encoder for categorical values."""

    def __init__(self) -> None:
        self.classes_: List[str] = []
        self.class_to_int_: Dict[str, int] = {}

    def fit(self, series: pd.Series) -> "_LabelEncoder":
        values = series.fillna("<missing>").astype(str)
        self.classes_ = [str(value) for value in pd.unique(values)]
        self.class_to_int_ = {value: index for index, value in enumerate(self.classes_)}
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        values = series.fillna("<missing>").astype(str)
        encoded = values.map(self.class_to_int_)
        return encoded.astype("Int64")

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)


class _OneHotEncoder:
    """Minimal one-hot encoder for low-cardinality categorical values."""

    def __init__(self, prefix: Optional[str] = None) -> None:
        self.prefix = prefix
        self.categories_: List[str] = []

    def fit(self, series: pd.Series) -> "_OneHotEncoder":
        values = series.fillna("<missing>").astype(str)
        self.categories_ = [str(value) for value in pd.unique(values)]
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        values = series.fillna("<missing>").astype(str)
        prefix = self.prefix or series.name
        encoded_columns: Dict[str, pd.Series] = {}

        for category in self.categories_:
            column_name = f"{prefix}_{category}"
            encoded_columns[column_name] = (values == category).astype("int64")

        return pd.DataFrame(encoded_columns)

    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        return self.fit(series).transform(series)


def prepare_training_data(df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    """
    Prepare a dataset for machine learning.

    Strategy:
    - Separate features (X) from the target (y).
    - Drop rows with missing target values because supervised learning requires
      a valid label for each training example.
    - Fill remaining missing feature values using mean for numeric columns and
      mode for categorical columns.
    - Encode categorical features automatically:
      - Use one-hot encoding for low-cardinality categorical columns.
      - Use label encoding for higher-cardinality categorical columns.
    - Encode the target with label encoding when it is categorical.

    The function returns feature and target objects plus metadata about the
    encoders that were applied.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the DataFrame.")

    working_df = df.copy()
    target_series = working_df[target_column]

    rows_before = len(working_df)
    if target_series.isna().any():
        working_df = working_df[target_series.notna()].reset_index(drop=True)
        target_series = working_df[target_column]

    X = working_df.drop(columns=[target_column])
    y = target_series
    encoders: Dict[str, Dict[str, object]] = {}

    if _is_categorical_series(y):
        target_encoder = _LabelEncoder()
        y = target_encoder.fit_transform(y)
        encoders["target"] = {
            "type": "label",
            "encoder": target_encoder,
            "classes": target_encoder.classes_,
        }

    X = X.copy()
    _fill_missing_values(X)

    categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for column in categorical_columns:
        unique_count = int(X[column].nunique(dropna=True))
        if unique_count <= 1:
            encoder = _LabelEncoder()
            X[column] = encoder.fit_transform(X[column])
            encoders[column] = {
                "type": "label",
                "encoder": encoder,
                "classes": encoder.classes_,
            }
            continue

        if unique_count <= 10:
            encoder = _OneHotEncoder(prefix=column)
            encoded_df = encoder.fit_transform(X[column])
            X = X.drop(columns=[column]).join(encoded_df)
            encoders[column] = {
                "type": "onehot",
                "encoder": encoder,
                "feature_names": encoded_df.columns.tolist(),
            }
        else:
            encoder = _LabelEncoder()
            X[column] = encoder.fit_transform(X[column])
            encoders[column] = {
                "type": "label",
                "encoder": encoder,
                "classes": encoder.classes_,
            }

    feature_names = X.columns.tolist()

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "encoders": encoders,
    }


def _fill_missing_values(df: pd.DataFrame) -> None:
    numeric_columns = df.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].mean())

    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical_columns:
        if df[column].isna().any():
            mode_values = df[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "<missing>"
            df[column] = df[column].fillna(fill_value)


def _is_categorical_series(series: pd.Series) -> bool:
    return (
        pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def train_models(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str) -> Dict[str, object]:
    """
    Train multiple machine learning models for the detected problem type.

    Training multiple models is important because different algorithms have
    different inductive biases and strengths. A simple linear model may be
    sufficient for smoothly varying numerical targets, while tree-based models
    are better for nonlinear interactions, categorical splits, and robustness
    to outliers.
    """

    models: Dict[str, object] = {}
    normalized_type = (problem_type or "").strip().lower()

    X = _to_numpy(X_train)
    y = _to_numpy(y_train)

    if "regression" in normalized_type:
        linear = _LinearRegressionModel()
        linear.fit(X, y)
        models["Linear Regression"] = linear

        tree = _DecisionTreeRegressor(max_depth=5, min_samples=5)
        tree.fit(X, y)
        models["Decision Tree Regressor"] = tree

        forest = _RandomForestRegressor(n_estimators=5, max_depth=5, min_samples=5)
        forest.fit(X, y)
        models["Random Forest Regressor"] = forest

    elif "classification" in normalized_type:
        logistic = _LogisticRegressionModel(max_iter=200, learning_rate=0.1)
        logistic.fit(X, y)
        models["Logistic Regression"] = logistic

        tree = _DecisionTreeClassifier(max_depth=5, min_samples=5)
        tree.fit(X, y)
        models["Decision Tree Classifier"] = tree

        forest = _RandomForestClassifier(n_estimators=5, max_depth=5, min_samples=5)
        forest.fit(X, y)
        models["Random Forest Classifier"] = forest

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    return {
        "trained_models": models
    }


def save_trained_model(model: Any, model_name: str) -> Dict[str, object]:
    """
    Save a trained model to storage/models using joblib.

    The folder is created automatically if it does not exist.
    """

    if joblib is None:
        raise ImportError(
            "joblib is required to save trained models. Install it with 'pip install joblib'."
        )

    storage_dir = Path("storage/models")
    storage_dir.mkdir(parents=True, exist_ok=True)

    model_filename = Path(model_name).stem + ".joblib"
    model_path = storage_dir / model_filename

    joblib.dump(model, model_path)

    return {
        "status": "Completed",
        "model_path": str(model_path),
        "model_name": model_filename,
    }


def _to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy(dtype=float)
    return np.asarray(data, dtype=float)


class _LinearRegressionModel:
    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearRegressionModel":
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        ones = np.ones((X.shape[0], 1), dtype=float)
        design = np.hstack([ones, X])
        regularization = 1e-8 * np.eye(design.shape[1])
        coeffs = np.linalg.pinv(design.T @ design + regularization) @ design.T @ y
        self.intercept_ = float(coeffs[0])
        self.coef_ = coeffs[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_


class _LogisticRegressionModel:
    def __init__(self, max_iter: int = 100, learning_rate: float = 0.05) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_multiclass: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticRegressionModel":
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        num_classes = len(self.classes_)
        self.is_multiclass = num_classes > 2

        if self.is_multiclass:
            self.coef_ = np.zeros((num_classes, X.shape[1]), dtype=float)
            self.intercept_ = np.zeros(num_classes, dtype=float)
            for class_index in range(num_classes):
                labels = (y_encoded == class_index).astype(float)
                self._fit_binary(X, labels, class_index)
        else:
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            self.intercept_ = 0.0
            self._fit_binary(X, y_encoded.astype(float))

        return self

    def _fit_binary(self, X: np.ndarray, y: np.ndarray, class_index: Optional[int] = None) -> None:
        coef = np.zeros(X.shape[1], dtype=float)
        intercept = 0.0

        for _ in range(self.max_iter):
            linear = X @ coef + intercept
            predictions = 1.0 / (1.0 + np.exp(-linear))
            gradient = X.T @ (predictions - y) / len(y)
            intercept_gradient = np.mean(predictions - y)
            coef -= self.learning_rate * gradient
            intercept -= self.learning_rate * intercept_gradient

        if class_index is None:
            self.coef_ = coef
            self.intercept_ = intercept
        else:
            assert self.coef_ is not None and self.intercept_ is not None
            self.coef_[class_index, :] = coef
            self.intercept_[class_index] = intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.is_multiclass:
            assert self.coef_ is not None and self.intercept_ is not None
            scores = X @ self.coef_.T + self.intercept_
            return self.classes_[np.argmax(scores, axis=1)]

        assert self.coef_ is not None
        linear = X @ self.coef_ + self.intercept_
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        return self.classes_[(probabilities >= 0.5).astype(int)]


class _DecisionTreeNode:
    def __init__(self) -> None:
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_DecisionTreeNode"] = None
        self.right: Optional["_DecisionTreeNode"] = None
        self.value: Optional[float] = None
        self.is_leaf: bool = False
        self.prediction: Optional[float] = None

    def predict_row(self, row: np.ndarray) -> float:
        if self.is_leaf or self.feature_index is None:
            return self.prediction  # type: ignore
        if row[self.feature_index] <= self.threshold:  # type: ignore
            return self.left.predict_row(row)  # type: ignore
        return self.right.predict_row(row)  # type: ignore


class _DecisionTreeRegressor:
    def __init__(self, max_depth: int = 5, min_samples: int = 5) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: Optional[_DecisionTreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DecisionTreeRegressor":
        self.root = self._build_tree(X, y, 0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _DecisionTreeNode:
        node = _DecisionTreeNode()
        if (
            depth >= self.max_depth
            or len(y) <= self.min_samples
            or np.unique(y).size == 1
        ):
            node.is_leaf = True
            node.prediction = float(np.nanmean(y))
            return node

        best_split = self._find_best_split(X, y)
        if best_split is None:
            node.is_leaf = True
            node.prediction = float(np.nanmean(y))
            return node

        feature_index, threshold, left_mask, right_mask = best_split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        best_score = float("inf")
        best_split = None
        for feature_index in range(X.shape[1]):
            values = np.unique(X[:, feature_index])
            if values.size <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            if thresholds.size > 20:
                thresholds = np.linspace(values.min(), values.max(), 20)
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples or right_mask.sum() < self.min_samples:
                    continue
                score = self._split_error(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score = score
                    best_split = (feature_index, threshold, left_mask, right_mask)
        return best_split

    def _split_error(self, left: np.ndarray, right: np.ndarray) -> float:
        def mse(values: np.ndarray) -> float:
            return np.mean((values - np.mean(values)) ** 2) if values.size else 0.0

        return mse(left) * left.size + mse(right) * right.size

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        return np.array([self.root.predict_row(row) for row in X], dtype=float)


class _DecisionTreeClassifier:
    def __init__(self, max_depth: int = 5, min_samples: int = 5) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: Optional[_DecisionTreeNode] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DecisionTreeClassifier":
        self.root = self._build_tree(X, y, 0)
        self.classes_ = np.unique(y)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _DecisionTreeNode:
        node = _DecisionTreeNode()
        unique_labels = np.unique(y)
        if (
            depth >= self.max_depth
            or len(y) <= self.min_samples
            or unique_labels.size == 1
        ):
            node.is_leaf = True
            node.prediction = float(np.bincount(y.astype(int)).argmax())
            return node

        best_split = self._find_best_split(X, y)
        if best_split is None:
            node.is_leaf = True
            node.prediction = float(np.bincount(y.astype(int)).argmax())
            return node

        feature_index, threshold, left_mask, right_mask = best_split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        best_score = float("inf")
        best_split = None
        for feature_index in range(X.shape[1]):
            values = np.unique(X[:, feature_index])
            if values.size <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            if thresholds.size > 20:
                thresholds = np.linspace(values.min(), values.max(), 20)
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples or right_mask.sum() < self.min_samples:
                    continue
                score = self._split_gini(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score = score
                    best_split = (feature_index, threshold, left_mask, right_mask)
        return best_split

    def _gini(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        counts = np.bincount(values.astype(int))
        probabilities = counts / counts.sum()
        return 1.0 - np.sum(probabilities ** 2)

    def _split_gini(self, left: np.ndarray, right: np.ndarray) -> float:
        total = left.size + right.size
        return (left.size / total) * self._gini(left) + (right.size / total) * self._gini(right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self.root.predict_row(row) for row in X], dtype=float)


class _RandomForestRegressor:
    def __init__(self, n_estimators: int = 5, max_depth: int = 5, min_samples: int = 5) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees: List[_DecisionTreeRegressor] = []
        self.feature_indices: List[np.ndarray] = []
        self.random_state = np.random.RandomState(42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomForestRegressor":
        n_features = X.shape[1]
        max_features = max(1, int(np.sqrt(n_features)))
        for _ in range(self.n_estimators):
            sample_indices = self.random_state.choice(len(X), len(X), replace=True)
            feature_indices = self.random_state.choice(n_features, max_features, replace=False)
            tree = _DecisionTreeRegressor(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(X[sample_indices][:, feature_indices], y[sample_indices])
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.column_stack(
            [tree.predict(X[:, features]) for tree, features in zip(self.trees, self.feature_indices)]
        )
        return np.mean(predictions, axis=1)


class _RandomForestClassifier:
    def __init__(self, n_estimators: int = 5, max_depth: int = 5, min_samples: int = 5) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees: List[_DecisionTreeClassifier] = []
        self.feature_indices: List[np.ndarray] = []
        self.random_state = np.random.RandomState(42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomForestClassifier":
        n_features = X.shape[1]
        max_features = max(1, int(np.sqrt(n_features)))
        for _ in range(self.n_estimators):
            sample_indices = self.random_state.choice(len(X), len(X), replace=True)
            feature_indices = self.random_state.choice(n_features, max_features, replace=False)
            tree = _DecisionTreeClassifier(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(X[sample_indices][:, feature_indices], y[sample_indices])
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.column_stack(
            [tree.predict(X[:, features]) for tree, features in zip(self.trees, self.feature_indices)]
        )
        majority_votes = np.apply_along_axis(lambda row: np.bincount(row.astype(int)).argmax(), axis=1, arr=predictions)
        return majority_votes
