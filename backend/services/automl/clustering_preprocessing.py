from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler


_MISSING_CATEGORY = "\x00__automl_missing__"
_NANOSECONDS_PER_DAY = 86_400_000_000_000
_LABEL_TOKENS = {
    "target",
    "label",
    "labels",
    "class",
    "classes",
    "outcome",
    "result",
    "groundtruth",
    "cluster",
    "clusters",
    "segment",
    "segments",
}
_DATE_TOKENS = {"date", "datetime", "timestamp"}
_IDENTIFIER_TOKENS = {"id", "identifier", "key"}
_BINARY_OUTCOME_TOKENS = {"churn", "churned", "exit", "exited"}


class ClusteringPreprocessor(BaseEstimator, TransformerMixin):
    """Fit a mixed-type, prediction-safe feature transform for clustering."""

    def __init__(
        self,
        max_categories: int = 64,
        identifier_unique_ratio: float = 0.98,
        date_parse_threshold: float = 0.8,
    ) -> None:
        self.max_categories = max_categories
        self.identifier_unique_ratio = identifier_unique_ratio
        self.date_parse_threshold = date_parse_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ClusteringPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Clustering preprocessing requires a pandas DataFrame.")
        if X.empty:
            raise ValueError("Clustering preprocessing cannot fit an empty dataset.")
        if not X.columns.is_unique:
            raise ValueError("Clustering preprocessing requires unique column names.")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2.")
        if not 0 < self.identifier_unique_ratio <= 1:
            raise ValueError("identifier_unique_ratio must be between 0 and 1.")
        if not 0 < self.date_parse_threshold <= 1:
            raise ValueError("date_parse_threshold must be between 0 and 1.")

        frame = X.copy()
        frame.columns = [str(column) for column in frame.columns]
        if not frame.columns.is_unique:
            raise ValueError("Column names must remain unique when converted to strings.")

        self.input_features_ = frame.columns.tolist()
        self.retained_features_: list[str] = []
        self.excluded_features_: list[dict[str, str]] = []
        self.feature_metadata_: list[dict[str, Any]] = []
        self.feature_types_: dict[str, str] = {}
        self.date_origins_: dict[str, float] = {}
        self.numeric_imputations_: dict[str, float] = {}
        self.categorical_encoders_: dict[str, OneHotEncoder] = {}
        self.missing_category_values_: dict[str, str] = {}
        numeric_values: dict[str, np.ndarray] = {}

        for column in self.input_features_:
            values = frame[column]
            non_missing = values.dropna()
            unique_count = int(non_missing.nunique(dropna=True))
            unique_ratio = unique_count / len(frame) if len(frame) else 0.0
            tokens = set(re.findall(r"[a-z0-9]+", column.lower()))
            exclusion_reason = self._semantic_exclusion(tokens, unique_count)
            if exclusion_reason:
                self._exclude(column, exclusion_reason, unique_count, unique_ratio)
                continue
            if non_missing.empty:
                self._exclude(column, "all values are missing", unique_count, unique_ratio)
                continue
            if unique_count <= 1:
                self._exclude(column, "constant column", unique_count, unique_ratio)
                continue

            parsed_dates = self._parse_dates(values)
            date_ratio = float(parsed_dates.notna().sum() / len(values))
            date_named = bool(tokens & _DATE_TOKENS)
            datetime_column = pd.api.types.is_datetime64_any_dtype(values)
            date_text = not pd.api.types.is_numeric_dtype(values)
            has_date_separators = date_text and self._contains_date_separators(
                values, self.date_parse_threshold
            )
            date_like = date_named or datetime_column or has_date_separators
            parsed_as_date = (
                date_ratio >= self.date_parse_threshold
                and parsed_dates.nunique(dropna=True) > 1
                and (
                    datetime_column
                    or (
                        date_text
                        and (date_named or has_date_separators)
                    )
                )
            )
            if parsed_as_date:
                days = self._date_days(parsed_dates)
                origin = float(np.nanmin(days))
                offsets = days - origin
                self.date_origins_[column] = origin
                numeric_values[column] = offsets
                self._retain(column, "date", unique_count, unique_ratio)
                continue
            if date_like:
                self._exclude(
                    column,
                    "date-like column could not be parsed reliably",
                    unique_count,
                    unique_ratio,
                )
                continue

            is_bool = pd.api.types.is_bool_dtype(values)
            numeric = pd.to_numeric(values, errors="coerce")
            numeric_ratio = float(numeric.notna().sum() / len(values))
            zip_like = bool(tokens & {"zip", "postal", "postcode"})
            numeric_candidate = not is_bool and not zip_like and (
                pd.api.types.is_numeric_dtype(values)
                or (numeric_ratio >= 0.8 and unique_count > 2)
            )
            if self._is_identifier(
                tokens,
                unique_count,
                unique_ratio,
                len(frame),
                numeric_candidate,
            ):
                self._exclude(column, "likely identifier or near-unique key", unique_count, unique_ratio)
                continue

            if numeric_candidate:
                array = numeric.to_numpy(dtype=float, na_value=np.nan, copy=True)
                array[~np.isfinite(array)] = np.nan
                numeric_values[column] = array
                self._retain(column, "numeric", unique_count, unique_ratio)
            else:
                self._retain(column, "categorical", unique_count, unique_ratio)

        numeric_columns = [
            column
            for column in self.retained_features_
            if self.feature_types_[column] in {"numeric", "date"}
        ]
        if numeric_columns:
            imputed_columns = []
            for column in numeric_columns:
                values = numeric_values[column]
                finite_values = values[np.isfinite(values)]
                fill_value = float(np.median(finite_values)) if finite_values.size else 0.0
                self.numeric_imputations_[column] = fill_value
                imputed_columns.append(np.where(np.isfinite(values), values, fill_value))
            self.numeric_scaler_ = RobustScaler()
            self.numeric_scaler_.fit(np.column_stack(imputed_columns))
        else:
            self.numeric_scaler_ = None

        for column in self.retained_features_:
            if self.feature_types_[column] != "categorical":
                continue
            present_values = set(frame[column].dropna().astype(str))
            missing_marker = _MISSING_CATEGORY
            while missing_marker in present_values:
                missing_marker = f"{missing_marker}_"
            self.missing_category_values_[column] = missing_marker
            values = frame[column].astype("string").fillna(missing_marker).astype(str)
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                max_categories=self.max_categories,
                sparse_output=False,
                dtype=np.float64,
            )
            encoder.fit(values.to_numpy().reshape(-1, 1))
            self.categorical_encoders_[column] = encoder

        self.feature_names_out_ = self._make_feature_names()
        self.n_features_in_ = len(self.input_features_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "input_features_"):
            raise ValueError("ClusteringPreprocessor must be fitted before transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Clustering preprocessing requires a pandas DataFrame.")
        if not X.columns.is_unique:
            raise ValueError("Clustering preprocessing requires unique column names.")

        frame = X.copy()
        frame.columns = [str(column) for column in frame.columns]
        if not frame.columns.is_unique:
            raise ValueError("Column names must remain unique when converted to strings.")

        output = pd.DataFrame(index=frame.index)
        numeric_columns = [
            column
            for column in self.retained_features_
            if self.feature_types_[column] in {"numeric", "date"}
        ]
        if numeric_columns:
            numeric_arrays = []
            for column in numeric_columns:
                if column not in frame:
                    values = pd.Series(np.nan, index=frame.index, dtype=float)
                elif self.feature_types_[column] == "date":
                    values = self._date_days(self._parse_dates(frame[column]))
                    values = values - self.date_origins_[column]
                else:
                    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                values = np.asarray(values, dtype=float).copy()
                values[~np.isfinite(values)] = np.nan
                values = np.where(
                    np.isfinite(values), values, self.numeric_imputations_[column]
                )
                numeric_arrays.append(values)
            scaled = self.numeric_scaler_.transform(np.column_stack(numeric_arrays))
            for index, column in enumerate(numeric_columns):
                output[f"numeric__{column}"] = scaled[:, index]

        for column, encoder in self.categorical_encoders_.items():
            if column not in frame:
                values = pd.Series(
                    self.missing_category_values_[column],
                    index=frame.index,
                    dtype=str,
                )
            else:
                values = frame[column].astype("string").fillna(
                    self.missing_category_values_[column]
                ).astype(str)
            encoded = encoder.transform(values.to_numpy().reshape(-1, 1))
            names = self._categorical_feature_names(column, encoder)
            for index, name in enumerate(names):
                output[name] = encoded[:, index]

        return output.reindex(columns=self.feature_names_out_, fill_value=0.0).astype(float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise ValueError("ClusteringPreprocessor must be fitted before requesting feature names.")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _semantic_exclusion(self, tokens: set[str], unique_count: int) -> str | None:
        if tokens & _LABEL_TOKENS or {"ground", "truth"} <= tokens:
            return "target, outcome, segment, or existing cluster label"
        if unique_count <= 2 and tokens & _BINARY_OUTCOME_TOKENS:
            return "binary outcome-like column"
        return None

    def _is_identifier(
        self,
        tokens: set[str],
        unique_count: int,
        unique_ratio: float,
        row_count: int,
        numeric_candidate: bool,
    ) -> bool:
        personal_name = (
            "name" in tokens
            and bool(tokens & {"customer", "person", "full", "first", "last"})
        )
        if personal_name:
            return unique_count >= 2
        named_key = bool(tokens & _IDENTIFIER_TOKENS)
        near_unique = (
            not numeric_candidate
            and row_count >= 10
            and unique_count >= 10
            and unique_ratio >= self.identifier_unique_ratio
        )
        return unique_ratio >= 0.8 and unique_count >= min(10, row_count) and (
            named_key or near_unique
        )

    @staticmethod
    def _parse_dates(values: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError, OverflowError):
            return pd.to_datetime(values, errors="coerce", utc=True)

    @staticmethod
    def _contains_date_separators(values: pd.Series, parse_ratio: float) -> bool:
        sample = values.dropna().astype(str)
        if sample.empty:
            return False
        date_pattern = re.compile(r"\d{1,4}[-/:]\d{1,2}")
        separated = int(sample.str.contains(date_pattern).sum())
        return separated / len(sample) >= parse_ratio

    @staticmethod
    def _date_days(values: pd.Series) -> np.ndarray:
        parsed = pd.Series(values)
        days = (
            parsed.astype("int64").to_numpy(dtype=float, copy=True)
            / _NANOSECONDS_PER_DAY
        )
        days[parsed.isna().to_numpy()] = np.nan
        return days

    def _retain(
        self,
        column: str,
        feature_type: str,
        unique_count: int,
        unique_ratio: float,
    ) -> None:
        self.retained_features_.append(column)
        self.feature_types_[column] = feature_type
        self.feature_metadata_.append(
            {
                "column": column,
                "type": feature_type,
                "unique_count": unique_count,
                "unique_ratio": round(unique_ratio, 4),
                "reason": f"retained as {feature_type}",
            }
        )

    def _exclude(
        self,
        column: str,
        reason: str,
        unique_count: int,
        unique_ratio: float,
    ) -> None:
        self.excluded_features_.append({"column": column, "reason": reason})
        self.feature_metadata_.append(
            {
                "column": column,
                "type": "excluded",
                "unique_count": unique_count,
                "unique_ratio": round(unique_ratio, 4),
                "reason": reason,
            }
        )

    @staticmethod
    def _categorical_feature_names(column: str, encoder: OneHotEncoder) -> list[str]:
        names = encoder.get_feature_names_out([column]).tolist()
        return [f"categorical__{name}" for name in names]

    def _make_feature_names(self) -> list[str]:
        names = []
        numeric_columns = [
            column
            for column in self.retained_features_
            if self.feature_types_[column] in {"numeric", "date"}
        ]
        names.extend(f"numeric__{name}" for name in numeric_columns)
        for column, encoder in self.categorical_encoders_.items():
            names.extend(self._categorical_feature_names(column, encoder))
        if len(names) != len(set(names)):
            raise ValueError("Clustering preprocessing generated duplicate feature names.")
        return names
