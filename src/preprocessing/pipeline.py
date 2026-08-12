"""Impute / scale / encode. Fitted on training rows only, always."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}


def _make_ohe(handle_unknown: str = "ignore", max_categories: Optional[int] = None):
    """OneHotEncoder across sklearn versions (sparse_output vs sparse)."""
    kwargs = {"handle_unknown": handle_unknown}
    if max_categories is not None:
        try:
            OneHotEncoder(max_categories=2)
            kwargs["max_categories"] = max_categories
            if handle_unknown == "ignore":
                kwargs["handle_unknown"] = "infrequent_if_exist"
        except TypeError:
            pass
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def split_column_types(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    numerical = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return numerical, categorical


class PreprocessingPipeline:
    """Column-typed preprocessing with stable, inspectable feature names."""

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        impute_strategy: str = "median",
        categorical_impute: str = "most_frequent",
        scaler_type: str = "standard",
        encode_categorical: bool = True,
        max_categories: Optional[int] = 30,
        drop_datetime: bool = True,
    ):
        self.numerical_cols = list(numerical_cols or [])
        self.categorical_cols = list(categorical_cols or [])
        self.impute_strategy = impute_strategy
        self.categorical_impute = categorical_impute
        self.scaler_type = str(scaler_type).lower()
        self.encode_categorical = encode_categorical
        self.max_categories = max_categories
        self.drop_datetime = drop_datetime
        self._ct: Optional[ColumnTransformer] = None
        self._feature_names: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.drop_datetime:
            dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
            if len(dt_cols):
                logger.info("Dropping datetime columns: %s", list(dt_cols))
                X = X.drop(columns=list(dt_cols))
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, X_train: pd.DataFrame) -> "PreprocessingPipeline":
        X_train = self._prepare(X_train)
        if not self.numerical_cols and not self.categorical_cols:
            self.numerical_cols, self.categorical_cols = split_column_types(X_train)

        num_cols = [c for c in self.numerical_cols if c in X_train.columns]
        cat_cols = [c for c in self.categorical_cols if c in X_train.columns]

        transformers = []
        if num_cols:
            steps = [("imputer", SimpleImputer(strategy=self.impute_strategy))]
            scaler_cls = SCALERS.get(self.scaler_type, StandardScaler)
            if scaler_cls is not None:
                steps.append(("scaler", scaler_cls()))
            transformers.append(("numerical", Pipeline(steps), num_cols))

        if cat_cols:
            encoder = (
                _make_ohe(max_categories=self.max_categories)
                if self.encode_categorical
                else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            )
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy=self.categorical_impute, fill_value="missing")),
                    ("encoder", encoder),
                ]),
                cat_cols,
            ))

        if not transformers:
            raise ValueError("No usable feature columns after type inspection.")

        self._ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        self._ct.fit(X_train)

        try:
            self._feature_names = [str(n) for n in self._ct.get_feature_names_out()]
        except Exception:
            self._feature_names = []

        self._fitted = True
        logger.info(
            "Preprocessor fitted: %d numeric + %d categorical -> %d output features",
            len(num_cols), len(cat_cols), len(self._feature_names) or -1,
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._ct is None:
            raise RuntimeError("Call fit() before transform().")
        out = self._ct.transform(self._prepare(X))
        if hasattr(out, "toarray"):
            out = out.toarray()
        out = np.asarray(out, dtype=np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def get_feature_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return list(self._feature_names)
