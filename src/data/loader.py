"""Generic tabular loader for CSV / TSV / Excel / Parquet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pq"}


class GenericDataLoader:
    """Load a tabular file into a DataFrame, format inferred from the suffix."""

    def __init__(self, file_path: str | Path, **read_kwargs: Any):
        self.file_path = Path(file_path)
        self.read_kwargs = read_kwargs
        self.df: Optional[pd.DataFrame] = None

    def load(self, force: bool = False) -> pd.DataFrame:
        if self.df is not None and not force:
            return self.df
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )

        if suffix in (".csv", ".txt"):
            df = pd.read_csv(self.file_path, **self.read_kwargs)
        elif suffix == ".tsv":
            df = pd.read_csv(self.file_path, sep="\t", **self.read_kwargs)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(self.file_path, **self.read_kwargs)
        else:
            df = pd.read_parquet(self.file_path, **self.read_kwargs)

        df.columns = [str(c).strip() for c in df.columns]
        self.df = df
        logger.info("Loaded %s: %d rows x %d cols", self.file_path.name, *df.shape)
        return df

    def summary(self) -> Dict[str, Any]:
        df = self.load()
        return {
            "path": str(self.file_path),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_counts": {c: int(v) for c, v in df.isna().sum().items()},
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        }


def infer_task(y: pd.Series, max_classes: int = 20) -> str:
    """Return 'classification' or 'regression' from the target's own values.

    Rules, in order:
      * non-numeric (object / category / bool) -> classification
      * <= 2 distinct values -> classification
      * integer-valued with few distinct values -> classification
      * otherwise -> regression

    Float 0.0/1.0 and string 'yes'/'no' both land in classification, which a
    naive dtype check would miss.
    """
    y = pd.Series(y).dropna()
    if y.empty:
        raise ValueError("Target column is entirely missing.")
    n_unique = int(y.nunique())
    if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if n_unique <= 2:
        return "classification"
    values = y.to_numpy()
    integral = bool(((values - values.round()) == 0).all())
    if integral and n_unique <= max_classes:
        return "classification"
    return "regression"
