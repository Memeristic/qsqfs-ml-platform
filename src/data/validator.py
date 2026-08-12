"""Dataset quality checks. Reports problems; never modifies the data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

CRITICAL = "CRITICAL"
WARNING = "WARNING"
NOTE = "NOTE"


@dataclass
class ValidationReport:
    issues: List[Dict[str, str]] = field(default_factory=list)

    def add(self, level: str, column: str | None, message: str) -> None:
        self.issues.append({"level": level, "column": column or "", "message": message})

    @property
    def messages(self) -> List[str]:
        return [f"{i['level']}: {i['message']}" for i in self.issues]

    @property
    def has_critical(self) -> bool:
        return any(i["level"] == CRITICAL for i in self.issues)

    def to_dict(self) -> Dict:
        return {
            "n_issues": len(self.issues),
            "has_critical": self.has_critical,
            "issues": self.issues,
        }


class DataValidator:
    def __init__(self, df: pd.DataFrame, target_col: str, near_constant_threshold: float = 0.99):
        self.df = df
        self.target = target_col
        self.near_constant_threshold = near_constant_threshold

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        df = self.df

        if len(df) == 0:
            report.add(CRITICAL, None, "Dataset is empty.")
            return report
        if self.target not in df.columns:
            report.add(
                CRITICAL, self.target,
                f"Target column '{self.target}' not found. "
                f"Available: {', '.join(map(str, df.columns[:25]))}",
            )
            return report

        n_dupes = int(df.duplicated().sum())
        if n_dupes:
            report.add(WARNING, None, f"{n_dupes} fully duplicated rows ({n_dupes/len(df):.1%}).")

        missing_target = int(df[self.target].isna().sum())
        if missing_target:
            report.add(
                WARNING, self.target,
                f"{missing_target} rows have a missing target and will be dropped.",
            )

        for col in df.columns:
            if col == self.target:
                continue
            series = df[col]
            n_unique = series.nunique(dropna=True)
            if n_unique <= 1:
                report.add(WARNING, col, f"Column '{col}' is constant; it carries no signal.")
                continue
            counts = series.value_counts(dropna=True)
            if len(counts):
                ratio = float(counts.iloc[0]) / max(1, int(series.notna().sum()))
                if ratio > self.near_constant_threshold:
                    report.add(
                        NOTE, col,
                        f"Column '{col}' is near-constant ({ratio:.1%} one value).",
                    )
            missing_ratio = float(series.isna().mean())
            if missing_ratio > 0.5:
                report.add(WARNING, col, f"Column '{col}' is {missing_ratio:.1%} missing.")
            if pd.api.types.is_numeric_dtype(series):
                n_inf = int(np.isinf(series.to_numpy(dtype="float64", na_value=np.nan)).sum())
                if n_inf:
                    report.add(WARNING, col, f"Column '{col}' contains {n_inf} infinite values.")
            if pd.api.types.is_object_dtype(series) and n_unique > 0.5 * len(df) and n_unique > 50:
                report.add(
                    NOTE, col,
                    f"Column '{col}' is high-cardinality text ({n_unique} distinct); "
                    "one-hot encoding it will explode the feature count.",
                )

        target = df[self.target].dropna()
        if target.nunique() <= 20:
            counts = target.value_counts()
            if len(counts) >= 2:
                imbalance = float(counts.min()) / float(counts.max())
                if imbalance < 0.1:
                    report.add(
                        WARNING, self.target,
                        f"Severe class imbalance: {counts.min()} vs {counts.max()} "
                        f"(ratio {imbalance:.3f}). Accuracy will be misleading; "
                        "read AUC/F1 and the majority-class baseline instead.",
                    )
            if len(counts) == 1:
                report.add(CRITICAL, self.target, "Target has only one class.")

        if len(df) < 50:
            report.add(
                WARNING, None,
                f"Only {len(df)} rows. Results from this dataset will not be stable.",
            )
        n_features = df.shape[1] - 1
        if n_features > len(df):
            report.add(
                WARNING, None,
                f"{n_features} features for {len(df)} rows (p > n). "
                "Feature selection will overfit without nested validation.",
            )
        return report


def naive_baseline(y: np.ndarray, is_classification: bool) -> Dict[str, float]:
    """The score to beat before any model is worth reporting."""
    y = np.asarray(y)
    if is_classification:
        values, counts = np.unique(y, return_counts=True)
        majority = values[int(np.argmax(counts))]
        return {
            "strategy": "majority_class",
            "majority_class": majority,
            "accuracy": float(np.mean(y == majority)),
        }
    mean_val = float(np.mean(y))
    return {
        "strategy": "mean_prediction",
        "mean": mean_val,
        "rmse": float(np.sqrt(np.mean((y - mean_val) ** 2))),
        "mae": float(np.mean(np.abs(y - mean_val))),
    }
