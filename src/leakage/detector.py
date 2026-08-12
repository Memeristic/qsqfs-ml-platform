"""Target-leakage detection.

Two classes of finding:

  * **excluded** - mechanical duplicates of the target. Dropped automatically,
    because a column that reproduces the target is never a legitimate feature.
  * **flagged**  - name-based proxies and high correlations. Reported for a
    human to decide on. Never dropped silently: a strong correlation can be the
    real clinical signal you are trying to find.

Correlations are computed on the training rows only. Screening on the full
frame lets the test set influence which columns survive, which is itself a
leak.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .rules import ICD_PATTERNS, load_rules, normalise, proxy_terms

logger = logging.getLogger(__name__)


@dataclass
class LeakageReport:
    domain: str = "generic"
    excluded_columns: List[str] = field(default_factory=list)
    flagged_columns: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    n_columns_screened: int = 0

    @property
    def warnings(self) -> List[str]:
        out = [f"EXCLUDED: '{c}' duplicates the target." for c in self.excluded_columns]
        out += [f"FLAGGED: '{f['column']}' - {f['reason']}" for f in self.flagged_columns]
        out += [f"NOTE: {n}" for n in self.notes]
        return out

    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "n_columns_screened": self.n_columns_screened,
            "excluded_columns": list(self.excluded_columns),
            "flagged_columns": list(self.flagged_columns),
            "notes": list(self.notes),
        }

    def text(self) -> str:
        lines = ["LEAKAGE REPORT", "=" * 60, f"Domain rule set: {self.domain}",
                 f"Columns screened: {self.n_columns_screened}", ""]
        if self.excluded_columns:
            lines.append("Automatically excluded (exact target duplicates):")
            lines += [f"  - {c}" for c in self.excluded_columns]
        else:
            lines.append("Automatically excluded: none.")
        lines.append("")
        if self.flagged_columns:
            lines.append("Flagged for review (NOT removed):")
            for f in self.flagged_columns:
                lines.append(f"  - {f['column']}: {f['reason']}")
        else:
            lines.append("Flagged for review: none.")
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


class LeakageDetector:
    def __init__(
        self,
        rules_path: Optional[str] = None,
        correlation_threshold: float = 0.95,
        exact_duplicate_only: bool = True,
    ):
        self.rules = load_rules(rules_path)
        self.correlation_threshold = float(correlation_threshold)
        self.exact_duplicate_only = exact_duplicate_only

    def detect(
        self,
        df: pd.DataFrame,
        target_col: str,
        domain: str = "generic",
        train_index: Optional[Sequence[int]] = None,
    ) -> LeakageReport:
        report = LeakageReport(domain=(domain or "generic").lower())
        if target_col not in df.columns:
            report.notes.append(f"Target '{target_col}' absent; screening skipped.")
            return report

        features = [c for c in df.columns if c != target_col]
        report.n_columns_screened = len(features)

        screen = df.loc[list(train_index)] if train_index is not None else df
        if train_index is not None:
            report.notes.append(
                f"Correlation screening used the {len(screen)} training rows only."
            )
        else:
            report.notes.append(
                "No training index supplied; correlations were computed on all rows. "
                "Prefer passing train_index so the test set cannot influence screening."
            )

        target = screen[target_col]

        # 1. Exact duplicates of the target.
        for col in features:
            series = screen[col]
            if series.dtype == target.dtype and series.equals(target):
                report.excluded_columns.append(col)
                continue
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
                both = pd.concat([series, target], axis=1).dropna()
                if len(both) and np.allclose(
                    both.iloc[:, 0].to_numpy(dtype=float),
                    both.iloc[:, 1].to_numpy(dtype=float),
                ):
                    report.excluded_columns.append(col)

        # 2. Deterministic categorical mapping (one value of X => one value of y).
        if not self.exact_duplicate_only:
            for col in features:
                if col in report.excluded_columns:
                    continue
                series = screen[col]
                if series.nunique(dropna=True) < 2 or series.nunique(dropna=True) > 50:
                    continue
                grouped = screen.groupby(col, observed=True)[target_col].nunique()
                if len(grouped) and (grouped <= 1).all():
                    report.flagged_columns.append({
                        "column": col,
                        "reason": "each of its values maps to exactly one target value "
                                  "(deterministic mapping)",
                        "kind": "deterministic_map",
                    })

        # 3. High linear correlation with a numeric target.
        if pd.api.types.is_numeric_dtype(target):
            target_numeric = pd.to_numeric(target, errors="coerce")
            for col in features:
                if col in report.excluded_columns:
                    continue
                series = screen[col]
                if not pd.api.types.is_numeric_dtype(series):
                    continue
                both = pd.concat(
                    [pd.to_numeric(series, errors="coerce"), target_numeric], axis=1
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if len(both) < 10 or both.iloc[:, 0].nunique() < 2:
                    continue
                corr = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
                if np.isfinite(corr) and abs(corr) >= self.correlation_threshold:
                    report.flagged_columns.append({
                        "column": col,
                        "reason": f"correlation with target r={corr:.3f} "
                                  f"(>= {self.correlation_threshold})",
                        "kind": "high_correlation",
                        "correlation": corr,
                    })

        # 4. Name-based proxies from the domain rule set.
        terms = proxy_terms(report.domain, self.rules)
        already = {f["column"] for f in report.flagged_columns}
        for col in features:
            if col in report.excluded_columns or col in already:
                continue
            name = normalise(col)
            hit = next((t for t in terms if t and t in name), None)
            if hit:
                report.flagged_columns.append({
                    "column": col,
                    "reason": f"name matches '{hit}' in the '{report.domain}' rule set",
                    "kind": "name_proxy",
                })

        # 5. ICD-style codes for the domain.
        import re as _re
        for pattern in ICD_PATTERNS.get(report.domain, []):
            regex = _re.compile(pattern, _re.IGNORECASE)
            for col in features:
                if col in report.excluded_columns:
                    continue
                series = screen[col]
                if not pd.api.types.is_object_dtype(series):
                    continue
                sample = series.dropna().astype(str).head(500)
                if len(sample) and sample.str.match(regex).mean() > 0.1:
                    report.flagged_columns.append({
                        "column": col,
                        "reason": f"contains diagnosis codes matching /{pattern}/",
                        "kind": "icd_code",
                    })
                    break

        # 6. Date/time columns.
        for col in features:
            name = normalise(col)
            if any(k in name for k in ("date", "time", "timestamp", "datetime")):
                report.notes.append(
                    f"'{col}' looks temporal. Use cv_type='time_series' or drop it; "
                    "a raw timestamp lets a model memorise cohort order."
                )

        if report.excluded_columns:
            logger.warning("Excluded as target duplicates: %s", report.excluded_columns)
        for flag in report.flagged_columns:
            logger.warning("Flagged '%s': %s", flag["column"], flag["reason"])
        return report


def apply_exclusions(df: pd.DataFrame, report: LeakageReport, drop_flagged: bool = False) -> pd.DataFrame:
    cols = list(report.excluded_columns)
    if drop_flagged:
        cols += [f["column"] for f in report.flagged_columns]
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    return df.drop(columns=cols) if cols else df
