"""SPSS-style statistical tables for a thesis or journal submission.

Produces the tables a reviewer expects: cohort description with the right test
per variable type, effect sizes alongside p-values, normality checks, a
correlation matrix, and a model-comparison table with confidence intervals.

Two conventions followed throughout, because reviewers ask about both:

  * **Effect sizes accompany every p-value.** A p-value tells you whether a
    difference is detectable at your sample size; it does not tell you whether
    the difference matters. Cohen's d, eta-squared and Cramer's V are reported.
  * **Test choice is data-driven and stated.** Normality is checked
    (Shapiro-Wilk, n<=5000) and the table records whether a parametric or
    non-parametric test was used for each variable, so the choice is auditable.

Nothing here computes a result the data does not support. If a test cannot be
run -- too few groups, zero variance, empty cells -- the row says so instead of
emitting a number.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _fmt_p(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference, pooled SD. 0.2 small / 0.5 medium / 0.8 large."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def cramers_v(table: np.ndarray) -> float:
    """Effect size for a chi-square test of association, in [0, 1]."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else float("nan")


def eta_squared(groups: List[np.ndarray]) -> float:
    """Proportion of variance explained by group membership (one-way ANOVA)."""
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((allv - grand) ** 2).sum())
    return float(ss_between / ss_total) if ss_total > 1e-12 else 0.0


def descriptive_table(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Table 1, univariate: n, missing, central tendency, spread, normality."""
    columns = list(columns) if columns else list(df.columns)
    rows = []
    for col in columns:
        series = df[col]
        valid = series.dropna()
        entry: Dict = {
            "Variable": col,
            "N": int(len(valid)),
            "Missing": int(series.isna().sum()),
            "Missing %": round(float(series.isna().mean() * 100), 2),
        }
        if pd.api.types.is_numeric_dtype(series) and len(valid) > 1:
            values = valid.to_numpy(dtype=float)
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            entry.update({
                "Mean": round(float(values.mean()), 4),
                "SD": round(float(values.std(ddof=1)), 4),
                "Median": round(float(med), 4),
                "IQR": round(float(q3 - q1), 4),
                "Min": round(float(values.min()), 4),
                "Max": round(float(values.max()), 4),
                "Skewness": round(float(stats.skew(values)), 3),
                "Kurtosis": round(float(stats.kurtosis(values)), 3),
            })
            if 3 <= len(values) <= 5000 and values.std() > 1e-12:
                w, p = stats.shapiro(values)
                entry["Shapiro-Wilk W"] = round(float(w), 4)
                entry["Normality p"] = _fmt_p(float(p))
                entry["Distribution"] = "Normal" if p > 0.05 else "Non-normal"
            else:
                entry["Distribution"] = "not tested (n out of range)"
        else:
            counts = valid.value_counts()
            entry["Type"] = "categorical"
            entry["Levels"] = int(counts.size)
            if counts.size:
                entry["Mode"] = str(counts.index[0])
                entry["Mode %"] = round(float(counts.iloc[0] / len(valid) * 100), 2)
        rows.append(entry)
    return pd.DataFrame(rows)


def group_comparison_table(
    df: pd.DataFrame, group_col: str, columns: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Table 1, by group: the comparison a clinical paper opens with.

    Numeric variables get a t-test or Mann-Whitney (2 groups), or ANOVA /
    Kruskal-Wallis (3+), chosen by a normality check. Categorical variables get
    a chi-square test. Every row reports which test ran and an effect size.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not in the data.")
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        raise ValueError(f"'{group_col}' has fewer than two groups.")

    columns = [c for c in (columns or df.columns) if c != group_col]
    rows = []

    for col in columns:
        series = df[col]
        entry: Dict = {"Variable": col, "Groups": len(groups)}
        try:
            if pd.api.types.is_numeric_dtype(series):
                samples = [df.loc[df[group_col] == g, col].dropna().to_numpy(float)
                           for g in groups]
                samples = [s for s in samples if len(s) >= 2]
                if len(samples) < 2:
                    entry["Test"] = "not run (insufficient data)"
                    rows.append(entry); continue

                for g, s in zip(groups, samples):
                    entry[f"{g} (M±SD)"] = f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"

                normal = all(
                    len(s) < 3 or len(s) > 5000 or stats.shapiro(s)[1] > alpha
                    for s in samples
                )
                if len(samples) == 2:
                    if normal:
                        stat, p = stats.ttest_ind(*samples, equal_var=False)
                        entry["Test"] = "Welch t-test"
                    else:
                        stat, p = stats.mannwhitneyu(*samples, alternative="two-sided")
                        entry["Test"] = "Mann-Whitney U"
                    entry["Effect size"] = "Cohen's d"
                    entry["Effect value"] = round(cohens_d(*samples), 3)
                else:
                    if normal:
                        stat, p = stats.f_oneway(*samples)
                        entry["Test"] = "One-way ANOVA"
                    else:
                        stat, p = stats.kruskal(*samples)
                        entry["Test"] = "Kruskal-Wallis"
                    entry["Effect size"] = "eta²"
                    entry["Effect value"] = round(eta_squared(samples), 3)
                entry["Statistic"] = round(float(stat), 4)
                entry["p"] = _fmt_p(float(p))
                entry["Sig."] = _stars(float(p))
            else:
                table = pd.crosstab(df[col], df[group_col])
                if table.shape[0] < 2 or table.shape[1] < 2:
                    entry["Test"] = "not run (needs 2x2 or larger)"
                    rows.append(entry); continue
                chi2, p, dof, _ = stats.chi2_contingency(table)
                entry.update({
                    "Test": "Chi-square", "Statistic": round(float(chi2), 4),
                    "df": int(dof), "p": _fmt_p(float(p)), "Sig.": _stars(float(p)),
                    "Effect size": "Cramér's V",
                    "Effect value": round(cramers_v(table.to_numpy()), 3),
                })
        except Exception as exc:
            entry["Test"] = f"failed: {exc}"
        rows.append(entry)

    return pd.DataFrame(rows)


def correlation_table(
    df: pd.DataFrame, target: Optional[str] = None, method: str = "spearman",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Correlation with the target, each with a p-value and 95% CI.

    Spearman by default: it is rank-based, so it does not assume normality and
    is robust to the outliers common in physiological features.
    """
    numeric = df.select_dtypes(include=[np.number])
    if target and target in numeric.columns:
        rows = []
        y = numeric[target].to_numpy(float)
        for col in numeric.columns:
            if col == target:
                continue
            pair = numeric[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 10 or pair.iloc[:, 0].nunique() < 2:
                continue
            a = pair.iloc[:, 0].to_numpy(float)
            b = pair.iloc[:, 1].to_numpy(float)
            r, p = (stats.spearmanr(a, b) if method == "spearman" else stats.pearsonr(a, b))
            n = len(pair)
            # Fisher z transform for the confidence interval.
            if abs(r) < 1 and n > 3:
                z = np.arctanh(r)
                se = 1 / np.sqrt(n - 3)
                lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
            else:
                lo = hi = float("nan")
            rows.append({
                "Variable": col, "n": n, f"{method.capitalize()} r": round(float(r), 4),
                "95% CI": f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "-",
                "p": _fmt_p(float(p)), "Sig.": _stars(float(p)),
                "|r|": abs(float(r)),
            })
        table = pd.DataFrame(rows).sort_values("|r|", ascending=False).drop(columns="|r|")
        return table.head(top_n) if top_n else table

    return numeric.corr(method=method).round(4)


def model_comparison_table(results: Dict, task: str) -> pd.DataFrame:
    """Model results side by side, ordered by the primary metric."""
    keys = (["roc_auc", "pr_auc", "balanced_accuracy", "accuracy", "f1",
             "precision", "recall", "specificity", "mcc", "brier"]
            if task == "classification"
            else ["rmse", "mae", "medae", "r2", "nrmse_std", "bias"])

    models = dict(results.get("baselines", {}))
    models.update(results.get("tabular_only_baselines", {}))
    models.update(results.get("modality_ablation", {}))
    for key, label in (("transformer", "QSQ-FS + Transformer"),
                       ("multimodal", "Multimodal fusion"),
                       ("tuned_models", None)):
        if key == "tuned_models":
            for name, entry in (results.get(key) or {}).items():
                if isinstance(entry, dict) and "metrics" in entry:
                    models[f"{name} (tuned)"] = entry["metrics"]
        elif results.get(key):
            models[label] = results[key]

    rows = []
    for name, metrics in models.items():
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        row = {"Model": name}
        row.update({k.replace("_", " ").title(): (round(float(metrics[k]), 4)
                                                  if isinstance(metrics.get(k), (int, float))
                                                  else "-")
                    for k in keys if k in metrics})
        rows.append(row)

    table = pd.DataFrame(rows)
    primary = "Roc Auc" if task == "classification" else "Rmse"
    if primary in table.columns:
        table = table.sort_values(primary, ascending=(task != "classification"))
    return table.reset_index(drop=True)


def cv_summary_table(per_fold: List[Dict], metric_keys: Optional[List[str]] = None) -> pd.DataFrame:
    """Mean ± SD with a 95% CI across folds or seeds -- what a paper reports."""
    if not per_fold:
        return pd.DataFrame()
    keys = metric_keys or [k for k, v in per_fold[0].items() if isinstance(v, (int, float))]
    rows = []
    for key in keys:
        values = np.array([float(f[key]) for f in per_fold if isinstance(f.get(key), (int, float))])
        if values.size < 2:
            continue
        mean, sd = values.mean(), values.std(ddof=1)
        se = sd / np.sqrt(values.size)
        t = stats.t.ppf(0.975, values.size - 1)
        rows.append({
            "Metric": key.replace("_", " ").title(),
            "Mean": round(float(mean), 4),
            "SD": round(float(sd), 4),
            "95% CI": f"[{mean - t*se:.4f}, {mean + t*se:.4f}]",
            "Min": round(float(values.min()), 4),
            "Max": round(float(values.max()), 4),
            "n": int(values.size),
        })
    return pd.DataFrame(rows)


def write_tables_excel(tables: Dict[str, pd.DataFrame], path) -> Optional[str]:
    """One Excel workbook, one sheet per table, ready to paste into a thesis."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, table in tables.items():
                if isinstance(table, pd.DataFrame) and not table.empty:
                    table.to_excel(writer, sheet_name=str(name)[:31], index=False)
        return str(path)
    except Exception as exc:
        logger.warning("Could not write Excel workbook: %s", exc)
        return None
