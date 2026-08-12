"""Paired comparisons between models."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("Paired test needs equal-length inputs.")
    if len(a) < 5:
        return {
            "test": "wilcoxon", "n_pairs": len(a),
            "warning": f"Only {len(a)} pairs; a p-value here is not meaningful.",
            "p_value": None,
        }
    if np.allclose(a, b):
        return {"test": "wilcoxon", "n_pairs": len(a), "p_value": 1.0,
                "note": "inputs are identical"}
    stat, p = stats.wilcoxon(a, b)
    diff = a - b
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(stat),
        "p_value": float(p),
        "n_pairs": int(len(a)),
        "median_difference": float(np.median(diff)),
        "mean_difference": float(diff.mean()),
    }


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return {"test": "paired_t", "n_pairs": len(a), "p_value": None,
                "warning": "too few pairs"}
    stat, p = stats.ttest_rel(a, b)
    diff = a - b
    sd = diff.std(ddof=1)
    return {
        "test": "paired_t",
        "statistic": float(stat),
        "p_value": float(p),
        "n_pairs": int(len(a)),
        "mean_difference": float(diff.mean()),
        "cohens_dz": float(diff.mean() / sd) if sd > 1e-12 else None,
    }


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict:
    """Step-down correction. Use it whenever you compare against >1 baseline."""
    valid = {k: v for k, v in p_values.items() if v is not None}
    if not valid:
        return {}
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, previous = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        adjusted = min(1.0, max(previous, (m - rank) * p))
        previous = adjusted
        out[name] = {
            "p_raw": float(p),
            "p_adjusted": float(adjusted),
            "significant_at_alpha": bool(adjusted < alpha),
        }
    return out
