"""Bootstrap confidence intervals over the test set."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def bootstrap_ci(
    y_true, y_pred, metric_fn: Callable, n_iterations: int = 1000,
    confidence: float = 0.95, seed: int = 42, y_proba=None,
) -> Dict:
    """Percentile CI by resampling test rows with replacement."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba) if y_proba is not None else None
    rng = np.random.default_rng(seed)
    n = len(y_true)

    values = []
    for _ in range(int(n_iterations)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2 and y_proba is not None:
            continue  # AUC is undefined on a single-class resample
        try:
            values.append(
                float(metric_fn(y_true[idx], y_pred[idx], y_proba[idx]))
                if y_proba is not None else float(metric_fn(y_true[idx], y_pred[idx]))
            )
        except Exception:
            continue

    if not values:
        return {"error": "no valid bootstrap resamples", "n_valid": 0}

    values = np.array(values)
    lo = (1.0 - confidence) / 2 * 100
    return {
        "point_estimate": float(
            metric_fn(y_true, y_pred, y_proba) if y_proba is not None
            else metric_fn(y_true, y_pred)
        ),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci_lower": float(np.percentile(values, lo)),
        "ci_upper": float(np.percentile(values, 100 - lo)),
        "confidence": float(confidence),
        "n_valid": int(len(values)),
        "n_requested": int(n_iterations),
    }
