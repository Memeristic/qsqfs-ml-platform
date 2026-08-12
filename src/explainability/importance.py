"""Permutation feature importance, computed on held-out data."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 20,
    seed: int = 42,
    higher_is_better: bool = True,
) -> Dict:
    """Drop in score when one column is shuffled.

    Works with any callable, so it covers the torch model as well as sklearn
    estimators. Must be run on test data: importances from training data
    describe what the model memorised, not what it uses to generalise.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(X.shape[1])]

    baseline = float(score_fn(y, predict_fn(X)))
    results = []

    for j in range(X.shape[1]):
        drops = []
        for _ in range(int(n_repeats)):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            try:
                permuted = float(score_fn(y, predict_fn(X_perm)))
            except Exception as exc:
                logger.debug("Permutation of column %d failed: %s", j, exc)
                continue
            drops.append(baseline - permuted if higher_is_better else permuted - baseline)
        if not drops:
            continue
        drops = np.array(drops)
        results.append({
            "feature": names[j],
            "index": int(j),
            "importance_mean": float(drops.mean()),
            "importance_std": float(drops.std(ddof=1)) if len(drops) > 1 else 0.0,
            "n_repeats": int(len(drops)),
        })

    results.sort(key=lambda r: r["importance_mean"], reverse=True)
    return {
        "baseline_score": baseline,
        "n_features": X.shape[1],
        "importances": results,
        "note": (
            "Positive importance = shuffling this column hurt the score. "
            "Values near zero or negative mean the model does not rely on it. "
            "Correlated features share credit and can both look unimportant."
        ),
    }


def selection_stability(masks: List[np.ndarray], feature_names: Optional[List[str]] = None) -> Dict:
    """How consistently features are chosen across seeds or folds.

    Low Jaccard with good accuracy usually means several feature subsets are
    equally predictive -- worth saying out loud rather than presenting one run's
    subset as *the* biomarker panel.
    """
    if len(masks) < 2:
        return {"error": "need at least 2 masks", "n_masks": len(masks)}

    stack = np.asarray(masks, dtype=bool)
    n_features = stack.shape[1]
    names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(n_features)]
    frequency = stack.mean(axis=0)

    jaccards = []
    for i in range(len(stack)):
        for j in range(i + 1, len(stack)):
            union = np.logical_or(stack[i], stack[j]).sum()
            if union:
                jaccards.append(np.logical_and(stack[i], stack[j]).sum() / union)

    ranked = sorted(
        ({"feature": names[i], "selection_frequency": float(frequency[i])}
         for i in range(n_features)),
        key=lambda r: r["selection_frequency"], reverse=True,
    )
    return {
        "n_masks": int(len(stack)),
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "std_jaccard": float(np.std(jaccards, ddof=1)) if len(jaccards) > 1 else 0.0,
        "mean_n_selected": float(stack.sum(axis=1).mean()),
        "always_selected": [names[i] for i in range(n_features) if frequency[i] == 1.0],
        "never_selected": int((frequency == 0).sum()),
        "feature_frequencies": ranked,
    }
