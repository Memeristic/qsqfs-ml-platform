"""Label-permutation sanity check.

Shuffle the target, rerun the *entire* pipeline, and confirm performance
collapses to chance. If a permuted run still scores well, something in the
pipeline is leaking and no headline number from it can be trusted.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np

logger = logging.getLogger(__name__)


def permutation_test(
    pipeline_fn: Callable[[np.ndarray], float],
    y: np.ndarray,
    observed_score: float,
    n_permutations: int = 20,
    seed: int = 42,
    higher_is_better: bool = True,
) -> Dict:
    """``pipeline_fn`` must take a permuted y and return a test-set score."""
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(int(n_permutations)):
        try:
            scores.append(float(pipeline_fn(rng.permutation(np.asarray(y)))))
        except Exception as exc:
            logger.warning("Permutation %d failed: %s", i, exc)

    if not scores:
        return {"error": "all permutations failed", "n_permutations": 0}

    scores = np.array(scores)
    extreme = (
        int((scores >= observed_score).sum()) if higher_is_better
        else int((scores <= observed_score).sum())
    )
    p_value = (extreme + 1) / (len(scores) + 1)
    return {
        "observed_score": float(observed_score),
        "permuted_mean": float(scores.mean()),
        "permuted_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "permuted_min": float(scores.min()),
        "permuted_max": float(scores.max()),
        "p_value": float(p_value),
        "n_permutations": int(len(scores)),
        "interpretation": (
            "Permuted runs score near chance, as expected."
            if p_value < 0.05 else
            "Permuted runs score close to the real run. Investigate for leakage "
            "before reporting any result from this pipeline."
        ),
    }
