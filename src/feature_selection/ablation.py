"""Ablation: which QSQ-FS mechanism actually earns its place?

Each variant is run over several seeds because a single-seed difference between
two stochastic searches is noise. The reported spread is the honest signal.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .qsqfs import QSQFS

logger = logging.getLogger(__name__)

VARIANTS: Dict[str, Dict] = {
    "full": {},
    "no_quorum_sensing": {"use_quorum_sensing": False},
    "no_quorum_quenching": {"use_quorum_quenching": False},
    "no_elitism": {"use_elitism": False},
    "no_sensing_no_quenching": {"use_quorum_sensing": False, "use_quorum_quenching": False},
    "random_search": {"use_quorum_sensing": False, "use_quorum_quenching": False,
                      "use_elitism": False},
}


def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
    qsqfs_config: Optional[Dict] = None,
    groups: Optional[np.ndarray] = None,
    seeds: Optional[List[int]] = None,
    variants: Optional[List[str]] = None,
) -> Dict:
    qsqfs_config = dict(qsqfs_config or {})
    seeds = seeds or [42, 52, 62]
    names = variants or list(VARIANTS)
    base = {k: v for k, v in qsqfs_config.items()
            if k in QSQFS.__init__.__code__.co_varnames}
    base.pop("seed", None)

    output: Dict = {
        "task": task, "seeds": list(seeds), "n_features": int(X.shape[1]),
        "n_samples": int(len(X)), "grouped_cv": groups is not None, "variants": {},
    }

    for name in names:
        if name not in VARIANTS:
            logger.warning("Unknown ablation variant '%s'; skipped.", name)
            continue
        runs = []
        for seed in seeds:
            selector = QSQFS(
                n_features=X.shape[1], task=task, seed=seed, verbose=False,
                **{**base, **VARIANTS[name]},
            )
            selector.fit(X, y, groups=groups)
            runs.append({
                "seed": seed,
                "fitness": selector.best_fitness,
                "score": selector.best_score,
                "n_selected": selector.best_n_selected,
                "runtime": selector.runtime_seconds,
                "evaluations": selector.total_evaluations,
                "mask": selector.best_mask.tolist(),
            })
            logger.info("  %s seed=%d fitness=%.4f k=%d",
                        name, seed, selector.best_fitness, selector.best_n_selected)

        def agg(key: str) -> tuple:
            values = np.array([r[key] for r in runs], dtype=float)
            return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

        fitness_mean, fitness_std = agg("fitness")
        score_mean, score_std = agg("score")
        k_mean, _ = agg("n_selected")
        rt_mean, _ = agg("runtime")

        from src.explainability.importance import selection_stability
        stability = selection_stability([np.array(r["mask"], dtype=bool) for r in runs])

        output["variants"][name] = {
            "fitness_mean": fitness_mean, "fitness_std": fitness_std,
            "score_mean": score_mean, "score_std": score_std,
            "n_selected_mean": k_mean, "runtime_mean": rt_mean,
            "mean_jaccard_across_seeds": stability.get("mean_jaccard"),
            "runs": [{k: v for k, v in r.items() if k != "mask"} for r in runs],
        }

    if "full" in output["variants"]:
        full = output["variants"]["full"]["fitness_mean"]
        for name, entry in output["variants"].items():
            entry["delta_vs_full"] = round(entry["fitness_mean"] - full, 6)
        output["note"] = (
            "delta_vs_full is the change in mean fitness when a mechanism is removed. "
            "Compare it against fitness_std before calling any difference real."
        )
    return output
