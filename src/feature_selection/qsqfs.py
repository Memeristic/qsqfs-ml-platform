"""QSQ-FS: Quorum Sensing / Quorum Quenching feature selection.

A population of binary feature masks ("colonies") is evolved under two
bacterial-signalling analogies:

**Quorum sensing.** High-fitness colonies emit an autoinducer field over the
feature axis. Field strength at feature *j* reflects how strongly the strong
colonies agree that *j* belongs in the mask. The field is smoothed across
generations with an EMA (``beta``) so it represents accumulated consensus
rather than one lucky generation, and it biases mutation toward the incumbent
best mask.

**Quorum quenching.** Masks that scored below the strong-colony threshold are
written to a suppression archive with a penalty. If the search revisits them
their *effective* fitness is reduced, discouraging cycling. Penalties decay
geometrically (``delta``) so a region is not banned forever.

Fitness is ``alpha * score + (1 - alpha) * parsimony``, where parsimony is the
fraction of features left out. ``score`` is bounded in [0, 1]:

  classification - cross-validated balanced accuracy (or another sklearn metric)
  regression     - ``1 - RMSE / std(y)``, clipped at 0. Zero means the mask does
                   no better than predicting the training mean; it is not R².

Design notes that matter for correctness:

  * The inner CV used for fitness accepts ``groups``. With repeated measures
    (e.g. many windows per subject) an ungrouped shuffled KFold puts near
    duplicate rows on both sides of a fold and inflates every mask's score.
  * ``self.best_fitness`` is assigned on every improvement. Earlier versions of
    this class only tracked a local variable, so callers read ``-inf``.
  * Colonies replaced during diversity injection are re-evaluated before they
    are allowed to influence the next generation's strong/weak classification.
  * The suppression archive keys on raw fitness, so a penalty never feeds back
    into the value that produced it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

logger = logging.getLogger(__name__)

CLASSIFICATION = "classification"
REGRESSION = "regression"


def default_estimator(task: str, seed: int = 42):
    """Cheap, low-variance wrapper model used to score candidate masks."""
    if task == CLASSIFICATION:
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    return KNeighborsRegressor(n_neighbors=5, weights="distance")


def forest_estimator(task: str, seed: int = 42):
    if task == CLASSIFICATION:
        return RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=-1)
    return RandomForestRegressor(n_estimators=60, random_state=seed, n_jobs=-1)


ESTIMATORS: Dict[str, Callable] = {"knn": default_estimator, "forest": forest_estimator}


@dataclass
class QSQFSHistory:
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    n_selected: List[int] = field(default_factory=list)
    archive_size: List[int] = field(default_factory=list)
    field_entropy: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "best_fitness": [float(v) for v in self.best_fitness],
            "mean_fitness": [float(v) for v in self.mean_fitness],
            "n_selected": [int(v) for v in self.n_selected],
            "archive_size": [int(v) for v in self.archive_size],
            "field_entropy": [float(v) for v in self.field_entropy],
        }


class QSQFS:
    """Quorum Sensing / Quorum Quenching feature selection."""

    def __init__(
        self,
        n_features: int,
        population_size: int = 30,
        n_iterations: int = 40,
        task: str = CLASSIFICATION,
        alpha: float = 0.85,
        w_ai: float = 0.5,
        beta: float = 0.7,
        delta: float = 0.97,
        strong_percentile: float = 85.0,
        n_min: Optional[int] = None,
        n_max: Optional[int] = None,
        min_fraction: float = 0.05,
        max_fraction: float = 0.5,
        use_quorum_sensing: bool = True,
        use_quorum_quenching: bool = True,
        use_elitism: bool = True,
        use_cache: bool = True,
        estimator: str = "knn",
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        stagnation_window: int = 12,
        diversity_injection_rate: float = 0.25,
        early_stop_patience: Optional[int] = None,
        seed: int = 42,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        if task not in (CLASSIFICATION, REGRESSION):
            raise ValueError(f"task must be '{CLASSIFICATION}' or '{REGRESSION}', got '{task}'.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if n_features < 2:
            raise ValueError("QSQ-FS needs at least 2 features.")

        self.n_features = int(n_features)
        self.pop_size = max(4, int(population_size))
        self.n_iterations = max(1, int(n_iterations))
        self.task = task
        self.alpha = float(alpha)
        self.w_ai = float(w_ai)
        self.beta = float(beta)
        self.delta = float(delta)
        self.strong_percentile = float(strong_percentile)
        self.use_quorum_sensing = bool(use_quorum_sensing)
        self.use_quorum_quenching = bool(use_quorum_quenching)
        self.use_elitism = bool(use_elitism)
        self.use_cache = bool(use_cache)
        self.estimator_name = estimator
        self.cv_folds = max(2, int(cv_folds))
        self.scoring = scoring
        self.stagnation_window = max(1, int(stagnation_window))
        self.diversity_injection_rate = float(diversity_injection_rate)
        self.early_stop_patience = early_stop_patience
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.verbose = verbose

        self.n_min = int(n_min) if n_min else max(1, int(min_fraction * n_features))
        self.n_max = int(n_max) if n_max else max(self.n_min + 1, int(max_fraction * n_features))
        self.n_max = min(self.n_max, self.n_features)
        self.n_min = min(self.n_min, self.n_max)

        # State
        self.rng = np.random.default_rng(self.seed)
        self.best_mask: Optional[np.ndarray] = None
        self.best_fitness: float = float("-inf")
        self.best_score: float = float("-inf")
        self.best_n_selected: int = 0
        self.history = QSQFSHistory()
        self.archive: Dict[bytes, float] = {}
        self._cache: Dict[bytes, tuple] = {}
        self.generation = 0

        # Counters
        self.cache_hits = 0
        self.cache_misses = 0
        self.archive_hits = 0
        self.diversity_injections = 0
        self.elite_replacements = 0
        self.total_evaluations = 0
        self.runtime_seconds = 0.0
        self.stopped_early_at: Optional[int] = None

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------
    def _repair(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool, copy=True)
        count = int(mask.sum())
        if count < self.n_min:
            zeros = np.flatnonzero(~mask)
            if zeros.size:
                pick = self.rng.choice(zeros, size=min(self.n_min - count, zeros.size), replace=False)
                mask[pick] = True
        elif count > self.n_max:
            ones = np.flatnonzero(mask)
            pick = self.rng.choice(ones, size=count - self.n_max, replace=False)
            mask[pick] = False
        if not mask.any():
            mask[self.rng.integers(0, self.n_features)] = True
        return mask

    def _random_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_features, dtype=bool)
        k = int(self.rng.integers(self.n_min, self.n_max + 1))
        mask[self.rng.choice(self.n_features, size=k, replace=False)] = True
        return mask

    def _init_population(self) -> np.ndarray:
        return np.array([self._random_mask() for _ in range(self.pop_size)])

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------
    def _make_cv(self, y: np.ndarray, groups: Optional[np.ndarray]):
        n = len(y)
        if groups is not None:
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                return GroupKFold(n_splits=max(2, min(self.cv_folds, n_groups)))
        if self.task == CLASSIFICATION:
            _, counts = np.unique(y, return_counts=True)
            k = max(2, min(self.cv_folds, int(counts.min())))
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)
        return KFold(n_splits=max(2, min(self.cv_folds, n)), shuffle=True, random_state=self.seed)

    def _raw_score(self, mask, X, y, groups, cv, y_std) -> float:
        """Cross-validated predictive score for a mask, bounded in [0, 1]."""
        X_sel = X[:, mask]
        if X_sel.shape[1] == 0:
            return 0.0
        estimator = clone(self._estimator)
        scoring = self.scoring or (
            "balanced_accuracy" if self.task == CLASSIFICATION else "neg_root_mean_squared_error"
        )
        try:
            scores = cross_val_score(
                estimator, X_sel, y, cv=cv, groups=groups,
                scoring=scoring, n_jobs=self.n_jobs, error_score="raise",
            )
        except Exception as exc:  # a degenerate mask should not kill the run
            logger.debug("Mask evaluation failed (%s); scoring it 0.", exc)
            return 0.0
        value = float(np.mean(scores))
        if self.task == REGRESSION:
            rmse = -value
            value = 1.0 - (rmse / (y_std + 1e-12))
        return float(np.clip(value, 0.0, 1.0))

    def _evaluate(self, mask, X, y, groups, cv, y_std) -> tuple:
        """Return (fitness, raw_score). Cached on the mask bytes."""
        key = np.packbits(mask).tobytes()
        if self.use_cache and key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        self.total_evaluations += 1

        score = self._raw_score(mask, X, y, groups, cv, y_std)
        parsimony = 1.0 - (int(mask.sum()) / self.n_features)
        fitness = self.alpha * score + (1.0 - self.alpha) * parsimony
        result = (float(fitness), float(score))
        if self.use_cache:
            self._cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Quorum quenching
    # ------------------------------------------------------------------
    def _suppressed(self, mask: np.ndarray, raw_fitness: float) -> float:
        if not self.use_quorum_quenching:
            return raw_fitness
        key = np.packbits(mask).tobytes()
        penalty = self.archive.get(key)
        if penalty is None:
            return raw_fitness
        self.archive_hits += 1
        return max(0.0, raw_fitness - penalty)

    def _archive_weak(self, mask: np.ndarray, raw_fitness: float, threshold: float) -> None:
        if not self.use_quorum_quenching or raw_fitness >= threshold:
            return
        key = np.packbits(mask).tobytes()
        penalty = threshold - raw_fitness
        self.archive[key] = max(self.archive.get(key, 0.0), penalty)

    def _decay_archive(self) -> None:
        if not self.use_quorum_quenching or not self.archive:
            return
        decayed = {k: v * self.delta for k, v in self.archive.items()}
        self.archive = {k: v for k, v in decayed.items() if v > 1e-6}

    # ------------------------------------------------------------------
    # Quorum sensing
    # ------------------------------------------------------------------
    def _autoinducer_field(self, strong_masks: np.ndarray, strong_fitness: np.ndarray) -> np.ndarray:
        if len(strong_masks) == 0:
            return np.zeros(self.n_features)
        f_min, f_max = float(strong_fitness.min()), float(strong_fitness.max())
        if f_max - f_min < 1e-9:
            weights = np.ones(len(strong_fitness))
        else:
            weights = (strong_fitness - f_min) / (f_max - f_min)
        noise = self.rng.random((len(strong_masks), self.n_features))
        contributions = (
            weights[:, None] * self.w_ai + (1.0 - self.w_ai) * noise
        ) * strong_masks.astype(float)
        return contributions.max(axis=0)

    @staticmethod
    def _smooth(new_field: np.ndarray, prev_field: Optional[np.ndarray], beta: float) -> np.ndarray:
        if prev_field is None:
            return new_field
        return beta * new_field + (1.0 - beta) * prev_field

    @staticmethod
    def _field_entropy(field: np.ndarray) -> float:
        """Normalised entropy of the field. 1.0 = no consensus, 0.0 = locked in."""
        p = np.clip(field, 1e-12, 1.0)
        p = p / p.sum()
        h = -np.sum(p * np.log(p))
        return float(h / np.log(len(p))) if len(p) > 1 else 0.0

    def _mutate(self, mask, fitness, field, best_mask) -> np.ndarray:
        """Vectorised three-way mutation: inherit / retain / explore."""
        r1 = self.rng.random(self.n_features)
        r2 = self.rng.random(self.n_features)
        r3 = self.rng.random(self.n_features)

        inherit = r1 < field if self.use_quorum_sensing else np.zeros(self.n_features, bool)
        retain = (~inherit) & (r2 < fitness * self.w_ai)
        explore = ~(inherit | retain)

        child = np.empty(self.n_features, dtype=bool)
        child[inherit] = best_mask[inherit]
        child[retain] = mask[retain]
        child[explore] = r3[explore] < 0.5
        return self._repair(child)

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Sequence] = None,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Run the search on TRAINING data only. Returns the boolean mask."""
        start = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        groups_arr = np.asarray(groups) if groups is not None else None

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} columns but QSQFS was built for {self.n_features}."
            )
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}.")

        self._estimator = ESTIMATORS.get(self.estimator_name, default_estimator)(self.task, self.seed)
        cv = self._make_cv(y, groups_arr)
        y_std = float(np.std(y.astype(float))) if self.task == REGRESSION else 1.0
        if self.task == REGRESSION and y_std < 1e-12:
            raise ValueError("Target has zero variance; regression is not defined.")

        if self.verbose:
            logger.info(
                "QSQ-FS start | %d features, %d rows, task=%s, pop=%d, iters=%d, "
                "bounds=[%d, %d], CV=%s%s",
                self.n_features, len(X), self.task, self.pop_size, self.n_iterations,
                self.n_min, self.n_max, type(cv).__name__,
                " (grouped)" if groups_arr is not None else "",
            )

        population = self._init_population()
        evaluated = [self._evaluate(m, X, y, groups_arr, cv, y_std) for m in population]
        fitness = np.array([f for f, _ in evaluated])
        raw_scores = np.array([s for _, s in evaluated])

        best_idx = int(np.argmax(fitness))
        self.best_mask = population[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])
        self.best_score = float(raw_scores[best_idx])
        self.best_n_selected = int(self.best_mask.sum())

        field: Optional[np.ndarray] = None
        stagnation = 0
        since_improvement = 0

        for generation in range(self.n_iterations):
            self.generation = generation
            mean_fitness = float(np.mean(fitness))
            threshold = float(np.percentile(fitness, self.strong_percentile))

            strong = fitness >= threshold
            for mask, raw in zip(population[~strong], fitness[~strong]):
                self._archive_weak(mask, float(raw), threshold)
            self._decay_archive()

            if self.use_quorum_sensing and strong.any():
                field = self._smooth(
                    self._autoinducer_field(population[strong], fitness[strong]), field, self.beta
                )
            else:
                field = np.zeros(self.n_features)

            # --- offspring -------------------------------------------------
            children: List[np.ndarray] = []
            if self.use_elitism:
                children.append(self.best_mask.copy())
            while len(children) < self.pop_size:
                idx = len(children) if not self.use_elitism else len(children) - 1
                idx = min(idx, self.pop_size - 1)
                parent, parent_fitness = population[idx], float(fitness[idx])
                if parent_fitness < mean_fitness:
                    child = self._mutate(parent, parent_fitness, field, self.best_mask)
                else:
                    child = parent.copy()
                    if self.rng.random() < 0.05:
                        child[self.rng.integers(0, self.n_features)] ^= True
                    child = self._repair(child)
                children.append(child)

            population = np.array(children[: self.pop_size])
            evaluated = [self._evaluate(m, X, y, groups_arr, cv, y_std) for m in population]
            raw_fitness = np.array([f for f, _ in evaluated])
            raw_scores = np.array([s for _, s in evaluated])
            fitness = np.array([
                self._suppressed(m, float(f)) for m, f in zip(population, raw_fitness)
            ])

            if self.use_elitism and not any(np.array_equal(m, self.best_mask) for m in population):
                worst = int(np.argmin(fitness))
                population[worst] = self.best_mask.copy()
                fitness[worst] = self.best_fitness
                raw_scores[worst] = self.best_score
                self.elite_replacements += 1

            # --- diversity injection --------------------------------------
            stagnation += 1
            if stagnation >= self.stagnation_window:
                n_replace = max(1, int(self.pop_size * self.diversity_injection_rate))
                order = np.argsort(fitness)
                replaced = 0
                for idx in order:
                    if replaced >= n_replace:
                        break
                    if self.use_elitism and np.array_equal(population[idx], self.best_mask):
                        continue
                    population[idx] = self._random_mask()
                    # Re-evaluate immediately: a stale fitness attached to a new
                    # genome would corrupt the next strong/weak classification.
                    f_new, s_new = self._evaluate(population[idx], X, y, groups_arr, cv, y_std)
                    fitness[idx] = self._suppressed(population[idx], f_new)
                    raw_scores[idx] = s_new
                    replaced += 1
                self.diversity_injections += 1
                stagnation = 0

            # --- track best -----------------------------------------------
            current = int(np.argmax(fitness))
            if fitness[current] > self.best_fitness:
                self.best_fitness = float(fitness[current])
                self.best_score = float(raw_scores[current])
                self.best_mask = population[current].copy()
                self.best_n_selected = int(self.best_mask.sum())
                stagnation = 0
                since_improvement = 0
            else:
                since_improvement += 1

            self.history.best_fitness.append(self.best_fitness)
            self.history.mean_fitness.append(float(np.mean(fitness)))
            self.history.n_selected.append(self.best_n_selected)
            self.history.archive_size.append(len(self.archive))
            self.history.field_entropy.append(self._field_entropy(field))

            if self.verbose and (generation % 5 == 0 or generation == self.n_iterations - 1):
                logger.info(
                    "  gen %3d | best=%.4f score=%.4f | k=%d | archive=%d | cache=%d/%d",
                    generation, self.best_fitness, self.best_score, self.best_n_selected,
                    len(self.archive), self.cache_hits, self.cache_hits + self.cache_misses,
                )

            if self.early_stop_patience and since_improvement >= self.early_stop_patience:
                self.stopped_early_at = generation
                logger.info("Early stop at generation %d (no improvement).", generation)
                break

        self.runtime_seconds = time.time() - start
        self.feature_names_ = list(feature_names) if feature_names else None
        if self.verbose:
            logger.info(
                "QSQ-FS done | fitness=%.4f score=%.4f | %d/%d features | %.1fs | %d evals",
                self.best_fitness, self.best_score, self.best_n_selected,
                self.n_features, self.runtime_seconds, self.total_evaluations,
            )
        return self.best_mask

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.best_mask is None:
            raise RuntimeError("Call fit() before transform().")
        return np.asarray(X)[:, self.best_mask]

    def selected_indices(self) -> np.ndarray:
        if self.best_mask is None:
            raise RuntimeError("Call fit() first.")
        return np.flatnonzero(self.best_mask)

    def selected_names(self, feature_names: Optional[Sequence[str]] = None) -> List[str]:
        names = feature_names or getattr(self, "feature_names_", None)
        idx = self.selected_indices()
        if names is None:
            return [f"feature_{i}" for i in idx]
        return [str(names[i]) for i in idx]

    def stats(self) -> Dict:
        total_lookups = self.cache_hits + self.cache_misses
        return {
            "best_fitness": float(self.best_fitness),
            "best_score": float(self.best_score),
            "score_definition": (
                "cross-validated balanced accuracy" if self.task == CLASSIFICATION
                else "1 - RMSE/std(y), clipped at 0"
            ),
            "n_selected": int(self.best_n_selected),
            "n_features_total": int(self.n_features),
            "selection_ratio": round(self.best_n_selected / self.n_features, 4),
            "n_iterations_run": len(self.history.best_fitness),
            "stopped_early_at": self.stopped_early_at,
            "runtime_seconds": round(float(self.runtime_seconds), 3),
            "total_evaluations": int(self.total_evaluations),
            "cache_hits": int(self.cache_hits),
            "cache_hit_rate": round(self.cache_hits / total_lookups, 4) if total_lookups else 0.0,
            "archive_hits": int(self.archive_hits),
            "final_archive_size": len(self.archive),
            "diversity_injections": int(self.diversity_injections),
            "elite_replacements": int(self.elite_replacements),
            "mechanisms": {
                "quorum_sensing": self.use_quorum_sensing,
                "quorum_quenching": self.use_quorum_quenching,
                "elitism": self.use_elitism,
                "cache": self.use_cache,
            },
            "hyperparameters": {
                "alpha": self.alpha, "w_ai": self.w_ai, "beta": self.beta,
                "delta": self.delta, "population_size": self.pop_size,
                "n_min": self.n_min, "n_max": self.n_max, "seed": self.seed,
            },
        }
