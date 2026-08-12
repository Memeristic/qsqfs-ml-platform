"""Legitimate ways to improve results, and one rule about how.

**The rule: tune on the training data only, never on the test set.**

Every knob in this module is optimised by cross-validation *inside the training
split*. The test set is scored once, at the end, with whatever the search chose.
Repeatedly adjusting settings, re-scoring the test set, and keeping the best
number is how published results become irreproducible -- the test set stops
being held out and becomes part of the fit. This module makes the honest path
the easy one, and it cannot inflate a score because it never sees the test data.

What genuinely helps, roughly in order of effect:

  1. Longer QSQ-FS search        more generations and a larger population
                                 explore more subsets. Costs time, not honesty.
  2. A stronger wrapper model    scoring subsets with a random forest instead of
                                 3-NN finds subsets that suit tree models.
  3. Hyperparameter search       the defaults are reasonable, not optimal for
                                 your data.
  4. Threshold tuning            for imbalanced classification, 0.5 is rarely
                                 the best cut point. Optimise it on validation
                                 folds, then apply it unchanged to the test set.
  5. Calibration                 makes predicted probabilities mean what they
                                 say. Improves Brier score and clinical utility;
                                 leaves AUC almost unchanged, by design.
  6. Ensembling                  averaging several models usually beats any one.

What does NOT count as improvement, and this platform will not do:

  * choosing the seed that gave the best test score
  * dropping "outlier" test rows that were predicted badly
  * tuning anything against the test set
  * reporting the best of N runs without saying it was the best of N
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR

logger = logging.getLogger(__name__)

CLASSIFICATION = "classification"

SEARCH_SPACES: Dict[str, Dict] = {
    "random_forest": {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 6, 12, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.6],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.01, 0.03, 0.06, 0.1, 0.2],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "max_iter": [200, 400],
    },
    "svm": {
        "C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 0.5, 1.0, 5.0, 20.0],
        "penalty": ["l2"],
    },
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
}


def _base_model(name: str, task: str, seed: int):
    if task == CLASSIFICATION:
        return {
            "random_forest": RandomForestClassifier(
                random_state=seed, n_jobs=-1, class_weight="balanced_subsample"),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
            "svm": SVC(kernel="rbf", probability=True, class_weight="balanced",
                       random_state=seed),
            "logistic_regression": LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed),
        }.get(name)
    return {
        "random_forest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
        "svm": SVR(kernel="rbf"),
        "ridge": Ridge(random_state=seed),
    }.get(name)


def tune_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = CLASSIFICATION,
    cv=None,
    groups: Optional[np.ndarray] = None,
    n_iter: int = 25,
    scoring: Optional[str] = None,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict:
    """Randomised hyperparameter search inside the training split only."""
    model = _base_model(name, task, seed)
    space = SEARCH_SPACES.get(name)
    if model is None or not space:
        return {"error": f"no search space for '{name}' on task '{task}'"}

    scoring = scoring or ("roc_auc" if task == CLASSIFICATION else "neg_root_mean_squared_error")
    n_candidates = int(np.prod([len(v) for v in space.values()]))
    search = RandomizedSearchCV(
        model, space, n_iter=min(n_iter, n_candidates), scoring=scoring,
        cv=cv or 5, random_state=seed, n_jobs=n_jobs, refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups) if groups is not None \
        else search.fit(X_train, y_train)

    logger.info("Tuned %s: best CV %s = %.4f", name, scoring, search.best_score_)
    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "scoring": scoring,
        "n_candidates_tried": int(len(search.cv_results_["params"])),
        "note": "Selected by cross-validation on training rows only; "
                "the test set was not consulted.",
    }


def tune_decision_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1"
) -> Dict:
    """Find the probability cut-point that maximises a metric.

    Call this on VALIDATION predictions, then apply the returned threshold
    unchanged to the test set. Choosing the threshold on test predictions is
    fitting to the test set and inflates every downstream metric.
    """
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 matthews_corrcoef, precision_score, recall_score)

    scorers = {
        "f1": lambda a, b: f1_score(a, b, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
        "mcc": matthews_corrcoef,
        "precision": lambda a, b: precision_score(a, b, zero_division=0),
        "recall": lambda a, b: recall_score(a, b, zero_division=0),
    }
    scorer = scorers.get(metric, scorers["f1"])
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba).ravel()

    grid = np.unique(np.round(np.linspace(0.05, 0.95, 91), 3))
    scores = [float(scorer(y_true, (proba >= t).astype(int))) for t in grid]
    best = int(np.argmax(scores))
    default = float(scorer(y_true, (proba >= 0.5).astype(int)))

    return {
        "best_threshold": float(grid[best]),
        "best_score": float(scores[best]),
        "metric": metric,
        "score_at_0.5": default,
        "improvement_over_0.5": float(scores[best] - default),
        "curve": {"thresholds": grid.tolist(), "scores": scores},
        "note": "Chosen on validation predictions. Apply to the test set as-is.",
    }


def calibrate(model, X_train, y_train, method: str = "isotonic", cv=5):
    """Wrap a classifier so its probabilities are trustworthy.

    Calibration mostly leaves AUC alone -- ranking is unchanged -- but it
    substantially improves the Brier score and makes "70% risk" actually mean
    70%. That matters more than AUC for anything clinical.
    """
    calibrated = CalibratedClassifierCV(clone(model), method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def build_ensemble(models: Dict, task: str = CLASSIFICATION, weights=None):
    """Soft-voting ensemble. Usually beats its own best member, modestly."""
    estimators = [(name, model) for name, model in models.items() if model is not None]
    if len(estimators) < 2:
        return None
    if task == CLASSIFICATION:
        return VotingClassifier(estimators, voting="soft", weights=weights, n_jobs=-1)
    return VotingRegressor(estimators, weights=weights, n_jobs=-1)


def suggested_settings(n_rows: int, n_features: int, task: str) -> Dict:
    """Search settings scaled to the dataset, so a big run does not stall.

    These are starting points, not tuned values. Larger populations and more
    generations explore more subsets and generally find better ones; they cost
    time, not validity.
    """
    if n_rows > 50_000 or n_features > 300:
        profile, qs = "large", {"population_size": 16, "n_iterations": 15,
                                "cv_folds": 3, "estimator": "knn"}
    elif n_rows > 5_000 or n_features > 100:
        profile, qs = "medium", {"population_size": 24, "n_iterations": 30,
                                 "cv_folds": 5, "estimator": "knn"}
    else:
        profile, qs = "small", {"population_size": 30, "n_iterations": 50,
                                "cv_folds": 5, "estimator": "forest"}

    return {
        "profile": profile,
        "qsqfs": qs,
        "advice": [
            f"Dataset profile: {profile} ({n_rows:,} rows x {n_features} features).",
            "More generations and a larger population search harder and often "
            "find better subsets. This is a time cost, not a shortcut.",
            "estimator='forest' scores subsets with a random forest instead of "
            "3-NN. Slower, but the chosen features suit tree models better.",
            "Enable hyperparameter tuning to fit the baselines properly; the "
            "defaults are sensible, not optimal for your data.",
            "If classes are imbalanced, tune the decision threshold and read "
            "balanced accuracy and MCC rather than raw accuracy.",
        ],
    }
