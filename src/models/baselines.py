"""Reference models. Without these a QSQ-FS + Transformer number means nothing.

Separate registries per task -- running a regressor on a binary target and
reporting R² was a real failure mode in earlier versions of this project.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

logger = logging.getLogger(__name__)


def classification_models(seed: int = 42) -> Dict:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def regression_models(seed: int = 42) -> Dict:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "svm": SVR(kernel="rbf"),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
    }


def available_models(task: str, seed: int = 42) -> Dict:
    return classification_models(seed) if task == "classification" else regression_models(seed)


def run_baselines(
    X_train, y_train, X_test, y_test,
    task: str = "classification",
    models: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Fit each baseline on the SAME features the main model receives."""
    from src.evaluation.metrics import compute_metrics

    registry = available_models(task, seed)
    names = models or list(registry)
    results: Dict[str, Dict] = {}

    for name in names:
        if name not in registry:
            logger.warning("Unknown baseline '%s' for task '%s'; skipped.", name, task)
            results[name] = {"error": f"not available for task '{task}'"}
            continue
        model = registry[name]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = None
            if task == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                y_proba = proba[:, 1] if proba.shape[1] == 2 else proba
            results[name] = compute_metrics(y_test, y_pred, y_proba, task=task)
        except Exception as exc:
            logger.warning("Baseline '%s' failed: %s", name, exc)
            results[name] = {"error": str(exc)}
    return results
