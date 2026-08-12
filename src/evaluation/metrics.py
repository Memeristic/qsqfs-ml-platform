"""Metrics for both tasks. Every value is computed from real predictions."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    binary = len(labels) <= 2
    average = "binary" if binary else "macro"

    out: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(labels) > 1 else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }

    if binary and len(np.unique(y_true)) == 2:
        cm = confusion_matrix(y_true, y_pred, labels=sorted(labels))
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            out["confusion_matrix"] = {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            }
            out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
            out["npv"] = float(tn / (tn + fn)) if (tn + fn) else 0.0

    if y_proba is not None:
        proba = np.asarray(y_proba)
        try:
            if binary and proba.ndim == 1:
                out["roc_auc"] = float(roc_auc_score(y_true, proba))
                out["pr_auc"] = float(average_precision_score(y_true, proba))
                out["brier"] = float(brier_score_loss(y_true, proba))
            elif proba.ndim == 2:
                out["roc_auc"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
                )
        except ValueError as exc:
            out["roc_auc"] = None
            out["roc_auc_error"] = str(exc)

    values, counts = np.unique(y_true, return_counts=True)
    out["majority_class_accuracy"] = float(counts.max() / counts.sum())
    out["class_distribution"] = {str(v): int(c) for v, c in zip(values, counts)}
    return out


def regression_metrics(y_true, y_pred) -> Dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    std = float(np.std(y_true))
    baseline_rmse = float(np.sqrt(np.mean((y_true - np.mean(y_true)) ** 2)))

    nonzero = np.abs(y_true) > 1e-9
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any() else None
    )

    return {
        "rmse": rmse,
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(np.median(np.abs(y_true - y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": mape,
        "nrmse_std": float(rmse / std) if std > 1e-12 else None,
        "mean_baseline_rmse": baseline_rmse,
        "rmse_vs_mean_baseline": float(rmse / baseline_rmse) if baseline_rmse > 1e-12 else None,
        "bias": float(np.mean(y_pred - y_true)),
        "n_test": int(len(y_true)),
        "target_mean": float(np.mean(y_true)),
        "target_std": std,
    }


def compute_metrics(y_true, y_pred, y_proba=None, task: str = "classification") -> Dict:
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_proba)
    return regression_metrics(y_true, y_pred)


def primary_metric(task: str) -> str:
    return "roc_auc" if task == "classification" else "rmse"


def aggregate_metrics(runs: List[Dict]) -> Dict:
    """Mean/std/min/max across seeds or folds. Reported as-is, no smoothing."""
    if not runs:
        return {}
    keys = [
        k for k in runs[0]
        if all(isinstance(r.get(k), (int, float)) and r.get(k) is not None for r in runs)
    ]
    summary: Dict = {"n_runs": len(runs)}
    for key in keys:
        values = np.array([float(r[key]) for r in runs], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "values": [float(v) for v in values],
        }
    return summary
