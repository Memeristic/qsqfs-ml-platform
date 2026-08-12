"""Figures. Every plot is drawn from arrays the caller actually computed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9})


def _save(fig, out_dir: Path, name: str, fmt: str = "png") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_convergence(history: Dict, out_dir: Path, fmt: str = "png") -> Optional[Path]:
    best = history.get("best_fitness") or []
    if len(best) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(best, label="best")
    if history.get("mean_fitness"):
        axes[0].plot(history["mean_fitness"], "--", alpha=0.7, label="population mean")
    axes[0].set(xlabel="generation", ylabel="fitness", title="QSQ-FS convergence")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.get("n_selected", []), color="tab:green")
    axes[1].set(xlabel="generation", ylabel="features in best mask", title="Subset size")
    axes[1].grid(alpha=0.3)

    axes[2].plot(history.get("archive_size", []), color="tab:red", label="archive size")
    ax2 = axes[2].twinx()
    if history.get("field_entropy"):
        ax2.plot(history["field_entropy"], color="tab:purple", alpha=0.7, label="field entropy")
        ax2.set_ylabel("normalised field entropy")
    axes[2].set(xlabel="generation", ylabel="suppression archive size",
                title="Quenching / sensing state")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "qsqfs_convergence", fmt)


def plot_training_curves(history: Dict, out_dir: Path, fmt: str = "png") -> Optional[Path]:
    if not history.get("train_loss"):
        return None
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    best = int(np.argmin(history["val_loss"]))
    ax.axvline(best, color="grey", ls=":", label=f"best epoch ({best + 1})")
    ax.set(xlabel="epoch", ylabel="loss", title="Transformer training")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, out_dir, "training_curves", fmt)


def plot_regression_diagnostics(y_true, y_pred, out_dir: Path, units: str = "",
                                fmt: str = "png") -> Optional[Path]:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if len(y_true) < 3:
        return None
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].scatter(y_true, y_pred, s=12, alpha=0.5, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "k--", lw=1, label="identity")
    axes[0].set(xlabel=f"observed {units}", ylabel=f"predicted {units}",
                title="Predicted vs observed")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].scatter(y_pred, residuals, s=12, alpha=0.5, edgecolors="none")
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel=f"predicted {units}", ylabel="residual",
                title="Residuals (look for structure)")
    axes[1].grid(alpha=0.3)

    axes[2].hist(residuals, bins=min(40, max(10, len(residuals) // 8)),
                 alpha=0.8, edgecolor="white")
    axes[2].axvline(0, color="k", ls="--", lw=1)
    axes[2].set(xlabel="residual", ylabel="count",
                title=f"Residuals (bias {residuals.mean():+.2f})")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "regression_diagnostics", fmt)


def plot_classification_diagnostics(y_true, y_pred, y_proba, out_dir: Path,
                                    fmt: str = "png") -> Optional[Path]:
    from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix")

    if y_proba is not None and len(np.unique(y_true)) == 2:
        proba = np.asarray(y_proba).ravel()
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
        axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        precision, recall, _ = precision_recall_curve(y_true, proba)
        prevalence = float(np.mean(y_true))
        axes[2].plot(recall, precision)
        axes[2].axhline(prevalence, color="k", ls="--", lw=1,
                        label=f"prevalence = {prevalence:.3f}")
        axes[2].set(xlabel="recall", ylabel="precision", title="Precision-recall")
        axes[2].legend(); axes[2].grid(alpha=0.3)
    else:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, "probabilities unavailable", ha="center", va="center")
            ax.set_axis_off()
    fig.tight_layout()
    return _save(fig, out_dir, "classification_diagnostics", fmt)


def plot_feature_importance(importances: List[Dict], out_dir: Path, top_n: int = 20,
                            fmt: str = "png") -> Optional[Path]:
    if not importances:
        return None
    top = importances[:top_n][::-1]
    names = [str(r["feature"])[:44] for r in top]
    means = [r["importance_mean"] for r in top]
    errors = [r.get("importance_std", 0.0) for r in top]
    fig, ax = plt.subplots(figsize=(7, max(3.2, 0.3 * len(top))))
    ax.barh(range(len(top)), means, xerr=errors, alpha=0.85, capsize=2)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names)
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="drop in score when permuted", title=f"Permutation importance (top {len(top)})")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out_dir, "feature_importance", fmt)


def plot_model_comparison(results: Dict[str, Dict], metric: str, out_dir: Path,
                          lower_is_better: bool = False, fmt: str = "png") -> Optional[Path]:
    pairs = [(k, v[metric]) for k, v in results.items()
             if isinstance(v, dict) and isinstance(v.get(metric), (int, float))]
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda kv: kv[1], reverse=not lower_is_better)
    names, values = zip(*pairs)
    colours = ["tab:blue" if "transformer" not in n.lower() else "tab:orange" for n in names]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.42 * len(names))))
    ax.barh(range(len(names)), values, color=colours, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set(xlabel=metric, title=f"Model comparison ({metric})")
    for i, value in enumerate(values):
        ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out_dir, f"model_comparison_{metric}", fmt)


def plot_target_distribution(y: Sequence[float], out_dir: Path, task: str,
                             units: str = "", fmt: str = "png") -> Optional[Path]:
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if task == "classification":
        values, counts = np.unique(y, return_counts=True)
        ax.bar([str(v) for v in values], counts, alpha=0.85)
        ax.set(xlabel="class", ylabel="count", title="Class distribution")
        for i, c in enumerate(counts):
            ax.text(i, c, f"{c}\n({c/counts.sum():.1%})", ha="center", va="bottom", fontsize=8)
    else:
        ax.hist(y, bins=min(50, max(10, len(y) // 10)), alpha=0.85, edgecolor="white")
        ax.axvline(float(np.mean(y)), color="r", ls="--", label=f"mean {np.mean(y):.2f}")
        ax.set(xlabel=f"target {units}", ylabel="count", title="Target distribution")
        ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "target_distribution", fmt)
