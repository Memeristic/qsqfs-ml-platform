"""Publication-quality figures: 300 DPI, vector formats, journal styling.

Every figure is drawn from arrays the pipeline actually computed. Styling
changes how a result looks, never what it is.

Defaults chosen for thesis and journal use:
  * 300 DPI raster (PNG/TIFF) plus a vector copy (PDF/SVG) of every figure
  * single-column 3.5 in and double-column 7.0 in widths
  * a colour-blind-safe palette, and line styles that survive greyscale printing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# Blue / gold / neutral, kept consistent with the app theme.
NAVY, BLUE, LIGHT_BLUE = "#0B3C6B", "#1F6FB2", "#7FB3DC"
GOLD, DARK_GOLD, LIGHT_GOLD = "#C9A227", "#8C6F14", "#E8D58A"
GREY, DARK = "#6B7280", "#111827"
PALETTE = [BLUE, GOLD, NAVY, LIGHT_BLUE, DARK_GOLD, GREY, "#2E8B57", "#B22222"]
# Distinct dashes so a multi-line figure is still readable printed in greyscale.
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

SINGLE_COLUMN, DOUBLE_COLUMN = 3.5, 7.0


def apply_style(font_size: int = 9, font_family: str = "DejaVu Sans") -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "savefig.transparent": False, "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": [font_family, "Arial"],
        "font.size": font_size,
        "axes.titlesize": font_size + 1, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": DARK,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "figure.autolayout": False,
    })


def _safe_bins(values: np.ndarray, cap: int = 50) -> int:
    """Bin count that numpy will accept for any input.

    A constant or near-constant array has no finite range to divide, and
    numpy raises rather than degrading gracefully. Perfectly constant
    residuals are common in tests and in degenerate models, so this is a real
    case, not a hypothetical one.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or float(np.ptp(values)) < 1e-12:
        return 1
    return int(min(cap, max(5, min(values.size // 10, int(np.sqrt(values.size)) + 1))))


def save_figure(fig, out_dir, name: str, formats: Sequence[str] = ("png", "pdf"),
                dpi: int = 300) -> List[str]:
    """Write one figure in every requested format. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        try:
            fig.savefig(path, format=fmt, dpi=dpi)
            written.append(str(path))
        except Exception as exc:
            logger.warning("Could not save %s as %s: %s", name, fmt, exc)
    plt.close(fig)
    return written


# ======================================================================
def fig_model_comparison(results: Dict, task: str, out_dir, formats=("png", "pdf"),
                         ci: Optional[Dict] = None) -> List[str]:
    """Grouped bars across models and metrics, with the dummy marked."""
    apply_style()
    metrics = (["roc_auc", "balanced_accuracy", "f1", "precision", "recall"]
               if task == "classification" else ["rmse", "mae", "r2"])
    models = dict(results.get("baselines", {}))
    for key, label in (("transformer", "QSQ-FS +\nTransformer"),
                       ("multimodal", "Multimodal\nfusion")):
        if results.get(key):
            models[label] = results[key]
    models = {k: v for k, v in models.items()
              if isinstance(v, dict) and "error" not in v}
    if len(models) < 2:
        return []

    names = list(models)
    available = [m for m in metrics
                 if any(isinstance(models[n].get(m), (int, float)) for n in names)]
    x = np.arange(len(names))
    width = 0.8 / max(1, len(available))

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 3.4))
    for i, metric in enumerate(available):
        values = [models[n].get(metric) if isinstance(models[n].get(metric), (int, float))
                  else np.nan for n in names]
        ax.bar(x + i * width - 0.4 + width / 2, values, width * 0.92,
               label=metric.replace("_", " ").title(),
               color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.4)

    dummy = next((n for n in names if "dummy" in n.lower()), None)
    if dummy and task == "classification":
        ax.axhline(0.5, color=GREY, ls="--", lw=0.9, zorder=0)
        ax.text(len(names) - 0.5, 0.51, "chance", color=GREY, fontsize=7, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("Score" if task == "classification" else "Value")
    ax.set_title("Model performance on the held-out test set")
    ax.legend(ncol=min(len(available), 5), loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_model_comparison", formats)


def fig_roc_pr_curves(curves: Dict[str, Dict], out_dir, formats=("png", "pdf"),
                      prevalence: Optional[float] = None) -> List[str]:
    """ROC and precision-recall for several models on one pair of axes."""
    apply_style()
    if not curves:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.2))

    for i, (name, data) in enumerate(curves.items()):
        colour, dash = PALETTE[i % len(PALETTE)], LINESTYLES[i % len(LINESTYLES)]
        if "fpr" in data:
            axes[0].plot(data["fpr"], data["tpr"], color=colour, ls=dash,
                         label=f"{name} (AUC={data.get('auc', float('nan')):.3f})")
        if "recall" in data:
            axes[1].plot(data["recall"], data["precision"], color=colour, ls=dash,
                         label=f"{name} (AP={data.get('ap', float('nan')):.3f})")

    axes[0].plot([0, 1], [0, 1], color=GREY, ls=":", lw=0.9, label="Chance")
    axes[0].set(xlabel="1 − Specificity", ylabel="Sensitivity",
                title="(a) ROC curves", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(loc="lower right", fontsize=7)

    if prevalence is not None:
        axes[1].axhline(prevalence, color=GREY, ls=":", lw=0.9,
                        label=f"Prevalence ({prevalence:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="(b) Precision–recall", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_roc_pr_curves", formats)


def fig_qsqfs_convergence(history: Dict, out_dir, formats=("png", "pdf")) -> List[str]:
    """Four-panel convergence diagnostic for the search itself."""
    apply_style()
    best = history.get("best_fitness") or []
    if len(best) < 2:
        return []
    generations = np.arange(len(best))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN, 4.6))

    axes[0, 0].plot(generations, best, color=BLUE, label="Best")
    if history.get("mean_fitness"):
        axes[0, 0].plot(generations, history["mean_fitness"], color=GOLD,
                        ls="--", label="Population mean")
    axes[0, 0].set(xlabel="Generation", ylabel="Fitness", title="(a) Convergence")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].plot(generations, history.get("n_selected", []), color=NAVY)
    axes[0, 1].set(xlabel="Generation", ylabel="Features selected",
                   title="(b) Subset size")

    axes[1, 0].plot(generations, history.get("archive_size", []), color=DARK_GOLD)
    axes[1, 0].set(xlabel="Generation", ylabel="Archive entries",
                   title="(c) Quorum-quenching archive")

    if history.get("field_entropy"):
        axes[1, 1].plot(generations, history["field_entropy"], color=LIGHT_BLUE)
        axes[1, 1].set(xlabel="Generation", ylabel="Normalised entropy",
                       title="(d) Autoinducer field consensus")
        axes[1, 1].text(0.98, 0.95, "lower = stronger consensus", fontsize=6,
                        color=GREY, ha="right", va="top",
                        transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].set_axis_off()
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_qsqfs_convergence", formats)


def fig_feature_importance(importances: List[Dict], out_dir, top_n: int = 20,
                           formats=("png", "pdf")) -> List[str]:
    apply_style()
    if not importances:
        return []
    top = importances[:top_n][::-1]
    names = [str(r["feature"])[:38] for r in top]
    means = [r["importance_mean"] for r in top]
    errors = [r.get("importance_std", 0.0) for r in top]
    colours = [BLUE if m > 0 else GREY for m in means]

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN * 1.6, max(2.6, 0.22 * len(top))))
    ax.barh(range(len(top)), means, xerr=errors, color=colours,
            edgecolor="white", linewidth=0.4,
            error_kw={"ecolor": DARK_GOLD, "elinewidth": 0.7, "capsize": 1.6})
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set(xlabel="Decrease in score when permuted",
           title=f"Permutation importance (top {len(top)})")
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_feature_importance", formats)


def fig_regression_diagnostics(y_true, y_pred, out_dir, units: str = "",
                               formats=("png", "pdf")) -> List[str]:
    """Four-panel regression diagnostic, including a Bland-Altman plot."""
    apply_style()
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if len(y_true) < 3:
        return []
    residuals = y_pred - y_true
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN, 5.0))

    axes[0, 0].scatter(y_true, y_pred, s=10, alpha=0.55, color=BLUE,
                       edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0, 0].plot(lims, lims, color=DARK, ls="--", lw=0.9, label="Identity")
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    axes[0, 0].plot(lims, [slope * v + intercept for v in lims], color=GOLD,
                    lw=1.1, label=f"Fit (slope={slope:.2f})")
    r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    axes[0, 0].set(xlabel=f"Observed {units}", ylabel=f"Predicted {units}",
                   title=f"(a) Predicted vs observed (R²={r2:.3f})")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].scatter(y_pred, residuals, s=10, alpha=0.55, color=BLUE,
                       edgecolors="none")
    axes[0, 1].axhline(0, color=DARK, ls="--", lw=0.9)
    axes[0, 1].set(xlabel=f"Predicted {units}", ylabel="Residual",
                   title="(b) Residuals vs fitted")

    axes[1, 0].hist(residuals, bins=_safe_bins(residuals, 40),
                    color=LIGHT_BLUE, edgecolor="white", linewidth=0.4)
    axes[1, 0].axvline(0, color=DARK, ls="--", lw=0.9)
    axes[1, 0].axvline(residuals.mean(), color=GOLD, lw=1.1,
                       label=f"Bias = {residuals.mean():+.3f}")
    axes[1, 0].set(xlabel="Residual", ylabel="Count",
                   title="(c) Residual distribution")
    axes[1, 0].legend(fontsize=7)

    # Bland-Altman: the standard agreement plot in clinical measurement work.
    mean_of_pair = (y_true + y_pred) / 2
    bias = float(residuals.mean())
    sd = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
    axes[1, 1].scatter(mean_of_pair, residuals, s=10, alpha=0.55, color=BLUE,
                       edgecolors="none")
    axes[1, 1].axhline(bias, color=GOLD, lw=1.1, label=f"Bias {bias:+.2f}")
    for offset, style in ((1.96, "--"), (-1.96, "--")):
        axes[1, 1].axhline(bias + offset * sd, color=DARK_GOLD, ls=style, lw=0.9)
    axes[1, 1].text(0.98, 0.95, "±1.96 SD limits of agreement", fontsize=6,
                    color=GREY, ha="right", va="top", transform=axes[1, 1].transAxes)
    axes[1, 1].set(xlabel=f"Mean of methods {units}", ylabel="Difference",
                   title="(d) Bland–Altman agreement")
    axes[1, 1].legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_regression_diagnostics", formats)


def fig_confusion_matrix(y_true, y_pred, out_dir, labels=None,
                         formats=("png", "pdf")) -> List[str]:
    apply_style()
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred)
    normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    names = labels or [str(i) for i in range(matrix.shape[0])]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 3.0))
    for ax, data, title, fmt in (
        (axes[0], matrix, "(a) Counts", "d"),
        (axes[1], normalised, "(b) Row-normalised", ".2f"),
    ):
        image = ax.imshow(data, cmap="Blues", vmin=0,
                          vmax=data.max() if data.size else 1)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = format(data[i, j], fmt)
                ax.text(j, i, value, ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > data.max() * 0.55 else DARK)
        ax.set(xticks=range(len(names)), yticks=range(len(names)),
               xlabel="Predicted", ylabel="Actual", title=title)
        ax.set_xticklabels(names, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.045)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_confusion_matrix", formats)


def fig_correlation_heatmap(df: pd.DataFrame, out_dir, top_n: int = 25,
                            method: str = "spearman", formats=("png", "pdf")) -> List[str]:
    apply_style()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    if numeric.shape[1] > top_n:
        numeric = numeric[numeric.var().sort_values(ascending=False).head(top_n).index]

    corr = numeric.corr(method=method)
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.85))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(corr)), yticks=range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5.5)
    ax.set_yticklabels(corr.index, fontsize=5.5)
    ax.set_title(f"{method.capitalize()} correlation "
                 f"({'top ' + str(top_n) + ' by variance' if df.shape[1] > top_n else 'all'})")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.045, label="r")
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_correlation_heatmap", formats)


def fig_cv_boxplot(per_fold: List[Dict], metric: str, out_dir,
                   formats=("png", "pdf")) -> List[str]:
    """Spread across folds or seeds -- shows stability, not just a point estimate."""
    apply_style()
    grouped: Dict[str, List[float]] = {}
    for entry in per_fold:
        for name, metrics in entry.items():
            if isinstance(metrics, dict) and isinstance(metrics.get(metric), (int, float)):
                grouped.setdefault(name, []).append(float(metrics[metric]))
    grouped = {k: v for k, v in grouped.items() if len(v) >= 2}
    if not grouped:
        return []

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN, 3.2))
    box = ax.boxplot(list(grouped.values()), patch_artist=True,
                     tick_labels=[k.replace("_", "\n") for k in grouped],
                     medianprops={"color": DARK, "linewidth": 1.2})
    for patch, colour in zip(box["boxes"], PALETTE * 5):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)
    for i, values in enumerate(grouped.values(), start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=9,
                   color=DARK, alpha=0.6, zorder=3)
    ax.set(ylabel=metric.replace("_", " ").title(),
           title=f"{metric.replace('_', ' ').title()} across runs")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir, f"fig_cv_spread_{metric}", formats)


def fig_target_distribution(y, out_dir, task: str, units: str = "",
                            formats=("png", "pdf")) -> List[str]:
    apply_style()
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN * 1.5, 2.8))
    if task == "classification":
        values, counts = np.unique(y, return_counts=True)
        bars = ax.bar([str(v) for v in values], counts, color=BLUE,
                      edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, count,
                    f"{count}\n({count/counts.sum():.1%})", ha="center",
                    va="bottom", fontsize=7)
        ax.set(xlabel="Class", ylabel="Count", title="Class distribution")
        ax.set_ylim(0, counts.max() * 1.2)
    else:
        # A near-constant target cannot be binned finely; numpy raises rather
        # than degrading, so choose the bin count from the actual data range.
        ax.hist(y, bins=_safe_bins(y), color=LIGHT_BLUE, edgecolor="white",
                linewidth=0.4)
        ax.axvline(float(np.mean(y)), color=GOLD, lw=1.2,
                   label=f"Mean {np.mean(y):.2f}")
        ax.axvline(float(np.median(y)), color=NAVY, ls="--", lw=1.0,
                   label=f"Median {np.median(y):.2f}")
        ax.set(xlabel=f"Target {units}", ylabel="Count", title="Target distribution")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_target_distribution", formats)


def generate_all(results: Dict, out_dir, task: str, y_true=None, y_pred=None,
                 y_proba=None, features_df=None, units: str = "",
                 formats=("png", "pdf")) -> Dict[str, List[str]]:
    """Produce every applicable publication figure. Returns name -> paths."""
    out_dir = Path(out_dir)
    written: Dict[str, List[str]] = {}

    def record(key, paths):
        if paths:
            written[key] = paths

    record("model_comparison", fig_model_comparison(results, task, out_dir, formats))
    record("qsqfs_convergence",
           fig_qsqfs_convergence(results.get("qsqfs_history", {}), out_dir, formats))
    record("feature_importance", fig_feature_importance(
        results.get("feature_importance", {}).get("importances", []), out_dir,
        formats=formats))

    if y_true is not None:
        record("target_distribution",
               fig_target_distribution(y_true, out_dir, task, units, formats))

    if y_true is not None and y_pred is not None:
        if task == "regression":
            record("regression_diagnostics",
                   fig_regression_diagnostics(y_true, y_pred, out_dir, units, formats))
        else:
            record("confusion_matrix",
                   fig_confusion_matrix(y_true, y_pred, out_dir, formats=formats))
            if y_proba is not None and len(np.unique(y_true)) == 2:
                from sklearn.metrics import (average_precision_score,
                                             precision_recall_curve, roc_auc_score,
                                             roc_curve)
                proba = np.asarray(y_proba).ravel()
                fpr, tpr, _ = roc_curve(y_true, proba)
                precision, recall, _ = precision_recall_curve(y_true, proba)
                curves = {"QSQ-FS + Transformer": {
                    "fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_true, proba),
                    "precision": precision, "recall": recall,
                    "ap": average_precision_score(y_true, proba),
                }}
                record("roc_pr_curves", fig_roc_pr_curves(
                    curves, out_dir, formats, prevalence=float(np.mean(y_true))))

    if features_df is not None and features_df.shape[1] >= 2:
        record("correlation_heatmap",
               fig_correlation_heatmap(features_df, out_dir, formats=formats))

    logger.info("Publication figures written: %d", sum(len(v) for v in written.values()))
    return written
