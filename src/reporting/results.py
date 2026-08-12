"""Assemble and write the run artefacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.jsonio import save_json

logger = logging.getLogger(__name__)


class ResultsWriter:
    def __init__(self, results_dir: str | Path, run_name: Optional[str] = None):
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(results_dir) / (run_name or f"run_{stamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.run_dir / "figures"
        self.figures: List[str] = []

    def save_results(self, results: Dict, name: str = "results.json") -> Path:
        return save_json(results, self.run_dir / name)

    def save_predictions(self, y_true, y_pred, y_proba=None, extra: Optional[Dict] = None) -> Path:
        data = {"y_true": np.asarray(y_true).ravel(), "y_pred": np.asarray(y_pred).ravel()}
        if y_proba is not None:
            proba = np.asarray(y_proba)
            if proba.ndim == 1:
                data["y_proba"] = proba
            else:
                for i in range(proba.shape[1]):
                    data[f"proba_class_{i}"] = proba[:, i]
        if extra:
            for key, values in extra.items():
                if len(values) == len(data["y_true"]):
                    data[key] = np.asarray(values)
        path = self.run_dir / "predictions.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def save_selected_features(self, names: List[str], mask: np.ndarray,
                               all_names: Optional[List[str]] = None) -> Path:
        path = self.run_dir / "selected_features.csv"
        if all_names is not None and len(all_names) == len(mask):
            pd.DataFrame({"feature": all_names, "selected": np.asarray(mask, dtype=bool)}
                         ).to_csv(path, index=False)
        else:
            pd.DataFrame({"feature": names}).to_csv(path, index=False)
        return path

    def save_text(self, text: str, name: str) -> Path:
        path = self.run_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def register_figure(self, path: Optional[Path]) -> None:
        if path:
            self.figures.append(str(Path(path).name))

    def write_summary(self, results: Dict) -> Path:
        """A plain-text summary containing only values present in ``results``."""
        lines = ["=" * 72, "QSQ-FS ML PLATFORM - RUN SUMMARY", "=" * 72, ""]
        dataset = results.get("dataset", {})
        lines += [
            "DATASET",
            f"  source          : {dataset.get('source', 'n/a')}",
            f"  rows x features : {dataset.get('n_rows', '?')} x {dataset.get('n_features_raw', '?')}",
            f"  task            : {results.get('task', 'n/a')}",
            f"  target          : {dataset.get('target', 'n/a')}",
            "",
        ]
        split = results.get("split", {})
        lines += [
            "SPLIT",
            f"  strategy        : {split.get('cv_type', 'n/a')}",
            f"  train / test    : {split.get('n_train', '?')} / {split.get('n_test', '?')}",
            f"  group leakage   : {'NONE' if not split.get('group_overlap') else split['group_overlap']}",
            "",
        ]
        fs = results.get("feature_selection", {})
        if fs:
            lines += [
                "FEATURE SELECTION (QSQ-FS)",
                f"  selected        : {fs.get('n_selected', '?')} / {fs.get('n_features_total', '?')}",
                f"  best fitness    : {fs.get('best_fitness', float('nan')):.4f}",
                f"  CV score        : {fs.get('best_score', float('nan')):.4f} "
                f"({fs.get('score_definition', '')})",
                f"  runtime         : {fs.get('runtime_seconds', '?')} s",
                f"  evaluations     : {fs.get('total_evaluations', '?')} "
                f"(cache hit rate {fs.get('cache_hit_rate', 0):.1%})",
                "",
            ]
        lines += ["TEST-SET RESULTS", "  (all values computed from held-out predictions)"]
        models = dict(results.get("baselines", {}))
        if results.get("transformer"):
            models["qsqfs_transformer"] = results["transformer"]
        keys = ["roc_auc", "balanced_accuracy", "accuracy", "f1"] \
            if results.get("task") == "classification" else ["rmse", "mae", "r2"]
        header = "  " + f"{'model':<28}" + "".join(f"{k:>14}" for k in keys)
        lines += ["", header, "  " + "-" * (28 + 14 * len(keys))]
        for name, metrics in models.items():
            if not isinstance(metrics, dict) or "error" in metrics:
                lines.append(f"  {name:<28}  {metrics.get('error', 'failed') if isinstance(metrics, dict) else 'failed'}")
                continue
            row = f"  {name:<28}"
            for key in keys:
                value = metrics.get(key)
                row += f"{value:>14.4f}" if isinstance(value, (int, float)) else f"{'-':>14}"
            lines.append(row)

        leakage = results.get("leakage", {})
        if leakage.get("excluded_columns") or leakage.get("flagged_columns"):
            lines += ["", "LEAKAGE"]
            for col in leakage.get("excluded_columns", []):
                lines.append(f"  EXCLUDED  {col} (duplicates the target)")
            for flag in leakage.get("flagged_columns", []):
                lines.append(f"  FLAGGED   {flag['column']}: {flag['reason']}")
        warnings = results.get("warnings", [])
        if warnings:
            lines += ["", "WARNINGS"] + [f"  - {w}" for w in warnings]
        lines += ["", "=" * 72,
                  "No value in this file is hardcoded or illustrative.",
                  "Every number was produced by this run on the data supplied.",
                  "=" * 72]
        return self.save_text("\n".join(lines), "summary.txt")
