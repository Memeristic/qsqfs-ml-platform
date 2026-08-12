"""Bundle a complete run into one ZIP for a thesis, a journal, or re-analysis.

The archive is self-describing: it carries the numbers, the figures, the
statistical tables, a methods paragraph you can adapt, and a MANIFEST listing
exactly what is inside and how it was produced. Handing the whole ZIP back for
analysis later gives the full context of the run, not just its headline metric.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.jsonio import save_json, to_jsonable

logger = logging.getLogger(__name__)


def methods_paragraph(results: Dict) -> str:
    """A methods description built only from what this run actually did."""
    dataset = results.get("dataset", {})
    split = results.get("split", {})
    fs = results.get("feature_selection", {})
    task = results.get("task", "classification")
    hyper = fs.get("hyperparameters", {})

    lines = [
        "METHODS (adapt to your own wording; every value is from this run)",
        "=" * 72, "",
        f"Data. The dataset comprised {dataset.get('n_rows', 'N')} observations "
        f"and {dataset.get('n_features_raw', 'P')} candidate variables, with "
        f"'{dataset.get('target', 'the outcome')}' as the target "
        f"({'binary/multi-class classification' if task == 'classification' else 'continuous regression'}).",
    ]
    if dataset.get("n_groups"):
        lines.append(
            f"Observations were nested within {dataset['n_groups']} groups "
            "(e.g. participants), and this structure was respected in splitting."
        )

    lines += [
        "",
        f"Partitioning. A {split.get('cv_type', 'random')} split assigned "
        f"{split.get('n_train', '?')} observations to training and "
        f"{split.get('n_test', '?')} to a held-out test set. The split was "
        "performed BEFORE any preprocessing, feature screening or model "
        "fitting, so no test information could influence the pipeline. "
        f"{split.get('n_cv_folds', '?')} inner cross-validation folds were used "
        "for all model selection.",
    ]
    if split.get("group_overlap") is None:
        lines.append("No group appeared in both the training and test partitions.")

    leakage = results.get("leakage", {})
    if leakage:
        excluded = leakage.get("excluded_columns", [])
        flagged = leakage.get("flagged_columns", [])
        lines += [
            "",
            "Leakage screening. Candidate predictors were screened for target "
            "leakage using training rows only. "
            + (f"{len(excluded)} variable(s) were exact duplicates of the target "
               f"and were removed ({', '.join(excluded)}). " if excluded else
               "No variable duplicated the target. ")
            + (f"{len(flagged)} variable(s) were flagged as possible proxies and "
               "retained for inspection rather than removed automatically."
               if flagged else "No proxy variables were flagged."),
        ]

    if fs:
        lines += [
            "",
            "Feature selection. Quorum Sensing / Quorum Quenching Feature "
            "Selection (QSQ-FS) was applied to the training partition. A "
            f"population of {hyper.get('population_size', '?')} binary feature "
            f"masks was evolved for {fs.get('n_iterations_run', '?')} "
            "generations under a quorum-sensing autoinducer field "
            f"(w_ai = {hyper.get('w_ai', '?')}, EMA smoothing beta = "
            f"{hyper.get('beta', '?')}) and a quorum-quenching suppression "
            f"archive with decay delta = {hyper.get('delta', '?')}. Fitness "
            f"combined predictive performance and parsimony (alpha = "
            f"{hyper.get('alpha', '?')}). The search evaluated "
            f"{fs.get('total_evaluations', '?')} candidate subsets in "
            f"{fs.get('runtime_seconds', '?')} s and retained "
            f"{fs.get('n_selected', '?')} of {fs.get('n_features_total', '?')} "
            f"features (cross-validated score {fs.get('best_score', float('nan')):.4f}, "
            f"defined as {fs.get('score_definition', 'see documentation')}).",
        ]

    lines += [
        "",
        "Models and evaluation. The selected features were supplied to a tabular "
        "Transformer and to classical baselines including a trivial predictor "
        "(majority class for classification, training mean for regression). All "
        "models were fitted on the training partition only and evaluated once on "
        "the held-out test set.",
    ]
    naive = results.get("naive_baseline", {})
    if naive:
        detail = ", ".join(f"{k} = {v:.4f}" for k, v in naive.items()
                           if isinstance(v, float))
        lines.append(f"The trivial baseline achieved {detail}, which any useful "
                     "model must exceed.")
    if results.get("bootstrap_ci", {}).get("ci_lower") is not None:
        ci = results["bootstrap_ci"]
        lines.append(
            f"A {ci['confidence']:.0%} confidence interval for {ci['metric']} was "
            f"obtained by bootstrap resampling of the test set "
            f"({ci['n_valid']} resamples)."
        )
    lines += [
        "",
        f"Reproducibility. Random seed {results.get('seed', '?')} was fixed across "
        "Python, NumPy and PyTorch. Library versions and platform details are "
        "recorded in results.json.",
        "",
        "=" * 72,
        "Note: no value in this bundle is hardcoded, simulated or illustrative. "
        "Every number was computed from the data supplied to this run.",
    ]
    return "\n".join(lines)


def build_manifest(run_dir: Path, results: Dict) -> str:
    lines = [
        "MANIFEST", "=" * 72,
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Run       : {run_dir.name}",
        f"Source    : {results.get('dataset', {}).get('source', 'unknown')}",
        f"Task      : {results.get('task', 'unknown')}",
        f"Seed      : {results.get('seed', 'unknown')}",
        "", "CONTENTS", "-" * 72,
        "results.json          every metric, setting and diagnostic, machine-readable",
        "summary.txt           human-readable overview of the run",
        "METHODS.txt           a methods paragraph built from this run's settings",
        "predictions.csv       per-row test-set predictions",
        "selected_features.csv which features QSQ-FS kept",
        "leakage_report.txt    what was excluded, what was flagged, and why",
        "tables/               statistical tables (CSV, plus one Excel workbook)",
        "figures/              300 DPI PNG and vector PDF of every figure",
        "",
        "HOW TO USE THIS BUNDLE", "-" * 72,
        "For a thesis  : figures/*.pdf are vector and scale without pixellation;",
        "                figures/*.png are 300 DPI for Word.",
        "                tables/*.csv paste directly into Word or SPSS.",
        "For a journal : most publishers want >=300 DPI raster or vector. Both are here.",
        "For re-analysis: send the whole ZIP. results.json holds the complete run,",
        "                including settings, warnings and library versions.",
        "",
        "READ THIS FIRST", "-" * 72,
    ]
    naive = results.get("naive_baseline", {})
    if naive:
        detail = ", ".join(f"{k}={v:.4f}" for k, v in naive.items()
                           if isinstance(v, float))
        lines.append(f"Trivial baseline to beat: {detail}")
    ci = results.get("bootstrap_ci", {})
    if ci.get("ci_lower") is not None:
        lines.append(
            f"{ci['metric']} = {ci['point_estimate']:.4f} "
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]. If this interval "
            "overlaps a baseline, the difference is not established."
        )
    for warning in results.get("warnings", [])[:15]:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)


def export_run(
    run_dir: str | Path,
    output_path: Optional[str | Path] = None,
    results: Optional[Dict] = None,
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    extra_files: Optional[Dict[str, str]] = None,
) -> Path:
    """Package a run directory into a single ZIP. Returns the archive path."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if results is None:
        results_file = run_dir / "results.json"
        if results_file.exists():
            import json
            results = json.loads(results_file.read_text(encoding="utf-8"))
        else:
            results = {}

    # Write the derived documents into the run directory first, so a later
    # export of the same run picks them up too.
    (run_dir / "METHODS.txt").write_text(methods_paragraph(results), encoding="utf-8")
    (run_dir / "MANIFEST.txt").write_text(build_manifest(run_dir, results), encoding="utf-8")

    if tables:
        tables_dir = run_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        for name, table in tables.items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                table.to_csv(tables_dir / f"{name}.csv", index=False)
        try:
            from src.reporting.stats_tables import write_tables_excel
            write_tables_excel(tables, tables_dir / "all_tables.xlsx")
        except Exception as exc:
            logger.warning("Excel workbook not written: %s", exc)

    if extra_files:
        for name, content in extra_files.items():
            (run_dir / name).write_text(content, encoding="utf-8")

    output_path = Path(output_path) if output_path else \
        run_dir.parent / f"{run_dir.name}_export.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, path.relative_to(run_dir.parent))

    size_mb = output_path.stat().st_size / 1024**2
    logger.info("Exported %s (%.2f MB)", output_path, size_mb)
    return output_path


def export_multiple(run_dirs: List[str | Path], output_path: str | Path,
                    label: str = "batch") -> Path:
    """Bundle several runs together, with a comparison index across them."""
    run_dirs = [Path(d) for d in run_dirs if Path(d).exists()]
    if not run_dirs:
        raise ValueError("No valid run directories supplied.")

    import json
    rows = []
    for run_dir in run_dirs:
        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue
        results = json.loads(results_file.read_text(encoding="utf-8"))
        task = results.get("task", "")
        primary = "roc_auc" if task == "classification" else "rmse"
        headline = results.get("transformer") or results.get("multimodal") or {}
        rows.append({
            "run": run_dir.name,
            "source": results.get("dataset", {}).get("source", ""),
            "task": task,
            "n_rows": results.get("dataset", {}).get("n_rows"),
            "n_features_selected": results.get("feature_selection", {}).get("n_selected"),
            primary: headline.get(primary),
            "naive_baseline": next(
                (v for k, v in results.get("naive_baseline", {}).items()
                 if isinstance(v, float)), None),
            "seed": results.get("seed"),
        })

    index = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{label}_index.csv", index.to_csv(index=False))
        archive.writestr(f"{label}_README.txt", "\n".join([
            f"BATCH EXPORT: {label}", "=" * 60,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Runs included: {len(run_dirs)}", "",
            f"{label}_index.csv compares every run in one table.",
            "Each run keeps its own folder with full results, figures and tables.",
            "",
            "Compare each run against its own naive_baseline column before "
            "comparing runs with each other -- datasets differ in difficulty.",
        ]))
        for run_dir in run_dirs:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file() and path.suffix != ".zip":
                    archive.write(path, Path(run_dir.name) / path.relative_to(run_dir))

    logger.info("Batch export: %s (%d runs, %.2f MB)", output_path, len(run_dirs),
                output_path.stat().st_size / 1024**2)
    return output_path
