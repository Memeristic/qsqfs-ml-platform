#!/usr/bin/env python3
"""QSQ-FS ML Platform - command line interface.

    python run_pipeline.py inspect  --data_root ./data/d1namo
    python run_pipeline.py d1namo   --data_root ./data/d1namo --horizon 30
    python run_pipeline.py tabular  --data_path ./data/hospital.csv --target readmitted
    python run_pipeline.py ablation --data_path ./data/hospital.csv --target readmitted

Every number this tool prints is computed from the data you supply. Nothing is
hardcoded, simulated, or filled in from a reference value.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import GenericDataLoader, infer_task          # noqa: E402
from src.pipeline import ExperimentConfig, run_experiment          # noqa: E402
from src.utils.config import load_config                           # noqa: E402
from src.utils.jsonio import save_json                             # noqa: E402
from src.utils.logging_setup import setup_logging                  # noqa: E402

DEFAULT_CONFIG = "config/default_config.yaml"


# ----------------------------------------------------------------------
def _shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--cv_type", default=None,
                        choices=["time_series", "group", "stratified", "random"])
    parser.add_argument("--test_size", type=float, default=None)
    parser.add_argument("--n_iterations", type=int, default=None,
                        help="QSQ-FS generations")
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Transformer epochs")
    parser.add_argument("--skip_transformer", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_importance", action="store_true")
    parser.add_argument("--tune", action="store_true",
                        help="Search baseline hyperparameters by inner CV "
                             "(training rows only). Slower, usually better.")
    parser.add_argument("--tune_iterations", type=int, default=20)
    parser.add_argument("--no_publication_figures", action="store_true")
    parser.add_argument("--no_stats_tables", action="store_true")
    parser.add_argument("--no_export", action="store_true",
                        help="Skip writing the ZIP bundle")
    parser.add_argument("--figure_formats", nargs="+", default=["png", "pdf"],
                        choices=["png", "pdf", "svg", "tiff", "eps"])
    parser.add_argument("--quiet", action="store_true")


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    qs = cfg.setdefault("feature_selection", {}).setdefault("qsqfs", {})
    tr = cfg.setdefault("models", {}).setdefault("transformer", {})
    ev = cfg.setdefault("evaluation", {})
    if args.n_iterations is not None:
        qs["n_iterations"] = args.n_iterations
    if args.population_size is not None:
        qs["population_size"] = args.population_size
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.test_size is not None:
        ev["test_size"] = args.test_size
    if args.cv_type is not None:
        ev["cv_type"] = args.cv_type
    if args.results_dir is not None:
        cfg.setdefault("output", {})["results_dir"] = args.results_dir
    return cfg


def _print_report(results: dict, task: str) -> None:
    print("\n" + "=" * 72)
    print("RESULTS (held-out test set)")
    print("=" * 72)
    naive = results.get("naive_baseline", {})
    print(f"Naive baseline ({naive.get('strategy', '?')}): " + ", ".join(
        f"{k}={v:.4f}" for k, v in naive.items() if isinstance(v, float)
    ))
    fs = results.get("feature_selection", {})
    if fs:
        print(f"QSQ-FS selected {fs.get('n_selected')}/{fs.get('n_features_total')} features "
              f"(fitness {fs.get('best_fitness', float('nan')):.4f}, "
              f"{fs.get('runtime_seconds')} s)")

    keys = (["roc_auc", "balanced_accuracy", "accuracy", "f1"]
            if task == "classification" else ["rmse", "mae", "r2"])
    models = dict(results.get("baselines", {}))
    if results.get("transformer"):
        models["qsqfs_transformer"] = results["transformer"]
    width = max(13, max(len(k) for k in keys) + 2)
    print("\n" + f"{'model':<28}" + "".join(f"{k:>{width}}" for k in keys))
    print("-" * (28 + width * len(keys)))
    for name, metrics in models.items():
        if not isinstance(metrics, dict) or "error" in metrics:
            print(f"{name:<28}  (failed)")
            continue
        row = f"{name:<28}"
        for key in keys:
            value = metrics.get(key)
            row += (f"{value:>{width}.4f}" if isinstance(value, (int, float))
                    else f"{'-':>{width}}")
        print(row)

    ci = results.get("bootstrap_ci")
    if ci and "ci_lower" in ci:
        print(f"\nTransformer {ci['metric']}: {ci['point_estimate']:.4f} "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
              f"({ci['confidence']:.0%} bootstrap CI, n={ci['n_valid']})")

    for warning in results.get("warnings", [])[:12]:
        print(f"  ! {warning}")
    print(f"\nArtefacts: {results['run_directory']}")
    print("=" * 72)


# ----------------------------------------------------------------------
def _export_features(X_df, y, groups, times, path: str, target_name: str) -> None:
    """Write the extracted window features to one CSV.

    Signal datasets are tens of gigabytes of raw recordings, but the feature
    table they reduce to is a few megabytes. Exporting it means the heavy
    extraction happens once, locally, and the resulting table can be uploaded
    to the Streamlit app (or shared, or version-controlled) like any other CSV.
    """
    frame = X_df.copy()
    frame.insert(0, "subject_id", groups)
    if times is not None:
        frame.insert(1, "window_time", pd.Series(times).astype(str).to_numpy())
    frame[target_name] = y
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    size_mb = out.stat().st_size / 1024**2
    print(f"\nExported {len(frame)} rows x {frame.shape[1]} cols -> {out} ({size_mb:.2f} MB)")
    print("Upload this to the Streamlit app, or run:")
    print(f"  python run_pipeline.py tabular --data_path {out} "
          f"--target {target_name} --group_col subject_id --cv_type group")


def cmd_inspect(args: argparse.Namespace) -> int:
    from src.data.d1namo import describe

    info = describe(args.data_root)
    print(f"\nD1NAMO inventory for {info['data_root']}")
    print(f"Subjects found : {info['n_subjects']}")
    print(f"Usable         : {len(info['usable_subjects'])} "
          f"(glucose + at least one sensor file)\n")
    print(f"{'subject':<10}{'cohort':<12}{'glucose':<9}signals")
    print("-" * 62)
    for subject in info["subjects"]:
        signals = ", ".join(f"{k}:{v}" for k, v in subject["signals"].items()) or "none"
        print(f"{subject['subject_id']:<10}{subject['cohort']:<12}"
              f"{subject['n_glucose_files']:<9}{signals}")
    if not info["usable_subjects"]:
        print("\nNothing usable was found. Check that you extracted the archive and "
              "pointed --data_root at the folder containing the cohort directories.")
        return 1
    out = Path(args.results_dir or "./run_results") / "d1namo_inventory.json"
    save_json(info, out)
    print(f"\nSaved to {out}")
    return 0


def cmd_tabular(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    df = GenericDataLoader(args.data_path).load()
    if args.target not in df.columns:
        print(f"Target '{args.target}' not in file. Columns: {list(df.columns)}")
        return 1

    df = df.dropna(subset=[args.target]).reset_index(drop=True)
    task = args.task if args.task != "auto" else infer_task(df[args.target])
    print(f"Task: {task} (target '{args.target}', {df[args.target].nunique()} distinct values)")

    groups = df[args.group_col].to_numpy() if args.group_col and args.group_col in df else None
    order = None
    if args.time_col and args.time_col in df.columns:
        order = pd.to_datetime(df[args.time_col], errors="coerce").to_numpy()

    drop = [args.target] + [c for c in (args.group_col, args.time_col) if c and c in df.columns]
    X_df = df.drop(columns=drop)

    ev = cfg.get("evaluation", {})
    default_cv = "stratified" if task == "classification" else "random"
    exp = ExperimentConfig(
        task=task, target=args.target, domain=args.domain,
        cv_type=ev.get("cv_type") or default_cv,
        test_size=ev.get("test_size", 0.2), gap=ev.get("gap", 0),
        n_folds=ev.get("n_folds", 5), seed=args.seed,
        results_dir=cfg.get("output", {}).get("results_dir", "./run_results"),
        run_name=args.run_name, drop_flagged=args.drop_flagged,
        skip_transformer=args.skip_transformer, skip_baselines=args.skip_baselines,
        skip_importance=args.skip_importance,
        bootstrap_iterations=ev.get("bootstrap_iterations", 1000),
        source=f"tabular:{Path(args.data_path).name}",
        tune_hyperparameters=args.tune, tune_iterations=args.tune_iterations,
        publication_figures=not args.no_publication_figures,
        stats_tables=not args.no_stats_tables, export_zip=not args.no_export,
        figure_formats=tuple(args.figure_formats),
    )
    results = run_experiment(
        X_df, df[args.target].to_numpy(), exp,
        model_config={**cfg.get("preprocessing", {}), **cfg.get("models", {})},
        qsqfs_config=cfg["feature_selection"]["qsqfs"],
        groups=groups, order=order, raw_df=df,
    )
    _print_report(results, task)
    return 0


def cmd_d1namo(args: argparse.Namespace) -> int:
    from src.data.d1namo import D1NAMOLoader

    cfg = _apply_overrides(load_config(args.config), args)
    loader = D1NAMOLoader(
        args.data_root, window_minutes=args.window, step_minutes=args.step,
        horizon_minutes=args.horizon, max_target_gap_minutes=args.max_gap,
        use_ecg=args.use_ecg, max_subjects=args.max_subjects,
        cohorts=args.cohorts,
    )
    X_df, y, groups, times = loader.load()
    print(f"\n{loader.report['n_windows']} windows x {loader.report['n_features']} features "
          f"from {loader.report['n_subjects']} subjects")
    print(f"Target: {loader.report['target_mean']:.1f} +/- {loader.report['target_std']:.1f} "
          f"{loader.report['target_units']} "
          f"(range {loader.report['target_min']:.1f}-{loader.report['target_max']:.1f})")
    for caveat in loader.report["caveats"]:
        print(f"  ! {caveat}")

    if args.export_features:
        _export_features(X_df, y, groups, times, args.export_features, "glucose")
        if args.export_only:
            return 0

    ev = cfg.get("evaluation", {})
    exp = ExperimentConfig(
        task="regression", target="glucose", domain="generic",
        cv_type=args.cv_type or "group", test_size=ev.get("test_size", 0.2),
        gap=ev.get("gap", 0), n_folds=ev.get("n_folds", 5), seed=args.seed,
        results_dir=cfg.get("output", {}).get("results_dir", "./run_results"),
        run_name=args.run_name, skip_transformer=args.skip_transformer,
        skip_baselines=args.skip_baselines, skip_importance=args.skip_importance,
        bootstrap_iterations=ev.get("bootstrap_iterations", 1000),
        units=loader.report["target_units"], source="d1namo",
        tune_hyperparameters=args.tune, tune_iterations=args.tune_iterations,
        publication_figures=not args.no_publication_figures,
        stats_tables=not args.no_stats_tables, export_zip=not args.no_export,
        figure_formats=tuple(args.figure_formats),
    )
    results = run_experiment(
        X_df, y, exp,
        model_config={**cfg.get("preprocessing", {}), **cfg.get("models", {})},
        qsqfs_config=cfg["feature_selection"]["qsqfs"],
        groups=groups, order=times.to_numpy(), raw_df=None,
        dataset_notes={"d1namo": loader.report},
    )
    _print_report(results, "regression")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Turn each QSQ-FS mechanism off in turn and rerun the same protocol."""
    from src.feature_selection.ablation import run_ablation

    cfg = _apply_overrides(load_config(args.config), args)
    df = GenericDataLoader(args.data_path).load().dropna(subset=[args.target])
    task = args.task if args.task != "auto" else infer_task(df[args.target])
    y = df[args.target].to_numpy()
    if task == "classification":
        labels = np.unique(y)
        y = np.searchsorted(labels, y)

    groups = df[args.group_col].to_numpy() if args.group_col and args.group_col in df else None

    # Screen for leakage here too. Skipping it would let a duplicated target
    # column give every ablation variant a perfect score, making the comparison
    # between mechanisms meaningless.
    from src.leakage.detector import LeakageDetector
    from src.preprocessing.pipeline import PreprocessingPipeline

    leakage = LeakageDetector().detect(df, args.target, domain="generic")
    drop = [args.target] + [c for c in leakage.excluded_columns if c in df.columns]
    if args.group_col and args.group_col in df.columns:
        drop.append(args.group_col)
    if len(drop) > 1:
        print(f"Excluded before ablation: {drop[1:]}")
    X = PreprocessingPipeline().fit_transform(df.drop(columns=list(dict.fromkeys(drop))))

    results = run_ablation(
        X, y, task=task, qsqfs_config=cfg["feature_selection"]["qsqfs"],
        groups=groups, seeds=args.seeds, variants=args.variants,
    )
    out = Path(cfg.get("output", {}).get("results_dir", "./run_results")) / "ablation.json"
    save_json(results, out)

    print(f"\n{'variant':<26}{'fitness':>12}{'cv score':>12}{'k':>7}{'seconds':>10}")
    print("-" * 67)
    for name, entry in results["variants"].items():
        print(f"{name:<26}{entry['fitness_mean']:>12.4f}{entry['score_mean']:>12.4f}"
              f"{entry['n_selected_mean']:>7.1f}{entry['runtime_mean']:>10.1f}")
    print(f"\nAcross {len(args.seeds)} seed(s). Saved to {out}")
    return 0



def cmd_multimodal(args: argparse.Namespace) -> int:
    """Manifest-driven multimodal run: tabular + text + image + genomic."""
    from src.data.multimodal import infer_modalities
    from src.multimodal_pipeline import MultimodalConfig, run_multimodal_experiment

    df = GenericDataLoader(args.manifest).load()
    if args.target not in df.columns:
        print(f"Target '{args.target}' not in manifest. Columns: {list(df.columns)[:30]}")
        return 1
    df = df.dropna(subset=[args.target]).reset_index(drop=True)

    exclude = [c for c in (args.group_col, args.id_col) if c and c in df.columns]
    modalities = infer_modalities(
        df, args.target, exclude=exclude,
        image_cols=args.image_cols, text_cols=args.text_cols,
        genomic_cols=args.genomic_cols,
    )
    if args.modalities:
        modalities = {k: v for k, v in modalities.items() if k in args.modalities}
        if not modalities:
            print(f"None of --modalities {args.modalities} were found.")
            return 1

    print("\nResolved modalities (check this matched what you intended):")
    for kind, cols in modalities.items():
        preview = ", ".join(map(str, cols[:6])) + (" ..." if len(cols) > 6 else "")
        print(f"  {kind:<9} {len(cols):>4} column(s)  {preview}")
    print("Override with --image_cols / --text_cols / --genomic_cols if wrong.\n")

    groups = df[args.group_col].to_numpy() if args.group_col and args.group_col in df else None
    cfg = MultimodalConfig(
        target=args.target, task=args.task, fusion=args.fusion,
        embedding_dim=args.embedding_dim, epochs=args.epochs or 30,
        batch_size=args.batch_size, image_size=args.image_size,
        image_root=args.image_root or str(Path(args.manifest).parent),
        cv_type=args.cv_type or "stratified", seed=args.seed,
        results_dir=args.results_dir or "./run_results", run_name=args.run_name,
        run_modality_ablation=not args.skip_modality_ablation,
        force_text_fallback=args.force_text_fallback,
    )
    results = run_multimodal_experiment(df, cfg, modalities, groups=groups)
    _print_multimodal_report(results)
    return 0


def _print_multimodal_report(results: dict) -> None:
    task = results.get("task", "classification")
    keys = (["roc_auc", "balanced_accuracy", "accuracy", "f1"]
            if task == "classification" else ["rmse", "mae", "r2"])
    print("\n" + "=" * 72)
    print("RESULTS (held-out test set)")
    print("=" * 72)
    naive = results.get("naive_baseline", {})
    print("Naive baseline: " + ", ".join(
        f"{k}={v:.4f}" for k, v in naive.items() if isinstance(v, float)))

    models = {"multimodal_fused": results.get("multimodal", {})}
    models.update(results.get("modality_ablation", {}))
    models.update({f"tabular_{k}": v
                   for k, v in results.get("tabular_only_baselines", {}).items()
                   if isinstance(v, dict) and "error" not in v})
    print("\n" + f"{chr(109)+chr(111)+chr(100)+chr(101)+chr(108):<30}" + "".join(f"{k:>22}" for k in keys))
    print("-" * (30 + 13 * len(keys)))
    for name, metrics in models.items():
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        row = f"{name:<30}"
        for key in keys:
            value = metrics.get(key)
            row += (f"{value:>{width}.4f}" if isinstance(value, (int, float))
                    else f"{'-':>{width}}")
        print(row)

    verdict = results.get("fusion_verdict")
    if verdict:
        print(f"\nFusion verdict: {verdict['note']}")
    weights = results.get("late_fusion_modality_weights")
    if weights:
        print("Late-fusion modality weights: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(weights.items(), key=lambda kv: -kv[1])))
    for warning in results.get("warnings", [])[:8]:
        print(f"  ! {warning}")
    print(f"\nArtefacts: {results['run_directory']}")
    print("=" * 72)


def cmd_physiocgm(args: argparse.Namespace) -> int:
    """PhysioCGM: ECG + PPG + EDA + motion + respiration against CGM glucose."""
    from src.data.physiocgm import PhysioCGMLoader, describe

    if args.inspect_only:
        info = describe(args.data_root)
        print(f"\nPhysioCGM inventory for {info['data_root']}")
        print(f"Subjects: {info['n_subjects']} | segments: {info['total_segments']}\n")
        print(f"{'subject':<12}segments")
        print("-" * 24)
        for subject in info["subjects"]:
            print(f"{subject['subject_id']:<12}{subject['n_segments']}")
        if not info["n_subjects"]:
            print("\nNothing found. This loader reads the PROCESSED segments "
                  "(dataset/processed/<subject>/<n>.pkl). If you only have "
                  "dataset/raw, run PhysioCGM's own preprocess.py first.")
            return 1
        return 0

    cfg_yaml = _apply_overrides(load_config(args.config), args)
    loader = PhysioCGMLoader(
        args.data_root, horizon_steps=args.horizon_steps, use_ecg=not args.no_ecg,
        max_subjects=args.max_subjects,
        max_segments_per_subject=args.max_segments,
    )
    X_df, y, groups, times = loader.load()

    if args.no_persistence and "glucose_now" in X_df.columns:
        X_df = X_df.drop(columns=["glucose_now"])
        print("Dropped 'glucose_now': measuring the sensors without persistence.")

    print(f"\n{loader.report.n_segments} segment pairs x {X_df.shape[1]} features "
          f"from {loader.report.n_subjects} subjects")
    print(f"Horizon: {loader.report.horizon_minutes:.0f} min | "
          f"target {y.mean():.1f} +/- {y.std():.1f} mg/dL")
    for caveat in loader.report.caveats:
        print(f"  ! {caveat}")

    blocks = loader.modality_blocks(list(X_df.columns))
    print("\nModality blocks: " + ", ".join(f"{k}({len(v)})" for k, v in blocks.items()))

    if args.export_features:
        _export_features(X_df, y, groups, times, args.export_features, "glucose")
        if args.export_only:
            return 0

    if args.model == "fusion":
        from src.multimodal_pipeline import MultimodalConfig, run_multimodal_experiment

        frame = X_df.copy()
        frame["__target__"] = y
        modalities = {"tabular": list(X_df.columns)}
        cfg = MultimodalConfig(
            target="__target__", task="regression", fusion=args.fusion,
            embedding_dim=args.embedding_dim, epochs=args.epochs or 40,
            batch_size=args.batch_size, cv_type=args.cv_type or "group",
            seed=args.seed, results_dir=args.results_dir or "./run_results",
            run_name=args.run_name, run_modality_ablation=False,
        )
        # One encoder per sensor family, so fusion operates over real modalities.
        results = run_multimodal_experiment(
            frame, cfg, {k: v for k, v in blocks.items()}, groups=groups
        )
        _print_multimodal_report(results)
        return 0

    ev = cfg_yaml.get("evaluation", {})
    exp = ExperimentConfig(
        task="regression", target="glucose", domain="diabetes",
        cv_type=args.cv_type or "group", test_size=ev.get("test_size", 0.2),
        n_folds=ev.get("n_folds", 5), seed=args.seed,
        results_dir=args.results_dir or cfg_yaml.get("output", {}).get(
            "results_dir", "./run_results"),
        run_name=args.run_name, skip_transformer=args.skip_transformer,
        skip_baselines=args.skip_baselines, skip_importance=args.skip_importance,
        bootstrap_iterations=ev.get("bootstrap_iterations", 1000),
        units="mg/dL", source="physiocgm",
        tune_hyperparameters=args.tune, tune_iterations=args.tune_iterations,
        publication_figures=not args.no_publication_figures,
        stats_tables=not args.no_stats_tables, export_zip=not args.no_export,
        figure_formats=tuple(args.figure_formats),
    )
    results = run_experiment(
        X_df, y, exp,
        model_config={**cfg_yaml.get("preprocessing", {}), **cfg_yaml.get("models", {})},
        qsqfs_config=cfg_yaml["feature_selection"]["qsqfs"],
        groups=groups, order=times.to_numpy(), raw_df=None,
        dataset_notes={"physiocgm": loader.report.to_dict()},
    )
    _print_report(results, "regression")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="List what the loader finds under a D1NAMO folder")
    p.add_argument("--data_root", required=True)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("tabular", help="Run on any CSV/Excel/Parquet table")
    p.add_argument("--data_path", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    p.add_argument("--domain", default="generic",
                   help="Leakage rule set: generic, diabetes, mortality, readmission, sepsis")
    p.add_argument("--group_col", default=None, help="Keep this entity out of both splits")
    p.add_argument("--time_col", default=None, help="Ordering key for time_series splits")
    p.add_argument("--drop_flagged", action="store_true",
                   help="Also drop columns flagged as possible proxies")
    _shared_args(p)
    p.set_defaults(func=cmd_tabular)

    p = sub.add_parser("d1namo", help="Run on the D1NAMO wearable dataset")
    p.add_argument("--data_root", required=True)
    p.add_argument("--window", type=int, default=30, help="Window length, minutes")
    p.add_argument("--step", type=int, default=10, help="Step between windows, minutes")
    p.add_argument("--horizon", type=int, default=30, help="Prediction horizon, minutes")
    p.add_argument("--max_gap", type=float, default=7.5,
                   help="Max minutes between the target time and a real CGM reading")
    p.add_argument("--use_ecg", action="store_true",
                   help="Parse raw 250 Hz ECG too (slow, several GB)")
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--cohorts", nargs="+", default=None, choices=["diabetes", "healthy"])
    p.add_argument("--export_features", default=None, metavar="PATH",
                   help="Write the extracted feature table to CSV (upload this "
                        "to the Streamlit app)")
    p.add_argument("--export_only", action="store_true",
                   help="Export the features and stop, without training")
    _shared_args(p)
    p.set_defaults(func=cmd_d1namo)

    p = sub.add_parser("ablation", help="Measure what each QSQ-FS mechanism contributes")
    p.add_argument("--data_path", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    p.add_argument("--group_col", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    p.add_argument("--variants", nargs="+", default=None)
    _shared_args(p)
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("multimodal",
                       help="Manifest-driven multimodal run (tabular/text/image/genomic)")
    p.add_argument("--manifest", required=True, help="CSV, one row per subject")
    p.add_argument("--target", required=True)
    p.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    p.add_argument("--fusion", default="early", choices=["early", "late", "hybrid"])
    p.add_argument("--modalities", nargs="+", default=None,
                   choices=["tabular", "text", "image", "genomic"],
                   help="Restrict to these modalities (for ablation)")
    p.add_argument("--image_cols", nargs="+", default=None)
    p.add_argument("--text_cols", nargs="+", default=None)
    p.add_argument("--genomic_cols", nargs="+", default=None)
    p.add_argument("--image_root", default=None,
                   help="Base for relative image paths (default: manifest folder)")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--group_col", default=None)
    p.add_argument("--id_col", default=None)
    p.add_argument("--skip_modality_ablation", action="store_true")
    p.add_argument("--force_text_fallback", action="store_true",
                   help="Use the offline hashed bag-of-words text encoder")
    _shared_args(p)
    p.set_defaults(func=cmd_multimodal)

    p = sub.add_parser("physiocgm",
                       help="PhysioCGM: ECG + PPG + EDA + motion vs CGM glucose")
    p.add_argument("--data_root", required=True,
                   help="Folder containing dataset/processed/<subject>/<n>.pkl")
    p.add_argument("--inspect_only", action="store_true",
                   help="List subjects and segment counts, then exit")
    p.add_argument("--horizon_steps", type=int, default=6,
                   help="CGM segments ahead to predict (each is 5 min; 6 = 30 min)")
    p.add_argument("--model", default="qsqfs", choices=["qsqfs", "fusion"],
                   help="qsqfs: selection + Transformer. fusion: per-modality encoders")
    p.add_argument("--fusion", default="early", choices=["early", "late", "hybrid"])
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_ecg", action="store_true", help="Skip raw ECG (faster)")
    p.add_argument("--no_persistence", action="store_true",
                   help="Drop glucose_now, so the sensors are measured alone")
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--max_segments", type=int, default=None)
    p.add_argument("--export_features", default=None, metavar="PATH",
                   help="Write the extracted feature table to CSV (upload this "
                        "to the Streamlit app)")
    p.add_argument("--export_only", action="store_true",
                   help="Export the features and stop, without training")
    _shared_args(p)
    p.set_defaults(func=cmd_physiocgm)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(logging.WARNING if getattr(args, "quiet", False) else logging.INFO)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
