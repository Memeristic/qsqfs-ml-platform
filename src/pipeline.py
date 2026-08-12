"""End-to-end experiment orchestration.

Strict ordering, because every step after the split can leak if run before it:

    1. load
    2. validate
    3. split                      <- nothing has touched the test set yet
    4. leakage screen (train rows only)
    5. fit preprocessing on train, transform test
    6. QSQ-FS on train only
    7. fit models on selected train features
    8. score once on test

The test set is touched exactly once, at step 8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.splitting import check_group_disjoint, make_splits
from src.data.validator import DataValidator, naive_baseline
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import compute_metrics, primary_metric
from src.explainability.importance import permutation_importance
from src.feature_selection.qsqfs import QSQFS
from src.leakage.detector import LeakageDetector, apply_exclusions
from src.models.baselines import run_baselines
from src.preprocessing.pipeline import PreprocessingPipeline
from src.reporting import plots
from src.reporting.results import ResultsWriter
from src.utils.run_metadata import get_metadata
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    task: str = "classification"
    target: str = "target"
    domain: str = "generic"
    cv_type: str = "stratified"
    test_size: float = 0.2
    gap: int = 0
    n_folds: int = 5
    seed: int = 42
    results_dir: str = "./run_results"
    run_name: Optional[str] = None
    drop_flagged: bool = False
    skip_transformer: bool = False
    skip_baselines: bool = False
    skip_importance: bool = False
    bootstrap_iterations: int = 1000
    units: str = ""
    source: str = "tabular"
    figure_format: str = "png"
    # Publication extras
    publication_figures: bool = True
    figure_formats: tuple = ("png", "pdf")
    stats_tables: bool = True
    tune_hyperparameters: bool = False
    tune_iterations: int = 20
    tune_threshold: bool = True
    export_zip: bool = True


def run_experiment(
    X_df: pd.DataFrame,
    y: np.ndarray,
    config: ExperimentConfig,
    model_config: Dict,
    qsqfs_config: Dict,
    groups: Optional[np.ndarray] = None,
    order: Optional[Sequence] = None,
    raw_df: Optional[pd.DataFrame] = None,
    dataset_notes: Optional[Dict] = None,
) -> Dict:
    set_seed(config.seed)
    is_classification = config.task == "classification"
    writer = ResultsWriter(config.results_dir, config.run_name)
    warnings: List[str] = []

    results: Dict = {
        "run_directory": str(writer.run_dir),
        "task": config.task,
        "seed": config.seed,
        "metadata": get_metadata({"source": config.source}),
        "dataset": {
            "source": config.source,
            "target": config.target,
            "n_rows": int(len(X_df)),
            "n_features_raw": int(X_df.shape[1]),
            "n_groups": int(len(np.unique(groups))) if groups is not None else None,
            **(dataset_notes or {}),
        },
    }

    # ---------------- 2. validate --------------------------------------
    if raw_df is not None:
        report = DataValidator(raw_df, config.target).validate()
        results["validation"] = report.to_dict()
        warnings.extend(report.messages)
        if report.has_critical:
            raise ValueError("Critical data problems:\n  " + "\n  ".join(report.messages))

    class_labels = None
    if is_classification:
        labels = np.unique(y)
        if len(labels) < 2:
            raise ValueError("Classification needs at least two classes in the target.")
        class_labels = {str(v): int(i) for i, v in enumerate(labels)}
        y = np.searchsorted(labels, y)
        results["dataset"]["class_mapping"] = class_labels

    results["naive_baseline"] = naive_baseline(y, is_classification)
    logger.info("Naive baseline to beat: %s", results["naive_baseline"])

    # ---------------- 3. split -----------------------------------------
    split = make_splits(
        n_samples=len(X_df), y=y, groups=groups, cv_type=config.cv_type,
        test_size=config.test_size, n_splits=config.n_folds, gap=config.gap,
        random_state=config.seed, is_classification=is_classification,
        order=np.asarray(order) if order is not None else None,
    )
    overlap = check_group_disjoint(split.train_idx, split.test_idx, groups)
    results["split"] = {**split.summary(), "group_overlap": overlap}
    warnings.extend(split.notes)
    if overlap:
        warnings.append(
            f"{len(overlap)} group(s) appear in both train and test. "
            "Use --cv_type group to prevent this."
        )

    train_idx, test_idx = split.train_idx, split.test_idx
    X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = np.asarray(groups)[train_idx] if groups is not None else None

    # ---------------- 4. leakage screen (train rows only) ---------------
    if raw_df is not None:
        detector = LeakageDetector()
        leakage = detector.detect(
            raw_df, config.target, domain=config.domain, train_index=list(train_idx)
        )
        results["leakage"] = leakage.to_dict()
        writer.save_text(leakage.text(), "leakage_report.txt")
        drop = list(leakage.excluded_columns)
        if config.drop_flagged:
            drop += [f["column"] for f in leakage.flagged_columns]
        drop = [c for c in dict.fromkeys(drop) if c in X_train_df.columns]
        if drop:
            logger.warning("Dropping %d leaking column(s): %s", len(drop), drop)
            X_train_df = X_train_df.drop(columns=drop)
            X_test_df = X_test_df.drop(columns=drop)
        warnings.extend(leakage.warnings)

    if X_train_df.shape[1] < 2:
        raise ValueError("Fewer than 2 usable features remain after leakage screening.")

    # ---------------- 5. preprocess (fit on train only) -----------------
    preprocessor = PreprocessingPipeline(
        impute_strategy=model_config.get("impute_strategy", "median"),
        scaler_type=model_config.get("scaler", "standard"),
    )
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    feature_names = preprocessor.get_feature_names() or [
        f"feature_{i}" for i in range(X_train.shape[1])
    ]
    results["preprocessing"] = {
        "n_features_after_encoding": int(X_train.shape[1]),
        "fitted_on": "training rows only",
        "scaler": model_config.get("scaler", "standard"),
        "impute_strategy": model_config.get("impute_strategy", "median"),
    }

    # ---------------- 6. QSQ-FS (train only) ---------------------------
    selector = QSQFS(
        n_features=X_train.shape[1], task=config.task, seed=config.seed,
        population_size=qsqfs_config.get("population_size", 30),
        n_iterations=qsqfs_config.get("n_iterations", 40),
        alpha=qsqfs_config.get("alpha", 0.85),
        w_ai=qsqfs_config.get("w_ai", 0.5),
        beta=qsqfs_config.get("beta", 0.7),
        delta=qsqfs_config.get("delta", 0.97),
        min_fraction=qsqfs_config.get("min_fraction", 0.05),
        max_fraction=qsqfs_config.get("max_fraction", 0.5),
        use_quorum_sensing=qsqfs_config.get("use_quorum_sensing", True),
        use_quorum_quenching=qsqfs_config.get("use_quorum_quenching", True),
        use_elitism=qsqfs_config.get("use_elitism", True),
        use_cache=qsqfs_config.get("use_cache", True),
        estimator=qsqfs_config.get("estimator", "knn"),
        cv_folds=qsqfs_config.get("cv_folds", 5),
        stagnation_window=qsqfs_config.get("stagnation_window", 12),
        diversity_injection_rate=qsqfs_config.get("diversity_injection_rate", 0.25),
        early_stop_patience=qsqfs_config.get("early_stop_patience"),
        n_jobs=qsqfs_config.get("n_jobs", 1),
    )
    mask = selector.fit(X_train, y_train, groups=groups_train, feature_names=feature_names)
    selected = selector.selected_names(feature_names)
    results["feature_selection"] = selector.stats()
    results["feature_selection"]["selected_features"] = selected
    results["qsqfs_history"] = selector.history.to_dict()

    X_train_sel, X_test_sel = X_train[:, mask], X_test[:, mask]
    logger.info("QSQ-FS kept %d/%d features.", len(selected), X_train.shape[1])
    writer.save_selected_features(selected, mask, feature_names)

    # ---------------- 7-8. models and one-shot test scoring -------------
    if not config.skip_baselines:
        logger.info("Fitting baselines on the selected features...")
        results["baselines"] = run_baselines(
            X_train_sel, y_train, X_test_sel, y_test,
            task=config.task, models=model_config.get("baselines"), seed=config.seed,
        )

    y_pred = y_proba = None
    trainer = None
    if not config.skip_transformer:
        try:
            import torch  # noqa: F401
            from src.models.trainer import TransformerTrainer
            from src.models.transformer import TabularTransformer

            tcfg = model_config.get("transformer", {})
            model = TabularTransformer(
                n_numerical=X_train_sel.shape[1],
                n_classes=len(np.unique(y_train)) if is_classification else 1,
                embedding_dim=tcfg.get("embedding_dim", 64),
                n_heads=tcfg.get("n_heads", 4),
                n_layers=tcfg.get("n_layers", 3),
                dropout=tcfg.get("dropout", 0.1),
                use_cls_token=tcfg.get("use_cls_token", True),
                regression=not is_classification,
            )
            trainer = TransformerTrainer(
                model=model,
                learning_rate=tcfg.get("learning_rate", 1e-3),
                weight_decay=tcfg.get("weight_decay", 1e-4),
                batch_size=tcfg.get("batch_size", 64),
                epochs=tcfg.get("epochs", 100),
                patience=tcfg.get("early_stopping_patience", 10),
                val_split=tcfg.get("val_split", 0.2),
                seed=config.seed,
            )
            history = trainer.fit(X_train_sel, y_train)
            raw = trainer.predict(X_test_sel)
            if is_classification:
                y_proba = raw[:, 1] if raw.shape[1] == 2 else raw
                y_pred = raw.argmax(axis=1)
            else:
                y_pred = raw
            results["transformer"] = compute_metrics(y_test, y_pred, y_proba, task=config.task)
            results["transformer_training"] = {
                "epochs_run": trainer.epochs_run,
                "best_epoch": trainer.best_epoch,
                "best_val_loss": trainer.best_val_loss,
                "n_parameters": model.count_parameters(),
                "device": str(trainer.device),
            }
            writer.register_figure(plots.plot_training_curves(
                history, writer.figures_dir, config.figure_format))
        except ImportError:
            warnings.append("PyTorch is not installed; the transformer was skipped.")
            logger.warning("PyTorch missing - skipping the transformer.")
        except Exception as exc:
            warnings.append(f"Transformer training failed: {exc}")
            logger.exception("Transformer training failed.")

    # ---------------- uncertainty and importance ------------------------
    if y_pred is not None and config.bootstrap_iterations > 0:
        metric_name = primary_metric(config.task)
        if is_classification and y_proba is not None:
            from sklearn.metrics import roc_auc_score
            ci = bootstrap_ci(y_test, y_pred, lambda a, b, c: roc_auc_score(a, c),
                              config.bootstrap_iterations, seed=config.seed, y_proba=y_proba)
        else:
            from sklearn.metrics import mean_squared_error
            ci = bootstrap_ci(y_test, y_pred,
                              lambda a, b: float(np.sqrt(mean_squared_error(a, b))),
                              config.bootstrap_iterations, seed=config.seed)
        results["bootstrap_ci"] = {"metric": metric_name, **ci}

    if trainer is not None and not config.skip_importance:
        logger.info("Computing permutation importance on the test set...")
        try:
            from sklearn.metrics import balanced_accuracy_score, mean_squared_error
            if is_classification:
                predict_fn = lambda M: trainer.predict(M).argmax(axis=1)  # noqa: E731
                score_fn = balanced_accuracy_score
            else:
                predict_fn = trainer.predict
                score_fn = lambda a, b: -float(np.sqrt(mean_squared_error(a, b)))  # noqa: E731
            importance = permutation_importance(
                predict_fn, X_test_sel, y_test, score_fn,
                feature_names=selected, n_repeats=model_config.get("n_permutations", 15),
                seed=config.seed,
            )
            results["feature_importance"] = importance
            writer.register_figure(plots.plot_feature_importance(
                importance["importances"], writer.figures_dir, fmt=config.figure_format))
        except Exception as exc:
            warnings.append(f"Permutation importance failed: {exc}")

    # ---------------- figures and artefacts -----------------------------
    writer.register_figure(plots.plot_convergence(
        results["qsqfs_history"], writer.figures_dir, config.figure_format))
    writer.register_figure(plots.plot_target_distribution(
        y, writer.figures_dir, config.task, config.units, config.figure_format))

    if y_pred is not None:
        if is_classification:
            writer.register_figure(plots.plot_classification_diagnostics(
                y_test, y_pred, y_proba, writer.figures_dir, config.figure_format))
        else:
            writer.register_figure(plots.plot_regression_diagnostics(
                y_test, y_pred, writer.figures_dir, config.units, config.figure_format))
        writer.save_predictions(y_test, y_pred, y_proba)

    comparison = dict(results.get("baselines", {}))
    if results.get("transformer"):
        comparison["qsqfs_transformer"] = results["transformer"]
    if len(comparison) >= 2:
        metric = primary_metric(config.task)
        writer.register_figure(plots.plot_model_comparison(
            comparison, metric, writer.figures_dir,
            lower_is_better=(metric == "rmse"), fmt=config.figure_format))

    # ---------------- hyperparameter tuning (training data only) --------
    if config.tune_hyperparameters:
        from src.tuning import tune_decision_threshold, tune_model
        from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

        logger.info("Tuning baseline hyperparameters by inner CV (training rows only)...")
        if groups_train is not None and len(np.unique(groups_train)) >= 3:
            inner_cv = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
        elif is_classification:
            inner_cv = StratifiedKFold(5, shuffle=True, random_state=config.seed)
        else:
            inner_cv = KFold(5, shuffle=True, random_state=config.seed)

        tuned: Dict = {}
        for name in ("random_forest", "hist_gradient_boosting", "svm"):
            try:
                outcome = tune_model(
                    name, X_train_sel, y_train, task=config.task, cv=inner_cv,
                    groups=groups_train, n_iter=config.tune_iterations,
                    seed=config.seed,
                )
                if "error" in outcome:
                    continue
                model = outcome.pop("model")
                predictions = model.predict(X_test_sel)
                proba = None
                if is_classification and hasattr(model, "predict_proba"):
                    raw = model.predict_proba(X_test_sel)
                    proba = raw[:, 1] if raw.shape[1] == 2 else raw
                outcome["metrics"] = compute_metrics(
                    y_test, predictions, proba, task=config.task)
                tuned[name] = outcome
            except Exception as exc:
                logger.warning("Tuning %s failed: %s", name, exc)
        if tuned:
            results["tuned_models"] = tuned
            results["tuning_note"] = (
                "Hyperparameters were selected by cross-validation on training "
                "rows only. The test set was scored once, afterwards."
            )

    if config.tune_threshold and is_classification and y_proba is not None:
        # Tuned on VALIDATION-fold predictions would be ideal; with a single
        # split we report the curve and the default so the trade-off is visible,
        # and we do NOT silently swap in the tuned threshold.
        from src.tuning import tune_decision_threshold
        try:
            results["threshold_analysis"] = tune_decision_threshold(
                y_test, y_proba, metric="f1")
            results["threshold_analysis"]["caution"] = (
                "This curve was computed on the test set for illustration. Do not "
                "report the tuned-threshold score as a headline result -- choose "
                "the threshold on validation data and apply it unchanged."
            )
        except Exception as exc:
            warnings.append(f"Threshold analysis failed: {exc}")

    # ---------------- statistical tables --------------------------------
    tables: Dict = {}
    if config.stats_tables:
        try:
            from src.reporting import stats_tables as st

            feature_frame = pd.DataFrame(X_train_sel, columns=selected)
            feature_frame["__target__"] = y_train
            tables["descriptive_statistics"] = st.descriptive_table(
                pd.DataFrame(X_train_sel, columns=selected))
            tables["correlation_with_target"] = st.correlation_table(
                feature_frame, "__target__", top_n=40)
            tables["model_comparison"] = st.model_comparison_table(results, config.task)
            if is_classification:
                grouped = pd.DataFrame(X_train_sel, columns=selected)
                grouped["group"] = y_train
                tables["group_comparison"] = st.group_comparison_table(
                    grouped, "group", columns=selected[:40])
            results["statistical_tables"] = {
                name: table.to_dict("records") for name, table in tables.items()
                if isinstance(table, pd.DataFrame) and len(table) <= 200
            }
        except Exception as exc:
            warnings.append(f"Statistical tables failed: {exc}")
            logger.exception("Statistical tables failed.")

    # ---------------- publication figures -------------------------------
    if config.publication_figures:
        try:
            from src.reporting import publication as pub

            pub_dir = writer.run_dir / "figures"
            written = pub.generate_all(
                results, pub_dir, config.task, y_true=y_test, y_pred=y_pred,
                y_proba=y_proba, features_df=pd.DataFrame(X_train_sel, columns=selected),
                units=config.units, formats=tuple(config.figure_formats),
            )
            results["publication_figures"] = {
                k: [Path(p).name for p in v] for k, v in written.items()
            }
            for paths in written.values():
                for path in paths:
                    if path.endswith(".png"):
                        writer.register_figure(Path(path))
        except Exception as exc:
            warnings.append(f"Publication figures failed: {exc}")
            logger.exception("Publication figures failed.")

    results["warnings"] = warnings
    results["figures"] = writer.figures
    writer.save_results(results)
    writer.write_summary(results)

    # ---------------- export bundle -------------------------------------
    if config.export_zip:
        try:
            from src.reporting.export import export_run

            archive = export_run(writer.run_dir, results=results, tables=tables)
            results["export_zip"] = str(archive)
            logger.info("Export bundle: %s", archive)
        except Exception as exc:
            warnings.append(f"Export bundle failed: {exc}")
    logger.info("Artefacts written to %s", writer.run_dir)
    results["_writer"] = writer
    return results
