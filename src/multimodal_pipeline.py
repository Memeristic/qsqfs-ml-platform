"""Multimodal experiment orchestration.

Same discipline as the tabular pipeline: split first, fit everything on the
training rows only, score once on the test set. Two things are added because
they are the questions a multimodal result has to answer.

**Per-modality ablation.** The full model is compared against one model per
single modality. If the image-only model matches the fused model, fusion
bought you nothing and the honest report says so.

**A tabular-only reference.** Multimodal models are frequently beaten by a
gradient-boosted tree on the tabular block alone. That reference is computed by
default so the comparison cannot be quietly omitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.multimodal import MultimodalDataset, build_schema, infer_modalities
from src.data.splitting import check_group_disjoint, make_splits
from src.data.validator import naive_baseline
from src.evaluation.metrics import compute_metrics, primary_metric
from src.models.baselines import run_baselines
from src.models.multimodal import MultimodalModel
from src.models.multimodal_trainer import MultimodalTrainer
from src.preprocessing.pipeline import PreprocessingPipeline
from src.reporting import plots
from src.reporting.results import ResultsWriter
from src.utils.run_metadata import get_metadata
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)


@dataclass
class MultimodalConfig:
    target: str = "target"
    task: str = "auto"
    fusion: str = "early"
    embedding_dim: int = 64
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    patience: int = 8
    image_size: int = 224
    image_root: Optional[str] = None
    cv_type: str = "stratified"
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42
    results_dir: str = "./run_results"
    run_name: Optional[str] = None
    genomic_encoding: str = "auto"
    run_modality_ablation: bool = True
    force_text_fallback: bool = False


def _fit_and_score(schema, df, y, matrices, cfg, train_idx, val_idx,
                   test_idx, is_classification, n_classes, label: str) -> Dict:
    dataset = MultimodalDataset(
        df, schema, y, matrices={k: v for k, v in matrices.items() if k in schema},
        image_root=cfg.image_root, image_size=cfg.image_size,
        regression=not is_classification,
    )
    if cfg.force_text_fallback and "text" in schema:
        schema["text"]["force_fallback"] = True

    model = MultimodalModel(
        schema=schema, n_classes=n_classes, regression=not is_classification,
        embedding_dim=cfg.embedding_dim, fusion_type=cfg.fusion,
    )
    trainer = MultimodalTrainer(
        model, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
        epochs=cfg.epochs, patience=cfg.patience, seed=cfg.seed, verbose=True,
    )
    logger.info("[%s] modalities=%s params=%s", label, sorted(schema),
                model.count_parameters())
    history = trainer.fit(dataset, train_idx, val_idx)
    raw = trainer.predict(dataset, test_idx)

    if is_classification:
        proba = raw[:, 1] if raw.ndim == 2 and raw.shape[1] == 2 else raw
        pred = raw.argmax(axis=1)
    else:
        proba, pred = None, raw

    metrics = compute_metrics(y[test_idx], pred, proba,
                              task="classification" if is_classification else "regression")
    return {
        "metrics": metrics, "history": history, "predictions": pred, "proba": proba,
        "model_description": model.describe(),
        "modality_weights": model.fusion.modality_weights(),
        "epochs_run": trainer.epochs_run, "best_epoch": trainer.best_epoch,
    }


def run_multimodal_experiment(
    df: pd.DataFrame,
    cfg: MultimodalConfig,
    modalities: Dict[str, List[str]],
    groups: Optional[np.ndarray] = None,
) -> Dict:
    set_seed(cfg.seed)
    writer = ResultsWriter(cfg.results_dir, cfg.run_name)
    warnings: List[str] = []

    from src.data.loader import infer_task

    task = cfg.task if cfg.task != "auto" else infer_task(df[cfg.target])
    is_classification = task == "classification"

    y = df[cfg.target].to_numpy()
    if is_classification:
        labels = np.unique(y)
        y = np.searchsorted(labels, y)
        n_classes = len(labels)
    else:
        y = y.astype(float)
        n_classes = 1

    schema = build_schema(modalities, df, cfg.genomic_encoding)
    if not schema:
        raise ValueError("No modalities were resolved from the manifest.")
    logger.info("Resolved schema: %s",
                {k: (v.get("input_dim") or v.get("columns")) for k, v in schema.items()})

    results: Dict = {
        "task": task, "seed": cfg.seed, "run_directory": str(writer.run_dir),
        "metadata": get_metadata({"source": "multimodal"}),
        "dataset": {
            "source": "multimodal manifest", "target": cfg.target,
            "n_rows": int(len(df)), "modalities": {k: len(v) for k, v in modalities.items()},
            "schema": {k: {kk: vv for kk, vv in v.items() if kk != "columns"}
                       for k, v in schema.items()},
            "modality_columns": modalities,
        },
        "naive_baseline": naive_baseline(y, is_classification),
        "fusion_type": cfg.fusion,
    }

    # ---- split -------------------------------------------------------
    split = make_splits(
        len(df), y, groups, cv_type=cfg.cv_type, test_size=cfg.test_size,
        random_state=cfg.seed, is_classification=is_classification,
    )
    inner = make_splits(
        len(split.train_idx), y[split.train_idx],
        groups[split.train_idx] if groups is not None else None,
        cv_type=cfg.cv_type, test_size=cfg.val_size, random_state=cfg.seed,
        is_classification=is_classification,
    )
    train_idx = split.train_idx[inner.train_idx]
    val_idx = split.train_idx[inner.test_idx]
    test_idx = split.test_idx
    overlap = check_group_disjoint(split.train_idx, test_idx, groups)
    results["split"] = {**split.summary(), "n_val": len(val_idx), "group_overlap": overlap}
    if overlap:
        warnings.append(f"{len(overlap)} group(s) span train and test.")

    # ---- preprocess every numeric block on training rows only ---------
    matrices: Dict[str, np.ndarray] = {}
    for name, spec in schema.items():
        columns = spec.get("columns", [])
        if spec["type"] == "genomic":
            block = df[columns].to_numpy(dtype=float)
            if spec.get("encoding") == "snp":
                matrices[name] = np.nan_to_num(block, nan=0.0).clip(0, 2)
            else:
                mean = np.nanmean(block[train_idx], axis=0)
                std = np.nanstd(block[train_idx], axis=0)
                std[std < 1e-9] = 1.0
                matrices[name] = np.nan_to_num((block - mean) / std, nan=0.0)
        elif spec["type"] == "tabular":
            preprocessor = PreprocessingPipeline()
            preprocessor.fit(df.iloc[train_idx][columns])
            matrices[name] = preprocessor.transform(df[columns])
            spec["input_dim"] = matrices[name].shape[1]

    tabular_matrix = matrices.get("tabular")

    # ---- full fused model --------------------------------------------
    full = _fit_and_score(dict(schema), df, y, matrices, cfg,
                          train_idx, val_idx, test_idx, is_classification, n_classes,
                          "fused:" + "+".join(sorted(schema)))
    results["multimodal"] = full["metrics"]
    results["model"] = full["model_description"]
    results["training"] = {"epochs_run": full["epochs_run"], "best_epoch": full["best_epoch"]}
    if full["modality_weights"]:
        results["late_fusion_modality_weights"] = full["modality_weights"]

    # ---- per-modality ablation ---------------------------------------
    if cfg.run_modality_ablation and len(schema) > 1:
        logger.info("Running per-modality ablation (%d single-modality models)...", len(schema))
        ablation = {}
        for name in schema:
            single = _fit_and_score(
                {name: dict(schema[name])}, df, y, matrices, cfg,
                train_idx, val_idx, test_idx, is_classification, n_classes, f"only:{name}",
            )
            ablation[f"{name}_only"] = single["metrics"]
        results["modality_ablation"] = ablation
        metric = primary_metric(task)
        fused_value = full["metrics"].get(metric)
        best_single = max(
            ((k, v.get(metric)) for k, v in ablation.items() if v.get(metric) is not None),
            key=lambda kv: kv[1] if metric != "rmse" else -kv[1], default=(None, None),
        )
        if fused_value is not None and best_single[1] is not None:
            improved = (fused_value > best_single[1] if metric != "rmse"
                        else fused_value < best_single[1])
            results["fusion_verdict"] = {
                "metric": metric, "fused": fused_value,
                "best_single_modality": best_single[0],
                "best_single_value": best_single[1],
                "fusion_helped": bool(improved),
                "note": (
                    "Fusion beat every single-modality model on this metric."
                    if improved else
                    f"Fusion did NOT beat '{best_single[0]}' alone. On this data the "
                    "extra modalities are not contributing; report the simpler model."
                ),
            }

    # ---- tabular-only classical reference -----------------------------
    if tabular_matrix is not None:
        logger.info("Fitting classical baselines on the tabular block alone...")
        results["tabular_only_baselines"] = run_baselines(
            tabular_matrix[train_idx], y[train_idx],
            tabular_matrix[test_idx], y[test_idx],
            task=task, seed=cfg.seed,
        )
        warnings.append(
            "Compare the fused model against tabular_only_baselines. A gradient-"
            "boosted tree on tabular data alone often matches or beats multimodal "
            "fusion on cohorts of this size."
        )

    # ---- figures and artefacts ----------------------------------------
    writer.register_figure(plots.plot_training_curves(full["history"], writer.figures_dir))
    writer.register_figure(plots.plot_target_distribution(y, writer.figures_dir, task))
    if is_classification:
        writer.register_figure(plots.plot_classification_diagnostics(
            y[test_idx], full["predictions"], full["proba"], writer.figures_dir))
    else:
        writer.register_figure(plots.plot_regression_diagnostics(
            y[test_idx], full["predictions"], writer.figures_dir))

    comparison = {"multimodal_fused": full["metrics"]}
    comparison.update(results.get("modality_ablation", {}))
    comparison.update({f"tabular_{k}": v
                       for k, v in results.get("tabular_only_baselines", {}).items()
                       if isinstance(v, dict) and "error" not in v})
    if len(comparison) >= 2:
        metric = primary_metric(task)
        writer.register_figure(plots.plot_model_comparison(
            comparison, metric, writer.figures_dir, lower_is_better=(metric == "rmse")))

    writer.save_predictions(y[test_idx], full["predictions"], full["proba"])
    results["warnings"] = warnings
    results["figures"] = writer.figures
    writer.save_results(results)
    logger.info("Artefacts written to %s", writer.run_dir)
    return results
