"""Regression tests. Several of these lock in fixes for bugs that previously
made the pipeline crash or silently report meaningless numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import infer_task
from src.data.splitting import check_group_disjoint, make_splits
from src.data.validator import DataValidator, naive_baseline
from src.evaluation.metrics import compute_metrics
from src.feature_selection.qsqfs import QSQFS
from src.leakage.detector import LeakageDetector
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.jsonio import to_jsonable


# ---------------------------------------------------------------- task
@pytest.mark.parametrize("values,expected", [
    ([0, 1, 0, 1] * 20, "classification"),
    ([0.0, 1.0, 0.0, 1.0] * 20, "classification"),          # float 0/1
    (["yes", "no", "yes", "no"] * 20, "classification"),     # string labels
    ([True, False] * 40, "classification"),
    (list(np.random.RandomState(0).randn(100)), "regression"),
])
def test_infer_task(values, expected):
    assert infer_task(pd.Series(values)) == expected


# ---------------------------------------------------------------- json
def test_json_safety_of_numpy_keys_and_nonfinite():
    payload = {np.int64(7): np.float64("nan"), "arr": np.arange(3), "inf": -np.inf}
    json.dumps(to_jsonable(payload))  # must not raise


# ---------------------------------------------------------------- qsqfs
def test_best_fitness_is_populated():
    """Regression test: best_fitness used to stay at -inf, poisoning every
    results file with an unserialisable value."""
    rng = np.random.RandomState(0)
    X, y = rng.randn(150, 12), rng.randint(0, 2, 150)
    selector = QSQFS(12, population_size=8, n_iterations=4, task="classification", verbose=False)
    selector.fit(X, y)
    assert np.isfinite(selector.best_fitness)
    assert selector.best_fitness > float("-inf")
    assert selector.best_mask.sum() >= 1
    json.dumps(to_jsonable(selector.stats()))


def test_qsqfs_respects_subset_bounds():
    rng = np.random.RandomState(1)
    X, y = rng.randn(120, 20), rng.randint(0, 2, 120)
    selector = QSQFS(20, population_size=8, n_iterations=4, n_min=4, n_max=9,
                     task="classification", verbose=False)
    mask = selector.fit(X, y)
    assert 4 <= mask.sum() <= 9


def test_qsqfs_is_deterministic_under_a_fixed_seed():
    rng = np.random.RandomState(2)
    X, y = rng.randn(120, 15), rng.randint(0, 2, 120)
    masks = []
    for _ in range(2):
        s = QSQFS(15, population_size=8, n_iterations=5, seed=123,
                  task="classification", verbose=False)
        masks.append(s.fit(X, y))
    assert np.array_equal(masks[0], masks[1])


def test_qsqfs_regression_path():
    rng = np.random.RandomState(3)
    X = rng.randn(150, 10)
    y = X[:, 0] * 2 + X[:, 3] - rng.randn(150) * 0.1
    selector = QSQFS(10, population_size=8, n_iterations=6, task="regression", verbose=False)
    selector.fit(X, y)
    assert 0.0 <= selector.best_score <= 1.0


# ---------------------------------------------------------------- splits
def test_group_split_keeps_groups_disjoint():
    y = np.random.RandomState(4).randint(0, 2, 200)
    groups = np.repeat(np.arange(20), 10)
    result = make_splits(200, y, groups, cv_type="group", is_classification=True)
    assert check_group_disjoint(result.train_idx, result.test_idx, groups) is None


def test_stratified_downgrades_for_continuous_targets():
    """Stratifying on a continuous target raises inside sklearn; we downgrade
    instead of crashing."""
    y = np.random.RandomState(5).randn(120)
    result = make_splits(120, y, cv_type="stratified", is_classification=False)
    assert result.cv_type == "random"
    assert any("classification-only" in n for n in result.notes)


def test_time_series_split_is_ordered_with_a_gap():
    y = np.random.RandomState(6).randn(200)
    result = make_splits(200, y, cv_type="time_series", is_classification=False,
                         order=np.arange(200), gap=10)
    assert result.train_idx.max() < result.test_idx.min()
    assert result.test_idx.min() - result.train_idx.max() > 10


# ---------------------------------------------------------------- leakage
def test_exact_duplicate_of_target_is_excluded():
    y = np.arange(100) % 2
    df = pd.DataFrame({"a": np.random.RandomState(7).randn(100), "leak": y, "y": y})
    report = LeakageDetector().detect(df, "y", domain="generic")
    assert "leak" in report.excluded_columns


def test_generic_domain_does_not_flag_clinical_terms():
    """The generic rule set must stay clinically empty: --domain generic
    promises not to warn about glucose columns."""
    df = pd.DataFrame({
        "glucose_level": np.random.RandomState(8).randn(100),
        "insulin_dose": np.random.RandomState(9).randn(100),
        "y": np.arange(100) % 2,
    })
    generic = LeakageDetector().detect(df, "y", domain="generic")
    diabetes = LeakageDetector().detect(df, "y", domain="diabetes")
    assert [f["column"] for f in generic.flagged_columns] == []
    assert {"glucose_level", "insulin_dose"} <= {f["column"] for f in diabetes.flagged_columns}


# ---------------------------------------------------------------- preprocessing
def test_preprocessing_handles_nan_inf_and_unseen_categories():
    train = pd.DataFrame({"a": [1.0, np.nan, 3, 4], "b": ["x", "y", "x", None],
                          "c": [np.inf, 1, 2, 3]})
    pipeline = PreprocessingPipeline()
    X = pipeline.fit_transform(train)
    assert np.isfinite(X).all()
    out = pipeline.transform(pd.DataFrame({"a": [2.0], "b": ["never_seen"], "c": [1.0]}))
    assert np.isfinite(out).all()
    assert out.shape[1] == X.shape[1]


# ---------------------------------------------------------------- metrics
def test_classification_metrics_include_the_floor():
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.zeros(100, dtype=int)
    metrics = compute_metrics(y_true, y_pred, task="classification")
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["majority_class_accuracy"] == pytest.approx(0.8)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_regression_metrics_report_the_mean_baseline():
    y = np.random.RandomState(10).randn(100)
    metrics = compute_metrics(y, np.full(100, y.mean()), task="regression")
    assert metrics["rmse_vs_mean_baseline"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["r2"] == pytest.approx(0.0, abs=1e-6)


def test_naive_baseline_matches_majority_share():
    y = np.array([0] * 70 + [1] * 30)
    assert naive_baseline(y, True)["accuracy"] == pytest.approx(0.7)


# ---------------------------------------------------------------- validator
def test_validator_flags_constant_column_and_missing_target():
    df = pd.DataFrame({"const": [1] * 50, "x": np.arange(50), "y": [0] * 25 + [1] * 25})
    issues = DataValidator(df, "y").validate()
    assert any("const" in i["message"] for i in issues.issues)
    missing = DataValidator(df, "not_a_column").validate()
    assert missing.has_critical
