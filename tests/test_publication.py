"""Publication output, statistics, tuning honesty, multi-file loading."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.multifile import load_dataset, merge_frames, read_any, stack_frames
from src.reporting import publication as pub
from src.reporting import stats_tables as st_
from src.reporting.export import export_run, methods_paragraph
from src.tuning import suggested_settings, tune_decision_threshold, tune_model


# ------------------------------------------------------------ figures
def test_figures_are_300_dpi_and_have_a_vector_copy(tmp_path):
    from PIL import Image

    results = {
        "baselines": {"dummy_mean": {"rmse": 25.0, "mae": 21.0, "r2": 0.0},
                      "svm": {"rmse": 22.0, "mae": 19.0, "r2": 0.2}},
        "transformer": {"rmse": 21.0, "mae": 18.0, "r2": 0.28},
    }
    paths = pub.fig_model_comparison(results, "regression", tmp_path,
                                     formats=("png", "pdf"))
    assert len(paths) == 2
    png = next(p for p in paths if p.endswith(".png"))
    assert Image.open(png).info["dpi"][0] == pytest.approx(300, abs=1)
    assert any(p.endswith(".pdf") for p in paths)


def test_regression_diagnostics_include_bland_altman(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(140, 25, 120)
    paths = pub.fig_regression_diagnostics(y, y + rng.normal(0, 10, 120), tmp_path)
    assert paths and Path(paths[0]).exists()


def test_generate_all_produces_multiple_figures(tmp_path):
    rng = np.random.default_rng(1)
    results = {
        "baselines": {"dummy_mean": {"rmse": 25.0}, "svm": {"rmse": 22.0}},
        "qsqfs_history": {"best_fitness": list(np.linspace(0.4, 0.7, 20)),
                          "mean_fitness": list(np.linspace(0.3, 0.6, 20)),
                          "n_selected": [10] * 20, "archive_size": list(range(20)),
                          "field_entropy": list(np.linspace(1, 0.6, 20))},
    }
    y = rng.normal(size=100)
    written = pub.generate_all(results, tmp_path, "regression", y, y + 0.3)
    assert len(written) >= 3


# --------------------------------------------------------- statistics
def test_descriptive_table_reports_normality():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"normal": rng.normal(0, 1, 200),
                       "skewed": rng.lognormal(0, 1, 200)})
    table = st_.descriptive_table(df)
    assert set(table["Variable"]) == {"normal", "skewed"}
    assert "Shapiro-Wilk W" in table.columns
    assert table.loc[table.Variable == "normal", "Distribution"].iloc[0] == "Normal"


def test_group_comparison_picks_the_right_test_and_reports_effect_size():
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({
        "value": np.concatenate([rng.normal(0, 1, n), rng.normal(1.0, 1, n)]),
        "category": rng.choice(["a", "b"], 2 * n),
        "group": [0] * n + [1] * n,
    })
    table = st_.group_comparison_table(df, "group")
    numeric_row = table[table.Variable == "value"].iloc[0]
    assert numeric_row["Test"] in ("Welch t-test", "Mann-Whitney U")
    assert numeric_row["Effect size"] == "Cohen's d"
    assert abs(numeric_row["Effect value"]) > 0.5   # a real difference was planted
    categorical_row = table[table.Variable == "category"].iloc[0]
    assert categorical_row["Test"] == "Chi-square"
    assert categorical_row["Effect size"] == "Cramér's V"


def test_effect_size_helpers():
    rng = np.random.default_rng(4)
    a = np.zeros(50)
    b = np.ones(50) + rng.normal(0, 0.1, 50)
    assert st_.cohens_d(a, b) < 0
    assert 0 <= st_.eta_squared([a, b]) <= 1


def test_correlation_table_has_ci_and_p():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "y": x * 2 + rng.normal(0, 0.5, 200),
                       "noise": rng.normal(size=200)})
    table = st_.correlation_table(df, "y")
    assert "95% CI" in table.columns and "p" in table.columns
    assert table.iloc[0]["Variable"] == "x"      # strongest correlate ranks first


def test_cv_summary_table_reports_spread():
    folds = [{"roc_auc": v} for v in (0.70, 0.74, 0.68, 0.72, 0.71)]
    table = st_.cv_summary_table(folds)
    assert table.iloc[0]["n"] == 5
    assert "95% CI" in table.columns


# ------------------------------------------------------------- tuning
def test_tuning_never_sees_the_test_set():
    """The search must fit only what it is handed; it has no test argument."""
    import inspect

    signature = inspect.signature(tune_model)
    assert "X_test" not in signature.parameters
    assert "y_test" not in signature.parameters


def test_tuning_stays_within_valid_score_range():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=250, n_features=12, n_informative=5,
                               random_state=0)
    result = tune_model("random_forest", X, y, "classification", n_iter=5, seed=0)
    assert 0.0 <= result["best_cv_score"] <= 1.0
    assert "training rows only" in result["note"]


def test_threshold_tuning_beats_or_matches_default():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    result = tune_decision_threshold(y, proba, "f1")
    assert result["best_score"] >= result["score_at_0.5"] - 1e-9
    assert 0.05 <= result["best_threshold"] <= 0.95


def test_suggested_settings_scale_with_dataset_size():
    assert suggested_settings(200, 20, "classification")["profile"] == "small"
    assert suggested_settings(100_000, 300, "regression")["profile"] == "large"


# --------------------------------------------------------- multi-file
def test_stack_adds_source_column_and_reports_mismatch():
    a = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    b = pd.DataFrame({"x": [3, 4], "y": [1, 0], "extra": [9, 9]})
    combined, report = stack_frames({"a.csv": a, "b.csv": b})
    assert len(combined) == 4
    assert "source_file" in combined.columns
    assert report["columns_missing_from_some_files"] == ["extra"]
    assert "warning" in report


def test_merge_reports_row_loss():
    left = pd.DataFrame({"pid": range(100), "age": range(100)})
    right = pd.DataFrame({"pid": range(50), "crp": range(50)})
    combined, report = merge_frames({"l.csv": left, "r.csv": right}, "pid")
    assert len(combined) == 50
    assert "warning" in report      # half the rows were dropped; must be flagged


def test_gzip_and_zip_are_readable(tmp_path):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gz = tmp_path / "data.csv.gz"
    frame.to_csv(gz, index=False, compression="gzip")
    assert len(read_any(gz)) == 1

    plain = tmp_path / "plain.csv"
    frame.to_csv(plain, index=False)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(plain, "one.csv")
        zf.write(plain, "two.csv")
    assert len(read_any(archive)) == 2


def test_load_dataset_auto_selects_mode(tmp_path):
    for name in ("s1.csv", "s2.csv"):
        pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(tmp_path / name, index=False)
    combined, report = load_dataset([tmp_path])
    assert report["mode"] == "stack"
    assert len(combined) == 4


# ------------------------------------------------------------- export
def test_export_bundle_contains_everything(tmp_path):
    from src.utils.jsonio import save_json

    run_dir = tmp_path / "run_x"
    (run_dir / "figures").mkdir(parents=True)
    results = {
        "task": "regression", "seed": 42,
        "dataset": {"source": "test", "n_rows": 100, "n_features_raw": 10,
                    "target": "y"},
        "split": {"cv_type": "group", "n_train": 80, "n_test": 20,
                  "n_cv_folds": 5, "group_overlap": None},
        "feature_selection": {"n_selected": 5, "n_features_total": 10,
                              "best_score": 0.3, "runtime_seconds": 1.0,
                              "total_evaluations": 50, "n_iterations_run": 10,
                              "hyperparameters": {"alpha": 0.85}},
        "naive_baseline": {"rmse": 10.0},
    }
    save_json(results, run_dir / "results.json")
    (run_dir / "figures" / "f.png").write_bytes(b"x")

    archive = export_run(run_dir, results=results,
                         tables={"desc": pd.DataFrame({"a": [1]})})
    names = zipfile.ZipFile(archive).namelist()
    assert any("METHODS.txt" in n for n in names)
    assert any("MANIFEST.txt" in n for n in names)
    assert any("tables/desc.csv" in n for n in names)
    assert any("figures/f.png" in n for n in names)


def test_methods_paragraph_uses_only_real_values():
    results = {"task": "classification", "seed": 7,
               "dataset": {"n_rows": 123, "n_features_raw": 45, "target": "outcome"},
               "split": {"cv_type": "group", "n_train": 98, "n_test": 25,
                         "n_cv_folds": 5},
               "feature_selection": {"n_selected": 9, "n_features_total": 45,
                                     "best_score": 0.71,
                                     "hyperparameters": {"alpha": 0.85}}}
    text = methods_paragraph(results)
    assert "123 observations" in text
    assert "98 observations to training" in text
    assert "9 of 45" in text
    assert "hardcoded, simulated or illustrative" in text
