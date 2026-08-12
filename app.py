"""QSQ-FS ML Platform — Streamlit interface.

Upload one file or many (CSV, TSV, Excel, Parquet, .gz, .zip), run the full
pipeline, and download a publication-ready bundle: 300 DPI figures, vector PDFs,
statistical tables and a methods paragraph.

Every number shown is computed from the data supplied. Nothing is precomputed,
simulated, or filled in from a reference value.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import infer_task                     # noqa: E402
from src.data.multifile import load_dataset                # noqa: E402
from src.data.validator import DataValidator               # noqa: E402
from src.leakage.detector import LeakageDetector           # noqa: E402
from src.pipeline import ExperimentConfig, run_experiment  # noqa: E402
from src.reporting import theme                            # noqa: E402
from src.tuning import suggested_settings                  # noqa: E402
from src.utils.config import load_config                   # noqa: E402
from src.utils.logging_setup import setup_logging          # noqa: E402

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

st.set_page_config(page_title="QSQ-FS ML Platform", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")
setup_logging()
st.markdown(theme.CSS, unsafe_allow_html=True)

CONFIG_PATH = Path(__file__).parent / "config" / "default_config.yaml"
RESULTS_DIR = Path(__file__).parent / "run_results"
UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xls", "parquet", "gz", "zip"]


# ======================================================================
# Helpers
# ======================================================================
def save_uploads(files) -> list[Path]:
    """Persist uploads to a temp folder so the multi-file loader can read them."""
    folder = Path(tempfile.mkdtemp(prefix="qsqfs_upload_"))
    paths = []
    for file in files:
        path = folder / file.name
        path.write_bytes(file.getvalue())
        paths.append(path)
    return paths


@st.cache_data(show_spinner=False)
def combine(paths: list[str], mode: str, key: str | None):
    return load_dataset([Path(p) for p in paths], mode=mode,
                        key=key or None, add_source=True)


def find_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "results.json").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def zip_bytes(run_dir: Path) -> bytes:
    """Read the export ZIP, building it on demand if it is not there yet."""
    existing = list(run_dir.parent.glob(f"{run_dir.name}_export.zip"))
    if existing:
        return existing[0].read_bytes()
    from src.reporting.export import export_run
    return Path(export_run(run_dir)).read_bytes()


def metrics_frame(results: dict, task: str) -> pd.DataFrame:
    from src.reporting.stats_tables import model_comparison_table
    return model_comparison_table(results, task)


# ======================================================================
# Run browser
# ======================================================================
def render_browser() -> None:
    st.markdown(theme.hero(
        "📂 Past runs",
        "Every completed run, including CLI runs this app cannot execute "
        "(D1NAMO, PhysioCGM, multimodal fusion)."), unsafe_allow_html=True)

    root = Path(st.sidebar.text_input("Results folder", str(RESULTS_DIR)))
    runs = find_runs(root)
    if not runs:
        st.info(f"No completed runs under `{root}`. Run something first — "
                "in this app, or from the command line.")
        return

    labels = [f"{p.name}  ·  {datetime.fromtimestamp(p.stat().st_mtime):%d %b %H:%M}"
              for p in runs]
    chosen = runs[labels.index(st.sidebar.selectbox("Select run", labels))]

    try:
        results = json.loads((chosen / "results.json").read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not read results.json: {exc}")
        return

    render_results(results, chosen, browsing=True)

    st.sidebar.divider()
    if st.sidebar.checkbox("Batch export several runs"):
        picked = st.sidebar.multiselect("Runs to bundle", [p.name for p in runs],
                                        default=[chosen.name])
        if picked and st.sidebar.button("Build batch ZIP"):
            from src.reporting.export import export_multiple
            out = Path(tempfile.mkdtemp()) / "batch_export.zip"
            export_multiple([root / n for n in picked], out)
            st.sidebar.download_button("⬇ Download batch ZIP", out.read_bytes(),
                                       file_name="qsqfs_batch_export.zip",
                                       mime="application/zip")


# ======================================================================
# Results renderer (shared by live runs and the browser)
# ======================================================================
def render_results(results: dict, run_dir: Path, browsing: bool = False) -> None:
    task = results.get("task", "classification")
    fs = results.get("feature_selection", {})
    naive = results.get("naive_baseline", {})

    st.markdown(theme.section("Headline"), unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Features selected",
             f"{fs.get('n_selected', '—')} / {fs.get('n_features_total', '—')}")
    b.metric("Best fitness", f"{fs.get('best_fitness', float('nan')):.4f}"
             if fs.get("best_fitness") is not None else "—")
    c.metric("Search time", f"{fs.get('runtime_seconds', 0):.1f} s")
    baseline_value = next((v for v in naive.values() if isinstance(v, float)), None)
    d.metric("Baseline to beat",
             f"{baseline_value:.4f}" if baseline_value is not None else "—",
             help="Majority class for classification, training mean for regression.")

    # Export is the point of the whole exercise — keep it prominent.
    try:
        st.download_button(
            "⬇  Download full results bundle (ZIP)", zip_bytes(run_dir),
            file_name=f"{run_dir.name}_export.zip", mime="application/zip",
            help="300 DPI figures, vector PDFs, statistical tables, Excel workbook, "
                 "METHODS.txt and results.json.",
            width="stretch",
        )
    except Exception as exc:
        st.warning(f"Export bundle unavailable: {exc}")

    tabs = st.tabs(["📊 Models", "🧬 Features", "📈 Figures", "📋 Tables",
                    "🔍 Search", "⚠️ Warnings", "🗂 Raw"])

    with tabs[0]:
        table = metrics_frame(results, task)
        if table.empty:
            st.info("No model results in this run.")
        else:
            primary = "Roc Auc" if task == "classification" else "Rmse"
            styled = table.style.format(precision=4)
            if primary in table.columns:
                styled = (styled.highlight_max(subset=[primary], color="#FBF6E3")
                          if task == "classification"
                          else styled.highlight_min(subset=[primary], color="#FBF6E3"))
            st.dataframe(styled, width="stretch", hide_index=True)

        ci = results.get("bootstrap_ci")
        if ci and ci.get("ci_lower") is not None:
            st.markdown(theme.note(
                f"<b>{ci['metric']} = {ci['point_estimate']:.4f}</b> "
                f"(95% bootstrap CI {ci['ci_lower']:.4f} – {ci['ci_upper']:.4f}, "
                f"{ci['n_valid']} resamples).<br>"
                "If this interval overlaps a baseline, the difference is "
                "<i>not established</i>."), unsafe_allow_html=True)

        tuned = results.get("tuned_models")
        if tuned:
            st.markdown(theme.section("Tuned models"), unsafe_allow_html=True)
            rows = [{"Model": name, "Inner-CV score": round(e["best_cv_score"], 4),
                     "Test score": round(next(
                         (v for k, v in e.get("metrics", {}).items()
                          if k in ("roc_auc", "rmse")), float("nan")), 4),
                     "Best parameters": str(e["best_params"])}
                    for name, e in tuned.items() if isinstance(e, dict)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(results.get("tuning_note", ""))

        verdict = results.get("fusion_verdict")
        if verdict:
            st.markdown(theme.verdict(verdict["note"]), unsafe_allow_html=True)

    with tabs[1]:
        selected = fs.get("selected_features", [])
        st.write(f"**{len(selected)} features retained by QSQ-FS**")
        if selected:
            st.dataframe(pd.DataFrame({"Feature": selected}),
                         width="stretch", hide_index=True, height=280)
        importance = results.get("feature_importance", {}).get("importances", [])
        if importance:
            st.markdown(theme.section("Permutation importance (test set)"),
                        unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(importance)[["feature", "importance_mean",
                                          "importance_std"]].style.format(precision=5),
                width="stretch", hide_index=True, height=320)
            st.caption(results["feature_importance"].get("note", ""))

    with tabs[2]:
        figures = sorted((run_dir / "figures").glob("*.png")) \
            if (run_dir / "figures").exists() else []
        publication = [f for f in figures if f.name.startswith("fig_")]
        others = [f for f in figures if not f.name.startswith("fig_")]
        if publication:
            st.markdown(theme.note(
                "Publication figures — 300 DPI PNG shown here; a vector "
                "<b>PDF</b> of each is in the download bundle."),
                unsafe_allow_html=True)
            for figure in publication:
                st.image(str(figure),
                         caption=figure.stem.replace("fig_", "").replace("_", " ").title(),
                         width="stretch")
        if others:
            with st.expander("Diagnostic figures"):
                for figure in others:
                    st.image(str(figure), caption=figure.stem.replace("_", " "),
                             width="stretch")
        if not figures:
            st.info("No figures in this run.")

    with tabs[3]:
        tables_dir = run_dir / "tables"
        files = sorted(tables_dir.glob("*.csv")) if tables_dir.exists() else []
        if files:
            for path in files:
                st.markdown(theme.section(path.stem.replace("_", " ").title()),
                            unsafe_allow_html=True)
                frame = pd.read_csv(path)
                st.dataframe(frame, width="stretch", hide_index=True)
                st.download_button(f"⬇ {path.name}", frame.to_csv(index=False),
                                   file_name=path.name, mime="text/csv",
                                   key=f"dl_{path.stem}")
            excel = tables_dir / "all_tables.xlsx"
            if excel.exists():
                st.download_button("⬇ All tables (Excel workbook)",
                                   excel.read_bytes(), file_name="all_tables.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument"
                                        ".spreadsheetml.sheet")
        else:
            st.info("No statistical tables in this run.")

    with tabs[4]:
        history = results.get("qsqfs_history", {})
        if history.get("best_fitness"):
            st.line_chart(pd.DataFrame({
                "Best fitness": history["best_fitness"],
                "Population mean": history.get("mean_fitness", []),
            }))
            st.line_chart(pd.DataFrame({"Features selected": history["n_selected"]}))
        st.json({k: v for k, v in fs.items() if k != "selected_features"})

    with tabs[5]:
        warnings = results.get("warnings", [])
        leakage = results.get("leakage", {})
        for column in leakage.get("excluded_columns", []):
            st.error(f"Removed — `{column}` duplicates the target.")
        for entry in leakage.get("flagged_columns", []):
            st.markdown(theme.flag(f"<b>{entry['column']}</b> — {entry['reason']}"),
                        unsafe_allow_html=True)
        for warning in warnings:
            st.warning(warning)
        if not warnings and not leakage.get("flagged_columns"):
            st.success("No warnings.")

    with tabs[6]:
        methods = run_dir / "METHODS.txt"
        if methods.exists():
            st.markdown(theme.section("Methods paragraph"), unsafe_allow_html=True)
            st.code(methods.read_text(encoding="utf-8"), language="text")
        st.json({k: v for k, v in results.items()
                 if k not in ("_writer", "qsqfs_history", "feature_importance",
                              "statistical_tables")})


# ======================================================================
# Main
# ======================================================================
st.sidebar.markdown("### Mode")
mode = st.sidebar.radio("Mode", ["Run a dataset", "Browse past runs"],
                        label_visibility="collapsed")

if mode == "Browse past runs":
    render_browser()
    st.stop()

st.markdown(theme.hero(
    "🧬 QSQ-FS ML Platform",
    "Quorum Sensing / Quorum Quenching feature selection · tabular Transformer · "
    "leakage screening · <span class='accent'>publication-ready output</span>"),
    unsafe_allow_html=True)

st.sidebar.markdown("## 1 · Data")
uploads = st.sidebar.file_uploader(
    "Upload one or more files", type=UPLOAD_TYPES, accept_multiple_files=True,
    help="CSV, TSV, Excel, Parquet, .gz or .zip. Upload several files and they "
         "will be combined.",
)

if not uploads:
    st.markdown(theme.section("Start here"), unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("""
**Upload one file, or several at once.** Multiple files are combined
automatically — stacked row-wise when they share columns (one file per site or
subject), or joined on a key when they hold different variables for the same
people.

Supported: `.csv` `.tsv` `.xlsx` `.parquet` `.gz` `.zip`

**What the pipeline does**

1. **Validate** — missingness, duplicates, imbalance, p > n
2. **Split first** — before anything inspects the data; grouped and time-series
   splits keep repeated measures from straddling the split
3. **Screen for leakage** — on training rows only; exact target duplicates are
   removed, suspected proxies are reported for you to judge
4. **Preprocess** — impute, scale, encode; fitted on training rows only
5. **Select features** — QSQ-FS under a quorum-sensing consensus field and a
   quorum-quenching suppression archive
6. **Model and compare** — Transformer against five baselines including a
   trivial predictor
7. **Export** — 300 DPI figures, vector PDFs, statistical tables, methods text
        """)
    with right:
        st.markdown(theme.note(
            "<b>Large signal archives</b><br>D1NAMO and PhysioCGM raw recordings "
            "are 10+ GB and cannot be uploaded. Extract features locally, then "
            "upload the small table:<br><br>"
            "<code>python run_pipeline.py physiocgm --data_root ./data/PhysioCGM "
            "--export_features feats.csv --export_only</code><br><br>"
            "The result is a few MB. Full runs also appear under "
            "<b>Browse past runs</b>."), unsafe_allow_html=True)
        if not TORCH_AVAILABLE:
            st.markdown(theme.flag(
                "PyTorch is not installed here, so the Transformer will be "
                "skipped. Feature selection, all baselines, figures and tables "
                "still run."), unsafe_allow_html=True)
    st.stop()

# ---------------- combine uploads ----------------
paths = save_uploads(uploads)
st.sidebar.markdown("## 2 · Combine")
if len(paths) > 1:
    combine_mode = st.sidebar.selectbox(
        "How to combine", ["auto", "stack", "merge"],
        help="stack = same columns, more rows. merge = different columns, same "
             "subjects (needs a key).")
    join_key = st.sidebar.text_input("Join key (merge only)", "")
else:
    combine_mode, join_key = "auto", ""

try:
    df, load_report = combine([str(p) for p in paths], combine_mode, join_key)
except Exception as exc:
    st.error(f"Could not load those files: {exc}")
    st.stop()

st.markdown(theme.section("Data loaded"), unsafe_allow_html=True)
a, b, c = st.columns(3)
a.metric("Files", load_report.get("n_files", 1))
b.metric("Rows", f"{load_report['n_rows']:,}")
c.metric("Columns", load_report["n_columns"])
if load_report.get("warning"):
    st.markdown(theme.flag(load_report["warning"]), unsafe_allow_html=True)
if load_report.get("mode") in ("stack", "merge"):
    with st.expander(f"Combination report ({load_report['mode']})"):
        st.json(load_report)

with st.expander("Preview and column summary"):
    st.dataframe(df.head(50), width="stretch")
    st.dataframe(pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing %": (df.isna().mean() * 100).round(2),
        "distinct": df.nunique(),
    }), width="stretch")

# ---------------- configuration ----------------
st.sidebar.markdown("## 3 · Target")
target = st.sidebar.selectbox("Target column", list(df.columns))
detected = infer_task(df[target].dropna()) if df[target].notna().any() else "classification"
task_choice = st.sidebar.radio("Task", ["auto", "classification", "regression"],
                               horizontal=True, help=f"Auto-detected: **{detected}**")
task = detected if task_choice == "auto" else task_choice
st.sidebar.caption(f"Running as **{task}** · {df[target].nunique()} distinct values")

st.sidebar.markdown("## 4 · Splitting")
others = [c for c in df.columns if c != target]
group_col = st.sidebar.selectbox(
    "Group column (optional)", ["(none)"] + others,
    index=(others.index("source_file") + 1) if "source_file" in others else 0,
    help="Rows sharing this value stay on the same side of the split. Use it "
         "for patient or subject IDs.")
group_col = None if group_col == "(none)" else group_col
time_col = st.sidebar.selectbox("Time column (optional)", ["(none)"] + others)
time_col = None if time_col == "(none)" else time_col

default_cv = "group" if group_col else ("stratified" if task == "classification" else "random")
cv_options = ["stratified", "random", "group", "time_series"]
cv_type = st.sidebar.selectbox("Split strategy", cv_options,
                               index=cv_options.index(default_cv))
test_size = st.sidebar.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)

st.sidebar.markdown("## 5 · Leakage")
domain = st.sidebar.selectbox("Domain rule set",
                              ["generic", "diabetes", "mortality", "readmission", "sepsis"])
drop_flagged = st.sidebar.checkbox("Also drop flagged proxy columns", value=False)

st.sidebar.markdown("## 6 · Search & model")
profile = suggested_settings(len(df), df.shape[1] - 1, task)
st.sidebar.caption(f"Suggested profile: **{profile['profile']}**")
n_iterations = st.sidebar.slider("QSQ-FS generations", 5, 100,
                                 profile["qsqfs"]["n_iterations"], 5)
population = st.sidebar.slider("Population size", 8, 60,
                               profile["qsqfs"]["population_size"], 4)
alpha = st.sidebar.slider("alpha — accuracy vs parsimony", 0.5, 1.0, 0.85, 0.05)
estimator = st.sidebar.selectbox("Subset scoring model", ["knn", "forest"],
                                 help="forest is slower but picks features that "
                                      "suit tree models.")
epochs = st.sidebar.slider("Transformer epochs", 10, 300, 60, 10,
                           disabled=not TORCH_AVAILABLE)
tune = st.sidebar.checkbox(
    "Tune baseline hyperparameters", value=False,
    help="Randomised search by cross-validation on training rows only. Slower, "
         "usually better, and cannot inflate the test score.")
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("## 7 · Output")
figure_formats = st.sidebar.multiselect("Figure formats", ["png", "pdf", "svg", "tiff"],
                                        default=["png", "pdf"])
skip_transformer = st.sidebar.checkbox("Skip the Transformer", value=not TORCH_AVAILABLE,
                                       disabled=not TORCH_AVAILABLE)

# ---------------- pre-flight ----------------
left, right = st.columns([3, 2])
with left:
    st.markdown(theme.section("Pre-flight checks"), unsafe_allow_html=True)
    report = DataValidator(df, target).validate()
    if report.has_critical:
        for issue in report.issues:
            if issue["level"] == "CRITICAL":
                st.error(issue["message"])
        st.stop()
    shown = report.issues[:8]
    for issue in shown:
        (st.warning if issue["level"] == "WARNING" else st.info)(issue["message"])
    if len(report.issues) > 8:
        st.caption(f"…and {len(report.issues) - 8} more (all recorded in results.json)")
    if not report.issues:
        st.success("No data-quality problems found.")

    leakage = LeakageDetector().detect(df, target, domain=domain)
    for column in leakage.excluded_columns:
        st.error(f"Will be removed — `{column}` duplicates the target.")
    for entry in leakage.flagged_columns[:6]:
        st.markdown(theme.flag(f"<b>{entry['column']}</b> — {entry['reason']}"),
                    unsafe_allow_html=True)
    if not leakage.excluded_columns and not leakage.flagged_columns:
        st.success(f"No leakage detected under the '{domain}' rule set.")

with right:
    st.markdown(theme.section("Target"), unsafe_allow_html=True)
    values = df[target].dropna()
    if task == "classification":
        counts = values.value_counts()
        st.bar_chart(counts)
        st.metric("Majority class share", f"{counts.max() / counts.sum():.1%}",
                  help="Any useful model must beat this.")
    else:
        st.bar_chart(np.histogram(values.astype(float), bins=30)[0])
        st.metric("Mean ± SD", f"{values.mean():.3f} ± {values.std():.3f}")

st.divider()
if st.button("▶  Run pipeline", type="primary", width="stretch"):
    cfg = load_config(CONFIG_PATH)
    qs = cfg["feature_selection"]["qsqfs"]
    qs.update({"n_iterations": n_iterations, "population_size": population,
               "alpha": alpha, "estimator": estimator})
    cfg["models"]["transformer"]["epochs"] = epochs

    clean = df.dropna(subset=[target]).reset_index(drop=True)
    groups = clean[group_col].to_numpy() if group_col else None
    order = pd.to_datetime(clean[time_col], errors="coerce").to_numpy() if time_col else None
    X_df = clean.drop(columns=[target] + [c for c in (group_col, time_col) if c])

    experiment = ExperimentConfig(
        task=task, target=target, domain=domain, cv_type=cv_type,
        test_size=test_size, seed=int(seed), results_dir=str(RESULTS_DIR),
        drop_flagged=drop_flagged, skip_transformer=skip_transformer,
        source=f"streamlit:{uploads[0].name}"
        + (f" (+{len(uploads)-1} more)" if len(uploads) > 1 else ""),
        tune_hyperparameters=tune,
        figure_formats=tuple(figure_formats or ["png"]),
    )

    progress = st.progress(0.0, "Splitting, screening and preprocessing…")
    try:
        progress.progress(0.3, "Running QSQ-FS and fitting models…")
        results = run_experiment(
            X_df, clean[target].to_numpy(), experiment,
            model_config={**cfg["preprocessing"], **cfg["models"]},
            qsqfs_config=qs, groups=groups, order=order, raw_df=clean,
        )
        progress.progress(1.0, "Complete")
    except Exception as exc:
        progress.empty()
        st.error(f"The run failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

    st.session_state["results"] = results
    st.session_state["run_dir"] = results["run_directory"]

if "results" in st.session_state:
    st.divider()
    st.markdown(theme.section("Results — held-out test set, scored once"),
                unsafe_allow_html=True)
    render_results(st.session_state["results"], Path(st.session_state["run_dir"]))
