# QSQ-FS ML Platform

Feature selection by **Quorum Sensing / Quorum Quenching**, paired with a tabular
Transformer, leakage screening, and baselines that are honest about the floor.

Runs on any tabular dataset and on the **D1NAMO** wearable dataset (ECG,
respiration, accelerometer, CGM glucose).

---

## The one rule this codebase follows

**No number is ever invented.** There are no hardcoded accuracies, no expected
RMSE ranges, no simulated predictions, no placeholder training loops. Every
metric, figure, and table is computed from predictions the code actually made on
data you actually supplied. If a model performs no better than predicting the
mean, that is what gets printed.

Two consequences worth knowing up front:

- Every comparison includes a **dummy predictor** (majority class, or the mean).
  If your model is not clearly above it, the result is not real.
- The naive baseline, the class balance, and the mean-baseline RMSE ratio are
  printed alongside every headline number, so a 0.85 accuracy on an 85/15 split
  cannot be mistaken for a finding.

---

## Guides

| File | Covers |
|---|---|
| `SETUP_GUIDE.md` | Install, first run, GitHub, deployment — assumes no experience |
| `DATASETS.md` | Exactly which D1NAMO / PhysioCGM files to download, and which to skip |
| `QUICKSTART.md` | Command reference once you are set up |

## New to this? Read `SETUP_GUIDE.md`

`SETUP_GUIDE.md` walks through installing Python, running your first
analysis, publishing to GitHub and deploying the web app, assuming no
prior experience. The rest of this README assumes you are comfortable
with a terminal.

## Publication output

Every run writes a bundle ready for a thesis or journal submission:

- **300 DPI figures** as PNG, plus a **vector PDF** of each (`svg`, `tiff`, `eps`
  also available via `--figure_formats`)
- **Statistical tables** — descriptives with normality tests, group comparisons
  with the correct test per variable type and an effect size beside every
  p-value, correlations with confidence intervals, model comparison — as CSV and
  as one Excel workbook
- **METHODS.txt** — a methods paragraph built from what the run actually did
- **A ZIP of all of it**, so a complete run can be archived or handed on for
  re-analysis

```bash
python run_pipeline.py tabular --data_path data.csv --target outcome --tune
```

`--tune` searches baseline hyperparameters by cross-validation **on training
rows only**. It cannot inflate the test score, because it never sees the test
set.

## Improving results honestly

Legitimate: more QSQ-FS generations, a larger population, `--estimator forest`,
`--tune`, threshold tuning on validation data, calibration, ensembling.

Not legitimate, and not supported here: picking the seed with the best test
score, tuning against the test set, dropping badly-predicted test rows, or
reporting the best of N runs without saying so. `src/tuning.py` documents the
distinction in full.

## Install

```bash
git clone <your-repo-url> qsqfs-ml-platform
cd qsqfs-ml-platform
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                             # 30 tests, ~3 s
```

Python 3.9+. No GPU required.

---

## Quick start without any data

The repo can generate a small synthetic dataset shaped like the D1NAMO archive,
so you can verify the code runs before committing to a 10 GB download:

```bash
python scripts/make_fixture.py --out ./data/d1namo_fixture --subjects 4 --hours 6
python run_pipeline.py inspect --data_root ./data/d1namo_fixture
python run_pipeline.py d1namo  --data_root ./data/d1namo_fixture --n_iterations 10
```

The fixture's signals are random. Any score it produces is meaningless — it
exists to test the wiring, and the pipeline will correctly report that nothing
beats the baseline.

---

## Usage

### Any tabular dataset

```bash
python run_pipeline.py tabular \
    --data_path ./data/hospital.csv \
    --target readmitted \
    --domain readmission \
    --group_col patient_id \
    --cv_type group
```

Classification vs regression is detected from the target (`--task` to override).
Detection handles float `0.0/1.0` and string `"yes"/"no"`, not just integer dtypes.

| Flag | Why you would use it |
|---|---|
| `--group_col` | Keeps one patient/subject out of both train and test |
| `--time_col` + `--cv_type time_series` | Chronological split, no peeking forward |
| `--domain` | Which leakage rule set to apply (see below) |
| `--drop_flagged` | Also drop suspected proxies, not just exact duplicates |
| `--skip_transformer` | Baselines only; much faster |

### D1NAMO

Download from Zenodo (the split archive is easier than the 10.2 GB `.tgz`):
<https://zenodo.org/records/5651217>

Extract, then **always inspect first**:

```bash
python run_pipeline.py inspect --data_root ./data/d1namo
```

This prints every subject found and which signal files each one has. If it finds
nothing, the layout differs from what the loader expects — fix that before
running anything else rather than debugging a zero-window error later.

```bash
python run_pipeline.py d1namo \
    --data_root ./data/d1namo \
    --window 30 --step 10 --horizon 30 \
    --cohorts diabetes \
    --cv_type group
```

`--use_ecg` additionally parses the raw 250 Hz ECG (several GB, slow). It is off
by default because the `_RR.csv` files already give you proper HRV.

### Ablation

Which mechanism actually earns its place?

```bash
python run_pipeline.py ablation --data_path ./data/hospital.csv \
    --target readmitted --seeds 42 52 62
```

Runs `full`, `no_quorum_sensing`, `no_quorum_quenching`, `no_elitism`,
`no_sensing_no_quenching` and `random_search` across several seeds, and reports
the spread. **Compare `delta_vs_full` against `fitness_std` before calling any
difference real** — two stochastic searches differing by less than their own
seed-to-seed noise have not been shown to differ.

### Web interface

```bash
streamlit run app.py
```

Two modes:

- **Run a dataset** — upload a CSV/Excel/Parquet table and run the full
  pipeline, with figures and metrics rendered in the browser.
- **Browse past runs** — view any completed run from `run_results/`, including
  CLI runs the app cannot itself execute (D1NAMO, PhysioCGM, multimodal). You
  get the summary, every figure, the model comparison, predictions and raw JSON.

Raw D1NAMO and PhysioCGM archives are far too large for a hosted container, but
they do not need to be uploaded. Extract the features locally and upload the
resulting few-megabyte table:

```bash
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM \
    --export_features pcgm_feats.csv --export_only
```

Set the group column to `subject_id` and the split strategy to `group` in the
app. Results are identical to running it directly in the CLI.

For Streamlit Community Cloud, point it at `requirements.txt`
(CPU-only torch wheel). Cloud containers are ephemeral, so "Browse past runs"
only shows runs made on the same machine.

---

## How the algorithm works

A population of binary feature masks ("colonies") evolves under two bacterial
signalling analogies.

**Quorum sensing.** High-fitness colonies emit an autoinducer field across the
feature axis; field strength at feature *j* reflects how strongly strong colonies
agree that *j* belongs. The field is EMA-smoothed across generations (`beta`), so
it encodes accumulated consensus rather than one lucky generation, and it biases
mutation toward the incumbent best mask.

**Quorum quenching.** Masks scoring below the strong-colony threshold enter a
suppression archive with a penalty. Revisiting them lowers their *effective*
fitness, discouraging cycling. Penalties decay geometrically (`delta`), so no
region is banned permanently.

Fitness is `alpha * score + (1 - alpha) * parsimony`, with `score` bounded in
[0, 1]:

- classification — cross-validated **balanced accuracy**
- regression — `1 - RMSE / std(y)`, clipped at 0. **Zero means no better than
  predicting the training mean.** This is not R².

Key parameters live in `config/default_config.yaml`.

---

## Leakage handling

Two distinct categories, deliberately:

- **Excluded automatically** — exact duplicates of the target. A column that
  reproduces the target is never a legitimate feature.
- **Flagged, never deleted** — name-based proxies and correlations above 0.95.
  A strong correlation may be exactly the clinical signal you are looking for;
  silently deleting it would hide your own result from you.

Correlation screening runs on **training rows only**. Screening the full frame
lets the test set influence which columns survive, which is itself a leak.

Rule sets: `generic`, `diabetes`, `mortality`, `readmission`, `sepsis`
(`config/leakage_rules.yaml`). **`generic` contains no clinical terms on
purpose** — running with `--domain generic` will *not* warn you about glucose
columns. Pass the domain that matches your target.

---

## Order of operations

Every step after the split can leak if run before it, so the order is fixed:

```
load → validate → SPLIT → leakage screen (train rows)
     → fit preprocessing on train → QSQ-FS on train
     → fit models → score once on test
```

The test set is touched exactly once, at the final step.

---

## What is in the box

```
run_pipeline.py            CLI: inspect | tabular | d1namo | ablation
app.py                     Streamlit interface
config/                    default_config.yaml, leakage_rules.yaml
src/
  pipeline.py              orchestration, ordering, artefact writing
  data/                    loader, validator, splitting, d1namo, features
  preprocessing/           impute / scale / encode, fit-on-train-only
  leakage/                 detector, domain rules
  feature_selection/       qsqfs, ablation
  models/                  transformer, trainer, baselines
  evaluation/              metrics, bootstrap, statistical, permutation
  explainability/          permutation importance, selection stability
  reporting/               plots, results writer
  utils/                   seeding, config, JSON-safe IO, run metadata
scripts/make_fixture.py    synthetic D1NAMO-shaped test data
tests/                     30 tests
```

Each run writes to `run_results/run_<timestamp>/`: `results.json`,
`summary.txt`, `predictions.csv`, `selected_features.csv`,
`leakage_report.txt`, and `figures/`.

---

## Notes on the D1NAMO loader

Three decisions differ from the obvious approach, each for a reason:

**Layout is discovered, not assumed.** The archive nests
`<cohort>/<subject>/sensor_data/<session>/…` and the depth differs between the
single-`.tgz` and split-zip releases. The loader walks the tree and classifies
files by suffix rather than globbing one hardcoded pattern and silently
producing zero samples.

**Alignment is by wall-clock timestamp.** Sensor sessions start and stop
throughout the day; glucose arrives every ~5 min. Index-based alignment — or
`scipy.signal.resample` stretching glucose onto the ECG length — fabricates
correspondence that does not exist and adds Fourier ringing to a non-periodic
CGM trace. Here each window carries real start/end times and the target is
interpolated at the true prediction time, subject to `--max_gap`.

**HRV comes from RR intervals**, either the Zephyr `_RR.csv` files or QRS
detection on 250 Hz ECG. Computing LF/HF from an ECG amplitude trace
downsampled to 1 Hz yields numbers that look like HRV but are not: the QRS
complex does not survive, and Nyquist at 1 Hz is 0.5 Hz, so the 0.15–0.4 Hz
"HF" band would be mostly aliased noise. `detect_r_peaks` returns `None` rather
than guessing when the sampling rate is too low.

**Windows overlap.** With `--window 30 --step 10`, consecutive windows share
two thirds of their samples. Rows from one subject are near-duplicates, so use
`--cv_type group` or `time_series`; a shuffled split will look better than it is.
The loader prints this warning on every run.

---

## Multimodal encoders

Four encoders, each with tests, each reachable from the CLI. None is dead code.

| Modality | Encoder | Backend | Falls back to |
|---|---|---|---|
| Image | `ImageEncoder` | torchvision ResNet/DenseNet | small CNN trained from scratch |
| Text | `TextEncoder` | `transformers` clinical BERT | hashed bag-of-words MLP, offline |
| Genomic | `GenomicEncoder` | expression / SNP / DNA sequence | — (pure torch) |
| Tabular | `TabularEncoder` | MLP | — |

**Fallbacks announce themselves.** If pretrained weights cannot be fetched —
no network, blocked host, air-gapped hospital — the encoder keeps the
architecture, records `pretrained: False` and a backbone name like
`resnet18_random_init`, and writes that into the run record. A result can never
imply a prior that was not loaded.

Fusion is `early` (concatenate), `late` (per-modality heads with readable
weights), or `hybrid` (cross-attention). Missing modalities use a learned
missing-token, so a row with no scan is not silently treated as an all-zero
scan.

### Running it

```bash
python run_pipeline.py multimodal \
    --manifest ./data/cohort/manifest.csv \
    --target outcome \
    --fusion early
```

One CSV, one row per subject. Columns are auto-assigned — file paths to image,
long strings to text, `gene_*`/`snp_*` to genomic, the rest to tabular — and the
resolved schema is printed before training so you can check it. Override with
`--image_cols` / `--text_cols` / `--genomic_cols`.

### Every multimodal run answers two questions

**Did fusion actually help?** Each single modality is trained alone and compared
against the fused model. The verdict is stated plainly:

```
Fusion verdict: Fusion did NOT beat 'text_only' alone. On this data the
extra modalities are not contributing; report the simpler model.
```

**Does it beat a tree on the tabular block?** Classical baselines on tabular
data alone are computed by default, because they frequently win on
hospital-sized cohorts and omitting them flatters the deep model.

Try it on the bundled fixture, which plants a known signal:

```bash
python scripts/make_multimodal_fixture.py --out ./data/mm_fixture
python run_pipeline.py multimodal --manifest ./data/mm_fixture/manifest.csv \
    --target outcome --force_text_fallback --image_size 64
```

The fixture's clinical notes leak the outcome **on purpose**. `text_only`
scores a perfect AUC and the ablation says so — a demonstration of why free-text
is the modality most likely to leak a label, and why no automatic screen will
catch it for you. Read a sample of your notes before trusting any text result.

---

## PhysioCGM

`physiocgm` runs on the PSI-TAMU PhysioCGM dataset: 10 participants with type 1
diabetes over 17 days, with **ECG (Zephyr, 250 Hz), PPG/BVP, EDA, temperature,
motion and CGM glucose**. This is the multimodal path on real recordings —
ECG, PPG and EDA are separate physical measurements of the same person at the
same moment, which is exactly what per-modality encoders are for.

```bash
# Always inspect first
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM --inspect_only

# QSQ-FS selection + Transformer over all sensor features
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM --horizon_steps 6

# One encoder per sensor family, fused
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM --model fusion --fusion late
```

This reads the **processed** segments (`dataset/processed/<subject>/<n>.pkl`)
produced by PhysioCGM's own `preprocess.py`. If you only have `dataset/raw`,
run their preprocessing first — the alignment work has already been done by the
dataset's authors and there is no reason to redo it.

`--horizon_steps` counts 5-minute CGM segments, so `6` is a 30-minute horizon.
Pairs separated by a real recording gap are dropped rather than bridged.

Two things worth knowing:

- **`glucose_now` is kept as a feature on purpose.** Any glucose model must
  beat persistence — predicting that glucose 30 minutes from now equals glucose
  now. Pass `--no_persistence` to measure the sensors alone. Expect the honest
  answer to be much worse; that gap *is* the result.
- **Access is gated.** The data is distributed via the TAMU PSI Lab drive and
  requires contacting Prof. Gutierrez-Osuna. This platform reads it; it cannot
  download it. Use `scripts/make_physiocgm_fixture.py` to check wiring first.

---

## Reproducibility

`--seed` fixes Python, NumPy and torch RNGs. Each `results.json` records library
versions, platform, git commit and CUDA availability. QSQ-FS is deterministic
under a fixed seed (there is a test for it). Because the search is stochastic,
report results across several seeds rather than the best single run.

---

## Citation

If you use the D1NAMO dataset, cite its authors:

> Dubosson, F., Ranvier, J.-E., Bromuri, S., Calbimonte, J.-P., Ruiz, J., &
> Schumacher, M. (2018). The open D1NAMO dataset: A multi-modal dataset for
> research on non-invasive type 1 diabetes management. *Informatics in Medicine
> Unlocked*, 13, 92–100.

The FT-Transformer architecture follows Gorishniy et al. (2021), *Revisiting
Deep Learning Models for Tabular Data*.

---

## Licence

MIT (see `LICENSE`). D1NAMO is distributed separately under its own terms; no
data is redistributed here.
