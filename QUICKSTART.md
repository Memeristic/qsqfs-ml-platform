# Quickstart

## 1. Install (5 minutes)

```bash
cd qsqfs-ml-platform
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check it works:

```bash
pytest -q                          # expect: 30 passed
```

## 2. Prove the pipeline runs, without downloading anything

```bash
python scripts/make_fixture.py --out ./data/d1namo_fixture
python run_pipeline.py inspect --data_root ./data/d1namo_fixture
python run_pipeline.py d1namo --data_root ./data/d1namo_fixture --n_iterations 10 --epochs 30
```

The fixture is random noise, so every model should land at the dummy baseline.
**That is the correct result** and confirms the pipeline is not inventing signal.

## 3. Run your own hospital CSV

```bash
python run_pipeline.py tabular \
    --data_path ./data/your_data.csv \
    --target your_target_column \
    --domain generic
```

Add `--group_col patient_id` if a patient can appear in more than one row.

## 4. Run real D1NAMO

Download from <https://zenodo.org/records/5651217>, extract, then:

```bash
python run_pipeline.py inspect --data_root ./data/d1namo     # ALWAYS do this first
python run_pipeline.py d1namo  --data_root ./data/d1namo --cohorts diabetes
```

If `inspect` shows 0 usable subjects, the folder layout differs from what the
loader expects. Send the output of `inspect` plus `find ./data/d1namo -name "*.csv" | head -20`
and the file-matching rules in `src/data/d1namo.py` can be adjusted.

## 5. Web interface

```bash
streamlit run app.py
```

## Reading your results

Open `run_results/run_<timestamp>/summary.txt`. Check, in this order:

1. **Does your model beat `dummy_majority` / `dummy_mean`?** If not, stop.
2. **Does the bootstrap CI overlap the best baseline?** If so, the difference is
   not established.
3. **Are any columns listed under FLAGGED?** Decide whether each is legitimate.
4. **Is `group_overlap` NONE?** If not, rerun with `--cv_type group`.

---

## Multimodal, without any real data

```bash
python scripts/make_multimodal_fixture.py --out ./data/mm_fixture
python run_pipeline.py multimodal --manifest ./data/mm_fixture/manifest.csv \
    --target outcome --force_text_fallback --image_size 64 --epochs 20
```

Watch the `fusion_verdict` line: the fixture's notes leak the outcome by design,
so `text_only` wins and the tool says fusion did not help. That is the tool
working.

## PhysioCGM

```bash
python scripts/make_physiocgm_fixture.py --out ./data/pcgm_fixture   # wiring check
python run_pipeline.py physiocgm --data_root ./data/pcgm_fixture --inspect_only
python run_pipeline.py physiocgm --data_root ./data/pcgm_fixture --no_ecg --model fusion
```

For the real data, request access from the PSI Lab (see the PhysioCGM README),
place it so `dataset/processed/<subject>/<n>.pkl` exists, and drop `--no_ecg`.

## Which dataset for which command

| Data | Command |
|---|---|
| Any CSV / Excel / Parquet | `tabular` |
| D1NAMO (Zenodo or Kaggle mirror) | `inspect`, then `d1namo` |
| PhysioCGM | `physiocgm --inspect_only`, then `physiocgm` |
| Manifest with images/notes/genes | `multimodal` |

---

## Where do I upload and test everything?

Three routes. Pick by data size.

### 1. A CSV that fits in a browser upload -> Streamlit, directly

```bash
streamlit run app.py
```

Upload, configure, run. Figures, metrics, feature importance and warnings all
render in the app.

### 2. A huge signal archive (D1NAMO, PhysioCGM) -> extract locally, upload the table

Raw recordings are 10+ GB and will never fit a hosted container. But the feature
table they reduce to is a couple of megabytes:

```bash
python run_pipeline.py d1namo --data_root ./data/d1namo \
    --export_features d1namo_feats.csv --export_only

python run_pipeline.py physiocgm --data_root ./data/PhysioCGM \
    --export_features pcgm_feats.csv --export_only
```

Upload that CSV to the app like any other table. Set **group column** to
`subject_id` and **split strategy** to `group`.

The heavy extraction happens once, locally; everything after it is portable.
Results are identical either way.

### 3. Anything the app cannot run -> run it in the CLI, view it in the app

Multimodal, fusion and full D1NAMO/PhysioCGM runs are CLI-only. Their output
still lands in `run_results/`, and the app reads it:

```bash
streamlit run app.py     # sidebar -> "Browse past runs"
```

Every run appears there with its summary, figures, model table, predictions and
full JSON.

### Hosting

| Where | Good for |
|---|---|
| `streamlit run app.py` locally | everything; reads your `run_results/` |
| Streamlit Community Cloud | CSV uploads only; use `requirements.txt` |

Community Cloud containers are ephemeral, so past runs made on your own machine
will not appear there. Use it as an upload-and-analyse front end, and run heavy
work locally.
