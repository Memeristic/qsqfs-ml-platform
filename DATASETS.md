# Which files to download

Both datasets ship far more than this study needs. Download the parts listed
here and skip the rest.

---

## D1NAMO

The split release lets you take only what you need:
**<https://zenodo.org/records/5651217>**

The single-archive version (`D1NAMO.tgz`, 10.2 GB) contains the same data and is
harder to resume if the download drops. Use the split release.

### Download these two — about 1.4 GB

| File | Size | Why |
|---|---|---|
| `diabetes_subset_sensor_data.zip` | ~1.1 GB | Zephyr summary, breathing, accelerometer and RR-interval files for the 9 T1D participants |
| `diabetes_subset_pictures-glucose-food-insulin.zip` | ~252 MB | contains `glucose.csv` — **the target variable** |

Both are required. Sensor data without glucose gives you features and nothing to
predict; glucose without sensors gives you a target and no features.

### Optional

| File | Size | When you would want it |
|---|---|---|
| `diabetes_subset_ecg_data.zip` | ~1.7 GB | only if the RR files turn out to be missing (see below), or if you specifically want raw-waveform ECG features |
| `healthy_subset_*` | ~7 GB | only if your research question compares T1D against healthy controls |

**Skip the healthy subset unless you need controls.** It is most of the archive
size and contributes nothing to a glucose-prediction study in diabetes.

### After extracting

```
python run_pipeline.py inspect --data_root ./data/d1namo
```

Read the `signals` column in that output.

- **If you see `rr:` counts** — you have RR intervals and get proper HRV
  (SDNN, RMSSD, pNN50, LF/HF). You do not need the ECG archive.
- **If `rr:` is absent or zero** — download `diabetes_subset_ecg_data.zip` and
  rerun with `--use_ecg`, which detects R-peaks from the 250 Hz waveform. Slower,
  same result.

I could not verify the RR file situation directly — Zenodo is not reachable from
where this code was built — so check rather than assume.

### Then

```bash
python run_pipeline.py d1namo --data_root ./data/d1namo ^
    --window 30 --step 10 --horizon 30 --cohorts diabetes --cv_type group ^
    --export_features ./data/d1namo_feats.csv
```

---

## PhysioCGM

**<https://github.com/PSI-TAMU/PhysioCGM>**

Access is gated: the data lives on the TAMU PSI Lab drive and you must contact
the authors. **Start that request early** — it is the longest lead time in this
project, and nothing about it is under your control.

### What to ask for

Request the **processed** release if it exists:

```
dataset/processed/<subject>/<n>.pkl
```

Each pickle is one aligned 5-minute CGM segment holding the Zephyr and Empatica
E4 signals. This platform reads that directly.

If you are only offered the raw release (`dataset/raw/`), run **their**
`preprocess.py` first. The alignment work has been done by the dataset's own
authors and there is no reason to redo it.

### If you must choose a subset

The full set is 10 participants over 17 days. In descending order of usefulness
for glucose prediction:

1. **Dexcom CGM** — the target; without it there is no study
2. **Zephyr Summary** — HR, BR, posture, activity; cheap and informative
3. **Zephyr RR** — proper HRV
4. **Empatica BVP** — PPG pulse rate and pulse-rate variability
5. **Empatica EDA + TEMP** — autonomic arousal signals
6. **Accelerometer** (either device) — activity context
7. **Zephyr raw ECG** — largest by far, and mostly redundant if you have RR

Items 1–6 are a complete study. Item 7 roughly doubles the storage for a modest
addition.

### Then

```bash
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM --inspect_only
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM ^
    --horizon_steps 6 --export_features ./data/pcgm_feats.csv
```

---

## Why `--export_features` matters

Feature extraction from raw recordings is slow — minutes to hours depending on
how much you downloaded. The resulting table is a few hundred kilobytes.

Extract once:

```bash
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM ^
    --export_features feats.csv --export_only
```

Then every subsequent experiment — different search settings, different models,
the Streamlit app, tuning — runs on that small CSV in seconds. Set the group
column to `subject_id` and the split strategy to `group`.

One caveat: the exported file is a snapshot of the window and horizon settings
you used. Change `--window` or `--horizon` and you must re-export. The filename
will not tell you which settings produced it, so name them accordingly
(`pcgm_w30_h30.csv`).

---

## A note on cohort size

Both datasets have around 10 participants. Grouped cross-validation means your
effective sample size is closer to 10 than to the thousands of windows the row
count suggests.

That is a real constraint on what can be concluded, not a flaw in the data or
the code — and stating it plainly in your limitations section is far stronger
than hoping a reviewer does not raise it.
