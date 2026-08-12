#!/usr/bin/env python3
"""Synthetic PhysioCGM-shaped fixture.

Reproduces the exact processed-segment structure documented in the PhysioCGM
README and produced by their preprocess.py:

    dataset/processed/<subject>/<n>.pkl  ->
        {Index, Timestamp, glucose,
         zephyr: {Accel{Time,Vertical,Lateral,Sagittal},
                  Breathing{Time,BreathingWaveform},
                  ECG{Time,EcgWaveform},
                  Summary{Time,HR,BR,Posture,Activity,HRConfidence,ECGNoise}},
         e4:     {ACC{Time,x,y,z}, HR{Time,HR}, BVP{Time,BVP},
                  EDA{Time,EDA}, TEMP{Time,TEMP}}}

Signals are synthetic. A weak coupling between heart rate and glucose is
planted so the pipeline can be seen to do something, but no score from this
fixture means anything about real glucose prediction. Use it to check wiring
before you have the real data, which requires contacting the PSI Lab.
"""

from __future__ import annotations

import argparse
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# numpy 2.5 emits a DeprecationWarning from inside pandas' own
# timedelta handling. It is noise from a library interaction, not a
# problem with this script, and it does not affect the output.
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=".*generic.*unit.*")

SEGMENT_SECONDS = 300


def make_segment(start: pd.Timestamp, glucose: float, index: int, rng) -> dict:
    def times(fs, n):
        return list(start + pd.to_timedelta(np.arange(n) / fs, unit="s"))

    # Weak, deliberate coupling: higher glucose -> slightly higher HR.
    hr_base = 70 + 0.06 * (glucose - 140)

    n_ecg = int(SEGMENT_SECONDS * 250)
    rr = 60.0 / hr_base
    ecg = np.zeros(n_ecg)
    for beat in np.arange(0, SEGMENT_SECONDS, rr):
        idx = int(beat * 250)
        if idx < n_ecg:
            ecg[idx] = 1.0
    ecg = np.convolve(ecg, np.hanning(20), "same") + rng.normal(0, 0.03, n_ecg)

    n_br = SEGMENT_SECONDS * 25
    tb = np.arange(n_br) / 25.0
    n_ac = SEGMENT_SECONDS * 100
    ta = np.arange(n_ac) / 100.0
    n_bvp = SEGMENT_SECONDS * 64
    tv = np.arange(n_bvp) / 64.0
    n_eda = SEGMENT_SECONDS * 4

    return {
        "Index": index,
        "Timestamp": start,
        "glucose": float(glucose),
        "zephyr": {
            "Accel": {
                "Time": times(100, n_ac),
                "Vertical": (0.9 + 0.1 * np.sin(2 * np.pi * 1.6 * ta) + rng.normal(0, .05, n_ac)).tolist(),
                "Lateral": (0.1 + rng.normal(0, .05, n_ac)).tolist(),
                "Sagittal": (0.1 + rng.normal(0, .05, n_ac)).tolist(),
            },
            "Breathing": {
                "Time": times(25, n_br),
                "BreathingWaveform": (np.sin(2 * np.pi * 0.25 * tb) + rng.normal(0, .12, n_br)).tolist(),
            },
            "ECG": {"Time": times(250, n_ecg), "EcgWaveform": ecg.tolist()},
            "Summary": {
                "Time": times(1, SEGMENT_SECONDS),
                "HR": (hr_base + rng.normal(0, 2, SEGMENT_SECONDS)).tolist(),
                "BR": (15 + rng.normal(0, 1, SEGMENT_SECONDS)).tolist(),
                "Posture": rng.integers(-20, 20, SEGMENT_SECONDS).tolist(),
                "Activity": np.abs(rng.normal(0.2, .08, SEGMENT_SECONDS)).tolist(),
                "HRConfidence": rng.integers(90, 101, SEGMENT_SECONDS).tolist(),
                "ECGNoise": np.abs(rng.normal(0.01, .005, SEGMENT_SECONDS)).tolist(),
            },
        },
        "e4": {
            "ACC": {
                "Time": times(32, SEGMENT_SECONDS * 32),
                "x": rng.normal(0, 8, SEGMENT_SECONDS * 32).tolist(),
                "y": rng.normal(0, 8, SEGMENT_SECONDS * 32).tolist(),
                "z": rng.normal(60, 8, SEGMENT_SECONDS * 32).tolist(),
            },
            "HR": {"Time": times(1, SEGMENT_SECONDS),
                   "HR": (hr_base + rng.normal(0, 1.5, SEGMENT_SECONDS)).tolist()},
            "BVP": {"Time": times(64, n_bvp),
                    "BVP": (np.sin(2 * np.pi * (hr_base / 60) * tv) * 40
                            + rng.normal(0, 6, n_bvp)).tolist()},
            "EDA": {"Time": times(4, n_eda),
                    "EDA": (2.0 + 0.4 * np.sin(np.linspace(0, 2, n_eda))
                            + rng.normal(0, .05, n_eda)).tolist()},
            "TEMP": {"Time": times(4, n_eda),
                     "TEMP": (32.5 + rng.normal(0, .15, n_eda)).tolist()},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./data/physiocgm_fixture")
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--segments", type=int, default=40, help="5-min segments per subject")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out) / "dataset" / "processed"

    for s in range(args.subjects):
        subject = f"c1s{s + 1:02d}"
        out_dir = root / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        start = pd.Timestamp("2024-03-01 08:00:00")
        glucose = 140 + 40 * np.sin(np.linspace(0, 4, args.segments)) \
            + rng.normal(0, 8, args.segments)
        for i in range(args.segments):
            segment = make_segment(start + pd.Timedelta(minutes=5 * i),
                                   float(glucose[i]), i, rng)
            with open(out_dir / f"{i}.pkl", "wb") as handle:
                pickle.dump(segment, handle)
        print(f"  {subject}: {args.segments} segments")

    print(f"\nFixture written to {Path(args.out).resolve()}")
    print("Synthetic signals. Scores from this fixture are meaningless;")
    print("it exists to verify the loader and pipeline wiring only.")
    print("Real data: contact the PSI Lab (see the PhysioCGM README).")


if __name__ == "__main__":
    main()
