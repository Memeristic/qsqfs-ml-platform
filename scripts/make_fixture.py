#!/usr/bin/env python3
"""Generate a small synthetic dataset shaped like the D1NAMO archive.

For testing the loader and the pipeline wiring only. The signals are
synthetic and any model score obtained on this fixture is meaningless --
it exists so you can verify the code runs before committing to the real
10 GB download.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# numpy 2.5 emits a DeprecationWarning from inside pandas' own
# timedelta handling. It is noise from a library interaction, not a
# problem with this script, and it does not affect the output.
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=".*generic.*unit.*")


def write_subject(root: Path, cohort: str, sid: str, hours: float, rng, sessions: int = 2) -> None:
    base = pd.Timestamp("2014-10-01 08:00:00")
    per_session = hours / sessions

    for s in range(sessions):
        start = base + pd.Timedelta(hours=s * (per_session + 0.5))
        tag = start.strftime("%Y_%m_%d-%H_%M_%S")
        sess = root / f"{cohort}_subset_sensor_data" / sid / "sensor_data" / tag
        sess.mkdir(parents=True, exist_ok=True)
        n_sec = int(per_session * 3600)

        t1 = start + pd.to_timedelta(np.arange(n_sec), unit="s")
        drift = np.sin(np.linspace(0, 4 * np.pi, n_sec))
        pd.DataFrame({
            "Time": t1.strftime("%d/%m/%Y %H:%M:%S.000"),
            "HR": 70 + 8 * drift + rng.normal(0, 3, n_sec),
            "BR": 15 + 2 * drift + rng.normal(0, 1, n_sec),
            "SkinTemp": 33 + rng.normal(0, 0.2, n_sec),
            "Activity": np.abs(0.2 + 0.15 * drift + rng.normal(0, 0.05, n_sec)),
            "PeakAccel": np.abs(0.5 + rng.normal(0, 0.2, n_sec)),
            "HRV": 60 + 15 * drift + rng.normal(0, 8, n_sec),
        }).to_csv(sess / f"{tag}_Summary.csv", index=False)

        n_rr = int(n_sec / 0.85)
        rr_ms = 850 - 80 * np.sin(np.linspace(0, 4 * np.pi, n_rr)) + rng.normal(0, 35, n_rr)
        pd.DataFrame({
            "Time": (start + pd.to_timedelta(np.cumsum(rr_ms) / 1000, unit="s"))
                    .strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3],
            "RR": rr_ms,
        }).to_csv(sess / f"{tag}_RR.csv", index=False)

        fs_br = 25
        n_br = n_sec * fs_br
        tb = np.arange(n_br) / fs_br
        pd.DataFrame({
            "Time": (start + pd.to_timedelta(tb, unit="s")).strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3],
            "BreathingWaveform": np.sin(2 * np.pi * 0.25 * tb) + rng.normal(0, 0.15, n_br),
        }).to_csv(sess / f"{tag}_Breathing.csv", index=False)

        fs_ac = 50
        n_ac = n_sec * fs_ac
        ta = np.arange(n_ac) / fs_ac
        pd.DataFrame({
            "Time": (start + pd.to_timedelta(ta, unit="s")).strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3],
            "X": np.sin(2 * np.pi * 1.7 * ta) * 0.3 + rng.normal(0, 0.1, n_ac),
            "Y": np.cos(2 * np.pi * 1.7 * ta) * 0.3 + rng.normal(0, 0.1, n_ac),
            "Z": 1.0 + rng.normal(0, 0.05, n_ac),
        }).to_csv(sess / f"{tag}_Accel.csv", index=False)

    # Glucose every 5 minutes across the whole span, in mmol/L as D1NAMO records it.
    n_g = int(hours * 12) + 24
    gt = base + pd.to_timedelta(np.arange(n_g) * 5, unit="m")
    mmol = 7.0 + 2.0 * np.sin(np.linspace(0, 5, n_g)) + rng.normal(0, 0.4, n_g)
    gdir = root / f"{cohort}_subset_pictures-glucose-food" / sid
    gdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": gt.strftime("%d/%m/%Y"),
        "time": gt.strftime("%H:%M:%S"),
        "type": "cgm",
        "glucose": np.round(mmol, 2),
    }).to_csv(gdir / "glucose.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./data/d1namo_fixture")
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.subjects):
        cohort = "diabetes" if i % 2 == 0 else "healthy"
        write_subject(root, cohort, f"{i+1:03d}", args.hours, rng)
        print(f"  subject {i+1:03d} ({cohort})")
    print(f"\nFixture written to {root.resolve()}")
    print("This is synthetic data. Any score from it is meaningless.")


if __name__ == "__main__":
    main()
