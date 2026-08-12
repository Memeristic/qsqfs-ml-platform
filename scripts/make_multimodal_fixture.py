#!/usr/bin/env python3
"""Synthetic multimodal cohort: tabular + text + images + genomics.

Unlike the D1NAMO fixture, this one contains a REAL planted signal so you can
confirm the multimodal path actually learns. The generative rule is printed on
creation, so you know exactly what the model is supposed to recover.
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

PHRASES_POS = ["progressive dyspnoea on exertion", "recurrent chest discomfort",
               "poor glycaemic control noted", "oedema of the lower limbs"]
PHRASES_NEG = ["routine review, no new complaints", "asymptomatic at follow up",
               "stable observations throughout", "no acute concerns raised"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./data/multimodal_fixture")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_genes", type=int, default=40)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is required: pip install pillow")

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    n = args.n

    age = rng.normal(62, 11, n)
    bmi = rng.normal(28, 5, n)
    lab_crp = rng.lognormal(1.2, 0.8, n)
    genes = rng.normal(0, 1, (n, args.n_genes))

    # Ground truth: two tabular terms, two genes, and image brightness.
    brightness = rng.uniform(0.2, 0.8, n)
    logit = (-0.3 + 0.06 * (age - 62) + 0.12 * (bmi - 28)
             + 0.8 * genes[:, 3] - 0.6 * genes[:, 7] + 3.0 * (brightness - 0.5))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    paths = []
    for i in range(n):
        base = np.full((args.image_size, args.image_size), brightness[i])
        base += rng.normal(0, 0.08, base.shape)
        array = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        name = f"scan_{i:04d}.png"
        Image.fromarray(array).convert("L").save(images_dir / name)
        paths.append(f"images/{name}")

    notes = []
    for i in range(n):
        pool = PHRASES_POS if y[i] else PHRASES_NEG
        chosen = rng.choice(pool, size=2, replace=False)
        notes.append(
            f"Patient aged {int(age[i])} attended clinic. "
            f"{chosen[0].capitalize()}. {chosen[1].capitalize()}. "
            "Plan discussed with the patient and documented."
        )

    data = {
        "subject_id": [f"P{i:04d}" for i in range(n)],
        "age": age.round(1), "bmi": bmi.round(1), "lab_crp": lab_crp.round(2),
        "sex": rng.choice(["M", "F"], n),
        "note_text": notes,
        "scan_path": paths,
    }
    for j in range(args.n_genes):
        data[f"gene_{j:03d}"] = genes[:, j].round(4)
    data["outcome"] = y

    pd.DataFrame(data).to_csv(root / "manifest.csv", index=False)

    print(f"Wrote {root.resolve()}/manifest.csv  ({n} rows, {y.sum()} positive)")
    print(f"Wrote {n} images to {images_dir}")
    print("\nPlanted signal (what a working model should recover):")
    print("  tabular : age and bmi, weakly")
    print("  genomic : gene_003 (+) and gene_007 (-)")
    print("  image   : mean brightness, strongly")
    print("  text    : phrasing differs by outcome, so notes leak the label by design")
    print("\nThe text signal is deliberate: it demonstrates how easily clinical")
    print("notes leak an outcome. Compare --modalities with and without text.")


if __name__ == "__main__":
    main()
