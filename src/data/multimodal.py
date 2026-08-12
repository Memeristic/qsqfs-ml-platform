"""Manifest-driven multimodal dataset.

A manifest is one CSV, one row per subject. Columns are assigned to modalities
either automatically or by explicit prefix:

    subject_id, target, age, bmi, lab_crp, ...        -> tabular
    note_text                                          -> text  (long strings)
    scan_path                                          -> image (file paths)
    gene_TP53, gene_BRCA1, ... | snp_rs1234, ...       -> genomic

Automatic assignment is a convenience, not a guarantee. The resolved schema is
printed before training and written into the run record so you can check it
caught what you meant. Override with --image_cols / --text_cols / --genomic_cols
whenever the guess is wrong.

Images are loaded lazily per batch. Loading a cohort of scans into RAM up front
is the fastest way to make this pipeline unusable on real data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii", ".nii.gz")
GENOMIC_PREFIXES = ("gene_", "snp_", "rs", "expr_", "chr")
TEXT_MIN_MEAN_LENGTH = 60


def infer_modalities(
    df: pd.DataFrame,
    target: str,
    exclude: Sequence[str] = (),
    image_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    genomic_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Assign every column to a modality. Explicit lists always win."""
    reserved = {target, *exclude}
    candidates = [c for c in df.columns if c not in reserved]

    assigned: Dict[str, List[str]] = {"image": [], "text": [], "genomic": [], "tabular": []}
    claimed = set()

    for kind, explicit in (("image", image_cols), ("text", text_cols),
                           ("genomic", genomic_cols)):
        for col in explicit or []:
            if col in candidates:
                assigned[kind].append(col)
                claimed.add(col)
            else:
                logger.warning("Column '%s' given as %s but not in the manifest.", col, kind)

    for col in candidates:
        if col in claimed:
            continue
        series = df[col]
        name = str(col).lower()

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.dropna().astype(str).head(200)
            if len(sample) and sample.str.lower().str.endswith(IMAGE_SUFFIXES).mean() > 0.5:
                assigned["image"].append(col)
                continue
            if len(sample) and sample.str.len().mean() > TEXT_MIN_MEAN_LENGTH:
                assigned["text"].append(col)
                continue

        if any(name.startswith(p) for p in GENOMIC_PREFIXES) and \
                pd.api.types.is_numeric_dtype(series):
            assigned["genomic"].append(col)
            continue

        assigned["tabular"].append(col)

    return {k: v for k, v in assigned.items() if v}


def build_schema(
    modalities: Dict[str, List[str]],
    df: pd.DataFrame,
    genomic_encoding: str = "expression",
) -> Dict[str, Dict]:
    """Turn a {block name -> columns} map into a per-modality encoder schema.

    Block names are not restricted to the four inferred kinds. Any name whose
    columns are numeric becomes its own tabular-encoded block, so a caller can
    pass domain-specific groupings -- ``{"ecg": [...], "ppg": [...],
    "eda": [...]}`` from PhysioCGM, for instance -- and get one encoder per
    sensor family rather than one undifferentiated feature vector.
    """
    schema: Dict[str, Dict] = {}
    for name, columns in modalities.items():
        if not columns:
            continue

        if name == "text":
            schema["text"] = {"type": "text", "columns": columns}
            continue

        if name == "image":
            if len(columns) > 1:
                logger.warning(
                    "Multiple image columns %s; using '%s'. One image encoder "
                    "handles one image column.", columns, columns[0],
                )
            schema["image"] = {"type": "image", "columns": columns[:1]}
            continue

        if name == "genomic":
            encoding = genomic_encoding
            if encoding == "auto":
                block = df[columns].to_numpy(dtype=float, na_value=np.nan)
                finite = block[np.isfinite(block)]
                integral = bool(
                    finite.size and np.all(finite == np.round(finite)) and finite.max() <= 2
                )
                encoding = "snp" if integral else "expression"
                logger.info("Genomic encoding auto-detected as '%s'.", encoding)
            schema["genomic"] = {"type": "genomic", "input_dim": len(columns),
                                 "encoding": encoding, "columns": columns}
            continue

        # Any other block: numeric, encoded by an MLP under its own name.
        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            logger.warning("Block '%s' has no numeric columns; skipped.", name)
            continue
        schema[name] = {"type": "tabular", "input_dim": len(numeric), "columns": numeric}

    return schema


def load_image(path: str | Path, size: int = 224) -> Tuple[torch.Tensor, bool]:
    """Load and resize one image. Returns (tensor, was_found)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for the image modality: pip install pillow") from exc

    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((size, size), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), True
    except Exception as exc:
        logger.debug("Could not load image %s: %s", path, exc)
        return torch.zeros(3, size, size), False


class MultimodalDataset(Dataset):
    """Serves one row across all modality blocks, plus a presence mask each.

    ``matrices`` maps a numeric block name to its already-preprocessed array.
    Text and image blocks are read from ``df`` lazily at access time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Dict],
        y: np.ndarray,
        matrices: Optional[Dict[str, np.ndarray]] = None,
        image_root: Optional[str | Path] = None,
        image_size: int = 224,
        regression: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.schema = schema
        self.y = np.asarray(y)
        self.matrices = matrices or {}
        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.regression = regression

        for name, spec in schema.items():
            if spec["type"] in ("tabular", "genomic") and name not in self.matrices:
                raise KeyError(f"No preprocessed matrix supplied for block '{name}'.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        inputs: Dict = {}
        presence: Dict = {}

        for name, spec in self.schema.items():
            kind = spec["type"]

            if kind == "genomic":
                dtype = torch.long if spec.get("encoding") == "snp" else torch.float32
                inputs[name] = torch.as_tensor(self.matrices[name][index], dtype=dtype)
                presence[name] = 1.0

            elif kind == "tabular":
                inputs[name] = torch.as_tensor(
                    self.matrices[name][index], dtype=torch.float32
                )
                presence[name] = 1.0

            elif kind == "text":
                parts = [str(self.df.at[index, c]) for c in spec["columns"]
                         if pd.notna(self.df.at[index, c])]
                inputs[name] = " ".join(parts)
                presence[name] = 1.0 if parts else 0.0

            elif kind == "image":
                raw = self.df.at[index, spec["columns"][0]]
                if pd.isna(raw):
                    inputs[name] = torch.zeros(3, self.image_size, self.image_size)
                    presence[name] = 0.0
                else:
                    path = Path(raw)
                    if self.image_root and not path.is_absolute():
                        path = self.image_root / path
                    tensor, found = load_image(path, self.image_size)
                    inputs[name] = tensor
                    presence[name] = 1.0 if found else 0.0

        target = torch.tensor(
            float(self.y[index]) if self.regression else int(self.y[index]),
            dtype=torch.float32 if self.regression else torch.long,
        )
        return inputs, presence, target


def collate(batch):
    """Stack tensor modalities; keep text as a list for the tokeniser."""
    inputs_list, presence_list, targets = zip(*batch)
    names = inputs_list[0].keys()

    inputs: Dict = {}
    for name in names:
        values = [item[name] for item in inputs_list]
        inputs[name] = list(values) if isinstance(values[0], str) else torch.stack(values)

    presence = {
        name: torch.tensor([item[name] for item in presence_list], dtype=torch.float32)
        for name in presence_list[0]
    }
    return inputs, presence, torch.stack(targets)
