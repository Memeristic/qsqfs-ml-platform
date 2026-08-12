"""Load datasets split across several files, in several formats.

Real datasets rarely arrive as one tidy CSV. This module handles:

  * many files stacked row-wise   (train.csv + test.csv, or one file per site)
  * many files joined column-wise (demographics.csv + labs.csv on patient_id)
  * compressed files              (.gz, .bz2, .zip, .xz)
  * mixed formats in one load     (.csv, .tsv, .xlsx, .parquet)
  * whole folders and archives    (point at a .zip and it reads what is inside)

Two combining modes, and the difference matters:

  ``stack``  same columns, more rows. Adds a ``source_file`` column so you can
             tell which file a row came from -- and so you can use it as the
             group column, which is what you want when each file is a site or a
             subject.
  ``merge``  different columns, same subjects. Requires a key column. Reports
             how many rows matched, because a silent partial join that drops
             two thirds of your cohort is a common and costly mistake.
"""

from __future__ import annotations

import gzip
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pq"}
COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}


def _read_buffer(buffer, name: str, **kwargs) -> Optional[pd.DataFrame]:
    suffix = Path(name.lower().removesuffix(".gz").removesuffix(".bz2")
                  .removesuffix(".xz")).suffix
    try:
        if suffix == ".tsv":
            return pd.read_csv(buffer, sep="\t", **kwargs)
        if suffix in (".csv", ".txt"):
            return pd.read_csv(buffer, **kwargs)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(buffer, **kwargs)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(buffer, **kwargs)
    except Exception as exc:
        logger.warning("Could not parse %s: %s", name, exc)
    return None


def read_any(path: str | Path, **kwargs) -> Dict[str, pd.DataFrame]:
    """Read one path into {display name -> DataFrame}.

    A .zip may contain several tables, so this returns a mapping rather than a
    single frame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    name = path.name.lower()
    frames: Dict[str, pd.DataFrame] = {}

    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith("/") or member.startswith("__MACOSX"):
                    continue
                stem = Path(member.lower())
                if stem.suffix not in TABULAR_SUFFIXES and \
                        stem.suffix not in COMPRESSED_SUFFIXES:
                    continue
                with archive.open(member) as handle:
                    data = handle.read()
                if member.lower().endswith(".gz"):
                    data = gzip.decompress(data)
                    member = member[:-3]
                frame = _read_buffer(io.BytesIO(data), member, **kwargs)
                if frame is not None and not frame.empty:
                    frames[f"{path.name}:{Path(member).name}"] = frame
        return frames

    if name.endswith((".gz", ".bz2", ".xz")):
        # pandas decompresses these transparently from a path.
        frame = _read_buffer(path, path.name, **kwargs)
        if frame is not None:
            frames[path.name] = frame
        return frames

    frame = _read_buffer(path, path.name, **kwargs)
    if frame is not None:
        frames[path.name] = frame
    return frames


def read_folder(folder: str | Path, pattern: str = "*", recursive: bool = True,
                **kwargs) -> Dict[str, pd.DataFrame]:
    """Read every tabular file in a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    paths = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))

    frames: Dict[str, pd.DataFrame] = {}
    for path in paths:
        if not path.is_file():
            continue
        suffix = Path(path.name.lower().removesuffix(".gz")).suffix
        if suffix not in TABULAR_SUFFIXES and path.suffix.lower() != ".zip":
            continue
        frames.update(read_any(path, **kwargs))
    logger.info("Read %d table(s) from %s", len(frames), folder)
    return frames


def stack_frames(frames: Dict[str, pd.DataFrame],
                 add_source: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Concatenate row-wise, reporting any column mismatch rather than hiding it."""
    if not frames:
        raise ValueError("No tables to stack.")

    column_sets = {name: set(frame.columns) for name, frame in frames.items()}
    common = set.intersection(*column_sets.values()) if column_sets else set()
    union = set.union(*column_sets.values()) if column_sets else set()
    inconsistent = sorted(union - common)

    prepared = []
    for name, frame in frames.items():
        frame = frame.copy()
        if add_source:
            frame["source_file"] = name
        prepared.append(frame)

    combined = pd.concat(prepared, ignore_index=True, sort=False)
    report = {
        "mode": "stack",
        "n_files": len(frames),
        "files": {name: len(frame) for name, frame in frames.items()},
        "n_rows": int(len(combined)),
        "n_columns": int(combined.shape[1]),
        "columns_in_every_file": sorted(common),
        "columns_missing_from_some_files": inconsistent,
        "source_column_added": add_source,
    }
    if inconsistent:
        report["warning"] = (
            f"{len(inconsistent)} column(s) are not present in every file and will "
            "contain missing values for the files that lack them: "
            f"{', '.join(inconsistent[:10])}"
            + (" ..." if len(inconsistent) > 10 else "")
        )
        logger.warning(report["warning"])
    return combined, report


def merge_frames(frames: Dict[str, pd.DataFrame], key: str,
                 how: str = "inner") -> Tuple[pd.DataFrame, Dict]:
    """Join column-wise on a shared key, reporting match rates honestly."""
    if not frames:
        raise ValueError("No tables to merge.")
    missing = [name for name, frame in frames.items() if key not in frame.columns]
    if missing:
        raise KeyError(f"Key '{key}' is missing from: {', '.join(missing)}")

    names = list(frames)
    combined = frames[names[0]].copy()
    steps = [{"file": names[0], "rows": int(len(combined)),
              "unique_keys": int(combined[key].nunique())}]

    for name in names[1:]:
        right = frames[name]
        before = len(combined)
        overlap = (set(combined.columns) & set(right.columns)) - {key}
        combined = combined.merge(right, on=key, how=how,
                                  suffixes=("", f"_{Path(name).stem[:12]}"))
        steps.append({
            "file": name,
            "rows_before": before,
            "rows_after": int(len(combined)),
            "unique_keys_in_file": int(right[key].nunique()),
            "duplicate_columns_suffixed": sorted(overlap),
        })

    report = {
        "mode": "merge", "key": key, "how": how, "n_files": len(frames),
        "steps": steps, "n_rows": int(len(combined)),
        "n_columns": int(combined.shape[1]),
    }
    first_rows = steps[0]["rows"]
    if len(combined) < first_rows * 0.9:
        report["warning"] = (
            f"The join kept {len(combined)} of {first_rows} rows "
            f"({len(combined)/max(1,first_rows):.1%}). Check that '{key}' has the "
            "same format in every file, or use how='left' to keep all rows."
        )
        logger.warning(report["warning"])
    return combined, report


def load_dataset(
    paths: Sequence[str | Path], mode: str = "auto", key: Optional[str] = None,
    how: str = "inner", add_source: bool = True, **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """Load and combine any mix of files, folders and archives.

    ``mode='auto'`` stacks when the files share most of their columns, and
    merges when they do not and a key was supplied.
    """
    frames: Dict[str, pd.DataFrame] = {}
    for path in paths:
        path = Path(path)
        frames.update(read_folder(path, **kwargs) if path.is_dir()
                      else read_any(path, **kwargs))

    if not frames:
        raise ValueError("No readable tables found in the supplied paths.")
    if len(frames) == 1:
        name, frame = next(iter(frames.items()))
        return frame, {"mode": "single", "files": {name: len(frame)},
                       "n_rows": int(len(frame)), "n_columns": int(frame.shape[1])}

    if mode == "auto":
        sets = [set(f.columns) for f in frames.values()]
        overlap = len(set.intersection(*sets)) / max(1, len(set.union(*sets)))
        mode = "stack" if overlap >= 0.8 or not key else "merge"
        logger.info("Auto-selected mode='%s' (column overlap %.0f%%).", mode, overlap * 100)

    if mode == "merge":
        if not key:
            raise ValueError("mode='merge' needs a key column to join on.")
        return merge_frames(frames, key, how)
    return stack_frames(frames, add_source)


def describe_files(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Preview what would be loaded, without combining anything."""
    rows = []
    for path in paths:
        path = Path(path)
        try:
            frames = read_folder(path) if path.is_dir() else read_any(path)
        except Exception as exc:
            rows.append({"file": str(path), "status": f"error: {exc}"})
            continue
        for name, frame in frames.items():
            rows.append({
                "file": name, "rows": len(frame), "columns": frame.shape[1],
                "status": "ok",
                "column_preview": ", ".join(map(str, frame.columns[:6]))
                + (" ..." if frame.shape[1] > 6 else ""),
            })
    return pd.DataFrame(rows)
