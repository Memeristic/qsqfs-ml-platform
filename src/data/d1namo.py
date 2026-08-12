"""D1NAMO loader (Zenodo records 1421616 / 5651217).

Design decisions, and why:

**The layout is discovered, not assumed.** The archive nests
``<cohort>/<subject>/sensor_data/<session>/<...>_Summary.csv`` and the exact
depth differs between the single-tgz and the split-zip releases. Rather than
glob one hardcoded pattern and fail silently with zero samples, this module
walks the tree, classifies every CSV by filename suffix, and groups by subject.
Run ``python run_pipeline.py inspect --data_root ...`` to print what it found
before committing to a full run.

**Alignment is by wall-clock timestamp.** Sensor sessions start and stop
throughout the day; glucose is sampled every ~5 min. Index-based alignment (or
``scipy.signal.resample`` to stretch glucose onto the ECG length) fabricates
correspondence that does not exist and adds Fourier ringing to a non-periodic
CGM trace. Here every window carries real start/end times and the target is
interpolated at the true prediction time, subject to a maximum gap.

**HRV comes from RR intervals.** Either the Zephyr ``_RR.csv`` file or QRS
detection on 250 Hz ECG. Never from a downsampled ECG amplitude trace.

**Windows carry their subject id.** ``groups`` is returned so downstream
splitting can keep a subject out of both train and test. Overlapping windows
from one person are near-duplicates; an ungrouped random split will report
optimistic numbers.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import (
    accel_features, basic_stats, breathing_features, detect_r_peaks,
    ecg_waveform_features, hrv_features,
)

logger = logging.getLogger(__name__)

# Nominal Zephyr BioHarness 3 rates; the actual rate is inferred per file and
# only falls back to these when timestamps are unusable.
NOMINAL_FS = {"ecg": 250.0, "breathing": 25.0, "accel": 100.0, "summary": 1.0}

# Zephyr BioHarness writes several summary variants. PhysioCGM's own
# verify_folder() requires "*_SummaryEnhanced.csv", which a "_summary.csv"-only
# match would silently skip -- so every variant is listed here explicitly.
SIGNAL_SUFFIXES = {
    "summary": ("_summary.csv", "_summaryenhanced.csv", "_summaryenhanced_.csv"),
    "breathing": ("_breathing.csv", "_br.csv", "_breathingwaveform.csv"),
    "accel": ("_accel.csv", "_acceleration.csv", "_accelerometer.csv"),
    "ecg": ("_ecg.csv", "_ecgwaveform.csv"),
    "rr": ("_rr.csv", "_rrdata.csv", "_rtor.csv"),
    "events": ("_events.csv", "_eventdata.csv", "_general.csv"),
}
GLUCOSE_NAMES = ("glucose.csv", "glucose_data.csv", "cgm.csv")
SESSION_TIME_RE = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})[-_](\d{2})[_-](\d{2})[_-](\d{2})")
# Subject folder naming differs by release: D1NAMO uses "001", PhysioCGM uses
# "c1s01". A digits-only pattern would assign every PhysioCGM file to "unknown".
SUBJECT_RES = (
    re.compile(r"^(\d{1,3})$"),                       # 001
    re.compile(r"^S[_-]?(\d{1,3})$", re.I),           # S1, S_01
    re.compile(r"^subject[_-]?(\d{1,3})$", re.I),     # subject_01
    re.compile(r"^(c\d+s\d+)$", re.I),                # c1s01  (PhysioCGM)
    re.compile(r"^(p\d{1,4})$", re.I),                # P001
)


# ======================================================================
# Discovery
# ======================================================================
@dataclass
class SubjectIndex:
    subject_id: str
    cohort: str = "unknown"
    glucose_files: List[Path] = field(default_factory=list)
    signal_files: Dict[str, List[Path]] = field(default_factory=lambda: defaultdict(list))

    def summary(self) -> Dict:
        return {
            "subject_id": self.subject_id,
            "cohort": self.cohort,
            "n_glucose_files": len(self.glucose_files),
            "signals": {k: len(v) for k, v in sorted(self.signal_files.items())},
        }


def _classify(path: Path) -> Optional[str]:
    name = path.name.lower()
    for kind, suffixes in SIGNAL_SUFFIXES.items():
        if any(name.endswith(s) for s in suffixes):
            return kind
    if name in GLUCOSE_NAMES or ("glucose" in name and name.endswith(".csv")):
        return "glucose"
    return None


def _subject_from_path(path: Path, root: Path) -> Optional[str]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in reversed(parts[:-1]):
        for pattern in SUBJECT_RES:
            match = pattern.match(part)
            if match:
                value = match.group(1)
                return value.zfill(3) if value.isdigit() else value.lower()
    return None


def _cohort_from_path(path: Path) -> str:
    lowered = str(path).lower()
    if "diabet" in lowered:
        return "diabetes"
    if "healthy" in lowered or "control" in lowered:
        return "healthy"
    return "unknown"


def discover(data_root: str | Path) -> Dict[str, SubjectIndex]:
    """Walk ``data_root`` and index every recognised D1NAMO CSV by subject."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    subjects: Dict[str, SubjectIndex] = {}
    n_seen = n_matched = 0

    for path in root.rglob("*.csv"):
        n_seen += 1
        kind = _classify(path)
        if kind is None:
            continue
        subject_id = _subject_from_path(path, root) or "unknown"
        entry = subjects.setdefault(
            subject_id, SubjectIndex(subject_id, _cohort_from_path(path))
        )
        if entry.cohort == "unknown":
            entry.cohort = _cohort_from_path(path)
        if kind == "glucose":
            entry.glucose_files.append(path)
        else:
            entry.signal_files[kind].append(path)
        n_matched += 1

    logger.info(
        "Discovery: %d CSV files scanned, %d recognised, %d subjects.",
        n_seen, n_matched, len(subjects),
    )
    if not subjects:
        logger.error(
            "Nothing recognised under %s. Expected Zephyr files ending in "
            "_Summary.csv/_Breathing.csv/_Accel.csv/_ECG.csv/_RR.csv plus a "
            "glucose.csv per subject.", root,
        )
    return dict(sorted(subjects.items()))


def describe(data_root: str | Path) -> Dict:
    """Human-readable inventory, for the ``inspect`` command."""
    subjects = discover(data_root)
    return {
        "data_root": str(Path(data_root).resolve()),
        "n_subjects": len(subjects),
        "subjects": [s.summary() for s in subjects.values()],
        "usable_subjects": [
            s.subject_id for s in subjects.values()
            if s.glucose_files and any(s.signal_files.values())
        ],
    }


# ======================================================================
# Parsing
# ======================================================================
def to_epoch_seconds(values) -> np.ndarray:
    """Datetimes -> float seconds since epoch.

    ``Series.astype("int64")`` is NOT always nanoseconds: pandas 2.x preserves
    datetime64[s]/[us]/[ms] resolution, so a naive ``/ 1e9`` silently rescales
    the timeline by a factor of 1000 and every target lookup falls out of range.
    Casting to datetime64[ns] first makes the unit explicit.
    """
    series = pd.to_datetime(pd.Series(values)).astype("datetime64[ns]")
    return series.astype("int64").to_numpy() / 1e9


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Zephyr writes day-first; pandas guesses month-first. Try both."""
    for kwargs in ({"dayfirst": True}, {"dayfirst": False}):
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", **kwargs)
        if parsed.notna().mean() > 0.8:
            return parsed
    return pd.to_datetime(series, errors="coerce")


def _time_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if str(col).strip().lower() in ("time", "timestamp", "datetime", "date_time"):
            return col
    return None


def _session_start(path: Path) -> Optional[pd.Timestamp]:
    for part in reversed(path.parts):
        match = SESSION_TIME_RE.search(part)
        if match:
            y, mo, d, h, mi, s = map(int, match.groups())
            try:
                return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s)
            except ValueError:
                return None
    return None


def read_signal_file(path: Path, kind: str, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Read one Zephyr CSV and attach a real DatetimeIndex-style 'time' column."""
    try:
        df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return None
    if df.empty:
        return None
    df.columns = [str(c).strip() for c in df.columns]

    time_col = _time_column(df)
    if time_col is not None:
        times = _parse_datetime(df[time_col])
        if times.notna().mean() < 0.5:
            times = None
        else:
            df = df.drop(columns=[time_col])
    else:
        times = None

    if times is None:
        start = _session_start(path)
        if start is None:
            logger.debug("No usable timestamps in %s; skipping.", path.name)
            return None
        fs = NOMINAL_FS.get(kind, 1.0)
        times = pd.Series(start + pd.to_timedelta(np.arange(len(df)) / fs, unit="s"))
        logger.debug("Synthesised timeline for %s at %.1f Hz.", path.name, fs)

    df = df.assign(time=pd.Series(times).values).dropna(subset=["time"])
    return df.sort_values("time").reset_index(drop=True) if len(df) else None


def infer_fs(times: pd.Series, fallback: float) -> float:
    if len(times) < 10:
        return fallback
    deltas = np.diff(to_epoch_seconds(times))
    deltas = deltas[(deltas > 0) & (deltas < 60)]
    if deltas.size < 5:
        return fallback
    median = float(np.median(deltas))
    return 1.0 / median if median > 0 else fallback


def load_glucose(paths: List[Path]) -> Optional[pd.DataFrame]:
    """Return a tidy ['time', 'glucose'] frame from one or more glucose CSVs."""
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue
        if df.empty:
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "date" in df.columns and "time" in df.columns:
            times = _parse_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        else:
            col = _time_column(df) or next(
                (c for c in df.columns if "time" in c or "date" in c), None
            )
            times = _parse_datetime(df[col]) if col else None
        if times is None or times.notna().mean() < 0.5:
            logger.warning("No parseable timestamps in %s.", path.name)
            continue

        value_col = next(
            (c for c in df.columns if "glucose" in c or c in ("value", "glycemia", "bg")), None
        )
        if value_col is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = numeric[0] if numeric else None
        if value_col is None:
            logger.warning("No glucose value column in %s (cols: %s).", path.name, list(df.columns))
            continue

        frames.append(pd.DataFrame({
            "time": times,
            "glucose": pd.to_numeric(df[value_col], errors="coerce"),
        }).dropna())

    if not frames:
        return None
    out = pd.concat(frames).dropna().sort_values("time").drop_duplicates("time")
    return out.reset_index(drop=True) if len(out) else None


def normalise_glucose_units(glucose: pd.Series) -> Tuple[pd.Series, str]:
    """D1NAMO records mmol/L. Convert to mg/dL and say so, rather than silently
    mixing units across sources."""
    median = float(glucose.median())
    if median < 35:  # mmol/L range
        return glucose * 18.018, "mg/dL (converted from mmol/L)"
    return glucose, "mg/dL"


# ======================================================================
# Windowing
# ======================================================================
def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["time"] >= start) & (df["time"] < end)]


def _numeric_window_features(df: pd.DataFrame, prefix: str, max_cols: int = 12) -> Dict[str, float]:
    """Generic stats over every numeric column of a Zephyr Summary-style file."""
    out: Dict[str, float] = {}
    numeric = df.select_dtypes(include=[np.number])
    for col in list(numeric.columns)[:max_cols]:
        values = numeric[col].to_numpy(dtype=float)
        if np.isfinite(values).sum() < 3:
            continue
        tag = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        out.update(basic_stats(values, f"{prefix}_{tag}"))
    return out


class D1NAMOLoader:
    """Build a window-level feature matrix and a glucose target from D1NAMO."""

    def __init__(
        self,
        data_root: str | Path,
        window_minutes: int = 30,
        step_minutes: int = 10,
        horizon_minutes: int = 30,
        max_target_gap_minutes: float = 7.5,
        use_ecg: bool = False,
        max_subjects: Optional[int] = None,
        min_coverage: float = 0.5,
        cohorts: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.window = pd.Timedelta(minutes=window_minutes)
        self.step = pd.Timedelta(minutes=step_minutes)
        self.horizon = pd.Timedelta(minutes=horizon_minutes)
        self.max_gap = pd.Timedelta(minutes=max_target_gap_minutes)
        self.use_ecg = use_ecg
        self.max_subjects = max_subjects
        self.min_coverage = min_coverage
        self.cohorts = [c.lower() for c in cohorts] if cohorts else None
        self.report: Dict = {}

    # ------------------------------------------------------------------
    def load(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
        """Returns (features_df, y, groups, window_end_times)."""
        subjects = discover(self.data_root)
        if self.cohorts:
            subjects = {k: v for k, v in subjects.items() if v.cohort in self.cohorts}
        usable = [s for s in subjects.values() if s.glucose_files and any(s.signal_files.values())]
        if self.max_subjects:
            usable = usable[: self.max_subjects]
        if not usable:
            raise ValueError(
                f"No subject under {self.data_root} has both glucose and sensor files. "
                "Run the 'inspect' command to see what was found."
            )

        rows: List[Dict] = []
        per_subject: List[Dict] = []
        unit_label = "mg/dL"

        for subject in usable:
            logger.info("Subject %s (%s)", subject.subject_id, subject.cohort)
            glucose = load_glucose(subject.glucose_files)
            if glucose is None or len(glucose) < 3:
                logger.warning("  no usable glucose; skipped.")
                per_subject.append({"subject": subject.subject_id, "n_windows": 0,
                                    "reason": "no glucose"})
                continue
            glucose["glucose"], unit_label = normalise_glucose_units(glucose["glucose"])

            n_before = len(rows)
            rows.extend(self._windows_for_subject(subject, glucose))
            per_subject.append({
                "subject": subject.subject_id,
                "cohort": subject.cohort,
                "n_glucose_readings": int(len(glucose)),
                "glucose_span_hours": round(
                    (glucose["time"].max() - glucose["time"].min()).total_seconds() / 3600, 2
                ),
                "n_windows": len(rows) - n_before,
            })
            logger.info("  -> %d windows", len(rows) - n_before)

        if not rows:
            raise ValueError(
                "No windows produced. Common causes: sensor sessions and glucose "
                "readings do not overlap in time, or the prediction horizon runs "
                "past the end of every glucose trace. Try a shorter --horizon."
            )

        df = pd.DataFrame(rows)
        meta_cols = ["subject_id", "cohort", "window_start", "window_end", "target_time", "glucose"]
        y = df["glucose"].to_numpy(dtype=float)
        groups = df["subject_id"].to_numpy()
        times = df["window_end"]
        features = df.drop(columns=[c for c in meta_cols if c in df.columns])

        # Drop features that are missing for most windows, then median-fill the rest.
        coverage = features.notna().mean()
        keep = coverage[coverage >= self.min_coverage].index.tolist()
        dropped = [c for c in features.columns if c not in keep]
        features = features[keep].astype(float)
        features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

        self.report = {
            "n_windows": int(len(df)),
            "n_features": int(features.shape[1]),
            "n_subjects": int(df["subject_id"].nunique()),
            "target_units": unit_label,
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y)),
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
            "window_minutes": self.window.total_seconds() / 60,
            "step_minutes": self.step.total_seconds() / 60,
            "horizon_minutes": self.horizon.total_seconds() / 60,
            "window_overlap_fraction": round(
                max(0.0, 1 - self.step / self.window), 3
            ),
            "ecg_used": self.use_ecg,
            "features_dropped_low_coverage": dropped,
            "per_subject": per_subject,
            "caveats": [
                "Windows overlap, so rows from one subject are correlated. Use "
                "cv_type='group' or 'time_series'; a shuffled split will be optimistic.",
                "Glucose targets are interpolated between CGM readings taken roughly "
                "every 5 minutes, within the configured maximum gap.",
            ],
        }
        logger.info(
            "Built %d windows x %d features from %d subjects (target in %s).",
            len(df), features.shape[1], df["subject_id"].nunique(), unit_label,
        )
        return features, y, groups, times

    # ------------------------------------------------------------------
    def _windows_for_subject(self, subject: SubjectIndex, glucose: pd.DataFrame) -> List[Dict]:
        """One row per window, with every available signal contributing columns.

        All signals share a single quantised time grid. Deriving a separate grid
        per signal file would give each modality slightly different window
        boundaries (an RR series starts at the first beat, not at second zero),
        and the rows would never merge -- you would end up with one sparse row
        per modality instead of one complete row per window.
        """
        wanted = ["summary", "breathing", "accel", "rr"] + (["ecg"] if self.use_ecg else [])
        loaded: List[Tuple[str, pd.DataFrame, float]] = []

        for kind in wanted:
            for path in subject.signal_files.get(kind, []):
                df = read_signal_file(path, kind)
                if df is None or len(df) < 10:
                    continue
                loaded.append((kind, df, infer_fs(df["time"], NOMINAL_FS.get(kind, 1.0))))

        if not loaded:
            return []

        g_start, g_end = glucose["time"].min(), glucose["time"].max()
        g_times = to_epoch_seconds(glucose["time"])
        g_values = glucose["glucose"].to_numpy()

        step_seconds = int(self.step.total_seconds())
        rows: List[Dict] = []

        # Group signal files into overlapping recording sessions so windows are
        # never built across a gap when the device was switched off.
        spans = sorted(
            ((df["time"].iloc[0], df["time"].iloc[-1], kind, df, fs) for kind, df, fs in loaded),
            key=lambda item: item[0],
        )
        sessions: List[Dict] = []
        for s_start, s_end, kind, df, fs in spans:
            attached = False
            for session in sessions:
                if s_start <= session["end"] and s_end >= session["start"]:
                    session["start"] = min(session["start"], s_start)
                    session["end"] = max(session["end"], s_end)
                    session["signals"].append((kind, df, fs))
                    attached = True
                    break
            if not attached:
                sessions.append({"start": s_start, "end": s_end,
                                 "signals": [(kind, df, fs)]})

        for session in sessions:
            begin = max(session["start"], g_start - self.window)
            finish = min(session["end"], g_end - self.horizon)
            if finish <= begin + self.window:
                continue
            # Anchor the grid to a round step boundary so windows are reproducible.
            cursor = begin.floor(f"{step_seconds}s")
            if cursor < begin:
                cursor += self.step

            while cursor + self.window <= finish:
                window_end = cursor + self.window
                target_time = window_end + self.horizon
                target = self._interpolate_target(g_times, g_values, target_time)
                if target is None:
                    cursor += self.step
                    continue

                features: Dict[str, float] = {}
                for kind, df, fs in session["signals"]:
                    segment = _slice(df, cursor, window_end)
                    extracted = self._features(segment, kind, fs)
                    for key, value in extracted.items():
                        features.setdefault(key, value)

                if features:
                    rows.append({
                        "subject_id": subject.subject_id,
                        "cohort": subject.cohort,
                        "window_start": cursor,
                        "window_end": window_end,
                        "target_time": target_time,
                        "glucose": target,
                        **features,
                    })
                cursor += self.step

        return rows

    def _interpolate_target(self, g_times, g_values, when: pd.Timestamp) -> Optional[float]:
        t = pd.Timestamp(when).value / 1e9  # Timestamp.value is always nanoseconds
        if t < g_times[0] or t > g_times[-1]:
            return None
        idx = int(np.searchsorted(g_times, t))
        neighbours = [g_times[i] for i in (idx - 1, idx) if 0 <= i < len(g_times)]
        if not neighbours or min(abs(t - n) for n in neighbours) > self.max_gap.total_seconds():
            return None
        return float(np.interp(t, g_times, g_values))

    def _features(self, segment: pd.DataFrame, kind: str, fs: float) -> Dict[str, float]:
        if len(segment) < 5:
            return {}
        numeric = segment.select_dtypes(include=[np.number])
        if numeric.empty:
            return {}

        if kind == "summary":
            return _numeric_window_features(segment, "sum")

        if kind == "rr":
            col = next(
                (c for c in numeric.columns if "rr" in str(c).lower() or "interval" in str(c).lower()),
                numeric.columns[0],
            )
            values = numeric[col].to_numpy(dtype=float)
            if np.nanmedian(np.abs(values)) < 10:  # seconds rather than ms
                values = values * 1000.0
            return hrv_features(np.abs(values), "hrv")

        if kind == "breathing":
            return breathing_features(numeric.iloc[:, 0].to_numpy(dtype=float), fs, "resp")

        if kind == "accel":
            axes = [c for c in numeric.columns if str(c).strip().lower() in
                    ("x", "y", "z", "accelx", "accely", "accelz", "vertical", "lateral", "sagittal")]
            if len(axes) >= 3:
                magnitude = np.sqrt((numeric[axes[:3]].to_numpy(dtype=float) ** 2).sum(axis=1))
            else:
                magnitude = numeric.iloc[:, 0].to_numpy(dtype=float)
            return accel_features(magnitude, fs, "accel")

        if kind == "ecg":
            values = numeric.iloc[:, 0].to_numpy(dtype=float)
            out = ecg_waveform_features(values, fs, "ecg")
            rr = detect_r_peaks(values, fs)
            if rr is not None:
                out.update(hrv_features(rr, "ecghrv"))
            return out
        return {}
