"""PhysioCGM loader (PSI-TAMU, Quamer et al., Scientific Data 2025).

10 participants with type 1 diabetes, 17 days, three sensor families:

  Zephyr BioHarness  ECG 250 Hz, breathing waveform 25 Hz, 3-axis accel,
                     summary (HR, BR, Posture, Activity, HRConfidence, ECGNoise)
  Empatica E4        BVP (PPG), EDA, TEMP, 3-axis ACC, HR
  Dexcom CGM         glucose in mg/dL every 5 minutes

This reads the **processed** ``dataset/processed/<subject>/<n>.pkl`` segments
produced by the project's own ``preprocess.py``. Each pickle is one 5-minute
CGM segment with the structure documented in their README, so alignment work
that this platform would otherwise have to redo has already been done upstream
and by its authors.

Why this dataset matters for the multimodal path: it is genuinely multimodal
*with real data*. ECG, PPG, EDA, temperature and motion are separate physical
measurements of the same person at the same moment, which is exactly the
situation per-modality encoders and fusion exist for. Each sensor family
becomes its own modality block, so ``run_pipeline.py multimodal`` runs on real
recordings rather than a synthetic stand-in.

Access note: the data is distributed via the TAMU PSI Lab drive and requires
contacting the authors. This module reads it; it cannot download it.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import (accel_features, basic_stats, breathing_features,
                       detect_r_peaks, ecg_waveform_features, hrv_features)

logger = logging.getLogger(__name__)

SEGMENT_MINUTES = 5.0
NOMINAL_FS = {"ecg": 250.0, "breathing": 25.0, "accel": 100.0,
              "bvp": 64.0, "eda": 4.0, "temp": 4.0, "e4_acc": 32.0}

# Sensor family -> modality block name used by the multimodal model.
MODALITY_BLOCKS = {
    "ecg": "ecg", "hrv": "ecg",
    "resp": "respiration",
    "accel": "motion", "e4acc": "motion",
    "bvp": "ppg", "ppg": "ppg",
    "eda": "eda", "temp": "eda",
    "sum": "summary", "e4hr": "summary",
}


def discover(data_root: str | Path) -> Dict[str, List[Path]]:
    """Find processed segment pickles, grouped by subject id (e.g. c1s01)."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    processed = root / "processed" if (root / "processed").is_dir() else root
    if (root / "dataset" / "processed").is_dir():
        processed = root / "dataset" / "processed"

    subjects: Dict[str, List[Path]] = {}
    for path in processed.rglob("*.pkl"):
        subject = path.parent.name
        if not re.match(r"^[a-z]?\d*s?\d+$", subject, re.I):
            subject = path.parent.name
        subjects.setdefault(subject, []).append(path)

    for subject in subjects:
        subjects[subject].sort(key=lambda p: (len(p.stem), p.stem))

    logger.info("PhysioCGM: %d subject(s), %d segment files.",
                len(subjects), sum(len(v) for v in subjects.values()))
    if not subjects:
        logger.error(
            "No .pkl segments under %s. Expected "
            "dataset/processed/<subject>/<n>.pkl produced by PhysioCGM's "
            "preprocess.py. If you only have dataset/raw, run their "
            "preprocess.py first.", processed,
        )
    return dict(sorted(subjects.items()))


def describe(data_root: str | Path) -> Dict:
    subjects = discover(data_root)
    return {
        "data_root": str(Path(data_root).resolve()),
        "n_subjects": len(subjects),
        "subjects": [
            {"subject_id": name, "n_segments": len(paths)}
            for name, paths in subjects.items()
        ],
        "total_segments": sum(len(v) for v in subjects.values()),
    }


def _array(node: Optional[Dict], key: str) -> Optional[np.ndarray]:
    """Pull one channel out of a segment dict, tolerating None and empties."""
    if not isinstance(node, dict):
        return None
    values = node.get(key)
    if values is None:
        return None
    array = np.asarray(values, dtype="float64" if key != "Time" else "object")
    return array if array.size else None


def _infer_fs(node: Optional[Dict], fallback: float) -> float:
    """Sampling rate from the segment's own timestamps, not an assumption."""
    if not isinstance(node, dict) or node.get("Time") is None:
        return fallback
    try:
        times = pd.to_datetime(pd.Series(node["Time"])).astype("datetime64[ns]")
        if len(times) < 5:
            return fallback
        deltas = np.diff(times.astype("int64").to_numpy()) / 1e9
        deltas = deltas[(deltas > 0) & (deltas < 10)]
        if deltas.size < 3:
            return fallback
        median = float(np.median(deltas))
        return 1.0 / median if median > 0 else fallback
    except Exception:
        return fallback


def segment_features(segment: Dict, use_ecg: bool = True) -> Dict[str, float]:
    """Window features for one 5-minute segment, prefixed by sensor family."""
    features: Dict[str, float] = {}
    zephyr = segment.get("zephyr") or {}
    e4 = segment.get("e4") or {}

    # --- Zephyr ECG -> waveform descriptors + real HRV from detected R peaks
    ecg_node = zephyr.get("ECG")
    ecg = _array(ecg_node, "EcgWaveform")
    if ecg is not None and use_ecg:
        fs = _infer_fs(ecg_node, NOMINAL_FS["ecg"])
        features.update(ecg_waveform_features(ecg, fs, "ecg"))
        rr = detect_r_peaks(ecg, fs)
        if rr is not None:
            features.update(hrv_features(rr, "hrv"))

    # --- Zephyr breathing
    breathing_node = zephyr.get("Breathing")
    breathing = _array(breathing_node, "BreathingWaveform")
    if breathing is not None:
        features.update(breathing_features(
            breathing, _infer_fs(breathing_node, NOMINAL_FS["breathing"]), "resp"))

    # --- Zephyr accelerometer (3 named axes)
    accel_node = zephyr.get("Accel")
    if isinstance(accel_node, dict):
        axes = [_array(accel_node, a) for a in ("Vertical", "Lateral", "Sagittal")]
        if all(a is not None for a in axes):
            length = min(len(a) for a in axes)
            magnitude = np.sqrt(sum(a[:length] ** 2 for a in axes))
            features.update(accel_features(
                magnitude, _infer_fs(accel_node, NOMINAL_FS["accel"]), "accel"))

    # --- Zephyr summary channels
    summary_node = zephyr.get("Summary")
    if isinstance(summary_node, dict):
        for channel in ("HR", "BR", "Posture", "Activity", "HRConfidence", "ECGNoise"):
            values = _array(summary_node, channel)
            if values is not None:
                features.update(basic_stats(values, f"sum_{channel.lower()}"))

    # --- Empatica E4: BVP (PPG), EDA, TEMP, ACC, HR
    bvp_node = e4.get("BVP")
    bvp = _array(bvp_node, "BVP")
    if bvp is not None:
        fs = _infer_fs(bvp_node, NOMINAL_FS["bvp"])
        features.update(basic_stats(bvp, "bvp"))
        # PPG pulse rate and pulse-rate variability, from the BVP peaks.
        from scipy import signal as sps
        centred = bvp - np.mean(bvp)
        if centred.size > int(fs * 5) and np.std(centred) > 1e-9:
            peaks, _ = sps.find_peaks(centred, distance=max(1, int(fs * 0.4)),
                                      prominence=np.std(centred) * 0.4)
            if peaks.size > 3:
                intervals = np.diff(peaks) / fs * 1000.0
                intervals = intervals[(intervals > 300) & (intervals < 2000)]
                if intervals.size >= 3:
                    features["ppg_pulse_rate_bpm"] = float(60000.0 / np.median(intervals))
                    features["ppg_pri_sdnn"] = float(np.std(intervals, ddof=1))
                    features["ppg_n_pulses"] = float(intervals.size + 1)

    eda = _array(e4.get("EDA"), "EDA")
    if eda is not None:
        features.update(basic_stats(eda, "eda"))
        if eda.size > 8:
            # Tonic level vs phasic activity: a coarse but standard split.
            window = max(3, eda.size // 8)
            tonic = pd.Series(eda).rolling(window, min_periods=1, center=True).mean().to_numpy()
            phasic = eda - tonic
            features["eda_tonic_mean"] = float(np.mean(tonic))
            features["eda_phasic_std"] = float(np.std(phasic))
            features["eda_phasic_max"] = float(np.max(np.abs(phasic)))

    temp = _array(e4.get("TEMP"), "TEMP")
    if temp is not None:
        features.update(basic_stats(temp, "temp"))

    e4_acc = e4.get("ACC")
    if isinstance(e4_acc, dict):
        axes = [_array(e4_acc, a) for a in ("x", "y", "z")]
        if all(a is not None for a in axes):
            length = min(len(a) for a in axes)
            magnitude = np.sqrt(sum(a[:length] ** 2 for a in axes))
            features.update(accel_features(
                magnitude, _infer_fs(e4_acc, NOMINAL_FS["e4_acc"]), "e4acc"))

    e4_hr = _array(e4.get("HR"), "HR")
    if e4_hr is not None:
        features.update(basic_stats(e4_hr, "e4hr"))

    return features


def block_for(feature_name: str) -> str:
    """Map a feature to its modality block for the multimodal model."""
    prefix = feature_name.split("_")[0]
    return MODALITY_BLOCKS.get(prefix, "summary")


@dataclass
class PhysioCGMReport:
    n_segments: int = 0
    n_features: int = 0
    n_subjects: int = 0
    per_subject: List[Dict] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    target_units: str = "mg/dL"
    horizon_minutes: float = 0.0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class PhysioCGMLoader:
    """Build a feature matrix and a glucose target from processed segments."""

    def __init__(
        self,
        data_root: str | Path,
        horizon_steps: int = 6,
        use_ecg: bool = True,
        max_subjects: Optional[int] = None,
        max_segments_per_subject: Optional[int] = None,
        min_coverage: float = 0.5,
        max_gap_minutes: float = 7.5,
    ):
        self.data_root = Path(data_root)
        self.horizon_steps = int(horizon_steps)
        self.use_ecg = use_ecg
        self.max_subjects = max_subjects
        self.max_segments_per_subject = max_segments_per_subject
        self.min_coverage = min_coverage
        self.max_gap_minutes = max_gap_minutes
        self.report = PhysioCGMReport()

    def load(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
        """Returns (features, y, groups, timestamps).

        The target is glucose ``horizon_steps`` segments ahead (each segment is
        one 5-minute CGM reading), so ``horizon_steps=6`` is a 30-minute
        prediction horizon. Pairs separated by a recording gap larger than
        ``max_gap_minutes`` per step are dropped rather than bridged.
        """
        subjects = discover(self.data_root)
        names = list(subjects)[: self.max_subjects] if self.max_subjects else list(subjects)
        if not names:
            raise ValueError(f"No PhysioCGM segments found under {self.data_root}.")

        rows: List[Dict] = []
        for subject in names:
            paths = subjects[subject]
            if self.max_segments_per_subject:
                paths = paths[: self.max_segments_per_subject]

            records = []
            for path in paths:
                try:
                    with open(path, "rb") as handle:
                        segment = pickle.load(handle)
                except Exception as exc:
                    logger.warning("Could not read %s: %s", path.name, exc)
                    continue
                glucose = segment.get("glucose")
                timestamp = segment.get("Timestamp")
                if glucose is None or timestamp is None:
                    continue
                features = segment_features(segment, self.use_ecg)
                if not features:
                    continue
                records.append({
                    "subject_id": subject,
                    "timestamp": pd.Timestamp(timestamp),
                    "glucose_now": float(glucose),
                    **features,
                })

            if not records:
                self.report.per_subject.append(
                    {"subject": subject, "n_segments": 0, "reason": "no readable segments"})
                continue

            frame = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
            # Pair each segment with the glucose reading `horizon_steps` later,
            # checking the real elapsed time rather than trusting row order.
            future = frame[["timestamp", "glucose_now"]].shift(-self.horizon_steps)
            elapsed = (future["timestamp"] - frame["timestamp"]).dt.total_seconds() / 60.0
            tolerance = SEGMENT_MINUTES * self.horizon_steps + self.max_gap_minutes
            valid = elapsed.notna() & (elapsed <= tolerance) & (elapsed > 0)

            frame = frame[valid].copy()
            frame["glucose_target"] = future.loc[valid, "glucose_now"].to_numpy()
            rows.extend(frame.to_dict("records"))

            self.report.per_subject.append({
                "subject": subject,
                "n_segments_read": len(records),
                "n_usable_pairs": int(valid.sum()),
                "n_dropped_gap": int((~valid).sum()),
            })
            logger.info("  %s: %d segments -> %d usable pairs",
                        subject, len(records), int(valid.sum()))

        if not rows:
            raise ValueError(
                "No usable segment pairs. Every segment lacked a glucose reading "
                f"{self.horizon_steps * SEGMENT_MINUTES:.0f} min later within the "
                "gap tolerance. Try a shorter --horizon_steps."
            )

        df = pd.DataFrame(rows)
        y = df["glucose_target"].to_numpy(dtype=float)
        groups = df["subject_id"].to_numpy()
        times = df["timestamp"]

        meta = ["subject_id", "timestamp", "glucose_target"]
        features = df.drop(columns=[c for c in meta if c in df.columns])

        coverage = features.notna().mean()
        keep = coverage[coverage >= self.min_coverage].index.tolist()
        self.report.dropped_features = [c for c in features.columns if c not in keep]
        features = features[keep].astype(float)
        features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

        self.report.n_segments = len(df)
        self.report.n_features = features.shape[1]
        self.report.n_subjects = int(df["subject_id"].nunique())
        self.report.horizon_minutes = self.horizon_steps * SEGMENT_MINUTES
        self.report.caveats = [
            "'glucose_now' is the CGM value at the START of the window and is kept "
            "as a feature on purpose: any glucose model must beat persistence. "
            "Drop it with --no_persistence to measure the sensors alone.",
            "Consecutive segments from one subject are autocorrelated. Use "
            "cv_type='group' so no participant appears in both train and test.",
            "Glucose is already mg/dL (Dexcom); no unit conversion is applied.",
        ]
        logger.info("PhysioCGM: %d pairs x %d features from %d subjects (horizon %.0f min).",
                    len(df), features.shape[1], self.report.n_subjects,
                    self.report.horizon_minutes)
        return features, y, groups, times

    @staticmethod
    def modality_blocks(feature_names: List[str]) -> Dict[str, List[str]]:
        """Group features into modality blocks for the multimodal model."""
        blocks: Dict[str, List[str]] = {}
        for name in feature_names:
            blocks.setdefault(block_for(name), []).append(name)
        return blocks
