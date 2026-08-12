"""PhysioCGM loader: segment parsing, modality blocks, horizon pairing."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.d1namo import _classify, _subject_from_path
from src.data.physiocgm import (PhysioCGMLoader, block_for, discover,
                                segment_features)


def _segment(start: pd.Timestamp, glucose: float, index: int) -> dict:
    rng = np.random.default_rng(index)
    n_sec = 60
    return {
        "Index": index, "Timestamp": start, "glucose": glucose,
        "zephyr": {
            "Accel": {"Time": list(start + pd.to_timedelta(np.arange(n_sec * 10) / 10, unit="s")),
                      "Vertical": rng.normal(1, .1, n_sec * 10).tolist(),
                      "Lateral": rng.normal(0, .1, n_sec * 10).tolist(),
                      "Sagittal": rng.normal(0, .1, n_sec * 10).tolist()},
            "Breathing": {"Time": list(start + pd.to_timedelta(np.arange(n_sec * 25) / 25, unit="s")),
                          "BreathingWaveform": np.sin(
                              2 * np.pi * .25 * np.arange(n_sec * 25) / 25).tolist()},
            "ECG": {"Time": None, "EcgWaveform": None},
            "Summary": {"Time": list(start + pd.to_timedelta(np.arange(n_sec), unit="s")),
                        "HR": rng.normal(70, 3, n_sec).tolist(),
                        "BR": rng.normal(15, 1, n_sec).tolist(),
                        "Posture": rng.integers(-10, 10, n_sec).tolist(),
                        "Activity": np.abs(rng.normal(.2, .05, n_sec)).tolist(),
                        "HRConfidence": rng.integers(95, 100, n_sec).tolist(),
                        "ECGNoise": np.abs(rng.normal(.01, .002, n_sec)).tolist()},
        },
        "e4": {
            "ACC": {"Time": list(start + pd.to_timedelta(np.arange(n_sec * 32) / 32, unit="s")),
                    "x": rng.normal(0, 8, n_sec * 32).tolist(),
                    "y": rng.normal(0, 8, n_sec * 32).tolist(),
                    "z": rng.normal(60, 8, n_sec * 32).tolist()},
            "HR": {"Time": list(start + pd.to_timedelta(np.arange(n_sec), unit="s")),
                   "HR": rng.normal(70, 2, n_sec).tolist()},
            "BVP": {"Time": list(start + pd.to_timedelta(np.arange(n_sec * 64) / 64, unit="s")),
                    "BVP": (np.sin(2 * np.pi * 1.2 * np.arange(n_sec * 64) / 64) * 40).tolist()},
            "EDA": {"Time": list(start + pd.to_timedelta(np.arange(n_sec * 4) / 4, unit="s")),
                    "EDA": rng.normal(2, .1, n_sec * 4).tolist()},
            "TEMP": {"Time": list(start + pd.to_timedelta(np.arange(n_sec * 4) / 4, unit="s")),
                     "TEMP": rng.normal(32.5, .1, n_sec * 4).tolist()},
        },
    }


@pytest.fixture
def dataset(tmp_path) -> Path:
    for subject in ("c1s01", "c1s02"):
        out = tmp_path / "dataset" / "processed" / subject
        out.mkdir(parents=True)
        start = pd.Timestamp("2024-03-01 08:00:00")
        for i in range(14):
            segment = _segment(start + pd.Timedelta(minutes=5 * i), 120.0 + i, i)
            (out / f"{i}.pkl").write_bytes(pickle.dumps(segment))
    return tmp_path


def test_discovery_finds_subjects_and_segments(dataset):
    found = discover(dataset)
    assert set(found) == {"c1s01", "c1s02"}
    assert all(len(v) == 14 for v in found.values())


def test_segment_features_cover_every_sensor_family(dataset):
    segment = pickle.loads(
        (dataset / "dataset" / "processed" / "c1s01" / "0.pkl").read_bytes())
    features = segment_features(segment, use_ecg=False)
    prefixes = {name.split("_")[0] for name in features}
    for expected in ("resp", "accel", "sum", "bvp", "eda", "temp", "e4acc", "e4hr"):
        assert expected in prefixes, f"missing {expected} features"
    assert all(np.isfinite(v) for v in features.values())


def test_features_map_to_the_right_modality_blocks():
    assert block_for("hrv_sdnn") == "ecg"
    assert block_for("bvp_mean") == "ppg"
    assert block_for("ppg_pulse_rate_bpm") == "ppg"
    assert block_for("eda_tonic_mean") == "eda"
    assert block_for("resp_rate_bpm") == "respiration"
    assert block_for("accel_rms") == "motion"


def test_loader_pairs_segments_at_the_requested_horizon(dataset):
    loader = PhysioCGMLoader(dataset, horizon_steps=6, use_ecg=False)
    X, y, groups, times = loader.load()
    assert len(X) == len(y) == len(groups)
    assert loader.report.horizon_minutes == 30.0
    assert set(groups) == {"c1s01", "c1s02"}
    assert np.isfinite(y).all()
    # 14 segments, 6 ahead -> 8 usable pairs per subject.
    assert len(X) == 16


def test_modality_blocks_partition_all_features(dataset):
    loader = PhysioCGMLoader(dataset, horizon_steps=3, use_ecg=False)
    X, _, _, _ = loader.load()
    blocks = loader.modality_blocks(list(X.columns))
    assert sum(len(v) for v in blocks.values()) == X.shape[1]
    assert len(blocks) >= 3


def test_persistence_feature_is_present_and_droppable(dataset):
    """glucose_now is kept on purpose: a glucose model must beat persistence."""
    loader = PhysioCGMLoader(dataset, horizon_steps=3, use_ecg=False)
    X, _, _, _ = loader.load()
    assert "glucose_now" in X.columns
    assert "persistence" in " ".join(loader.report.caveats).lower()


def test_impossible_horizon_raises_a_useful_error(dataset):
    with pytest.raises(ValueError, match="horizon"):
        PhysioCGMLoader(dataset, horizon_steps=500, use_ecg=False).load()


# ---- discovery fixes driven by the real PhysioCGM/Zephyr conventions ----
def test_summaryenhanced_is_recognised_as_a_summary_file():
    """PhysioCGM's verify_folder() requires *_SummaryEnhanced.csv; matching only
    '_summary.csv' would skip every session."""
    assert _classify(Path("2014_10_01-08_00_00_SummaryEnhanced.csv")) == "summary"
    assert _classify(Path("x_Summary.csv")) == "summary"


def test_non_numeric_subject_ids_are_recovered():
    """PhysioCGM subject folders are 'c1s01', not '001'."""
    root = Path("/data")
    assert _subject_from_path(
        Path("/data/raw/c1s01/zephyr/2024_01_01-10_00_00/a_ECG.csv"), root) == "c1s01"
    assert _subject_from_path(
        Path("/data/diabetes_subset_sensor_data/001/sensor_data/s/a_Accel.csv"), root) == "001"
