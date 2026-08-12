"""Signal feature extraction is validated against synthetic signals with a
known ground truth, so a silent regression in the DSP shows up here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import (accel_features, breathing_features, detect_r_peaks,
                               hrv_features)
from src.data.d1namo import normalise_glucose_units, to_epoch_seconds


@pytest.mark.parametrize("true_bpm", [10, 15, 20])
def test_breathing_rate_recovers_the_true_rate(true_bpm):
    rng = np.random.default_rng(0)
    fs = 25.0
    t = np.arange(0, 180, 1 / fs)
    signal = np.sin(2 * np.pi * (true_bpm / 60) * t) + rng.normal(0, 0.15, t.size)
    assert breathing_features(signal, fs)["resp_rate_bpm"] == pytest.approx(true_bpm, abs=1.0)


def test_r_peak_detection_recovers_the_true_rr_interval():
    rng = np.random.default_rng(1)
    fs, true_rr = 250.0, 0.8
    t = np.arange(0, 30, 1 / fs)
    ecg = np.zeros_like(t)
    for beat in np.arange(0, 30, true_rr):
        ecg[int(beat * fs)] = 1.0
    ecg = np.convolve(ecg, np.hanning(20), "same") + rng.normal(0, 0.02, t.size)
    rr = detect_r_peaks(ecg, fs)
    assert rr is not None
    assert float(np.mean(rr)) == pytest.approx(true_rr * 1000, abs=20)


def test_r_peak_detection_refuses_low_sampling_rates():
    """QRS complexes do not survive downsampling to 1 Hz. Returning None is the
    correct answer; producing 'HRV' from such a signal would be fabrication."""
    assert detect_r_peaks(np.random.default_rng(2).normal(size=600), fs=1.0) is None


def test_hrv_features_are_physiologically_sane():
    rng = np.random.default_rng(3)
    rr = 800 + rng.normal(0, 40, 400)
    features = hrv_features(rr)
    assert features["hrv_mean_rr"] == pytest.approx(800, abs=15)
    assert features["hrv_mean_hr"] == pytest.approx(75, abs=3)
    assert features["hrv_sdnn"] > 0
    assert "hrv_lf_hf_ratio" in features


def test_hrv_rejects_too_short_a_series():
    assert hrv_features(np.array([800.0, 810.0, 790.0])) == {}


def test_accel_features_present():
    fs = 50.0
    t = np.arange(0, 60, 1 / fs)
    magnitude = 1 + 0.3 * np.sin(2 * np.pi * 1.5 * t)
    features = accel_features(magnitude, fs)
    assert features["accel_rms"] > 0
    assert 0.0 <= features["accel_active_fraction"] <= 1.0


def test_glucose_units_are_converted_from_mmol():
    import pandas as pd

    mmol = pd.Series([5.0, 7.5, 10.0])
    converted, label = normalise_glucose_units(mmol)
    assert converted.iloc[0] == pytest.approx(90.1, abs=0.5)
    assert "mmol/L" in label
    already_mgdl = pd.Series([90.0, 135.0, 180.0])
    unchanged, label2 = normalise_glucose_units(already_mgdl)
    assert unchanged.iloc[0] == pytest.approx(90.0)
    assert "converted" not in label2


def test_epoch_conversion_is_nanosecond_safe():
    """pandas 2.x keeps datetime64[s]/[us] resolution, so a naive
    astype('int64') / 1e9 rescales the timeline by 1000x and every target
    lookup falls out of range."""
    import pandas as pd

    times = pd.to_datetime(["2014-10-01 08:00:00", "2014-10-01 08:00:01"])
    seconds = to_epoch_seconds(times)
    assert float(np.diff(seconds)[0]) == pytest.approx(1.0)
