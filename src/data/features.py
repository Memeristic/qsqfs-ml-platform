"""Window-level feature extraction from physiological signals.

Naming is deliberately literal. A feature called ``rr_rmssd`` is RMSSD computed
from an actual RR-interval series; a feature called ``ecg_bandpower_5_15hz`` is
band power of the raw ECG waveform and is *not* an HRV measure. Frequency-domain
HRV (LF/HF) is only produced from RR intervals, because computing it from a
downsampled ECG amplitude trace -- as some earlier drafts of this project did --
yields numbers that look like HRV but are not.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import signal as sps
from scipy import stats

EPS = 1e-12


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def basic_stats(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """Distribution shape plus a linear trend term."""
    x = _finite(x)
    if x.size < 3:
        return {}
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    slope = float(np.polyfit(np.arange(x.size), x, 1)[0]) if x.size >= 5 else 0.0
    out = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(q50),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_slope": slope,
        f"{prefix}_last": float(x[-1]),
        f"{prefix}_delta": float(x[-1] - x[0]),
    }
    if x.size >= 8 and np.std(x) > EPS:
        out[f"{prefix}_skew"] = float(stats.skew(x))
        out[f"{prefix}_kurtosis"] = float(stats.kurtosis(x))
    return out


# ----------------------------------------------------------------------
# Heart-rate variability from RR intervals (the correct source)
# ----------------------------------------------------------------------
def hrv_features(rr_ms: np.ndarray, prefix: str = "hrv") -> Dict[str, float]:
    """Time- and frequency-domain HRV from an RR-interval series in milliseconds."""
    rr = _finite(rr_ms)
    rr = rr[(rr > 300) & (rr < 2000)]  # physiological plausibility filter
    if rr.size < 10:
        return {}

    diff = np.diff(rr)
    out = {
        f"{prefix}_mean_rr": float(np.mean(rr)),
        f"{prefix}_sdnn": float(np.std(rr, ddof=1)),
        f"{prefix}_rmssd": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_pnn50": float(np.mean(np.abs(diff) > 50) * 100),
        f"{prefix}_pnn20": float(np.mean(np.abs(diff) > 20) * 100),
        f"{prefix}_cvnn": float(np.std(rr, ddof=1) / (np.mean(rr) + EPS)),
        f"{prefix}_mean_hr": float(60000.0 / (np.mean(rr) + EPS)),
        f"{prefix}_n_beats": float(rr.size),
    }

    # Poincaré descriptors
    sd1 = float(np.sqrt(0.5) * np.std(diff, ddof=1))
    sdnn = out[f"{prefix}_sdnn"]
    sd2_sq = 2 * sdnn**2 - sd1**2
    out[f"{prefix}_sd1"] = sd1
    out[f"{prefix}_sd2"] = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else 0.0
    out[f"{prefix}_sd1_sd2"] = float(sd1 / (out[f"{prefix}_sd2"] + EPS))

    # Frequency domain: resample the tachogram to an even 4 Hz grid first.
    if rr.size >= 30:
        t = np.cumsum(rr) / 1000.0
        t -= t[0]
        if t[-1] > 30.0:
            fs = 4.0
            grid = np.arange(0, t[-1], 1.0 / fs)
            tacho = np.interp(grid, t, rr)
            tacho = tacho - tacho.mean()
            nperseg = min(len(tacho), int(fs * 60))
            if nperseg >= 32:
                freqs, psd = sps.welch(tacho, fs=fs, nperseg=nperseg)
                def band(lo, hi):
                    sel = (freqs >= lo) & (freqs < hi)
                    return float(np.trapezoid(psd[sel], freqs[sel])) if sel.any() else 0.0
                vlf, lf, hf = band(0.003, 0.04), band(0.04, 0.15), band(0.15, 0.4)
                total = vlf + lf + hf
                out[f"{prefix}_vlf_power"] = vlf
                out[f"{prefix}_lf_power"] = lf
                out[f"{prefix}_hf_power"] = hf
                out[f"{prefix}_total_power"] = total
                out[f"{prefix}_lf_hf_ratio"] = float(lf / (hf + EPS))
                out[f"{prefix}_lf_nu"] = float(100 * lf / (lf + hf + EPS))
                out[f"{prefix}_hf_nu"] = float(100 * hf / (lf + hf + EPS))
    return out


# ----------------------------------------------------------------------
def breathing_features(x: np.ndarray, fs: float, prefix: str = "resp") -> Dict[str, float]:
    """Breathing rate and variability from a raw respiration waveform."""
    x = _finite(x)
    if x.size < int(fs * 10):
        return {}
    out = basic_stats(x, prefix)
    # Low-pass first: raw thoracic impedance carries cardiac and motion ripple
    # that find_peaks would otherwise count as extra breaths.
    centred = x - np.mean(x)
    if fs > 2.0:
        try:
            b, a = sps.butter(2, min(0.8 / (fs / 2.0), 0.99), btype="low")
            centred = sps.filtfilt(b, a, centred)
        except Exception:
            pass
    min_distance = max(1, int(fs * 1.5))  # <= 40 breaths/min
    peaks, _ = sps.find_peaks(centred, distance=min_distance, prominence=np.std(centred) * 0.5)
    if peaks.size > 2:
        intervals = np.diff(peaks) / fs
        intervals = intervals[(intervals > 1.0) & (intervals < 15.0)]
        if intervals.size >= 2:
            # Median: one missed or doubled breath should not move the estimate.
            out[f"{prefix}_rate_bpm"] = float(60.0 / np.median(intervals))
            out[f"{prefix}_interval_std"] = float(np.std(intervals))
            out[f"{prefix}_interval_cv"] = float(np.std(intervals) / (np.mean(intervals) + EPS))
            out[f"{prefix}_n_breaths"] = float(intervals.size + 1)
    if x.size >= 64:
        freqs, psd = sps.welch(centred, fs=fs, nperseg=min(x.size, int(fs * 30)))
        sel = (freqs >= 0.1) & (freqs <= 0.7)  # 6-42 breaths/min
        if sel.any() and psd[sel].sum() > EPS:
            out[f"{prefix}_dominant_freq_hz"] = float(freqs[sel][np.argmax(psd[sel])])
            out[f"{prefix}_band_power"] = float(np.trapezoid(psd[sel], freqs[sel]))
    return out


def accel_features(mag: np.ndarray, fs: float, prefix: str = "accel") -> Dict[str, float]:
    """Movement intensity from an accelerometer magnitude series."""
    mag = _finite(mag)
    if mag.size < int(fs * 5):
        return {}
    out = basic_stats(mag, prefix)
    out[f"{prefix}_rms"] = float(np.sqrt(np.mean(mag**2)))
    centred = mag - np.mean(mag)
    out[f"{prefix}_mad"] = float(np.mean(np.abs(centred)))
    threshold = np.mean(mag) + 0.5 * np.std(mag)
    out[f"{prefix}_active_fraction"] = float(np.mean(mag > threshold))
    out[f"{prefix}_zero_crossings"] = float(np.mean(np.diff(np.signbit(centred)) != 0))
    if mag.size >= 64:
        freqs, psd = sps.welch(centred, fs=fs, nperseg=min(mag.size, int(fs * 10)))
        total = psd.sum()
        if total > EPS:
            out[f"{prefix}_dominant_freq_hz"] = float(freqs[np.argmax(psd)])
            p = psd / total
            p = p[p > EPS]
            out[f"{prefix}_spectral_entropy"] = float(-np.sum(p * np.log(p)) / np.log(len(p)))
            for lo, hi, tag in [(0.5, 3.0, "walk"), (3.0, 8.0, "fast")]:
                sel = (freqs >= lo) & (freqs < hi)
                if sel.any():
                    out[f"{prefix}_power_{tag}"] = float(np.trapezoid(psd[sel], freqs[sel]))
    return out


def ecg_waveform_features(x: np.ndarray, fs: float, prefix: str = "ecg") -> Dict[str, float]:
    """Amplitude and band-power descriptors of the raw ECG waveform.

    Not HRV. Use ``hrv_features`` on an RR series for that.
    """
    x = _finite(x)
    if x.size < int(fs * 5):
        return {}
    out = basic_stats(x, prefix)
    if x.size >= 256:
        centred = x - np.mean(x)
        freqs, psd = sps.welch(centred, fs=fs, nperseg=min(x.size, int(fs * 4)))
        total = float(np.trapezoid(psd, freqs))
        out[f"{prefix}_total_power"] = total
        for lo, hi in [(0.5, 4.0), (4.0, 15.0), (15.0, 40.0)]:
            sel = (freqs >= lo) & (freqs < hi)
            if sel.any():
                power = float(np.trapezoid(psd[sel], freqs[sel]))
                tag = f"{prefix}_bandpower_{lo:g}_{hi:g}hz"
                out[tag] = power
                out[tag + "_rel"] = float(power / (total + EPS))
    return out


def detect_r_peaks(ecg: np.ndarray, fs: float) -> Optional[np.ndarray]:
    """Pan-Tompkins style QRS detection. Returns RR intervals in ms.

    Only valid on ECG sampled at its native rate (D1NAMO: 250 Hz). Returns None
    when the sampling rate is too low for the QRS complex to survive.
    """
    ecg = _finite(ecg)
    if fs < 100 or ecg.size < int(fs * 10):
        return None
    nyq = fs / 2.0
    try:
        b, a = sps.butter(2, [5.0 / nyq, min(15.0 / nyq, 0.99)], btype="band")
        filtered = sps.filtfilt(b, a, ecg)
    except Exception:
        return None
    energy = np.convolve(np.diff(filtered) ** 2, np.ones(int(0.12 * fs)) / int(0.12 * fs), "same")
    if np.std(energy) < EPS:
        return None
    peaks, _ = sps.find_peaks(
        energy, distance=int(0.25 * fs), height=np.mean(energy) + 0.5 * np.std(energy)
    )
    if peaks.size < 5:
        return None
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    return rr if rr.size >= 5 else None
