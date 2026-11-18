"""
Loudness measurement and normalization adhering to ITU-R BS.1770 / EBU R128.
Standard targets: -16.0 LUFS integrated (±1.0 LUFS) with a -1.0 dBTP true-peak ceiling.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pyloudnorm as pyln


def measure_loudness(audio_48k: np.ndarray, sample_rate: int = 48000) -> float:
    """
    Measures integrated loudness in LUFS using ITU-R BS.1770 standard.
    Returns -70.0 for near-silent audio.
    """
    if len(audio_48k) == 0:
        return -70.0

    meter = pyln.Meter(sample_rate)
    # pyloudnorm expects shape (samples, channels)
    data = audio_48k[:, np.newaxis] if audio_48k.ndim == 1 else audio_48k

    # Guard against silence or very short audio (<0.4s)
    if len(data) < int(sample_rate * 0.4) or np.max(np.abs(data)) < 1e-4:
        return -70.0

    try:
        loudness = meter.integrated_loudness(data)
        if np.isneginf(loudness) or np.isnan(loudness):
            return -70.0
        return float(loudness)
    except Exception:
        return -70.0


def apply_peak_limiter(audio: np.ndarray, max_peak_dbtp: float = -1.0) -> np.ndarray:
    """
    Limits true peaks to max_peak_dbtp (default -1.0 dBTP = ~0.8912 max amplitude).
    """
    ceiling = 10.0 ** (max_peak_dbtp / 20.0)
    current_peak = np.max(np.abs(audio))

    if current_peak > ceiling and current_peak > 0:
        scale = ceiling / current_peak
        return audio * scale
    return audio


def normalize_to_target_lufs(
    audio_48k: np.ndarray,
    target_lufs: float = -16.0,
    sample_rate: int = 48000,
    max_gain_db: float = 18.0,
) -> Tuple[np.ndarray, float]:
    """
    Normalizes audio to target integrated LUFS with peak limiting.
    Returns (normalized_audio_48k, measured_lufs).
    """
    if len(audio_48k) == 0:
        return audio_48k, -70.0

    measured = measure_loudness(audio_48k, sample_rate)

    if measured <= -60.0:
        # Audio is effectively silent, do not blow up background noise
        return audio_48k, measured

    gain_db = target_lufs - measured
    # Bounded gain clamp to avoid extreme amplification of background noise
    clamped_gain_db = np.clip(gain_db, -18.0, max_gain_db)
    gain_linear = 10.0 ** (clamped_gain_db / 20.0)

    scaled_audio = audio_48k * gain_linear

    # Peak ceiling check (-1.0 dBTP)
    limited_audio = apply_peak_limiter(scaled_audio, max_peak_dbtp=-1.0)
    final_lufs = measure_loudness(limited_audio, sample_rate)

    return limited_audio.astype(np.float32), round(final_lufs, 2)
