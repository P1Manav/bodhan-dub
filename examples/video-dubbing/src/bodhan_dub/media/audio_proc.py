"""
Audio signal processing utilities: silence trimming, clipping detection,
resampling from 24kHz to 48kHz, time-stretching with ffmpeg atempo, and crossfade stitching.
"""

from __future__ import annotations

import io
import subprocess
from typing import List

import numpy as np
import scipy.signal
import soundfile as sf


def trim_edge_silence(
    samples: np.ndarray,
    threshold_db: float = -45.0,
    min_keep_samples: int = 240,  # 10ms at 24kHz
) -> np.ndarray:
    """
    Trims leading and trailing silence below threshold_db from a mono audio array.
    """
    if len(samples) == 0:
        return samples

    abs_samples = np.abs(samples)
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    active_indices = np.where(abs_samples > threshold_amp)[0]
    if len(active_indices) == 0:
        return np.zeros(0, dtype=samples.dtype)

    start_idx = max(0, active_indices[0] - min_keep_samples)
    end_idx = min(len(samples), active_indices[-1] + min_keep_samples)

    return samples[start_idx:end_idx]


def check_clipping(samples: np.ndarray, threshold: float = 0.999) -> bool:
    """Detects whether audio exceeds digital ceiling indicating harsh distortion."""
    if len(samples) == 0:
        return False
    peak = float(np.max(np.abs(samples)))
    return peak >= threshold


def detect_internal_silence(
    samples: np.ndarray,
    sample_rate: int = 24000,
    min_silence_s: float = 1.5,
    threshold_db: float = -45.0,
) -> bool:
    """Detects if there is an unnatural pause longer than min_silence_s inside speech."""
    if len(samples) == 0:
        return False

    threshold_amp = 10.0 ** (threshold_db / 20.0)
    window_len = int(min_silence_s * sample_rate)

    if len(samples) < window_len:
        return False

    is_silent = np.abs(samples) < threshold_amp
    # Check if any contiguous run of length window_len is all silent
    silent_runs = 0
    for val in is_silent:
        if val:
            silent_runs += 1
            if silent_runs >= window_len:
                return True
        else:
            silent_runs = 0

    return False


def resample_24k_to_48k(samples: np.ndarray) -> np.ndarray:
    """
    High-fidelity 2x upsampling from 24,000 Hz to 48,000 Hz float32 using polyphase resampler.
    """
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)

    # Convert to float32 normalized in [-1.0, 1.0]
    if samples.dtype == np.int16:
        float_samples = samples.astype(np.float32) / 32768.0
    else:
        float_samples = samples.astype(np.float32)

    # 24k -> 48k is exactly 2x integer ratio (up=2, down=1)
    resampled = scipy.signal.resample_poly(float_samples, up=2, down=1)
    return resampled.astype(np.float32)


def time_stretch_audio(
    samples: np.ndarray,
    speed: float,
    sample_rate: int = 48000,
) -> np.ndarray:
    """
    Time-stretches audio by factor `speed` (1.0x to 1.2x) using ffmpeg atempo filter.
    Preserves pitch accurately. If speed == 1.0, returns copy.
    """
    if abs(speed - 1.0) < 0.005 or len(samples) == 0:
        return samples.copy()

    # Clamp speed to allowed stretch range [0.85, 1.20]
    clamped_speed = max(0.85, min(1.20, speed))

    # Write input WAV to memory buffer
    in_buf = io.BytesIO()
    sf.write(in_buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    in_bytes = in_buf.getvalue()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        "pipe:0",
        "-filter:a",
        f"atempo={clamped_speed:.4f}",
        "-f",
        "wav",
        "pipe:1",
    ]

    proc = subprocess.run(cmd, input=in_bytes, capture_output=True, check=True)
    out_buf = io.BytesIO(proc.stdout)
    stretched, _ = sf.read(out_buf, dtype="float32")
    return stretched


def crossfade_concat(
    segments: List[np.ndarray],
    crossfade_ms: int = 10,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Concatenates audio arrays with short linear crossfades at boundaries to eliminate clicks.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return segments[0]

    xfade_samples = int((crossfade_ms / 1000.0) * sample_rate)
    result = list(segments[0])

    for nxt in segments[1:]:
        if len(nxt) == 0:
            continue
        if len(result) < xfade_samples or len(nxt) < xfade_samples:
            result.extend(nxt)
            continue

        # Linear crossfade over xfade_samples
        overlap_a = np.array(result[-xfade_samples:], dtype=np.float32)
        overlap_b = nxt[:xfade_samples].astype(np.float32)

        fade_out = np.linspace(1.0, 0.0, xfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)

        crossfaded = (overlap_a * fade_out) + (overlap_b * fade_in)
        # Replace tail of result with crossfaded samples, then append remainder of nxt
        result = result[:-xfade_samples] + list(crossfaded) + list(nxt[xfade_samples:])

    return np.array(result, dtype=np.float32)
