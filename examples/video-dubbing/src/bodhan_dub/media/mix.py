"""
Audio mixing pipeline.
Ducks/mutes original speech regions with smooth 40ms crossfades while preserving
background music, ambience, and skipped segment audio, then mixes dubbed speech stem
and normalizes to -16 LUFS integrated.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from bodhan_dub.media.loudness import normalize_to_target_lufs


def mute_intervals_with_crossfade(
    audio: np.ndarray,
    mute_spans: List[Tuple[float, float]],
    sample_rate: int = 48000,
    fade_ms: float = 40.0,
) -> np.ndarray:
    """
    Applies smooth gain envelope to mute speech regions in original audio.
    fade_ms: duration of linear fade-in and fade-out.
    """
    out = audio.copy()
    fade_samples = max(1, int(round((fade_ms / 1000.0) * sample_rate)))
    num_samples = len(audio)

    # 1.0 = full volume, 0.0 = muted
    gain = np.ones(num_samples, dtype=np.float32)

    for start_s, end_s in mute_spans:
        s_idx = int(round(start_s * sample_rate))
        e_idx = int(round(end_s * sample_rate))

        s_idx = max(0, min(num_samples, s_idx))
        e_idx = max(0, min(num_samples, e_idx))

        if e_idx <= s_idx:
            continue

        # Fade out before start
        fade_out_start = max(0, s_idx - fade_samples)
        fade_out_len = s_idx - fade_out_start
        if fade_out_len > 0:
            fade_curve = np.linspace(1.0, 0.0, fade_out_len, endpoint=False, dtype=np.float32)
            gain[fade_out_start:s_idx] = np.minimum(gain[fade_out_start:s_idx], fade_curve)

        # Full mute during speech
        gain[s_idx:e_idx] = 0.0

        # Fade in after end
        fade_in_end = min(num_samples, e_idx + fade_samples)
        fade_in_len = fade_in_end - e_idx
        if fade_in_len > 0:
            fade_curve = np.linspace(0.0, 1.0, fade_in_len, endpoint=False, dtype=np.float32)
            gain[e_idx:fade_in_end] = np.minimum(gain[e_idx:fade_in_end], fade_curve)

    if out.ndim == 1:
        return out * gain
    else:
        # Multi-channel
        return out * gain[:, np.newaxis]


def mix_dubbed_audio(
    original_audio_path: Path,
    speech_stem_48k: np.ndarray,
    speech_intervals_to_mute: List[Tuple[float, float]],
    output_wav_path: Path,
    target_lufs: float = -16.0,
    sample_rate: int = 48000,
) -> float:
    """
    Mutes original speech, mixes dubbed speech stem, normalizes loudness to -16 LUFS,
    and writes the final audio WAV.
    Returns final integrated LUFS.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read original 48kHz audio
    orig_data, orig_sr = sf.read(str(original_audio_path), dtype="float32")
    if orig_sr != sample_rate:
        raise ValueError(
            f"Original audio sample rate {orig_sr} does not match required {sample_rate}"
        )

    is_stereo = orig_data.ndim > 1 and orig_data.shape[1] >= 2
    total_samples = len(orig_data)

    # Match stem length to original audio length
    if len(speech_stem_48k) < total_samples:
        pad_len = total_samples - len(speech_stem_48k)
        speech_stem_48k = np.pad(speech_stem_48k, (0, pad_len), mode="constant")
    elif len(speech_stem_48k) > total_samples:
        speech_stem_48k = speech_stem_48k[:total_samples]

    # 2. Mute speech intervals in original audio
    muted_bg = mute_intervals_with_crossfade(
        audio=orig_data,
        mute_spans=speech_intervals_to_mute,
        sample_rate=sample_rate,
        fade_ms=40.0,
    )

    # 3. Sum speech stem with background
    if is_stereo:
        dubbed_stereo = np.column_stack([speech_stem_48k, speech_stem_48k])
        mixed = muted_bg[:, :2] + dubbed_stereo
    else:
        mono_bg = muted_bg[:, 0] if muted_bg.ndim > 1 else muted_bg
        mixed = mono_bg + speech_stem_48k

    # 4. Loudness normalization to -16 LUFS
    normalized, final_lufs = normalize_to_target_lufs(
        mixed,
        target_lufs=target_lufs,
        sample_rate=sample_rate,
    )

    # 5. Write to output WAV
    sf.write(str(output_wav_path), normalized, sample_rate, subtype="PCM_16")

    return final_lufs
