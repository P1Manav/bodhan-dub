"""
Audio mixing pipeline.
Ducks/mutes original speech regions with smooth 40ms crossfades while preserving
background music, ambience, and skipped segment audio, then mixes dubbed speech stem
and normalizes to -16 LUFS integrated.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from bodhan_dub.media.loudness import apply_true_peak_limiter, normalize_to_target_lufs


def mute_intervals_with_crossfade(
    audio: np.ndarray,
    mute_spans: List[Tuple[float, float]],
    sample_rate: int = 48000,
    fade_ms: float = 40.0,
) -> np.ndarray:
    """
    Applies smooth equal-power (cos/sin) gain envelope to mute speech regions in original audio.
    fade_ms: duration of fade-in and fade-out (default 40.0ms).
    """
    out = audio.copy()
    fade_samples = max(1, int(round((fade_ms / 1000.0) * sample_rate)))
    num_samples = len(audio)

    # 1.0 = full volume, 0.0 = muted
    gain = np.ones(num_samples, dtype=np.float32)

    # Precalculate equal-power fade curves: cos for fade-out, sin for fade-in
    t_curve = np.linspace(0.0, np.pi / 2.0, fade_samples, endpoint=False, dtype=np.float32)
    fade_out_curve = np.cos(t_curve)  # 1.0 down to ~0.0
    fade_in_curve = np.sin(t_curve)   # 0.0 up to 1.0

    for start_s, end_s in mute_spans:
        s_idx = int(round(start_s * sample_rate))
        e_idx = int(round(end_s * sample_rate))

        s_idx = max(0, min(num_samples, s_idx))
        e_idx = max(0, min(num_samples, e_idx))

        if e_idx <= s_idx:
            continue

        # Equal-power fade out before start
        fade_out_start = max(0, s_idx - fade_samples)
        fade_out_len = s_idx - fade_out_start
        if fade_out_len > 0:
            curve = fade_out_curve[-fade_out_len:]
            gain[fade_out_start:s_idx] = np.minimum(gain[fade_out_start:s_idx], curve)

        # Full mute during speech
        gain[s_idx:e_idx] = 0.0

        # Equal-power fade in after end
        fade_in_end = min(num_samples, e_idx + fade_samples)
        fade_in_len = fade_in_end - e_idx
        if fade_in_len > 0:
            curve = fade_in_curve[:fade_in_len]
            gain[e_idx:fade_in_end] = np.minimum(gain[e_idx:fade_in_end], curve)

    if out.ndim == 1:
        return (out * gain).astype(np.float32)
    else:
        # Multi-channel
        return (out * gain[:, np.newaxis]).astype(np.float32)


def mix_dubbed_audio(
    original_audio_path: Path,
    speech_stem_48k: np.ndarray,
    speech_intervals_to_mute: List[Tuple[float, float]],
    output_wav_path: Path,
    target_lufs: float = -16.0,
    sample_rate: int = 48000,
    separate_music_vocal_path: Optional[Path] = None,
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

    # Pre-normalize the speech stem to -16 LUFS (±1) if non-silent
    if np.max(np.abs(speech_stem_48k)) > 1e-4:
        norm_stem, _ = normalize_to_target_lufs(speech_stem_48k, target_lufs=target_lufs, sample_rate=sample_rate)
        speech_stem_48k = norm_stem

    # 2. Mute speech intervals in original audio (or vocal stem if separated)
    if separate_music_vocal_path and Path(separate_music_vocal_path).exists():
        vocal_data, _ = sf.read(str(separate_music_vocal_path), dtype="float32")
        # Duck only vocal stem, background instrumental remains unmuted
        muted_vocal = mute_intervals_with_crossfade(
            audio=vocal_data,
            mute_spans=speech_intervals_to_mute,
            sample_rate=sample_rate,
            fade_ms=40.0,
        )
        muted_bg = orig_data - vocal_data + muted_vocal
    else:
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

    # 4. Final loudness normalization to -16 LUFS with true-peak limiter (ceiling -1.0 dBTP)
    normalized, final_lufs = normalize_to_target_lufs(
        mixed,
        target_lufs=target_lufs,
        sample_rate=sample_rate,
    )
    limited = apply_true_peak_limiter(normalized, max_peak_dbtp=-1.0)

    # 5. Write to output WAV (48kHz float32/PCM_16)
    sf.write(str(output_wav_path), limited, sample_rate, subtype="PCM_16")

    return final_lufs
