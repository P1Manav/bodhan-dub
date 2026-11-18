"""
Speech QA evaluation and bounded retry planner.
Back-transcribes synthesized audio with Indic-Transcribe, measures CER and acoustic defects,
and plans intelligent retries.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.signal
import soundfile as sf
from pydantic import BaseModel, Field

from bodhan_dub.gateway.bodhan import BodhanClient
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.media.audio_proc import check_clipping, detect_internal_silence
from bodhan_dub.text.cer import compute_cer, detect_tail_omission
from bodhan_dub.text.normalize import detect_repeated_ngrams


class QAMetrics(BaseModel):
    cer: float
    passed: bool
    has_repetition: bool = False
    has_tail_omission: bool = False
    duration_ratio: float = 1.0  # actual / predicted
    is_silent: bool = False
    is_clipped: bool = False
    is_empty: bool = False
    failure_reasons: List[str] = Field(default_factory=list)


@dataclass
class SynthesisCandidate:
    attempt: int
    text: str
    audio_48k: np.ndarray
    duration_s: float
    metrics: QAMetrics


def convert_48k_to_16k_wav_bytes(samples_48k: np.ndarray) -> bytes:
    """Downsamples 48kHz float32 audio to 16kHz mono WAV bytes for ASR evaluation."""
    if len(samples_48k) == 0:
        return b""
    # 48k -> 16k is 1/3 decimation (up=1, down=3)
    downsampled = scipy.signal.resample_poly(samples_48k, up=1, down=3)
    # Convert float32 to int16
    int16_samples = np.clip(downsampled * 32767.0, -32768.0, 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", format="WAV", subtype="PCM_16", samplerate=16000, channels=1
    ) as f:
        f.write(int16_samples)
    return buf.getvalue()


class SpeechQA:
    """Evaluates speech synthesis against reference text and plans retries."""

    def __init__(
        self,
        bodhan_client: BodhanClient,
        logger: PipelineLogger,
        cer_warn: float = 0.15,
        cer_fail: float = 0.30,
    ):
        self.bodhan_client = bodhan_client
        self.logger = logger
        self.cer_warn = cer_warn
        self.cer_fail = cer_fail

    async def evaluate(
        self,
        audio_48k: np.ndarray,
        reference_text: str,
        predicted_duration: float,
        lang: str,
        segment_id: str,
    ) -> QAMetrics:
        """
        Back-transcribes synthesized audio and checks CER, repetitions, omissions, and clipping.
        """
        reasons: List[str] = []
        actual_dur = len(audio_48k) / 48000.0

        if actual_dur <= 0.05 or len(audio_48k) == 0:
            return QAMetrics(cer=1.0, passed=False, is_empty=True, failure_reasons=["empty_audio"])

        # Acoustic checks
        clipped = check_clipping(audio_48k)
        if clipped:
            reasons.append("audio_clipping")

        silent = detect_internal_silence(audio_48k, sample_rate=48000, min_silence_s=1.5)
        if silent:
            reasons.append("internal_silence")

        dur_ratio = actual_dur / max(0.1, predicted_duration)
        if dur_ratio < 0.6 or dur_ratio > 1.6:
            reasons.append(f"duration_mismatch_ratio_{dur_ratio:.2f}")

        # Back-transcription with Indic-Transcribe
        wav_16k_bytes = convert_48k_to_16k_wav_bytes(audio_48k)
        try:
            resp = await self.bodhan_client.transcribe(
                audio_bytes=wav_16k_bytes,
                language=lang,
                response_format="json",
                duration_seconds=actual_dur,
                stage="qa",
                lang=lang,
                segment_id=segment_id,
            )
            hypothesis_text = resp.get("text", "").strip()
        except Exception as e:
            self.logger.warning(
                f"Back-transcription failed during QA: {e}",
                stage="qa",
                lang=lang,
                segment_id=segment_id,
            )
            hypothesis_text = ""

        # Compute CER
        cer = compute_cer(reference_text, hypothesis_text)
        if cer >= self.cer_fail:
            reasons.append(f"cer_high_{cer:.3f}")
        elif cer >= self.cer_warn:
            reasons.append(f"cer_warn_{cer:.3f}")

        # Repetition detector
        has_repeat, repeated_str = detect_repeated_ngrams(
            hypothesis_text, min_n=1, max_n=3, repeat_threshold=3
        )
        if has_repeat:
            reasons.append(f"repetition_{repeated_str}")

        # Tail omission detector
        has_tail_omission, tail_str = detect_tail_omission(reference_text, hypothesis_text)
        if has_tail_omission:
            reasons.append(f"tail_omission_{tail_str}")

        passed = (
            (cer < self.cer_fail) and (not has_repeat) and (not has_tail_omission) and (not clipped)
        )

        return QAMetrics(
            cer=round(cer, 4),
            passed=passed,
            has_repetition=has_repeat,
            has_tail_omission=has_tail_omission,
            duration_ratio=round(dur_ratio, 2),
            is_silent=silent,
            is_clipped=clipped,
            is_empty=False,
            failure_reasons=reasons,
        )

    def plan_retry_action(
        self,
        metrics: QAMetrics,
        attempt: int,
    ) -> str:
        """
        Chooses corrective action from: 'split_text', 'paraphrase_source', 'condense', 'retry_same'.
        """
        if metrics.has_tail_omission or metrics.has_repetition:
            return "split_text"
        if metrics.cer >= self.cer_fail:
            return "paraphrase_source" if attempt == 1 else "condense"
        if metrics.duration_ratio > 1.3:
            return "condense"
        return "retry_same"
