"""
TTS synthesis stage and dynamic CPS (chars-per-second) calibration.
Predicts duration, splits text safely at punctuation (≤ 180 chars),
and stitches audio with crossfades and edge silence trimming.
"""
from __future__ import annotations

import io
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
import regex
import soundfile as sf

from bodhan_dub.gateway.bodhan import BodhanClient
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.media.audio_proc import (
    crossfade_concat,
    resample_24k_to_48k,
    trim_edge_silence,
)


class CPSCalibrator:
    """Tracks running median characters-per-second per (lang, voice)."""

    def __init__(self):
        # Initial calibration seeds from Phase 0 live measurements
        self.history: Dict[Tuple[str, str], List[float]] = {
            ("hi", "Kavya"): [11.37],
            ("hi", "Amit"): [13.50],
            ("bn", "Ishita"): [11.20],
            ("bn", "Sourav"): [13.80],
            ("ta", "Kavya"): [11.50],
            ("te", "Kavya"): [12.00],
        }
        self.default_cps = 12.0

    def get_cps(self, lang: str, voice: str) -> float:
        """Returns running median CPS for given voice profile."""
        key = (lang, voice)
        if key in self.history and self.history[key]:
            return float(median(self.history[key]))
        # Fallback to any measurement for this voice
        voice_matches = [
            v for (_lang, v_name), hist in self.history.items() if v_name == voice for v in hist
        ]
        if voice_matches:
            return float(median(voice_matches))
        return self.default_cps

    def record_measurement(self, lang: str, voice: str, char_count: int, duration_s: float) -> None:
        """Updates calibration history with a verified synthesis run."""
        if duration_s <= 0.1 or char_count <= 2:
            return
        cps = round(char_count / duration_s, 2)
        # Sane bounds check for speech rate (6 to 25 cps)
        if 6.0 <= cps <= 25.0:
            key = (lang, voice)
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(cps)
            # Keep last 15 observations for running window
            if len(self.history[key]) > 15:
                self.history[key].pop(0)

    def predict_duration(self, text: str, lang: str, voice: str) -> float:
        """Predicts audio duration before calling synthesis API."""
        chars = len(text.strip())
        cps = self.get_cps(lang, voice)
        return round(chars / cps, 3)


def split_text_at_punctuation(text: str, max_chars: int = 180) -> List[str]:
    """
    Splits text at clause/sentence punctuation (danda ।, ., ?, ,, !)
    ensuring each piece is <= max_chars to prevent server truncation.
    """
    t_clean = text.strip()
    if len(t_clean) <= max_chars:
        return [t_clean]

    # Split preserving delimiters using regex \X
    # Indic danda: \u0964, double danda: \u0965
    tokens = regex.split(r"([।॥\.\?!,;\n]\s*)", t_clean)
    pieces: List[str] = []
    curr = ""

    for token in tokens:
        if not token:
            continue
        if len(curr) + len(token) <= max_chars:
            curr += token
        else:
            if curr.strip():
                pieces.append(curr.strip())
            curr = token

    if curr.strip():
        pieces.append(curr.strip())

    return pieces if pieces else [t_clean]


class TTSSynthesizer:
    """Synthesizes text into 48kHz float32 audio with duration prediction."""

    def __init__(
        self,
        bodhan_client: BodhanClient,
        logger: PipelineLogger,
        calibrator: Optional[CPSCalibrator] = None,
    ):
        self.bodhan_client = bodhan_client
        self.logger = logger
        self.calibrator = calibrator or CPSCalibrator()

    async def synthesize(
        self,
        text: str,
        lang: str,
        voice: str,
        segment_id: str,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Synthesizes text into 48 kHz mono float32 audio.
        Returns (audio_48k, actual_duration, predicted_duration).
        """
        predicted_dur = self.calibrator.predict_duration(text, lang, voice)

        # Split into <=180 character chunks at punctuation
        pieces = split_text_at_punctuation(text, max_chars=180)

        audio_pieces: List[np.ndarray] = []
        for idx, piece in enumerate(pieces):
            piece_id = f"{segment_id}_p{idx}" if len(pieces) > 1 else segment_id
            raw_wav_bytes = await self.bodhan_client.speak(
                text=piece,
                voice=voice,
                language_code=lang,
                stage="tts",
                lang=lang,
                segment_id=piece_id,
            )

            # Read 24kHz audio from bytes
            with io.BytesIO(raw_wav_bytes) as buf:
                samples_24k, sr = sf.read(buf, dtype="float32")
                if len(samples_24k.shape) > 1:
                    samples_24k = samples_24k[:, 0]

            # Trim leading/trailing edge silence
            trimmed = trim_edge_silence(samples_24k)
            if len(trimmed) > 0:
                audio_pieces.append(trimmed)

        # Stitch pieces with 10ms smooth crossfades
        stitched_24k = crossfade_concat(audio_pieces, crossfade_ms=10, sample_rate=24000)

        # Resample to 48kHz float32 for video pipeline standard
        audio_48k = resample_24k_to_48k(stitched_24k)
        actual_dur = len(audio_48k) / 48000.0

        # Update calibration history
        self.calibrator.record_measurement(lang, voice, len(text), actual_dur)

        return audio_48k, actual_dur, predicted_dur

