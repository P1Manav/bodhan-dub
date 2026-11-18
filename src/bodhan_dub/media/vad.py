"""
Voice Activity Detection (VAD) and speech/pause boundary detection.
Primary engine: Silero-VAD via onnxruntime.
Fallback engine: ffmpeg silencedetect filter.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf


@dataclass
class TimeRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeechRegion(TimeRange):
    pass


@dataclass
class Pause(TimeRange):
    pass


class VADResult:
    def __init__(
        self,
        total_duration: float,
        speech_regions: List[SpeechRegion],
        pauses: List[Pause],
        engine_used: str = "silero_onnx",
    ):
        self.total_duration = total_duration
        self.speech_regions = speech_regions
        self.pauses = pauses
        self.engine_used = engine_used

    @property
    def has_speech(self) -> bool:
        return len(self.speech_regions) > 0 and sum(r.duration for r in self.speech_regions) >= 0.3

    @property
    def total_speech_duration(self) -> float:
        return sum(r.duration for r in self.speech_regions)


def detect_silence_ffmpeg(
    wav_path: Path,
    noise_db: float = -32.0,
    min_silence_duration: float = 0.3,
) -> Tuple[List[Pause], float]:
    """
    Runs ffmpeg silencedetect to find pauses in audio.
    Returns (list_of_pauses, total_duration).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_duration}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    pauses: List[Pause] = []
    current_start: Optional[float] = None
    duration = 0.0

    # Parse duration from ffprobe or ffmpeg output
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if dur_match:
        hours = float(dur_match.group(1))
        mins = float(dur_match.group(2))
        secs = float(dur_match.group(3))
        duration = hours * 3600 + mins * 60 + secs

    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([0-9\.]+)", line)
            if m:
                current_start = float(m.group(1))
        elif "silence_end:" in line and current_start is not None:
            m = re.search(r"silence_end:\s*([0-9\.]+)", line)
            if m:
                end = float(m.group(1))
                pauses.append(Pause(start=round(current_start, 3), end=round(end, 3)))
                current_start = None

    # If audio ended while inside silence
    if current_start is not None and duration > current_start:
        pauses.append(Pause(start=round(current_start, 3), end=round(duration, 3)))

    return pauses, duration


def invert_pauses_to_speech(pauses: List[Pause], total_duration: float) -> List[SpeechRegion]:
    """Inverts silence intervals into speech regions."""
    speech: List[SpeechRegion] = []
    current_pos = 0.0

    # Sort pauses by start time
    sorted_pauses = sorted(pauses, key=lambda p: p.start)

    for pause in sorted_pauses:
        if pause.start > current_pos + 0.05:  # at least 50ms speech
            speech.append(SpeechRegion(start=round(current_pos, 3), end=round(pause.start, 3)))
        current_pos = max(current_pos, pause.end)

    if total_duration > current_pos + 0.05:
        speech.append(SpeechRegion(start=round(current_pos, 3), end=round(total_duration, 3)))

    return speech


class SileroVADDetector:
    """ONNX-based Silero VAD implementation with automatic ffmpeg fallback."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self._session = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempts to load Silero VAD onnx session."""
        try:
            import onnxruntime as ort

            # Check paths for silero_vad.onnx
            candidates = [
                self.model_path,
                Path("silero_vad.onnx"),
                Path(__file__).resolve().parent.parent.parent / "silero_vad.onnx",
                Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx",
            ]
            valid_path = None
            for c in candidates:
                if c and Path(c).exists():
                    valid_path = Path(c)
                    break

            if valid_path:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self._session = ort.InferenceSession(
                    str(valid_path), opts, providers=["CPUExecutionProvider"]
                )
        except Exception:
            self._session = None

    def process(
        self,
        wav_path: Path,
        threshold: float = 0.5,
        min_silence_ms: int = 300,
        min_speech_ms: int = 250,
    ) -> VADResult:
        """
        Runs Voice Activity Detection on a 16kHz mono WAV file.
        Uses ONNX session if loaded, else falls back to ffmpeg silencedetect.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        # If ONNX session is active, run native frame-level inference
        if self._session is not None:
            try:
                audio, sr = sf.read(str(wav_path))
                if sr != 16000:
                    raise ValueError(f"Silero VAD requires 16000 Hz, got {sr}")
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                audio = audio.astype(np.float32)
                total_duration = len(audio) / 16000.0

                # Silero window is 512 samples at 16kHz (32ms)
                window_size = 512
                state = np.zeros((2, 1, 128), dtype=np.float32)

                speech_frames: List[bool] = []
                for i in range(0, len(audio) - window_size + 1, window_size):
                    chunk = audio[i : i + window_size].reshape(1, -1)
                    inputs = {
                        "input": chunk,
                        "state": state,
                        "sr": np.array([16000], dtype=np.int64),
                    }
                    out, state = self._session.run(None, inputs)
                    prob = float(out[0][0])
                    speech_frames.append(prob >= threshold)

                # Convert boolean frames to regions
                frame_dur = window_size / 16000.0
                regions: List[SpeechRegion] = []
                in_speech = False
                speech_start = 0.0

                for idx, is_sp in enumerate(speech_frames):
                    t = idx * frame_dur
                    if is_sp and not in_speech:
                        in_speech = True
                        speech_start = t
                    elif not is_sp and in_speech:
                        in_speech = False
                        if t - speech_start >= (min_speech_ms / 1000.0):
                            regions.append(SpeechRegion(round(speech_start, 3), round(t, 3)))

                if in_speech and total_duration - speech_start >= (min_speech_ms / 1000.0):
                    regions.append(SpeechRegion(round(speech_start, 3), round(total_duration, 3)))

                # Derive pauses
                pauses: List[Pause] = []
                curr = 0.0
                for r in regions:
                    if r.start - curr >= (min_silence_ms / 1000.0):
                        pauses.append(Pause(round(curr, 3), round(r.start, 3)))
                    curr = r.end
                if total_duration - curr >= (min_silence_ms / 1000.0):
                    pauses.append(Pause(round(curr, 3), round(total_duration, 3)))

                return VADResult(
                    total_duration=round(total_duration, 3),
                    speech_regions=regions,
                    pauses=pauses,
                    engine_used="silero_onnx",
                )
            except Exception:
                # Fall back to ffmpeg silencedetect on any runtime error
                pass

        # Fallback Engine: ffmpeg silencedetect
        pauses, duration = detect_silence_ffmpeg(
            wav_path,
            noise_db=-30.0,
            min_silence_duration=min_silence_ms / 1000.0,
        )
        if duration <= 0.0:
            info = sf.info(str(wav_path))
            duration = info.duration

        speech_regions = invert_pauses_to_speech(pauses, duration)

        return VADResult(
            total_duration=round(duration, 3),
            speech_regions=speech_regions,
            pauses=pauses,
            engine_used="ffmpeg_silencedetect",
        )
