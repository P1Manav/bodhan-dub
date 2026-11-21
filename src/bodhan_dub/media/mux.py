"""
Multiplexer stage for video and dubbed audio.
Combines video stream and normalized 48kHz audio into MP4 with AI-dubbed metadata,
verifies stream sync within ±100ms, and performs bounded repair if needed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from bodhan_dub.logging import PipelineLogger


class MuxResult(BaseModel):
    output_path: str
    video_duration_s: float
    audio_duration_s: float
    duration_delta_ms: float
    verified: bool
    repair_attempts: int = 0
    error: Optional[str] = None


def probe_media_streams(file_path: Path) -> dict:
    """Probes media streams and metadata tags using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,start_time:stream=index,codec_type,duration,start_time:format_tags:stream_tags",
        "-of",
        "json",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


ISO_639_2_MAP = {
    "hi": "hin",
    "bn": "ben",
    "te": "tel",
    "ta": "tam",
    "mr": "mar",
    "gu": "guj",
    "kn": "kan",
    "ml": "mal",
    "pa": "pan",
    "or": "ori",
    "as": "asm",
    "ur": "urd",
    "en": "eng",
}


def get_stream_duration(probe_data: dict, codec_type: str) -> float:
    """Extracts duration for a given codec_type from ffprobe output."""
    for s in probe_data.get("streams", []):
        if s.get("codec_type") == codec_type and "duration" in s:
            try:
                return float(s["duration"])
            except ValueError:
                pass
    # Fallback to format duration
    return float(probe_data.get("format", {}).get("duration", 0.0))


def get_stream_start_time(probe_data: dict, codec_type: str) -> float:
    """Extracts start_time for a given codec_type from ffprobe output."""
    for s in probe_data.get("streams", []):
        if s.get("codec_type") == codec_type and "start_time" in s:
            try:
                return float(s["start_time"])
            except ValueError:
                pass
    try:
        return float(probe_data.get("format", {}).get("start_time", 0.0))
    except ValueError:
        return 0.0


def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_mp4_path: Path,
    lang: str,
    logger: PipelineLogger,
    reencode_video: bool = False,
) -> None:
    """Executes ffmpeg multiplexing command with duration lock and format preservation."""
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe input video duration and start_time
    probe = probe_media_streams(video_path)
    input_v_dur = get_stream_duration(probe, "video")
    input_v_start = get_stream_start_time(probe, "video")

    itsoffset_args = ["-itsoffset", f"{input_v_start:.3f}"] if input_v_start > 0.05 else []

    v_codec = (
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast"]
        if reencode_video
        else ["-c:v", "copy"]
    )

    iso_lang = ISO_639_2_MAP.get(lang, lang)

    cmd = [
        "ffmpeg",
        "-y",
        *itsoffset_args,
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        *v_codec,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-metadata",
        "comment=AI-dubbed with Bodhan Indic-Speak",
        "-metadata:s:a:0",
        f"language={iso_lang}",
        "-metadata",
        f"title=Dubbed in {lang} using Bodhan Indic AI",
        "-t",
        f"{input_v_dur:.3f}",
        str(output_mp4_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        if not reencode_video:
            # Retry with re-encoding video (handles VFR or stream incompatibility)
            logger.warning(
                f"Video copy muxing failed: {e.stderr[:200]}; retrying with H.264 re-encoding",
                stage="mux",
                lang=lang,
            )
            mux_video_audio(
                video_path,
                audio_path,
                output_mp4_path,
                lang,
                logger,
                reencode_video=True,
            )
        else:
            raise e


def mux_and_verify(
    video_path: Path,
    audio_path: Path,
    output_mp4_path: Path,
    lang: str,
    logger: PipelineLogger,
    raise_on_failure: bool = True,
) -> MuxResult:
    """
    Multiplexes video and audio, validates duration match within ±100ms,
    runs bounded repair if delta exceeds tolerance, and raises error if desynced.
    """
    repair_attempts = 0
    current_audio_path = audio_path
    last_delta_ms = 0.0

    # Probe input video duration for strict duration match check
    in_probe = probe_media_streams(video_path)
    target_video_dur = get_stream_duration(in_probe, "video")

    while repair_attempts <= 2:
        mux_video_audio(video_path, current_audio_path, output_mp4_path, lang, logger)

        # ffprobe verification
        try:
            probe = probe_media_streams(output_mp4_path)
            v_dur = get_stream_duration(probe, "video")
            a_dur = get_stream_duration(probe, "audio")
            delta_ms = abs(v_dur - a_dur) * 1000.0
            last_delta_ms = delta_ms

            # Check both streams and verify output matches input duration within ±100ms
            input_delta_ms = abs(v_dur - target_video_dur) * 1000.0

            if delta_ms <= 100.0 and input_delta_ms <= 100.0:
                return MuxResult(
                    output_path=str(output_mp4_path),
                    video_duration_s=round(v_dur, 3),
                    audio_duration_s=round(a_dur, 3),
                    duration_delta_ms=round(delta_ms, 1),
                    verified=True,
                    repair_attempts=repair_attempts,
                )

            # If duration delta > 100ms and repairs left, pad/trim audio WAV to exact video duration
            if repair_attempts < 2:
                repair_attempts += 1
                logger.warning(
                    f"Mux duration delta {delta_ms:.1f}ms exceeds 100ms limit; attempting repair {repair_attempts}",
                    stage="mux_repair",
                    lang=lang,
                )
                repaired_wav_path = (
                    output_mp4_path.parent / f"repaired_audio_a{repair_attempts}.wav"
                )
                audio_data, sr = sf.read(str(current_audio_path), dtype="float32")
                target_samples = int(round(v_dur * sr))

                if len(audio_data) < target_samples:
                    pad_len = target_samples - len(audio_data)
                    repaired_data = np.pad(audio_data, (0, pad_len), mode="constant")
                else:
                    repaired_data = audio_data[:target_samples]

                sf.write(str(repaired_wav_path), repaired_data, sr, subtype="PCM_16")
                current_audio_path = repaired_wav_path
            else:
                break

        except Exception as e:
            if raise_on_failure:
                raise RuntimeError(f"ffprobe verification failed: {e}") from e
            return MuxResult(
                output_path=str(output_mp4_path),
                video_duration_s=0.0,
                audio_duration_s=0.0,
                duration_delta_ms=0.0,
                verified=False,
                repair_attempts=repair_attempts,
                error=f"ffprobe verification failed: {e}",
            )

    msg = f"Mux verification failed: duration delta {last_delta_ms:.1f}ms exceeds ±100ms tolerance after repairs"
    if raise_on_failure:
        raise RuntimeError(msg)

    return MuxResult(
        output_path=str(output_mp4_path),
        video_duration_s=round(v_dur, 3) if "v_dur" in locals() else 0.0,
        audio_duration_s=round(a_dur, 3) if "a_dur" in locals() else 0.0,
        duration_delta_ms=round(last_delta_ms, 1),
        verified=False,
        repair_attempts=repair_attempts,
        error=msg,
    )
