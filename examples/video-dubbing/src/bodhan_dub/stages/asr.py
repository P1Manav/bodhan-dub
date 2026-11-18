"""
ASR Stage for bodhan_dub.
Performs chunk slicing, parallel Indic-Transcribe calls, chunk-splitting retries,
hallucination filtering, and overlap deduplication.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List

import rapidfuzz
import soundfile as sf

from bodhan_dub.gateway.bodhan import BodhanClient
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.media.chunks import AudioChunk
from bodhan_dub.media.vad import SpeechRegion, VADResult
from bodhan_dub.text.normalize import detect_repeated_ngrams


@dataclass
class TranscribedSegment:
    text: str
    start: float
    end: float
    timestamp_source: str  # "api_verbose" or "vad_derived"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def slice_wav_bytes(audio_path: Path, start_s: float, end_s: float) -> bytes:
    """Reads a slice of a 16kHz mono WAV file and returns WAV bytes in memory."""
    with sf.SoundFile(str(audio_path)) as sf_file:
        sr = sf_file.samplerate
        start_frame = int(start_s * sr)
        n_frames = max(1, int((end_s - start_s) * sr))

        sf_file.seek(start_frame)
        slice_data = sf_file.read(n_frames, dtype="int16")

    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", format="WAV", subtype="PCM_16", samplerate=sr, channels=1
    ) as out_sf:
        out_sf.write(slice_data)
    return buf.getvalue()


def calculate_speech_overlap_ratio(
    start: float, end: float, speech_regions: List[SpeechRegion]
) -> float:
    """Calculates what fraction of [start, end] intersects with active VAD speech regions."""
    duration = end - start
    if duration <= 0.0:
        return 0.0

    overlap = 0.0
    for r in speech_regions:
        inter_start = max(start, r.start)
        inter_end = min(end, r.end)
        if inter_end > inter_start:
            overlap += inter_end - inter_start

    return min(1.0, overlap / duration)


def deduplicate_overlap_text(prev_text: str, curr_text: str) -> str:
    """
    Removes duplicate overlapping phrases between adjacent chunks using rapidfuzz.
    """
    prev_words = prev_text.strip().split()
    curr_words = curr_text.strip().split()

    if not prev_words or not curr_words:
        return curr_text

    # Compare trailing up to 6 words of prev with leading up to 6 words of curr
    max_check = min(len(prev_words), len(curr_words), 8)
    for n in range(max_check, 1, -1):
        prev_tail = " ".join(prev_words[-n:]).lower()
        curr_head = " ".join(curr_words[:n]).lower()
        # High similarity indicates duplicate boundary phrase
        if rapidfuzz.fuzz.ratio(prev_tail, curr_head) >= 85:
            # Strip the leading duplicate words from current text
            return " ".join(curr_words[n:])

    return curr_text


class ASRStage:
    """Orchestrates parallel chunk transcription, fault retries, and deduplication."""

    def __init__(
        self,
        bodhan_client: BodhanClient,
        logger: PipelineLogger,
        speech_overlap_min_ratio: float = 0.40,
    ):
        self.bodhan_client = bodhan_client
        self.logger = logger
        self.speech_overlap_min_ratio = speech_overlap_min_ratio

    async def transcribe_chunk(
        self,
        audio_path: Path,
        start_s: float,
        end_s: float,
        chunk_id: int,
        vad_result: VADResult,
        depth: int = 0,
    ) -> List[TranscribedSegment]:
        """
        Transcribes an individual audio chunk with retry and chunk-splitting on empty responses.
        """
        duration = end_s - start_s
        has_speech_in_chunk = any(
            r.end > start_s and r.start < end_s for r in vad_result.speech_regions
        )

        audio_bytes = slice_wav_bytes(audio_path, start_s, end_s)

        for attempt in range(1, 3):
            try:
                resp = await self.bodhan_client.transcribe(
                    audio_bytes=audio_bytes,
                    language="en",
                    response_format="verbose_json",
                    duration_seconds=duration,
                    stage="asr",
                    segment_id=f"chunk_{chunk_id}",
                )

                text = resp.get("text", "").strip()
                api_segments = resp.get("segments", [])

                # If text returned, parse segments
                if text:
                    results: List[TranscribedSegment] = []
                    if api_segments:
                        for s in api_segments:
                            s_text = s.get("text", "").strip()
                            if not s_text:
                                continue
                            seg_start = start_s + float(s.get("start", 0.0))
                            seg_end = min(end_s, start_s + float(s.get("end", duration)))
                            results.append(
                                TranscribedSegment(
                                    text=s_text,
                                    start=round(seg_start, 3),
                                    end=round(seg_end, 3),
                                    timestamp_source="api_verbose",
                                )
                            )
                    else:
                        # Single segment for entire chunk
                        results.append(
                            TranscribedSegment(
                                text=text,
                                start=round(start_s, 3),
                                end=round(end_s, 3),
                                timestamp_source="vad_derived",
                            )
                        )
                    return results

                # If empty text but VAD sees speech, retry or split
                if has_speech_in_chunk and attempt == 1:
                    self.logger.warning(
                        f"Chunk {chunk_id} returned empty text despite VAD detecting speech; retrying...",
                        stage="asr",
                        segment_id=f"chunk_{chunk_id}",
                    )
                    await asyncio.sleep(0.5)
                    continue

            except Exception as e:
                self.logger.warning(
                    f"Chunk {chunk_id} transcription attempt {attempt} failed: {e}",
                    stage="asr",
                    segment_id=f"chunk_{chunk_id}",
                )
                if attempt == 1:
                    await asyncio.sleep(1.0)
                    continue

        # Split chunk in half if it failed and depth allows (max 2 splits)
        if has_speech_in_chunk and depth < 2 and duration > 6.0:
            mid = round((start_s + end_s) / 2.0, 3)
            self.logger.info(
                f"Splitting chunk {chunk_id} into [{start_s}-{mid}] and [{mid}-{end_s}]",
                stage="asr",
                segment_id=f"chunk_{chunk_id}",
            )
            left_res = await self.transcribe_chunk(
                audio_path, start_s, mid, chunk_id, vad_result, depth=depth + 1
            )
            right_res = await self.transcribe_chunk(
                audio_path, mid, end_s, chunk_id, vad_result, depth=depth + 1
            )
            return left_res + right_res

        self.logger.warning(
            f"Chunk {chunk_id} [{start_s}-{end_s}] produced no transcription. Continuing with gap.",
            stage="asr",
            segment_id=f"chunk_{chunk_id}",
        )
        return []

    async def run(
        self,
        audio_path: Path,
        chunks: List[AudioChunk],
        vad_result: VADResult,
    ) -> List[TranscribedSegment]:
        """
        Runs transcription on all planned chunks in parallel, applies hallucination filters,
        and deduplicates overlapping boundaries.
        """
        if not chunks:
            return []

        # Run chunks in parallel bounded by client semaphores
        tasks = [
            self.transcribe_chunk(
                audio_path=audio_path,
                start_s=ch.start,
                end_s=ch.end,
                chunk_id=ch.id,
                vad_result=vad_result,
            )
            for ch in chunks
        ]
        chunk_results = await asyncio.gather(*tasks)

        # Flatten and filter segments
        all_segments: List[TranscribedSegment] = []
        for seg_list in chunk_results:
            for seg in seg_list:
                # 1. Hallucination Check: VAD speech overlap
                overlap_ratio = calculate_speech_overlap_ratio(
                    seg.start, seg.end, vad_result.speech_regions
                )
                if overlap_ratio < self.speech_overlap_min_ratio:
                    self.logger.warning(
                        f"Dropping hallucinated segment '{seg.text}' (overlap {overlap_ratio:.1%} < {self.speech_overlap_min_ratio:.1%}): no_speech_overlap",
                        stage="asr",
                    )
                    continue

                # 2. Hallucination Check: Looped n-grams
                is_loop, phrase = detect_repeated_ngrams(seg.text)
                if is_loop:
                    self.logger.warning(
                        f"Dropping looping segment '{seg.text}' (repeated phrase '{phrase}'): asr_loop",
                        stage="asr",
                    )
                    continue

                # 3. Non-empty check
                if not seg.text.strip():
                    continue

                all_segments.append(seg)

        # Sort chronologically by start time
        all_segments.sort(key=lambda s: s.start)

        # Overlap deduplication across chunk boundaries
        deduped: List[TranscribedSegment] = []
        for idx, seg in enumerate(all_segments):
            if not deduped:
                deduped.append(seg)
                continue

            prev = deduped[-1]
            # If segments overlap in time, deduplicate text
            if seg.start < prev.end:
                clean_text = deduplicate_overlap_text(prev.text, seg.text)
                if clean_text:
                    seg.text = clean_text
                    deduped.append(seg)
            else:
                deduped.append(seg)

        return deduped
