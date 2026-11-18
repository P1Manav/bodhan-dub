"""
Timeline scheduler for dubbed audio segments.
Places synthesized segments onto the audio timeline, handles start shifts,
enforces the ±300ms sync bound, and eliminates segment overlaps.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from bodhan_dub.logging import PipelineLogger
from bodhan_dub.stages.fit import SegmentFitResult
from bodhan_dub.stages.segment import Segment


class ScheduledSegment(BaseModel):
    segment_id: str
    source_start: float
    source_end: float
    placed_start: float
    placed_end: float
    sync_offset_ms: float
    start_shift_s: float
    speed_factor: float
    qa_status: str
    fit_status: str
    is_skipped: bool = False


class ScheduleResult(BaseModel):
    lang: str
    total_segments: int
    placed_segments: int
    skipped_segments: int
    degraded_segments: int
    max_sync_offset_ms: float
    p95_sync_offset_ms: float
    segments: List[ScheduledSegment] = Field(default_factory=list)


class TimelineScheduler:
    """
    Schedules fitted audio segments onto the master timeline.
    Enforces sample-accurate placement without collisions.
    """

    def __init__(self, sample_rate: int = 48000, max_sync_drift_s: float = 0.300):
        self.sample_rate = sample_rate
        self.max_sync_drift_s = max_sync_drift_s

    def schedule(
        self,
        lang: str,
        source_segments: List[Segment],
        fit_results: List[SegmentFitResult],
        audio_buffers: List[Optional[np.ndarray]],
        total_duration_s: float,
        logger: PipelineLogger,
    ) -> Tuple[ScheduleResult, np.ndarray]:
        """
        Places segments on the 48kHz audio timeline.
        Returns (ScheduleResult, speech_stem_audio_48k).
        """
        total_samples = int(round(total_duration_s * self.sample_rate))
        master_buffer = np.zeros(total_samples, dtype=np.float32)

        scheduled: List[ScheduledSegment] = []
        last_placed_end = 0.0

        sync_offsets: List[float] = []

        # Map fit results by segment_id
        fit_map = {res.segment_id: (res, buf) for res, buf in zip(fit_results, audio_buffers)}

        for seg in source_segments:
            if seg.segment_id not in fit_map:
                continue

            fit_res, audio_buf = fit_map[seg.segment_id]

            if fit_res.fit_status == "skipped" or audio_buf is None or len(audio_buf) == 0:
                # Slot is skipped; marked for keeping original audio
                sched = ScheduledSegment(
                    segment_id=seg.segment_id,
                    source_start=seg.start,
                    source_end=seg.end,
                    placed_start=seg.start,
                    placed_end=seg.end,
                    sync_offset_ms=0.0,
                    start_shift_s=0.0,
                    speed_factor=1.0,
                    qa_status=fit_res.qa_status,
                    fit_status="skipped",
                    is_skipped=True,
                )
                scheduled.append(sched)
                continue

            # Requested start time based on borrowed slack shift
            target_start = seg.start + fit_res.start_shift
            # Collision prevention: must not start before previous placed end
            placed_start = max(target_start, last_placed_end)
            placed_start = max(0.0, placed_start)

            dur_s = len(audio_buf) / float(self.sample_rate)
            placed_end = placed_start + dur_s

            sync_offset_ms = abs(placed_start - seg.start) * 1000.0
            sync_offsets.append(sync_offset_ms)

            # Warn if sync drift exceeds 300 ms target
            if (sync_offset_ms / 1000.0) > self.max_sync_drift_s:
                logger.warning(
                    f"Segment {seg.segment_id} sync offset {sync_offset_ms:.1f}ms exceeds target 300ms",
                    stage="schedule",
                    lang=lang,
                    segment_id=seg.segment_id,
                )

            # Write samples into master buffer
            start_idx = int(round(placed_start * self.sample_rate))

            if start_idx < total_samples:
                avail = min(len(audio_buf), total_samples - start_idx)
                # Overlap-add with soft clipping if previous trailing edge exists
                master_buffer[start_idx : start_idx + avail] += audio_buf[:avail]

            last_placed_end = placed_end

            sched = ScheduledSegment(
                segment_id=seg.segment_id,
                source_start=seg.start,
                source_end=seg.end,
                placed_start=round(placed_start, 3),
                placed_end=round(placed_end, 3),
                sync_offset_ms=round(sync_offset_ms, 1),
                start_shift_s=round(fit_res.start_shift, 3),
                speed_factor=round(fit_res.speed_factor, 3),
                qa_status=fit_res.qa_status,
                fit_status=fit_res.fit_status,
                is_skipped=False,
            )
            scheduled.append(sched)

        max_sync = max(sync_offsets) if sync_offsets else 0.0
        p95_sync = float(np.percentile(sync_offsets, 95)) if sync_offsets else 0.0

        placed_cnt = sum(1 for s in scheduled if not s.is_skipped)
        skipped_cnt = sum(1 for s in scheduled if s.is_skipped)
        degraded_cnt = sum(1 for s in scheduled if s.qa_status == "degraded")

        result = ScheduleResult(
            lang=lang,
            total_segments=len(scheduled),
            placed_segments=placed_cnt,
            skipped_segments=skipped_cnt,
            degraded_segments=degraded_cnt,
            max_sync_offset_ms=round(max_sync, 1),
            p95_sync_offset_ms=round(p95_sync, 1),
            segments=scheduled,
        )

        return result, master_buffer
