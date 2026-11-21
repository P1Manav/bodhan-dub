"""
Timeline scheduler for dubbed audio segments.
Places synthesized segments onto the audio timeline, handles start shifts,
enforces the ±300ms sync bound, and eliminates segment overlaps without cascading drift.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from bodhan_dub.logging import PipelineLogger
from bodhan_dub.media.audio_proc import time_stretch_audio
from bodhan_dub.stages.fit import SegmentFitResult
from bodhan_dub.stages.segment import SegmentSlot


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
    drift_exceeded: bool = False
    overlap_forced: bool = False


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
    Enforces sample-accurate placement without cascading drift or compound stretching.
    """

    def __init__(self, sample_rate: int = 48000, max_sync_drift_s: float = 0.300):
        self.sample_rate = sample_rate
        self.max_sync_drift_s = max_sync_drift_s

    def schedule(
        self,
        lang: str,
        source_segments: List[SegmentSlot],
        fit_results: List[SegmentFitResult],
        audio_buffers: List[Optional[np.ndarray]],
        total_duration_s: float,
        logger: PipelineLogger,
    ) -> Tuple[ScheduleResult, np.ndarray]:
        """
        B6 & B13: Places segments on the 48kHz audio timeline.
        Enforces non-cascading drift, single-stretch from raw synthesis,
        bounded overlap resolution (borrow, stretch <= 1.20x, crossfade <= 100ms),
        and ghost speech prevention.
        Returns (ScheduleResult, speech_stem_audio_48k).
        """
        total_samples = int(round(total_duration_s * self.sample_rate))
        master_buffer = np.zeros(total_samples, dtype=np.float32)

        scheduled: List[ScheduledSegment] = []
        last_placed_end = 0.0
        sync_offsets: List[float] = []

        # Map fit results and buffers by segment_id
        fit_map = {res.segment_id: (res, buf) for res, buf in zip(fit_results, audio_buffers)}

        for idx, seg in enumerate(source_segments):
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

            # B6: Always re-stretch from raw unstretched audio if available
            raw_audio = getattr(fit_res, "raw_audio", None)
            if raw_audio is None or len(raw_audio) == 0:
                raw_audio = audio_buf.copy()

            # B13: Per-segment sync budget: target start time is source_segment.start
            # Target start incorporates borrowed slack but is clamped strictly to [start - 300ms, start + 300ms]
            desired_shift = fit_res.start_shift
            target_start = seg.start + desired_shift
            clamped_start = max(
                seg.start - self.max_sync_drift_s,
                min(seg.start + self.max_sync_drift_s, target_start),
            )

            # B13: Non-cascading drift prevention:
            # Do NOT cascade previous segment's end time into the future!
            # If last_placed_end > clamped_start, we allow up to 100ms overlap crossfade or adjust placed_start,
            # but do NOT push placed_start beyond [seg.start - 300ms, seg.start + 300ms].
            if last_placed_end > clamped_start:
                # Try placing at last_placed_end only if within +300ms window
                max_allowed_start = seg.start + self.max_sync_drift_s
                if last_placed_end <= max_allowed_start:
                    placed_start = last_placed_end
                else:
                    # Previous segment overran too much; clamp to max allowed start (do not cascade)
                    placed_start = max_allowed_start
            else:
                placed_start = clamped_start

            placed_start = max(0.0, placed_start)
            dur_s = len(audio_buf) / float(self.sample_rate)

            # Lookahead: next valid segment start boundary
            next_seg_target_start = total_duration_s
            for next_seg in source_segments[idx + 1:]:
                if next_seg.segment_id in fit_map and fit_map[next_seg.segment_id][0].fit_status != "skipped":
                    next_seg_target_start = next_seg.start
                    break

            overlap_forced = False
            drift_exceeded = False

            # B13: Overlap resolution if segment would overlap next segment
            if placed_start + dur_s > next_seg_target_start:
                # 1. Borrow into gap before (shift earlier within 250ms limit and previous end)
                earliest_possible = max(last_placed_end, seg.start - min(0.250, self.max_sync_drift_s))
                shift_earlier = max(0.0, placed_start - earliest_possible)
                if shift_earlier > 0:
                    needed_shift = min(shift_earlier, (placed_start + dur_s) - next_seg_target_start)
                    placed_start -= needed_shift

                # 2. If still overlapping, stretch up to 1.20x from RAW audio
                if placed_start + dur_s > next_seg_target_start:
                    avail_time = max(0.1, next_seg_target_start - placed_start)
                    raw_dur_s = len(raw_audio) / float(self.sample_rate)
                    required_speed = raw_dur_s / avail_time
                    capped_speed = min(1.20, max(1.0, required_speed))
                    if capped_speed > fit_res.speed_factor:
                        audio_buf = time_stretch_audio(raw_audio, capped_speed, sample_rate=self.sample_rate)
                        dur_s = len(audio_buf) / float(self.sample_rate)
                        fit_res.speed_factor = round(capped_speed, 3)

                # 3. If still overlapping, borrow from next segment's leading silence (up to 300ms)
                if placed_start + dur_s > next_seg_target_start:
                    gap_overrun = (placed_start + dur_s) - next_seg_target_start
                    if gap_overrun <= 0.100:
                        # Up to 100ms overlap crossfade
                        overlap_forced = True
                        logger.info(
                            f"Segment {seg.segment_id} overlap of {gap_overrun*1000:.1f}ms resolved with crossfade",
                            stage="schedule",
                            lang=lang,
                            segment_id=seg.segment_id,
                        )
                    else:
                        # Cannot fit even after max stretch and crossfade: flag overrun
                        fit_res.fit_status = "overrun"
                        overrun_ms = gap_overrun * 1000.0
                        logger.warning(
                            f"Segment {seg.segment_id} overruns by {overrun_ms:.1f}ms into following slot",
                            stage="schedule",
                            lang=lang,
                            segment_id=seg.segment_id,
                        )

            # B11: Ghost speech check: ensure dubbed segment doesn't drift excessively past source speech
            max_allowed_end = seg.end + self.max_sync_drift_s + 0.100
            if (placed_start + dur_s) > max_allowed_end:
                drift_exceeded = True
                logger.warning(
                    f"Segment {seg.segment_id} exceeds VAD speech window ({placed_start+dur_s:.3f}s > {max_allowed_end:.3f}s); flagged drift_exceeded",
                    stage="schedule",
                    lang=lang,
                    segment_id=seg.segment_id,
                )

            # Check media end boundary (never extend past video duration)
            placed_end = placed_start + dur_s
            if placed_end > total_duration_s:
                excess_s = placed_end - total_duration_s
                earliest_allowed = max(last_placed_end, placed_start - 0.250)
                shift_avail = placed_start - earliest_allowed
                if shift_avail >= excess_s:
                    placed_start -= excess_s
                    placed_end = placed_start + dur_s
                else:
                    if shift_avail > 0:
                        placed_start -= shift_avail
                        placed_end = placed_start + dur_s
                    fit_res.qa_status = "degraded"
                    if "truncation_risk" not in fit_res.reasons:
                        fit_res.reasons.append("truncation_risk")

            sync_offset_ms = (placed_start - seg.start) * 1000.0
            sync_offsets.append(abs(sync_offset_ms))

            # B13: Write samples into master buffer with crossfade on overlap
            start_idx = int(round(placed_start * self.sample_rate))
            if start_idx < total_samples:
                avail = min(len(audio_buf), total_samples - start_idx)
                seg_samples = audio_buf[:avail].copy()

                # If overlapping with previous placed audio, apply smooth crossfade over the overlap region
                overlap_samples = max(0, int(round((last_placed_end - placed_start) * self.sample_rate)))
                if overlap_samples > 0 and overlap_samples <= int(0.100 * self.sample_rate):
                    overlap_len = min(overlap_samples, avail)
                    fade_in = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)
                    fade_out = np.linspace(1.0, 0.0, overlap_len, dtype=np.float32)
                    master_buffer[start_idx : start_idx + overlap_len] = (
                        master_buffer[start_idx : start_idx + overlap_len] * fade_out
                        + seg_samples[:overlap_len] * fade_in
                    )
                    master_buffer[start_idx + overlap_len : start_idx + avail] = seg_samples[overlap_len:avail]
                else:
                    master_buffer[start_idx : start_idx + avail] += seg_samples

            last_placed_end = placed_end

            sched = ScheduledSegment(
                segment_id=seg.segment_id,
                source_start=seg.start,
                source_end=seg.end,
                placed_start=round(placed_start, 3),
                placed_end=round(placed_end, 3),
                sync_offset_ms=round(abs(sync_offset_ms), 1),
                start_shift_s=round(placed_start - seg.start, 3),
                speed_factor=round(fit_res.speed_factor, 3),
                qa_status=fit_res.qa_status,
                fit_status=fit_res.fit_status,
                is_skipped=False,
                drift_exceeded=drift_exceeded,
                overlap_forced=overlap_forced,
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
