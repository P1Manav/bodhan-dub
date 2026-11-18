"""
Unit tests for TimelineScheduler.
"""

import numpy as np

from bodhan_dub.logging import PipelineLogger
from bodhan_dub.stages.fit import SegmentFitResult
from bodhan_dub.stages.qa import QAMetrics
from bodhan_dub.stages.schedule import TimelineScheduler
from bodhan_dub.stages.segment import Segment


def test_timeline_scheduler_placement():
    logger = PipelineLogger(run_id="test_run")
    scheduler = TimelineScheduler(sample_rate=48000, max_sync_drift_s=0.300)

    # 2 segments: seg1 at [1.0, 3.0], seg2 at [4.0, 6.0]
    seg1 = Segment(
        id="s1",
        index=0,
        start=1.0,
        end=3.0,
        duration=2.0,
        slot=[1.0, 3.0],
        slack_before=0.5,
        slack_after=0.5,
        text="First sentence.",
    )
    seg2 = Segment(
        id="s2",
        index=1,
        start=4.0,
        end=6.0,
        duration=2.0,
        slot=[4.0, 6.0],
        slack_before=0.5,
        slack_after=0.5,
        text="Second sentence.",
    )

    qa1 = QAMetrics(cer=0.02, passed=True)
    qa2 = QAMetrics(cer=0.01, passed=True)

    fit1 = SegmentFitResult(
        segment_id="s1",
        source_en="First sentence.",
        final_en="First sentence.",
        final_translation="पहला वाक्य।",
        raw_duration=2.0,
        final_duration=2.0,
        slot_duration=2.0,
        start_shift=0.0,
        speed_factor=1.0,
        qa_metrics=qa1,
        qa_status="passed",
        fit_status="ok",
        attempts=1,
    )
    fit2 = SegmentFitResult(
        segment_id="s2",
        source_en="Second sentence.",
        final_en="Second sentence.",
        final_translation="दूसरा वाक्य।",
        raw_duration=1.8,
        final_duration=1.8,
        slot_duration=2.0,
        start_shift=-0.1,  # shifted 100ms earlier
        speed_factor=1.0,
        qa_metrics=qa2,
        qa_status="passed",
        fit_status="borrowed_slack",
        attempts=1,
    )

    # Generate 48kHz audio buffers
    buf1 = 0.5 * np.ones(int(2.0 * 48000), dtype=np.float32)
    buf2 = 0.5 * np.ones(int(1.8 * 48000), dtype=np.float32)

    sched_res, master_buffer = scheduler.schedule(
        lang="hi",
        source_segments=[seg1, seg2],
        fit_results=[fit1, fit2],
        audio_buffers=[buf1, buf2],
        total_duration_s=8.0,
        logger=logger,
    )

    assert sched_res.total_segments == 2
    assert sched_res.placed_segments == 2
    assert sched_res.skipped_segments == 0
    assert sched_res.max_sync_offset_ms <= 150.0  # shifted by 100ms
    assert len(master_buffer) == int(8.0 * 48000)

    # Check placement of segment 1
    assert sched_res.segments[0].placed_start == 1.0
    assert sched_res.segments[0].placed_end == 3.0

    # Check placement of segment 2 (target start was 4.0 - 0.1 = 3.9)
    assert sched_res.segments[1].placed_start == 3.9
    assert abs(sched_res.segments[1].placed_end - 5.7) < 0.05
