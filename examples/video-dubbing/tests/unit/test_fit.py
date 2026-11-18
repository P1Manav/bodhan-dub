"""
Unit tests for SegmentFitController (duration fit & QA loop).
"""

import numpy as np
import pytest

from bodhan_dub.logging import PipelineLogger
from bodhan_dub.stages.fit import SegmentFitController
from bodhan_dub.stages.qa import QAMetrics


class DummyTTS:
    def __init__(self, sample_rate=48000, fixed_dur=1.0):
        self.sample_rate = sample_rate
        self.fixed_dur = fixed_dur

    async def synthesize(self, text, lang, voice, segment_id):
        samples = int(self.sample_rate * self.fixed_dur)
        t = np.linspace(0, self.fixed_dur, samples, endpoint=False, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 300 * t)
        return audio, self.fixed_dur, self.fixed_dur


class DummyQA:
    def __init__(self, passed=True, cer=0.0):
        self.passed = passed
        self.cer = cer

    async def evaluate(self, audio_48k, reference_text, predicted_duration, lang, segment_id):
        return QAMetrics(
            cer=self.cer,
            passed=self.passed,
            has_repetition=False,
            has_tail_omission=False,
            duration_ratio=1.0,
            is_silent=False,
            is_clipped=False,
            is_empty=len(audio_48k) == 0,
        )

    def plan_retry_action(self, metrics, attempt):
        return "retry_same"


@pytest.mark.asyncio
async def test_fit_controller_natural_speed():
    logger = PipelineLogger(run_id="test_run")
    # slot 2.0s, audio 1.0s -> ratio = 0.5 <= 1.0 -> ok
    tts = DummyTTS(fixed_dur=1.0)
    qa = DummyQA(passed=True)
    controller = SegmentFitController(
        bodhan_client=None,
        llm_client=None,
        logger=logger,
        tts_synthesizer=tts,
        speech_qa=qa,
    )

    res, audio = await controller.process_segment(
        segment_id="s1",
        source_en="Hello world",
        initial_translation="नमस्ते दुनिया",
        slot_duration=2.0,
        slack_before=0.5,
        slack_after=0.5,
        lang="hi",
        voice="Kavya",
    )

    assert res.fit_status == "ok"
    assert res.speed_factor == 1.0
    assert res.qa_status == "passed"
    assert audio is not None
    assert abs(res.final_duration - 1.0) < 0.05


@pytest.mark.asyncio
async def test_fit_controller_stretched():
    logger = PipelineLogger(run_id="test_run")
    # slot 1.0s, audio 1.15s -> ratio = 1.15 in (1.0, 1.20] -> stretched
    tts = DummyTTS(fixed_dur=1.15)
    qa = DummyQA(passed=True)
    controller = SegmentFitController(
        bodhan_client=None,
        llm_client=None,
        logger=logger,
        tts_synthesizer=tts,
        speech_qa=qa,
    )

    res, audio = await controller.process_segment(
        segment_id="s2",
        source_en="Hello world",
        initial_translation="नमस्ते दुनिया",
        slot_duration=1.0,
        slack_before=0.1,
        slack_after=0.1,
        lang="hi",
        voice="Kavya",
    )

    assert res.fit_status == "stretched"
    assert 1.0 < res.speed_factor <= 1.20
    assert audio is not None


@pytest.mark.asyncio
async def test_fit_controller_borrow_slack():
    logger = PipelineLogger(run_id="test_run")
    # slot 1.0s, audio 1.5s -> ratio 1.5 > 1.20
    # slack_before 0.250s, slack_after 0.500s -> effective slot = 1.75s
    # eff_ratio = 1.5 / 1.75 = 0.857 <= 1.20 -> borrowed_slack
    tts = DummyTTS(fixed_dur=1.5)
    qa = DummyQA(passed=True)
    controller = SegmentFitController(
        bodhan_client=None,
        llm_client=None,
        logger=logger,
        tts_synthesizer=tts,
        speech_qa=qa,
    )

    res, audio = await controller.process_segment(
        segment_id="s3",
        source_en="Hello world long sentence",
        initial_translation="नमस्ते दुनिया एक लंबा वाक्य",
        slot_duration=1.0,
        slack_before=0.3,
        slack_after=0.5,
        lang="hi",
        voice="Kavya",
    )

    assert res.fit_status == "borrowed_slack"
    assert res.start_shift <= 0.0
    assert res.speed_factor <= 1.20


@pytest.mark.asyncio
async def test_fit_controller_overrun_capped_at_1_20():
    logger = PipelineLogger(run_id="test_run")
    # slot 1.0s, audio 3.0s, zero slack -> eff_ratio 3.0 > 1.20 -> overrun, max stretch 1.20
    tts = DummyTTS(fixed_dur=3.0)
    qa = DummyQA(passed=True)
    controller = SegmentFitController(
        bodhan_client=None,
        llm_client=None,
        logger=logger,
        tts_synthesizer=tts,
        speech_qa=qa,
    )

    res, audio = await controller.process_segment(
        segment_id="s4",
        source_en="Massive text",
        initial_translation="अत्यधिक विशाल वाक्य",
        slot_duration=1.0,
        slack_before=0.0,
        slack_after=0.0,
        lang="hi",
        voice="Kavya",
    )

    assert res.fit_status == "overrun"
    assert res.speed_factor == 1.20  # Never exceed 1.20!


@pytest.mark.asyncio
async def test_fit_controller_condensation_and_paraphrase(
    test_settings, test_cache, test_ledger, test_logger
):
    from fake_bodhan.server import FakeBodhanTransport

    from bodhan_dub.gateway.bodhan import BodhanClient
    from bodhan_dub.gateway.cost import BudgetGuard
    from bodhan_dub.gateway.llm import LLMClient

    transport = FakeBodhanTransport()
    b_client = BodhanClient(
        settings=test_settings,
        cache=test_cache,
        ledger=test_ledger,
        budget_guard=BudgetGuard(test_ledger),
        logger=test_logger,
        transport=transport,
    )
    llm_client = LLMClient(
        settings=test_settings,
        cache=test_cache,
        ledger=test_ledger,
        logger=test_logger,
        transport=transport,
    )

    try:
        controller = SegmentFitController(
            bodhan_client=b_client,
            llm_client=llm_client,
            logger=test_logger,
        )

        # Test _condense_english_source directly
        condensed = await controller._condense_english_source(
            source_en="Welcome to artificial intelligence.",
            target_char_budget=20,
            lang="hi",
            segment_id="s_condense",
        )
        assert condensed is not None
        assert len(condensed) > 0

        # Test _paraphrase_english_source directly
        paraphrased = await controller._paraphrase_english_source(
            source_en="Welcome to artificial intelligence.",
            lang="hi",
            segment_id="s_paraphrase",
        )
        assert paraphrased is not None
        assert len(paraphrased) > 0
    finally:
        await b_client.close()
        await llm_client.close()

