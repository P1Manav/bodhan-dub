"""
Unit tests for TTS synthesis, CPS calibrator, and punctuation splitting.
"""

import io

import numpy as np
import pytest
import soundfile as sf

from bodhan_dub.logging import PipelineLogger
from bodhan_dub.stages.tts import CPSCalibrator, TTSSynthesizer, split_text_at_punctuation


def test_cps_calibrator_running_median():
    calibrator = CPSCalibrator()
    # Initial seed check
    hi_cps = calibrator.get_cps("hi", "Kavya")
    assert 10.0 <= hi_cps <= 13.0

    # Record 3 measurements
    calibrator.record_measurement("hi", "Kavya", 100, 10.0)  # 10.0 cps
    calibrator.record_measurement("hi", "Kavya", 120, 10.0)  # 12.0 cps
    calibrator.record_measurement("hi", "Kavya", 140, 10.0)  # 14.0 cps

    # Median of history should be around 11.37 or 12.0
    cps = calibrator.get_cps("hi", "Kavya")
    assert 10.0 <= cps <= 14.0

    # Test duration prediction
    pred = calibrator.predict_duration("यह पंद्रह अक्षरों का वाक्य है।", "hi", "Kavya")
    assert pred > 0.5


def test_split_text_at_punctuation_short_text():
    short = "नमस्ते, आप कैसे हैं?"
    assert split_text_at_punctuation(short, max_chars=180) == [short]


def test_split_text_at_punctuation_long_text():
    part1 = "पहला भाग जो काफी लंबा है और इसमें कई सारे शब्द शामिल हैं।"
    part2 = "दूसरा भाग जो इसके बाद आता है और बात को आगे बढ़ाता है।"
    part3 = "तीसरा भाग जो अंत में आता है और वाक्य को पूर्ण करता है।"
    full = f"{part1} {part2} {part3}"

    pieces = split_text_at_punctuation(full, max_chars=80)
    assert len(pieces) >= 2
    for p in pieces:
        assert len(p) <= 80


@pytest.mark.asyncio
async def test_tts_synthesizer_flow():
    # Create fake audio bytes for speak response (24kHz sine wave)
    t = np.linspace(0, 1.0, 24000, endpoint=False, dtype=np.float32)
    sine_24k = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, sine_24k, 24000, format="WAV")
    audio_bytes = buf.getvalue()

    class MockBodhanClient:
        async def speak(self, text, voice, language_code, stage, lang, segment_id):
            return audio_bytes

    logger = PipelineLogger(run_id="test_run")
    synthesizer = TTSSynthesizer(
        bodhan_client=MockBodhanClient(),
        logger=logger,
    )

    audio_48k, actual_dur, pred_dur = await synthesizer.synthesize(
        text="नमस्ते दुनिया",
        lang="hi",
        voice="Kavya",
        segment_id="seg_01",
    )

    assert len(audio_48k) > 0
    # Resampled to 48kHz, should be approx 48000 samples (~1 sec)
    assert abs(len(audio_48k) - 48000) < 1000
    assert abs(actual_dur - 1.0) < 0.1
