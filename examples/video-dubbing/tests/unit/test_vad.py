"""
Unit tests for Voice Activity Detection and pause inversion.
"""

from bodhan_dub.media.vad import Pause, invert_pauses_to_speech


def test_invert_pauses_to_speech():
    total_dur = 10.0
    pauses = [
        Pause(start=2.0, end=4.0),
        Pause(start=7.0, end=8.5),
    ]

    speech = invert_pauses_to_speech(pauses, total_dur)
    assert len(speech) == 3

    assert speech[0].start == 0.0
    assert speech[0].end == 2.0

    assert speech[1].start == 4.0
    assert speech[1].end == 7.0

    assert speech[2].start == 8.5
    assert speech[2].end == 10.0


def test_empty_pauses_creates_full_speech():
    speech = invert_pauses_to_speech([], 12.0)
    assert len(speech) == 1
    assert speech[0].start == 0.0
    assert speech[0].end == 12.0
