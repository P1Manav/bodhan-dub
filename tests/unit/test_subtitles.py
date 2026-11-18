"""
Unit tests for subtitles and transcript formatting.
"""

from bodhan_dub.stages.segment import SegmentSlot
from bodhan_dub.stages.subtitles import (
    format_srt_timestamp,
    format_vtt_timestamp,
    write_srt,
    write_transcript_en_json,
    write_vtt,
)
from bodhan_dub.text.graphemes import get_grapheme_clusters, wrap_text_graphemes


def test_timestamp_formatting():
    # 83.456 seconds = 1 min, 23.456 sec
    assert format_srt_timestamp(83.456) == "00:01:23,456"
    assert format_vtt_timestamp(83.456) == "00:01:23.456"

    # 3661.050 seconds = 1 hour, 1 min, 1.05 sec
    assert format_srt_timestamp(3661.050) == "01:01:01,050"
    assert format_vtt_timestamp(3661.050) == "01:01:01.050"


def test_grapheme_clusters_indic():
    # Devanagari conjunct: स् + त + े = स्ते is 1 extended grapheme cluster \X
    # So नमस्ते has 3 clusters: न, म, स्ते
    clusters = get_grapheme_clusters("नमस्ते")
    assert len(clusters) == 3
    assert clusters == ["न", "म", "स्ते"]


def test_wrap_text_graphemes():
    long_text = "This is a sentence that wraps across lines."
    wrapped = wrap_text_graphemes(long_text, max_line_chars=25, max_lines=2)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert lines[0] == "This is a sentence that"
    assert lines[1] == "wraps across lines."


def test_write_transcript_and_subtitles(tmp_path):
    slots = [
        SegmentSlot(
            id="seg_0001",
            index=0,
            text="Welcome to artificial intelligence.",
            start=1.0,
            end=4.5,
            duration=3.5,
            slot=[1.0, 4.5],
            slack_before=0.5,
            slack_after=0.42,
            timestamp_source="api_verbose",
        )
    ]

    srt_path = tmp_path / "subtitles.en.srt"
    write_srt(slots, srt_path)
    assert srt_path.exists()
    srt_content = srt_path.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:04,500" in srt_content
    assert "Welcome to artificial intelligence." in srt_content

    vtt_path = tmp_path / "subtitles.en.vtt"
    write_vtt(slots, vtt_path)
    assert vtt_path.exists()
    vtt_content = vtt_path.read_text(encoding="utf-8")
    assert "WEBVTT" in vtt_content
    assert "00:00:01.000 --> 00:00:04.500" in vtt_content

    json_path = tmp_path / "transcript.en.json"
    write_transcript_en_json(slots, json_path)
    assert json_path.exists()
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["id"] == "seg_0001"
    assert data[0]["slot"] == [1.0, 4.5]
