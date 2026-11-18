"""
End-to-end smoke test for bodhan_dub using the offline FakeBodhan server.
Generates a synthetic 30-second video fixture using ffmpeg and runs CLI preflight and estimate.
"""

import subprocess
from pathlib import Path

import pytest
from fake_bodhan.server import FakeBodhanTransport

from bodhan_dub.cli import async_main, build_parser


def generate_synthetic_mp4(out_path: Path, duration_s: int = 30) -> Path:
    """Generates a small synthetic MP4 video file using ffmpeg."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_s}:size=320x240:rate=25",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_s}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


@pytest.mark.asyncio
async def test_smoke_cli_estimate(tmp_path):
    """Verifies that the CLI executes --estimate-only successfully on a synthetic 30s fixture."""
    fixture_video = generate_synthetic_mp4(tmp_path / "smoke_sample.mp4", duration_s=30)
    assert fixture_video.exists()

    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            str(fixture_video),
            "--langs",
            "hi,ta,bn",
            "--out",
            str(tmp_path / "out"),
            "--estimate-only",
        ]
    )

    exit_code = await async_main(args)
    assert exit_code == 0, f"Expected exit code 0 for --estimate-only, got {exit_code}"


@pytest.mark.asyncio
async def test_smoke_cli_preflight_success(tmp_path, monkeypatch):
    """Verifies that CLI initializes cleanly on valid inputs with mock keys."""
    # Set mock keys for offline smoke testing
    monkeypatch.setenv("BODHAN_TRANSCRIBE_KEY", "sk-mock-transcribe-key")
    monkeypatch.setenv("BODHAN_TRANSLATE_KEY", "sk-mock-translate-key")
    monkeypatch.setenv("BODHAN_SPEAK_KEY", "sk-mock-speak-key")

    fixture_video = generate_synthetic_mp4(tmp_path / "smoke_sample_preflight.mp4", duration_s=65)
    assert fixture_video.exists()

    parser = build_parser()
    out_dir = tmp_path / "out_preflight"
    args = parser.parse_args(
        [
            "--input",
            str(fixture_video),
            "--langs",
            "hi,te",
            "--out",
            str(out_dir),
        ]
    )

    exit_code = await async_main(args, transport=FakeBodhanTransport())
    assert exit_code == 0
    # Check structured log created
    log_file = out_dir / "run.log.jsonl"
    assert log_file.exists(), "Structured JSONL log file must be created"
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists(), "Run manifest must be created"
    transcript_file = out_dir / "transcript.en.json"
    assert transcript_file.exists(), "English transcript transcript.en.json must be created"
    srt_file = out_dir / "subtitles.en.srt"
    assert srt_file.exists(), "English subtitles.en.srt must be created"
