"""
Unit tests for video/audio muxing and verification.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from bodhan_dub.logging import PipelineLogger
from bodhan_dub.media.mux import mux_and_verify
from tests.e2e.test_smoke import generate_synthetic_mp4


def test_mux_and_verify(tmp_path: Path):
    video_path = generate_synthetic_mp4(tmp_path / "source.mp4", duration_s=3)
    assert video_path.exists()

    # Generate 3 seconds of 48kHz audio WAV
    audio_path = tmp_path / "audio.wav"
    samples = np.zeros(48000 * 3, dtype=np.float32)
    sf.write(str(audio_path), samples, 48000)

    out_mp4 = tmp_path / "dubbed_video.mp4"
    logger = PipelineLogger(run_id="test_run")

    res = mux_and_verify(
        video_path=video_path,
        audio_path=audio_path,
        output_mp4_path=out_mp4,
        lang="hi",
        logger=logger,
    )

    assert res.verified is True
    assert out_mp4.exists()
    assert res.duration_delta_ms <= 100.0
