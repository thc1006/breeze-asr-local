"""Real end-to-end pipeline against the user's golden sample.

Runs the full audio→model→transcribe→segments flow with a real 1.6 GB Q8_0
download. Skipped by default — opt in with `pytest --run-slow`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from breeze_asr.audio import convert_to_16k_mono_wav
from breeze_asr.model import ensure_ggml
from breeze_asr.transcriber import run_whisper


pytestmark = pytest.mark.slow


def _skip_without_flag(config: pytest.Config) -> None:
    if not config.getoption("--run-slow", default=False):
        pytest.skip("integration tests require --run-slow")


def test_breeze_q8_0_transcribes_1woman_m4a(
    pytestconfig: pytest.Config, real_m4a: Path
) -> None:
    _skip_without_flag(pytestconfig)

    wav_path, duration = convert_to_16k_mono_wav(real_m4a)
    try:
        assert 3.0 < duration < 30.0, "1woman.m4a should be a short clip"
        model_path = ensure_ggml(variant="q8_0")
        segments = run_whisper(
            wav_path=wav_path,
            model_path=model_path,
            language="zh",
        )
    finally:
        wav_path.unlink(missing_ok=True)

    # Content assertions — regression-locks the known-good transcription.
    assert len(segments) >= 2, f"expected >=2 segments, got {len(segments)}"
    full_text = "".join(s.text for s in segments)
    assert "那個女人是誰" in full_text
    assert "老師" in full_text

    # Timestamps are monotonic and cover the clip.
    for prev, curr in zip(segments, segments[1:]):
        assert curr.start_time >= prev.start_time
    assert segments[-1].end_time <= duration + 0.5
