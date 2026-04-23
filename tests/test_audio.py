"""TDD: audio.convert_to_16k_mono_wav.

Contract:
  - Accepts any format ffmpeg understands.
  - Returns (Path, duration_seconds) where the path is a 16 kHz, 1-channel,
    PCM_S16LE WAV in a temp location.
  - Raises FileNotFoundError if src is missing.
  - Raises AudioConversionError on ffmpeg failure.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from asr_local.audio import AudioConversionError, convert_to_16k_mono_wav


class TestConvertResamplesAndDownmixes:
    def test_output_exists_and_is_wav(self, wav_44k_stereo: Path) -> None:
        out, duration = convert_to_16k_mono_wav(wav_44k_stereo)
        try:
            assert out.exists()
            assert out.suffix.lower() == ".wav"
            assert duration == pytest.approx(2.0, abs=0.05)
        finally:
            out.unlink(missing_ok=True)

    def test_output_is_16khz_mono_pcm16(self, wav_44k_stereo: Path) -> None:
        out, _ = convert_to_16k_mono_wav(wav_44k_stereo)
        try:
            info = sf.info(str(out))
            assert info.samplerate == 16000
            assert info.channels == 1
            assert info.subtype == "PCM_16"
        finally:
            out.unlink(missing_ok=True)

    def test_passthrough_for_already_target_format(self, wav_16k_mono: Path) -> None:
        # Still converts (safer than heuristics) but duration + format must match.
        out, duration = convert_to_16k_mono_wav(wav_16k_mono)
        try:
            info = sf.info(str(out))
            assert info.samplerate == 16000 and info.channels == 1
            assert duration == pytest.approx(1.5, abs=0.05)
        finally:
            out.unlink(missing_ok=True)


class TestConvertErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_to_16k_mono_wav(tmp_path / "does_not_exist.wav")

    def test_non_audio_file_raises_conversion_error(self, tmp_path: Path) -> None:
        bogus = tmp_path / "not_audio.wav"
        bogus.write_bytes(b"this is not audio data at all")
        with pytest.raises(AudioConversionError):
            convert_to_16k_mono_wav(bogus)

    def test_ffmpeg_timeout_surfaces_as_conversion_error(
        self, tmp_path: Path, mocker
    ) -> None:
        import subprocess as sp

        src = tmp_path / "a.wav"
        src.write_bytes(b"\x00")  # existence check only
        mocker.patch(
            "asr_local.audio.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="ffmpeg", timeout=300),
        )
        with pytest.raises(AudioConversionError, match="timed out"):
            convert_to_16k_mono_wav(src)


class TestConvertRealM4a:
    """Integration test against the user's real m4a clip."""

    def test_m4a_converts_to_valid_wav(self, real_m4a: Path) -> None:
        out, duration = convert_to_16k_mono_wav(real_m4a)
        try:
            info = sf.info(str(out))
            assert info.samplerate == 16000
            assert info.channels == 1
            assert info.subtype == "PCM_16"
            # 1woman.m4a is ~5-10 s of speech; sanity-bound duration.
            assert 1.0 < duration < 120.0
        finally:
            out.unlink(missing_ok=True)
