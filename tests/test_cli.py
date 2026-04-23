"""TDD: cli.parse_args and cli.main orchestration."""
from __future__ import annotations

from pathlib import Path

import pytest

from asr_local import cli
from asr_local.cli import choose_audio_ctx
from asr_local.segment import TimestampedSegment


class TestChooseAudioCtx:
    def test_short_clip_uses_small_context(self) -> None:
        assert choose_audio_ctx(5.8) == 512

    def test_medium_clip_uses_mid_context(self) -> None:
        assert choose_audio_ctx(15.0) == 1024

    def test_long_clip_uses_full_context(self) -> None:
        assert choose_audio_ctx(45.0) == 0

    def test_boundary_ten_seconds(self) -> None:
        assert choose_audio_ctx(9.999) == 512
        assert choose_audio_ctx(10.0) == 1024

    def test_boundary_thirty_seconds(self) -> None:
        assert choose_audio_ctx(29.999) == 1024
        assert choose_audio_ctx(30.0) == 0


class TestParseArgs:
    def test_audio_path_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args([])

    def test_default_quant_is_q8_0(self, tmp_path: Path) -> None:
        args = cli.parse_args([str(tmp_path / "a.wav")])
        assert args.quant == "q8_0"

    def test_quant_fp16_accepted(self, tmp_path: Path) -> None:
        args = cli.parse_args([str(tmp_path / "a.wav"), "--quant", "fp16"])
        assert args.quant == "fp16"

    def test_unknown_quant_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args([str(tmp_path / "a.wav"), "--quant", "bogus"])

    def test_default_language_zh(self, tmp_path: Path) -> None:
        args = cli.parse_args([str(tmp_path / "a.wav")])
        assert args.language == "zh"

    def test_default_threads_8(self, tmp_path: Path) -> None:
        args = cli.parse_args([str(tmp_path / "a.wav")])
        assert args.threads == 8

    def test_custom_output(self, tmp_path: Path) -> None:
        out = tmp_path / "custom.txt"
        args = cli.parse_args([str(tmp_path / "a.wav"), "--output", str(out)])
        assert args.output == out


class TestMain:
    def test_missing_audio_exits_nonzero(self, tmp_path: Path, capsys) -> None:
        rc = cli.main([str(tmp_path / "nope.wav")])
        assert rc != 0
        err = capsys.readouterr().err
        assert "not found" in err.lower()

    def test_happy_path_orchestration(
        self, tmp_path: Path, mocker
    ) -> None:
        # Input file exists
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        wav_out = tmp_path / "converted.wav"
        wav_out.write_bytes(b"\x00")
        model = tmp_path / "ggml-model-q8_0.bin"
        model.write_bytes(b"\x00")
        txt_out = tmp_path / "a.transcript.txt"

        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav",
            return_value=(wav_out, 5.0),
        )
        mocker.patch("asr_local.cli.ensure_ggml", return_value=model)
        mocker.patch(
            "asr_local.cli.run_whisper",
            return_value=[
                TimestampedSegment(0.0, 2.5, "你好"),
                TimestampedSegment(2.5, 5.0, "測試"),
            ],
        )
        mocker.patch("asr_local.cli.save_transcript", return_value=txt_out)

        rc = cli.main([str(audio)])
        assert rc == 0

    def test_empty_transcription_exits_with_warning(
        self, tmp_path: Path, mocker
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav",
            return_value=(tmp_path / "converted.wav", 1.0),
        )
        model = tmp_path / "m"
        model.write_bytes(b"\x00")
        mocker.patch("asr_local.cli.ensure_ggml", return_value=model)
        mocker.patch("asr_local.cli.run_whisper", return_value=[])
        mocker.patch("asr_local.cli.save_transcript", return_value=None)

        rc = cli.main([str(audio)])
        assert rc == 2

    def test_quant_flag_forwarded_to_ensure_ggml(
        self, tmp_path: Path, mocker
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav",
            return_value=(tmp_path / "converted.wav", 1.0),
        )
        model = tmp_path / "m"
        model.write_bytes(b"\x00")
        ensure = mocker.patch("asr_local.cli.ensure_ggml", return_value=model)
        mocker.patch("asr_local.cli.run_whisper", return_value=[])
        mocker.patch("asr_local.cli.save_transcript", return_value=None)

        cli.main([str(audio), "--quant", "fp16"])
        ensure.assert_called_once()
        assert ensure.call_args.kwargs.get("variant") == "fp16"

    def test_temp_wav_cleaned_up_after_success(
        self, tmp_path: Path, mocker
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        tmp_wav = tmp_path / "converted.wav"
        tmp_wav.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav", return_value=(tmp_wav, 1.0)
        )
        model = tmp_path / "m"
        model.write_bytes(b"\x00")
        mocker.patch("asr_local.cli.ensure_ggml", return_value=model)
        mocker.patch(
            "asr_local.cli.run_whisper",
            return_value=[TimestampedSegment(0, 1, "x")],
        )
        mocker.patch("asr_local.cli.save_transcript", return_value=tmp_path / "o.txt")

        cli.main([str(audio)])
        assert not tmp_wav.exists()
