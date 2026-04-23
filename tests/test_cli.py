"""TDD: cli.parse_args and cli.main orchestration."""
from __future__ import annotations

from pathlib import Path

import pytest

from asr_local import cli
from asr_local.cli import (
    choose_audio_ctx,
    choose_flash_attn,
    choose_processors,
    choose_vad,
)
from asr_local.segment import TimestampedSegment


class TestChooseVad:
    def test_short_clip_disables_vad(self) -> None:
        # VAD overhead (~1ms/30s) doesn't pay off on a 5-second clip.
        assert choose_vad(5.8) is False

    def test_boundary_30s_enables_vad(self) -> None:
        assert choose_vad(29.999) is False
        assert choose_vad(30.0) is True

    def test_long_clip_enables_vad(self) -> None:
        assert choose_vad(300.0) is True
        assert choose_vad(3600.0) is True


class TestChooseProcessors:
    """(p, t) split must satisfy p*t <= cpu_count and favour parallelism
    for long audio where chunk-parallel overlap actually helps."""

    def test_short_clip_is_single_processor_all_threads(self) -> None:
        # < 30 s: no point spawning a second chunk-processor.
        p, t = choose_processors(5.8, cpu_count=8)
        assert p == 1 and t == 8

    def test_long_clip_uses_two_processors(self) -> None:
        # >= 30 s: 2x4 beat 1x8 by ~25% (measured on 57.6 s clip).
        p, t = choose_processors(57.6, cpu_count=8)
        assert p == 2 and t == 4

    def test_very_short_cpu_count_degrades_gracefully(self) -> None:
        # 4-core machine: can't do 2x4; stay sequential.
        p, t = choose_processors(60.0, cpu_count=4)
        assert p * t <= 4
        assert p >= 1 and t >= 1

    @pytest.mark.parametrize(
        "duration,cpu",
        [(5.0, 8), (15.0, 8), (30.0, 8), (60.0, 8), (60.0, 12), (300.0, 8)],
    )
    def test_product_never_exceeds_cpu_count(self, duration: float, cpu: int) -> None:
        p, t = choose_processors(duration, cpu_count=cpu)
        assert p * t <= cpu

    def test_minimum_threads_is_one(self) -> None:
        p, t = choose_processors(60.0, cpu_count=2)
        assert t >= 1 and p >= 1

    def test_boundary_exactly_30_seconds(self) -> None:
        # 30 s is the point where parallel chunks become worthwhile.
        p_short, _ = choose_processors(29.999, cpu_count=8)
        p_long, _ = choose_processors(30.0, cpu_count=8)
        assert p_short == 1
        assert p_long == 2

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError):
            choose_processors(-1.0, cpu_count=8)


class TestChooseFlashAttn:
    """Flash attention degrades CJK transcription quality and is also slower
    for Mandarin on Snapdragon X (whisper.cpp issue #3020, empirically
    confirmed 15% speedup for Breeze-ASR-25 Q8_0 with -nfa on zh)."""

    @pytest.mark.parametrize("lang", ["zh", "ja", "ko"])
    def test_cjk_disables_flash_attn(self, lang: str) -> None:
        assert choose_flash_attn(lang) is False

    @pytest.mark.parametrize("lang", ["en", "de", "fr", "es", "auto"])
    def test_non_cjk_keeps_flash_attn(self, lang: str) -> None:
        assert choose_flash_attn(lang) is True

    def test_unknown_language_defaults_safe(self) -> None:
        # Unknown / exotic language codes keep the whisper.cpp default (ON).
        assert choose_flash_attn("zzzzz") is True


class TestChooseAudioCtx:
    """-ac value must cover the whole clip. whisper mel rate is 100 fps."""

    def test_long_clip_returns_full_context(self) -> None:
        # 30 s or more needs full context (3000 frames).
        assert choose_audio_ctx(30.0) == 0
        assert choose_audio_ctx(45.0) == 0
        assert choose_audio_ctx(120.0) == 0

    def test_returned_value_covers_duration(self) -> None:
        # Invariant: ac / 100 fps must be >= duration for every short clip,
        # otherwise the encoder silently truncates the audio tail.
        for dur in [0.5, 1.0, 2.5, 5.0, 5.8, 7.3, 9.9, 15.0, 25.0, 29.9]:
            ac = choose_audio_ctx(dur)
            if ac == 0:
                continue  # full 30 s is always enough
            covered_s = ac / 100.0
            assert covered_s >= dur, (
                f"duration {dur}s > covered {covered_s}s (ac={ac})"
            )

    def test_output_is_multiple_of_64(self) -> None:
        # SIMD alignment: whisper encoder kernels want 64-aligned seq lens.
        for dur in [0.5, 5.8, 15.0, 25.0]:
            ac = choose_audio_ctx(dur)
            assert ac == 0 or ac % 64 == 0

    def test_5s_clip_regression_with_pad(self) -> None:
        # Previous bug: returned 512 (5.12 s, truncated tail) for 5.8 s clip.
        # Also lock in the safety pad: 580 frames needed + 64 pad = 644
        # minimum before 64-alignment rounds it to 704.
        assert choose_audio_ctx(5.8) >= 580 + 64

    def test_zero_duration_returns_small_nonzero(self) -> None:
        # Pathological: 0 s should still return a valid small ac.
        ac = choose_audio_ctx(0.0)
        assert ac > 0 and ac < 3000
        assert ac % 64 == 0

    def test_negative_duration_raises(self) -> None:
        import pytest as _pt
        with _pt.raises(ValueError):
            choose_audio_ctx(-1.0)


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

    def test_default_threads_is_auto(self, tmp_path: Path) -> None:
        # --threads default is None (auto-tuned via choose_processors).
        args = cli.parse_args([str(tmp_path / "a.wav")])
        assert args.threads is None

    def test_default_processors_is_auto(self, tmp_path: Path) -> None:
        args = cli.parse_args([str(tmp_path / "a.wav")])
        assert args.processors is None

    def test_explicit_threads_override(self, tmp_path: Path) -> None:
        args = cli.parse_args([str(tmp_path / "a.wav"), "--threads", "6"])
        assert args.threads == 6

    def test_explicit_processors_override(self, tmp_path: Path) -> None:
        args = cli.parse_args([str(tmp_path / "a.wav"), "--processors", "4"])
        assert args.processors == 4

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
        mocker.patch("asr_local.cli.ensure_vad_model", return_value=tmp_path / "vad.bin")
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

    def test_audio_conversion_error_exits_with_clean_message(
        self, tmp_path: Path, mocker, capsys
    ) -> None:
        from asr_local.audio import AudioConversionError

        audio = tmp_path / "bad.wav"
        audio.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav",
            side_effect=AudioConversionError("ffmpeg: invalid stream"),
        )
        rc = cli.main([str(audio)])
        assert rc == 3
        err = capsys.readouterr().err
        assert "audio conversion failed" in err
        assert "invalid stream" in err

    def test_whisper_cli_not_found_exits_with_code_4(
        self, tmp_path: Path, mocker, capsys
    ) -> None:
        # WhisperCliNotFoundError is a WhisperCliError subclass and must
        # flow through the same error handler.
        from asr_local.transcriber import WhisperCliNotFoundError

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        wav = tmp_path / "converted.wav"
        wav.write_bytes(b"\x00")
        model = tmp_path / "m.bin"
        model.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav", return_value=(wav, 1.0)
        )
        mocker.patch("asr_local.cli.ensure_ggml", return_value=model)
        mocker.patch(
            "asr_local.cli.run_whisper",
            side_effect=WhisperCliNotFoundError("binary missing"),
        )
        rc = cli.main([str(audio)])
        assert rc == 4
        assert "whisper-cli failed" in capsys.readouterr().err

    def test_ggml_validation_error_exits_with_code_5(
        self, tmp_path: Path, mocker, capsys
    ) -> None:
        from asr_local.model import GgmlValidationError

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        wav = tmp_path / "converted.wav"
        wav.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav", return_value=(wav, 1.0)
        )
        mocker.patch(
            "asr_local.cli.ensure_ggml",
            side_effect=GgmlValidationError("bad magic"),
        )
        rc = cli.main([str(audio)])
        assert rc == 5
        assert "model validation failed" in capsys.readouterr().err

    def test_network_error_exits_with_code_6(
        self, tmp_path: Path, mocker, capsys
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        wav = tmp_path / "converted.wav"
        wav.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav", return_value=(wav, 1.0)
        )
        mocker.patch(
            "asr_local.cli.ensure_ggml",
            side_effect=ConnectionError("HF unreachable"),
        )
        rc = cli.main([str(audio)])
        assert rc == 6
        assert "download or IO failed" in capsys.readouterr().err

    def test_timeout_flag_forwarded_to_run_whisper(
        self, tmp_path: Path, mocker
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        wav = tmp_path / "converted.wav"
        wav.write_bytes(b"\x00")
        model = tmp_path / "m.bin"
        model.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav", return_value=(wav, 1.0)
        )
        mocker.patch("asr_local.cli.ensure_ggml", return_value=model)
        rw = mocker.patch("asr_local.cli.run_whisper", return_value=[])
        mocker.patch("asr_local.cli.save_transcript", return_value=None)

        cli.main([str(audio), "--timeout", "42"])
        assert rw.call_args.kwargs.get("timeout") == 42.0

    def test_whisper_error_exits_with_clean_message(
        self, tmp_path: Path, mocker, capsys
    ) -> None:
        from asr_local.transcriber import WhisperCliError

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00")
        wav = tmp_path / "converted.wav"
        wav.write_bytes(b"\x00")
        model = tmp_path / "m.bin"
        model.write_bytes(b"\x00")
        mocker.patch(
            "asr_local.cli.convert_to_16k_mono_wav", return_value=(wav, 1.0)
        )
        mocker.patch("asr_local.cli.ensure_ggml", return_value=model)
        mocker.patch(
            "asr_local.cli.run_whisper",
            side_effect=WhisperCliError("model load failed"),
        )
        rc = cli.main([str(audio)])
        assert rc == 4
        err = capsys.readouterr().err
        assert "whisper-cli failed" in err

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
        mocker.patch("asr_local.cli.ensure_vad_model", return_value=tmp_path / "vad.bin")
        mocker.patch(
            "asr_local.cli.run_whisper",
            return_value=[TimestampedSegment(0, 1, "x")],
        )
        mocker.patch("asr_local.cli.save_transcript", return_value=tmp_path / "o.txt")

        cli.main([str(audio)])
        assert not tmp_wav.exists()
