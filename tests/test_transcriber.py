"""TDD: transcriber.run_whisper and its parsing + cmd-build helpers."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from asr_local.segment import TimestampedSegment
from asr_local.transcriber import (
    WhisperCliError,
    WhisperCliNotFoundError,
    _build_command,
    _parse_json_output,
    resolve_default_binary,
    run_whisper,
)

FIXTURES = Path(__file__).parent / "fixtures" / "json"


class TestParseJsonOutput:
    def test_parses_sample_transcription(self) -> None:
        segments = _parse_json_output(FIXTURES / "sample_output.json")
        # Empty-text segments are dropped.
        assert len(segments) == 2
        first, second = segments
        assert first.start_time == pytest.approx(0.0)
        assert first.end_time == pytest.approx(2.5)
        assert first.text == "你好,歡迎收聽本節目。"
        assert second.start_time == pytest.approx(2.5)
        assert second.end_time == pytest.approx(5.12)
        assert second.text == "今天我們要討論的主題是台灣華語。"

    def test_returns_timestamped_segments(self) -> None:
        segments = _parse_json_output(FIXTURES / "sample_output.json")
        assert all(isinstance(s, TimestampedSegment) for s in segments)

    def test_missing_transcription_key_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"result": {}}), encoding="utf-8")
        assert _parse_json_output(path) == []

    def test_strips_leading_whitespace_in_text(self, tmp_path: Path) -> None:
        path = tmp_path / "ws.json"
        path.write_text(
            json.dumps(
                {
                    "transcription": [
                        {"offsets": {"from": 0, "to": 1000}, "text": "   leading spaces"}
                    ]
                }
            ),
            encoding="utf-8",
        )
        segs = _parse_json_output(path)
        assert segs[0].text == "leading spaces"

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json at all", encoding="utf-8")
        with pytest.raises(WhisperCliError):
            _parse_json_output(path)


class TestBuildCommand:
    def test_contains_required_flags(self, tmp_path: Path) -> None:
        wav = tmp_path / "x.wav"
        model = tmp_path / "m.bin"
        wav.touch()
        model.touch()
        cmd = _build_command(
            binary_path=Path("whisper-cli.exe"),
            wav_path=wav,
            model_path=model,
            language="zh",
            threads=8,
            output_prefix=tmp_path / "out",
        )
        # assert each arg pair is present
        assert "-m" in cmd and str(model) in cmd
        assert "-f" in cmd and str(wav) in cmd
        assert "-l" in cmd and "zh" in cmd
        assert "-t" in cmd and "8" in cmd
        assert "-oj" in cmd
        assert "-of" in cmd and str(tmp_path / "out") in cmd

    def test_threads_forwarded(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=4,
            output_prefix=tmp_path / "o",
        )
        i = cmd.index("-t")
        assert cmd[i + 1] == "4"

    def test_binary_path_is_first(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("my-whisper.exe"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
        )
        assert cmd[0] == "my-whisper.exe"

    def test_audio_ctx_omitted_when_zero(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
            audio_ctx=0,
        )
        assert "-ac" not in cmd

    def test_audio_ctx_forwarded_when_positive(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
            audio_ctx=512,
        )
        i = cmd.index("-ac")
        assert cmd[i + 1] == "512"

    def test_flash_attn_default_true_omits_nfa(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="en",
            threads=1,
            output_prefix=tmp_path / "o",
            flash_attn=True,
        )
        assert "-nfa" not in cmd

    def test_flash_attn_false_adds_nfa(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
            flash_attn=False,
        )
        assert "-nfa" in cmd

    def test_processors_default_omits_flag(self, tmp_path: Path) -> None:
        # whisper-cli default is -p 1; only emit -p when explicitly > 1.
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=4,
            output_prefix=tmp_path / "o",
            processors=1,
        )
        assert "-p" not in cmd

    def test_processors_two_emits_flag(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=4,
            output_prefix=tmp_path / "o",
            processors=2,
        )
        i = cmd.index("-p")
        assert cmd[i + 1] == "2"

    def test_vad_default_none_omits_flag(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
        )
        assert "--vad" not in cmd
        assert "-vm" not in cmd

    def test_decode_default_beam_omits_bs_bo(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
        )
        # Default: let whisper-cli use beam=5 best_of=5. Don't emit flags.
        assert "-bs" not in cmd
        assert "-bo" not in cmd

    def test_decode_greedy_sets_bs1_bo1(self, tmp_path: Path) -> None:
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
            greedy=True,
        )
        i_bs = cmd.index("-bs")
        i_bo = cmd.index("-bo")
        assert cmd[i_bs + 1] == "1"
        assert cmd[i_bo + 1] == "1"

    def test_vad_model_path_adds_flags(self, tmp_path: Path) -> None:
        vad = tmp_path / "silero.bin"
        vad.touch()
        cmd = _build_command(
            binary_path=Path("w"),
            wav_path=tmp_path / "x.wav",
            model_path=tmp_path / "m.bin",
            language="zh",
            threads=1,
            output_prefix=tmp_path / "o",
            vad_model_path=vad,
        )
        assert "--vad" in cmd
        i = cmd.index("-vm")
        assert cmd[i + 1] == str(vad)


class TestRunWhisperPriority:
    def test_default_priority_passes_zero_creationflags(
        self, tmp_path: Path, mocker
    ) -> None:
        wav = tmp_path / "x.wav"
        model = tmp_path / "m.bin"
        binary = tmp_path / "whisper-cli.exe"
        for p in (wav, model, binary):
            p.touch()

        def fake_run(cmd, **kwargs):
            out_prefix = Path(cmd[cmd.index("-of") + 1])
            out_prefix.with_suffix(".json").write_text(
                (FIXTURES / "sample_output.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

        run = mocker.patch("asr_local.transcriber.subprocess.run", side_effect=fake_run)
        run_whisper(wav_path=wav, model_path=model, binary_path=binary)
        assert run.call_args.kwargs.get("creationflags") == 0

    def test_high_priority_sets_high_priority_class(
        self, tmp_path: Path, mocker
    ) -> None:
        wav = tmp_path / "x.wav"
        model = tmp_path / "m.bin"
        binary = tmp_path / "whisper-cli.exe"
        for p in (wav, model, binary):
            p.touch()

        def fake_run(cmd, **kwargs):
            out_prefix = Path(cmd[cmd.index("-of") + 1])
            out_prefix.with_suffix(".json").write_text(
                (FIXTURES / "sample_output.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

        run = mocker.patch("asr_local.transcriber.subprocess.run", side_effect=fake_run)
        run_whisper(
            wav_path=wav, model_path=model, binary_path=binary, priority="high"
        )
        assert run.call_args.kwargs.get("creationflags") == 0x00000080


class TestRunWhisperOrchestration:
    """Subprocess mocked — we test orchestration, error routing, output file handling."""

    def test_missing_binary_raises(self, tmp_path: Path) -> None:
        wav = tmp_path / "x.wav"
        wav.touch()
        model = tmp_path / "m.bin"
        model.touch()
        with pytest.raises(WhisperCliNotFoundError):
            run_whisper(
                wav_path=wav,
                model_path=model,
                binary_path=tmp_path / "does-not-exist.exe",
            )

    def test_nonzero_exit_raises_with_stderr(
        self, tmp_path: Path, mocker
    ) -> None:
        wav = tmp_path / "x.wav"
        model = tmp_path / "m.bin"
        binary = tmp_path / "whisper-cli.exe"
        for p in (wav, model, binary):
            p.touch()

        mocker.patch(
            "asr_local.transcriber.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"model load failed"
            ),
        )
        with pytest.raises(WhisperCliError, match="model load failed"):
            run_whisper(wav_path=wav, model_path=model, binary_path=binary)

    def test_success_parses_output_file(
        self, tmp_path: Path, mocker
    ) -> None:
        wav = tmp_path / "x.wav"
        model = tmp_path / "m.bin"
        binary = tmp_path / "whisper-cli.exe"
        for p in (wav, model, binary):
            p.touch()

        # simulate whisper-cli creating the JSON output next to -of prefix
        def fake_run(cmd, **kwargs):
            out_prefix = Path(cmd[cmd.index("-of") + 1])
            out_json = out_prefix.with_suffix(".json")
            out_json.write_text(
                (FIXTURES / "sample_output.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b""
            )

        mocker.patch("asr_local.transcriber.subprocess.run", side_effect=fake_run)
        segments = run_whisper(wav_path=wav, model_path=model, binary_path=binary)
        assert len(segments) == 2
        assert segments[0].text == "你好,歡迎收聽本節目。"

    def test_timeout_wrapped_as_whisper_cli_error(
        self, tmp_path: Path, mocker
    ) -> None:
        wav = tmp_path / "x.wav"
        model = tmp_path / "m.bin"
        binary = tmp_path / "whisper-cli.exe"
        for p in (wav, model, binary):
            p.touch()
        mocker.patch(
            "asr_local.transcriber.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="whisper-cli", timeout=5),
        )
        with pytest.raises(WhisperCliError, match="timed out"):
            run_whisper(
                wav_path=wav, model_path=model, binary_path=binary, timeout=5
            )


class TestResolveDefaultBinary:
    """Lazy resolution lets the first post-build call pick up the new binary."""

    def test_raises_when_neither_binary_exists(self, tmp_path: Path, mocker) -> None:
        mocker.patch(
            "asr_local.transcriber._NATIVE_ARM64", tmp_path / "never" / "whisper-cli.exe"
        )
        mocker.patch(
            "asr_local.transcriber._PRISM_X64", tmp_path / "nope" / "whisper-cli.exe"
        )
        with pytest.raises(WhisperCliNotFoundError):
            resolve_default_binary()

    def test_prefers_native_over_prism(self, tmp_path: Path, mocker) -> None:
        native = tmp_path / "native-arm64" / "whisper-cli.exe"
        prism = tmp_path / "Release" / "whisper-cli.exe"
        native.parent.mkdir(parents=True)
        prism.parent.mkdir(parents=True)
        native.touch()
        prism.touch()
        mocker.patch("asr_local.transcriber._NATIVE_ARM64", native)
        mocker.patch("asr_local.transcriber._PRISM_X64", prism)
        assert resolve_default_binary() == native

    def test_falls_back_to_prism_when_native_missing(
        self, tmp_path: Path, mocker
    ) -> None:
        native = tmp_path / "native-arm64" / "whisper-cli.exe"  # absent
        prism = tmp_path / "Release" / "whisper-cli.exe"
        prism.parent.mkdir(parents=True)
        prism.touch()
        mocker.patch("asr_local.transcriber._NATIVE_ARM64", native)
        mocker.patch("asr_local.transcriber._PRISM_X64", prism)
        assert resolve_default_binary() == prism
