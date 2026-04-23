"""whisper.cpp CLI wrapper: audio + model → list[TimestampedSegment]."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .segment import TimestampedSegment

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_ARM64 = _PROJECT_ROOT / "bin" / "native-arm64" / "whisper-cli.exe"
_PRISM_X64 = _PROJECT_ROOT / "bin" / "Release" / "whisper-cli.exe"
# Prefer native ARM64 build (NEON + dotprod + i8mm — ~10× faster than x64-Prism
# for large-v2 Q8_0 on Snapdragon X). Fall back to the prebuilt x64 binary
# under Windows Prism emulation if native wasn't compiled.
DEFAULT_WHISPER_CLI = _NATIVE_ARM64 if _NATIVE_ARM64.exists() else _PRISM_X64


class WhisperCliError(RuntimeError):
    """whisper-cli exited non-zero, produced malformed output, or similar."""


class WhisperCliNotFoundError(WhisperCliError):
    """whisper-cli binary could not be located at the given path."""


def _build_command(
    *,
    binary_path: Path,
    wav_path: Path,
    model_path: Path,
    language: str,
    threads: int,
    output_prefix: Path,
    audio_ctx: int = 0,
) -> list[str]:
    cmd = [
        str(binary_path),
        "-m", str(model_path),
        "-f", str(wav_path),
        "-l", language,
        "-t", str(threads),
        "-oj",
        "-of", str(output_prefix),
        "-np",
    ]
    if audio_ctx > 0:
        cmd += ["-ac", str(audio_ctx)]
    return cmd


def _parse_json_output(json_path: Path) -> list[TimestampedSegment]:
    try:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise WhisperCliError(f"cannot parse JSON from {json_path}: {e}") from e

    transcription = raw.get("transcription", []) or []
    segments: list[TimestampedSegment] = []
    for chunk in transcription:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        offsets = chunk.get("offsets") or {}
        start_ms = offsets.get("from", 0) or 0
        end_ms = offsets.get("to", start_ms) or start_ms
        if end_ms < start_ms:
            end_ms = start_ms
        segments.append(
            TimestampedSegment(
                start_time=start_ms / 1000.0,
                end_time=end_ms / 1000.0,
                text=text,
            )
        )
    return segments


def run_whisper(
    *,
    wav_path: Path,
    model_path: Path,
    language: str = "zh",
    threads: int = 8,
    audio_ctx: int = 0,
    binary_path: Path | None = None,
    timeout: float | None = None,
) -> list[TimestampedSegment]:
    """Transcribe `wav_path` with `model_path` via whisper-cli subprocess.

    `audio_ctx` caps the encoder attention window in mel frames (50 Hz).
    0 = full 30 s context (default, required for long audio >=30 s).
    512 = ~10 s context (3× faster on short clips, safe if clip <10 s).
    """
    binary = Path(binary_path) if binary_path else DEFAULT_WHISPER_CLI
    if not binary.exists():
        raise WhisperCliNotFoundError(f"whisper-cli binary not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="asr_local_out_") as td:
        out_prefix = Path(td) / "transcribe"
        cmd = _build_command(
            binary_path=binary,
            wav_path=Path(wav_path),
            model_path=Path(model_path),
            language=language,
            threads=threads,
            output_prefix=out_prefix,
            audio_ctx=audio_ctx,
        )
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            raise WhisperCliError(
                f"whisper-cli exited {result.returncode}: {stderr or '(no stderr)'}"
            )

        json_path = out_prefix.with_suffix(".json")
        if not json_path.exists():
            raise WhisperCliError(
                f"whisper-cli succeeded but JSON output missing at {json_path}"
            )
        return _parse_json_output(json_path)
