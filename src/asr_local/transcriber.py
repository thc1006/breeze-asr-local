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

# Windows process priority constants. `subprocess.HIGH_PRIORITY_CLASS` is
# available on Windows Python but referencing it by name is fragile on
# non-Windows test hosts, so we hardcode the values for portability. These
# are no-ops when the OS doesn't honor them.
_PRIORITY_FLAGS: dict[str, int] = {
    "normal": 0,
    "high": 0x00000080,  # HIGH_PRIORITY_CLASS
}


class WhisperCliError(RuntimeError):
    """whisper-cli exited non-zero, produced malformed output, or similar."""


class WhisperCliNotFoundError(WhisperCliError):
    """whisper-cli binary could not be located at the given path."""


def resolve_default_binary() -> Path:
    """Return the best whisper-cli binary available locally.

    Prefers the native ARM64 build (produced by scripts/setup.ps1, NEON +
    dotprod + i8mm, ~10x faster on Snapdragon X). Falls back to a prebuilt
    x64 binary at bin/Release/ running under Windows Prism emulation if the
    native build is absent. Raises WhisperCliNotFoundError if neither exists.

    Resolved lazily so that builds completed after module import are picked
    up on the next call.
    """
    if _NATIVE_ARM64.exists():
        return _NATIVE_ARM64
    if _PRISM_X64.exists():
        return _PRISM_X64
    raise WhisperCliNotFoundError(
        "No whisper-cli binary found. Run 'scripts\\setup.ps1' to build the "
        f"native ARM64 binary at {_NATIVE_ARM64}, or unzip whisper-bin-x64.zip "
        f"from the whisper.cpp release at {_PRISM_X64.parent} for Prism fallback."
    )


def _build_command(
    *,
    binary_path: Path,
    wav_path: Path,
    model_path: Path,
    language: str,
    threads: int,
    output_prefix: Path,
    audio_ctx: int = 0,
    flash_attn: bool = True,
    processors: int = 1,
    vad_model_path: Path | None = None,
    greedy: bool = False,
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
    if not flash_attn:
        cmd += ["-nfa"]
    if processors > 1:
        cmd += ["-p", str(processors)]
    if vad_model_path is not None:
        cmd += ["--vad", "-vm", str(vad_model_path)]
    if greedy:
        # Disable beam search: beam=1, best_of=1 (~15-20% faster on long
        # Mandarin audio; empirically identical transcripts on Breeze-ASR-25
        # Q8_0 golden samples. Code-switched zh-en audio untested — keep beam
        # as the default.
        cmd += ["-bs", "1", "-bo", "1"]
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
    processors: int = 1,
    audio_ctx: int = 0,
    flash_attn: bool = True,
    vad_model_path: Path | None = None,
    greedy: bool = False,
    priority: str = "normal",
    binary_path: Path | None = None,
    timeout: float | None = None,
) -> list[TimestampedSegment]:
    """Transcribe `wav_path` with `model_path` via whisper-cli subprocess.

    `audio_ctx` caps the encoder attention window in mel frames. Whisper's
    mel spectrogram runs at 100 fps (hop_length=160 @ 16 kHz), so each
    frame covers 0.01 s of audio.
      0    = full 30 s context (required when clip length >= ~29 s).
      3000 = same as 0 (whisper's native ceiling).
      640  = 6.4 s context (safe for clips up to 6.4 s).
    Any audio past `audio_ctx / 100` seconds is silently truncated.
    See `asr_local.cli.choose_audio_ctx` for a safe auto-tuner.
    """
    if binary_path is not None:
        binary = Path(binary_path)
        if not binary.exists():
            raise WhisperCliNotFoundError(f"whisper-cli binary not found: {binary}")
    else:
        binary = resolve_default_binary()

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
            flash_attn=flash_attn,
            processors=processors,
            vad_model_path=vad_model_path,
            greedy=greedy,
        )
        flags = _PRIORITY_FLAGS.get(priority, 0)
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=timeout, creationflags=flags
            )
        except subprocess.TimeoutExpired as e:
            raise WhisperCliError(
                f"whisper-cli timed out after {timeout}s"
            ) from e

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
