"""Audio format normalization: any supported format -> 16 kHz mono PCM_S16LE WAV."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"


class AudioConversionError(RuntimeError):
    """ffmpeg could not read or transcode the input audio."""


def _is_target_format(path: Path) -> bool:
    """Fast-path probe: does `path` already match our target format?

    Returns True for 16 kHz mono PCM_S16LE WAV — no ffmpeg re-encode needed.
    Reads only the RIFF header via soundfile (microseconds); returns False
    for any read error so the caller falls back to the ffmpeg path.
    """
    if path.suffix.lower() != ".wav":
        return False
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return (
        info.samplerate == TARGET_SAMPLE_RATE
        and info.channels == TARGET_CHANNELS
        and info.subtype == TARGET_SUBTYPE
    )


def convert_to_16k_mono_wav(src_path: Path | str) -> tuple[Path, float, bool]:
    """Convert `src_path` to a 16 kHz mono PCM_S16LE WAV.

    Returns `(wav_path, duration_seconds, owns_tempfile)`. If the input is
    already target format (`_is_target_format`), returns `(src, duration, False)`
    as a passthrough — caller must NOT delete the returned path. Otherwise
    creates a tempfile via ffmpeg and returns `(tmp, duration, True)` — caller
    is responsible for deleting when done.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"audio source not found: {src}")

    # Fast path: skip ffmpeg entirely when the input is already target format.
    if _is_target_format(src):
        return src, sf.info(str(src)).duration, False

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    fd, out_path_str = tempfile.mkstemp(suffix=".wav", prefix="breeze_asr_")
    os.close(fd)
    out_path = Path(out_path_str)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(src),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-loglevel", "error",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        out_path.unlink(missing_ok=True)
        raise AudioConversionError(f"ffmpeg timed out after 300s on {src}") from e

    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise AudioConversionError(f"ffmpeg failed for {src}: {stderr or '(no stderr)'}")

    try:
        duration = sf.info(str(out_path)).duration
    except Exception as e:
        out_path.unlink(missing_ok=True)
        raise AudioConversionError(f"cannot read converted WAV: {e}") from e

    return out_path, duration, True
