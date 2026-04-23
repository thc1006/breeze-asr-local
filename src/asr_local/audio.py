"""Audio format normalization: any supported format -> 16 kHz mono PCM_S16LE WAV."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


class AudioConversionError(RuntimeError):
    """ffmpeg could not read or transcode the input audio."""


def convert_to_16k_mono_wav(src_path: Path | str) -> tuple[Path, float]:
    """Convert `src_path` to a temporary 16 kHz mono PCM_S16LE WAV.

    Returns (wav_path, duration_seconds). Caller is responsible for deleting
    the returned path when done.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"audio source not found: {src}")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    fd, out_path_str = tempfile.mkstemp(suffix=".wav", prefix="asr_local_")
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

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise AudioConversionError(f"ffmpeg failed for {src}: {stderr or '(no stderr)'}")

    try:
        duration = sf.info(str(out_path)).duration
    except Exception as e:
        out_path.unlink(missing_ok=True)
        raise AudioConversionError(f"cannot read converted WAV: {e}") from e

    return out_path, duration
