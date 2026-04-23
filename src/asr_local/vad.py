"""Silero Voice Activity Detection model acquisition for whisper.cpp.

Source: https://huggingface.co/ggml-org/whisper-vad (GGML-converted silero VAD
weights). whisper.cpp consumes these via `--vad --vad-model <path>` and uses
them to skip silent regions of the audio, typically halving wall-clock time
on meeting / podcast content without degrading CER.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

from huggingface_hub import hf_hub_download

from .model import GgmlValidationError, validate_ggml_magic, validate_ggml_size

VAD_REPO_ID: Final[str] = "ggml-org/whisper-vad"
DEFAULT_VAD_FILENAME: Final[str] = "ggml-silero-v6.2.0.bin"

# silero v5.1.2 and v6.2.0 are both GGML-converted to the same byte count.
VAD_EXPECTED_SIZES: Final[dict[str, int]] = {
    "ggml-silero-v5.1.2.bin": 885_098,
    "ggml-silero-v6.2.0.bin": 885_098,
}


def ensure_vad_model(
    filename: str = DEFAULT_VAD_FILENAME,
    cache_dir: Path | str | None = None,
) -> Path:
    """Download (if needed) and validate a Silero VAD GGML file.

    Raises GgmlValidationError if the downloaded file has an unexpected
    magic or, for a known filename, an unexpected size.
    """
    local_path_str = hf_hub_download(
        repo_id=VAD_REPO_ID,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    path = Path(local_path_str)
    validate_ggml_magic(path)
    if filename in VAD_EXPECTED_SIZES:
        validate_ggml_size(path, VAD_EXPECTED_SIZES[filename])
    return path
