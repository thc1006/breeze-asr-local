"""Breeze-ASR-25 GGML model acquisition + validation.

Source repo: alan314159/Breeze-ASR-25-whispercpp (HF).
All four variants share identical byte counts across the community GGML repos
(danielkao0421/lsheep/talkyon), so the table below is authoritative.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

from huggingface_hub import hf_hub_download

REPO_ID: Final[str] = "alan314159/Breeze-ASR-25-whispercpp"
# q4_0 isn't published by alan314159 but lsheep hosts it and hits the ARM
# KleidiAI + NEON sdot fast path on Snapdragon X (measured ~1.4-1.7x vs q8_0
# with identical transcript on Mandarin golden samples).
REPO_FOR_Q4_0: Final[str] = "lsheep/Breeze-ASR-25-ggml"

VARIANT_FILES: Final[dict[str, str]] = {
    "q8_0": "ggml-model-q8_0.bin",
    "fp16": "ggml-model.bin",
    "q5_k": "ggml-model-q5_k.bin",
    "q4_k": "ggml-model-q4_k.bin",
    "q4_0": "ggml-model-q4_0.bin",
}

EXPECTED_SIZES: Final[dict[str, int]] = {
    "q8_0": 1_656_129_708,
    "fp16": 3_094_623_708,
    "q5_k": 1_080_732_108,
    "q4_k": 888_932_908,
    "q4_0": 888_932_908,
}


def _repo_for_variant(variant: str) -> str:
    """Pick the HF repo a given variant lives in (most share one, q4_0 differs)."""
    if variant == "q4_0":
        return REPO_FOR_Q4_0
    return REPO_ID

# Accepted leading magics across GGML format generations.
# whisper.cpp legacy format stores uint32 0x67676d6c little-endian → b"lmgg".
# Some dumps show b"ggml" (big-endian storage). GGUF is the newer unified container.
_KNOWN_MAGICS: Final[frozenset[bytes]] = frozenset({b"lmgg", b"ggml", b"GGUF", b"GGJT"})


class GgmlValidationError(RuntimeError):
    """The downloaded file does not look like a valid whisper.cpp GGML model."""


def validate_ggml_magic(path: Path | str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GGML file not found: {p}")
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic not in _KNOWN_MAGICS:
        raise GgmlValidationError(
            f"unexpected magic bytes at start of {p.name}: {magic!r} "
            f"(expected one of {sorted(m.decode('latin1') for m in _KNOWN_MAGICS)})"
        )


def validate_ggml_size(path: Path | str, expected: int) -> None:
    p = Path(path)
    actual = p.stat().st_size
    if actual != expected:
        raise GgmlValidationError(
            f"size mismatch for {p.name}: expected {expected:,} bytes, got {actual:,}"
        )


def ensure_ggml(variant: str = "q8_0", cache_dir: Path | str | None = None) -> Path:
    """Download (if needed) and validate the requested GGML variant.

    Returns the local Path to the model file. On cache hit, returns fast.
    """
    if variant not in VARIANT_FILES:
        raise ValueError(
            f"unknown variant {variant!r}, available: {sorted(VARIANT_FILES)}"
        )

    local_path_str = hf_hub_download(
        repo_id=_repo_for_variant(variant),
        filename=VARIANT_FILES[variant],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    path = Path(local_path_str)
    validate_ggml_magic(path)
    validate_ggml_size(path, EXPECTED_SIZES[variant])
    return path
