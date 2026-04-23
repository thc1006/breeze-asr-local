"""TDD: vad.ensure_vad_model — download + validate silero VAD GGML."""
from __future__ import annotations

from pathlib import Path

import pytest

from asr_local.model import GgmlValidationError
from asr_local.vad import (
    DEFAULT_VAD_FILENAME,
    VAD_EXPECTED_SIZES,
    VAD_REPO_ID,
    ensure_vad_model,
)


class TestVadConstants:
    def test_repo_is_whisper_vad(self) -> None:
        assert VAD_REPO_ID == "ggml-org/whisper-vad"

    def test_default_is_v6(self) -> None:
        # v6.2.0 is the newer silero model; keep it as default for new installs.
        assert DEFAULT_VAD_FILENAME == "ggml-silero-v6.2.0.bin"

    def test_both_known_sizes_are_identical(self) -> None:
        # Both silero variants ship at 885,098 bytes on ggml-org/whisper-vad.
        v5 = VAD_EXPECTED_SIZES["ggml-silero-v5.1.2.bin"]
        v6 = VAD_EXPECTED_SIZES["ggml-silero-v6.2.0.bin"]
        assert v5 == v6 == 885_098


class TestEnsureVadModel:
    """Mock hf_hub_download; we only verify orchestration + validation."""

    def test_hf_hub_download_called_with_repo_and_filename(
        self, tmp_path: Path, mocker
    ) -> None:
        fake = tmp_path / DEFAULT_VAD_FILENAME
        fake.write_bytes(b"lmgg" + b"\x00" * (VAD_EXPECTED_SIZES[DEFAULT_VAD_FILENAME] - 4))
        dl = mocker.patch("asr_local.vad.hf_hub_download", return_value=str(fake))
        result = ensure_vad_model()
        assert result == fake
        kwargs = dl.call_args.kwargs
        assert kwargs["repo_id"] == VAD_REPO_ID
        assert kwargs["filename"] == DEFAULT_VAD_FILENAME

    def test_explicit_v5_filename(self, tmp_path: Path, mocker) -> None:
        fake = tmp_path / "ggml-silero-v5.1.2.bin"
        fake.write_bytes(b"lmgg" + b"\x00" * (VAD_EXPECTED_SIZES["ggml-silero-v5.1.2.bin"] - 4))
        dl = mocker.patch("asr_local.vad.hf_hub_download", return_value=str(fake))
        ensure_vad_model(filename="ggml-silero-v5.1.2.bin")
        assert dl.call_args.kwargs["filename"] == "ggml-silero-v5.1.2.bin"

    def test_wrong_magic_raises(self, tmp_path: Path, mocker) -> None:
        fake = tmp_path / "vad.bin"
        fake.write_bytes(b"<htm" + b"\x00" * (VAD_EXPECTED_SIZES[DEFAULT_VAD_FILENAME] - 4))
        mocker.patch("asr_local.vad.hf_hub_download", return_value=str(fake))
        with pytest.raises(GgmlValidationError):
            ensure_vad_model()

    def test_wrong_size_raises_for_known_filename(
        self, tmp_path: Path, mocker
    ) -> None:
        fake = tmp_path / DEFAULT_VAD_FILENAME
        fake.write_bytes(b"lmgg" + b"\x00" * 1000)  # truncated
        mocker.patch("asr_local.vad.hf_hub_download", return_value=str(fake))
        with pytest.raises(GgmlValidationError):
            ensure_vad_model()

    def test_unknown_filename_skips_size_check(
        self, tmp_path: Path, mocker
    ) -> None:
        # A user-supplied custom VAD file of unknown size should pass magic-only.
        fake = tmp_path / "my-custom-vad.bin"
        fake.write_bytes(b"lmgg" + b"\x00" * 42)
        mocker.patch("asr_local.vad.hf_hub_download", return_value=str(fake))
        result = ensure_vad_model(filename="my-custom-vad.bin")
        assert result == fake
