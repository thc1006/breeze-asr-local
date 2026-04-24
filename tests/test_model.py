"""TDD: model.ensure_ggml (download + validate Breeze-ASR-25 GGML)."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from breeze_asr.model import (
    EXPECTED_SIZES,
    VARIANT_FILES,
    GgmlValidationError,
    ensure_ggml,
    validate_ggml_magic,
    validate_ggml_size,
)


def _write_header(path: Path, magic: bytes, body_size: int = 0) -> Path:
    """Create a small fake file with a given leading magic."""
    path.write_bytes(magic + b"\x00" * body_size)
    return path


class TestValidateMagic:
    @pytest.mark.parametrize("magic", [b"lmgg", b"ggml", b"GGUF", b"GGJT"])
    def test_accepts_known_magics(self, tmp_path: Path, magic: bytes) -> None:
        p = _write_header(tmp_path / "m.bin", magic)
        validate_ggml_magic(p)  # must not raise

    def test_rejects_html_error_page(self, tmp_path: Path) -> None:
        p = _write_header(tmp_path / "m.bin", b"<!DO")
        with pytest.raises(GgmlValidationError):
            validate_ggml_magic(p)

    def test_rejects_zero_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        with pytest.raises(GgmlValidationError):
            validate_ggml_magic(p)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_ggml_magic(tmp_path / "nope.bin")


class TestValidateSize:
    def test_exact_match_passes(self, tmp_path: Path) -> None:
        p = tmp_path / "m.bin"
        p.write_bytes(b"x" * 42)
        validate_ggml_size(p, expected=42)

    def test_size_mismatch_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "m.bin"
        p.write_bytes(b"x" * 10)
        with pytest.raises(GgmlValidationError):
            validate_ggml_size(p, expected=42)


class TestVariantTable:
    def test_q8_0_is_default_preferred(self) -> None:
        assert "q8_0" in VARIANT_FILES
        assert VARIANT_FILES["q8_0"] == "ggml-model-q8_0.bin"

    def test_fp16_fallback_present(self) -> None:
        assert "fp16" in VARIANT_FILES
        assert VARIANT_FILES["fp16"] == "ggml-model.bin"

    def test_q4_0_variant_present(self) -> None:
        # KleidiAI-fast path: measured 1.4-1.7x vs q8_0 with identical
        # transcript on the Breeze-ASR-25 golden sample.
        assert "q4_0" in VARIANT_FILES
        assert VARIANT_FILES["q4_0"] == "ggml-model-q4_0.bin"

    def test_every_variant_has_expected_size(self) -> None:
        for v in VARIANT_FILES:
            assert v in EXPECTED_SIZES
            assert EXPECTED_SIZES[v] > 500_000_000  # all > 500 MB


class TestEnsureGgmlOrchestration:
    """Unit-level: mock hf_hub_download so we don't fetch 1.6 GB."""

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError):
            ensure_ggml(variant="nonsense")

    def test_delegates_to_hf_hub_download(
        self, tmp_path: Path, mocker
    ) -> None:
        fake = tmp_path / "fake-q8_0.bin"
        fake.write_bytes(b"lmgg" + b"\x00" * (EXPECTED_SIZES["q8_0"] - 4))
        mock_dl = mocker.patch(
            "breeze_asr.model.hf_hub_download", return_value=str(fake)
        )
        result = ensure_ggml(variant="q8_0")
        assert result == fake
        mock_dl.assert_called_once()
        args, kwargs = mock_dl.call_args
        assert kwargs.get("repo_id") == "alan314159/Breeze-ASR-25-whispercpp"
        assert kwargs.get("filename") == "ggml-model-q8_0.bin"

    def test_q4_0_downloads_from_lsheep_repo(
        self, tmp_path: Path, mocker
    ) -> None:
        fake = tmp_path / "fake-q4_0.bin"
        fake.write_bytes(b"lmgg" + b"\x00" * (EXPECTED_SIZES["q4_0"] - 4))
        mock_dl = mocker.patch(
            "breeze_asr.model.hf_hub_download", return_value=str(fake)
        )
        ensure_ggml(variant="q4_0")
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["repo_id"] == "lsheep/Breeze-ASR-25-ggml"
        assert kwargs["filename"] == "ggml-model-q4_0.bin"

    def test_invalid_magic_after_download_raises(
        self, tmp_path: Path, mocker
    ) -> None:
        fake = tmp_path / "fake.bin"
        # right size but wrong magic — something went wrong mid-download
        fake.write_bytes(b"<htm" + b"\x00" * (EXPECTED_SIZES["q8_0"] - 4))
        mocker.patch("breeze_asr.model.hf_hub_download", return_value=str(fake))
        with pytest.raises(GgmlValidationError):
            ensure_ggml(variant="q8_0")

    def test_size_mismatch_after_download_raises(
        self, tmp_path: Path, mocker
    ) -> None:
        fake = tmp_path / "fake.bin"
        # correct magic but truncated
        fake.write_bytes(b"lmgg" + b"\x00" * 1000)
        mocker.patch("breeze_asr.model.hf_hub_download", return_value=str(fake))
        with pytest.raises(GgmlValidationError):
            ensure_ggml(variant="q8_0")
