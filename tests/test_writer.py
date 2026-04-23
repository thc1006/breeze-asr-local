"""TDD: writer.save_transcript — one timestamp line per segment, UTF-8."""
from __future__ import annotations

from pathlib import Path

import pytest

from asr_local.segment import TimestampedSegment
from asr_local.writer import save_transcript


def _sample_segments() -> list[TimestampedSegment]:
    return [
        TimestampedSegment(0.0, 2.5, "你好,歡迎收聽本節目。"),
        TimestampedSegment(2.5, 5.12, "今天我們要討論的主題是台灣華語。"),
    ]


class TestSaveTranscript:
    def test_returns_none_for_empty_segments(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.txt"
        assert save_transcript([], out) is None
        assert not out.exists()

    def test_writes_one_line_per_segment(self, tmp_path: Path) -> None:
        out = tmp_path / "t.txt"
        save_transcript(_sample_segments(), out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_line_format_matches_contract(self, tmp_path: Path) -> None:
        out = tmp_path / "t.txt"
        save_transcript(_sample_segments(), out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "[00:00:00 - 00:00:02] 你好,歡迎收聽本節目。"
        assert lines[1] == "[00:00:02 - 00:00:05] 今天我們要討論的主題是台灣華語。"

    def test_utf8_chinese_preserved_roundtrip(self, tmp_path: Path) -> None:
        out = tmp_path / "cjk.txt"
        segs = [TimestampedSegment(0, 1, "台灣繁體中文 · 測試 — 混英 mixed 123")]
        save_transcript(segs, out)
        assert "台灣繁體中文" in out.read_text(encoding="utf-8")
        assert "mixed 123" in out.read_text(encoding="utf-8")

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "t.txt"
        save_transcript(_sample_segments(), out)
        assert out.exists()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        out = tmp_path / "t.txt"
        out.write_text("stale", encoding="utf-8")
        save_transcript(_sample_segments(), out)
        assert "stale" not in out.read_text(encoding="utf-8")

    def test_returns_path_on_success(self, tmp_path: Path) -> None:
        out = tmp_path / "t.txt"
        result = save_transcript(_sample_segments(), out)
        assert result == out
