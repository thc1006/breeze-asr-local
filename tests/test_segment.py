"""TDD: time formatting helper and TimestampedSegment dataclass."""
import dataclasses

import pytest

from asr_local.segment import TimestampedSegment, format_time_display


class TestFormatTimeDisplay:
    def test_zero(self):
        assert format_time_display(0.0) == "00:00:00"

    def test_one_minute(self):
        assert format_time_display(60.0) == "00:01:00"

    def test_one_hour(self):
        assert format_time_display(3600.0) == "01:00:00"

    def test_mixed_hours_minutes_seconds(self):
        assert format_time_display(3723.0) == "01:02:03"

    def test_fractional_seconds_truncated(self):
        assert format_time_display(5.9) == "00:00:05"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            format_time_display(-1.0)

    def test_none_raises(self):
        with pytest.raises(TypeError):
            format_time_display(None)  # type: ignore[arg-type]


class TestTimestampedSegmentConstruction:
    def test_basic(self):
        s = TimestampedSegment(0.0, 1.5, "hello")
        assert s.start_time == 0.0
        assert s.end_time == 1.5
        assert s.text == "hello"

    def test_frozen(self):
        s = TimestampedSegment(0.0, 1.0, "a")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.text = "b"  # type: ignore[misc]

    def test_negative_start_raises(self):
        with pytest.raises(ValueError):
            TimestampedSegment(-0.5, 1.0, "x")

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError):
            TimestampedSegment(2.0, 1.0, "x")

    def test_equal_start_end_allowed(self):
        s = TimestampedSegment(1.0, 1.0, "pause")
        assert s.start_time == s.end_time

    def test_empty_text_allowed(self):
        s = TimestampedSegment(0.0, 1.0, "")
        assert s.text == ""


class TestTimestampedSegmentRendering:
    def test_to_timestamp_line_chinese_preserved(self):
        s = TimestampedSegment(0.0, 65.0, "你好台灣")
        assert s.to_timestamp_line() == "[00:00:00 - 00:01:05] 你好台灣"

    def test_to_timestamp_line_format(self):
        s = TimestampedSegment(125.0, 130.0, "word")
        assert s.to_timestamp_line() == "[00:02:05 - 00:02:10] word"
