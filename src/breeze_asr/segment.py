"""Time formatting and timestamped ASR segments."""
from __future__ import annotations

from dataclasses import dataclass


def _validate_seconds(seconds: float) -> None:
    if seconds is None:
        raise TypeError("seconds must be a number, got None")
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")


def format_time_display(seconds: float) -> str:
    """Human-readable HH:MM:SS (integer seconds, truncated)."""
    _validate_seconds(seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True, slots=True)
class TimestampedSegment:
    """One ASR output segment with start/end times in seconds."""

    start_time: float
    end_time: float
    text: str

    def __post_init__(self) -> None:
        _validate_seconds(self.start_time)
        _validate_seconds(self.end_time)
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) precedes start_time ({self.start_time})"
            )

    def to_timestamp_line(self) -> str:
        return (
            f"[{format_time_display(self.start_time)} - "
            f"{format_time_display(self.end_time)}] {self.text}"
        )
