"""Write timestamped segments to a UTF-8 plain-text transcript."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .segment import TimestampedSegment


def save_transcript(
    segments: Iterable[TimestampedSegment],
    output_path: Path | str,
) -> Path | None:
    """Write one `[HH:MM:SS - HH:MM:SS] text` line per segment.

    Returns the output Path, or None if `segments` is empty (no file created).
    """
    segments = list(segments)
    if not segments:
        return None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.to_timestamp_line() + "\n")
    return out
