"""Local Taiwanese Mandarin ASR using whisper.cpp + Breeze-ASR-25."""
from .audio import AudioConversionError, convert_to_16k_mono_wav
from .model import GgmlValidationError, ensure_ggml
from .segment import TimestampedSegment, format_time_display
from .transcriber import WhisperCliError, WhisperCliNotFoundError, run_whisper
from .writer import save_transcript

__version__ = "0.1.0"
__all__ = [
    "AudioConversionError",
    "GgmlValidationError",
    "TimestampedSegment",
    "WhisperCliError",
    "WhisperCliNotFoundError",
    "convert_to_16k_mono_wav",
    "ensure_ggml",
    "format_time_display",
    "run_whisper",
    "save_transcript",
]
