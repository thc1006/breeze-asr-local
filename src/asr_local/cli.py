"""Command-line entry point for asr-local."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from .audio import AudioConversionError, convert_to_16k_mono_wav
from .model import GgmlValidationError, VARIANT_FILES, ensure_ggml
from .transcriber import WhisperCliError, run_whisper
from .vad import ensure_vad_model
from .writer import save_transcript

# whisper mel spectrogram runs at hop_length=160 over 16 kHz input => 100 fps.
# Full 30 s context = 3000 mel frames. `-ac N` caps the encoder's attention
# window at N frames; any audio past N/100 seconds is silently truncated, so
# the chosen value must cover the entire clip.
_MEL_FPS = 100
_FULL_CTX_FRAMES = 3000
_SAFETY_PAD_FRAMES = 64

# whisper.cpp's flash-attention kernel (`-fa` default ON in v1.8.4) loses
# output quality on CJK scripts (whisper.cpp issue #3020). For Breeze-ASR-25
# Q8_0 on Mandarin this is also ~15% *slower* than the non-FA path on
# Snapdragon X, so disabling it is a pure win.
_CJK_LANGUAGES = frozenset({"zh", "ja", "ko"})


def choose_flash_attn(language: str) -> bool:
    """Whether whisper.cpp should use flash attention for this language.

    Returns False for CJK (zh / ja / ko) where flash-attn degrades accuracy.
    Returns True otherwise, matching whisper.cpp's own default.
    """
    return language.lower() not in _CJK_LANGUAGES


_PARALLEL_MIN_DURATION_S = 30.0

# Silero VAD adds ~1 ms/30 s overhead and catches silence that otherwise burns
# encoder compute. For short clips there is rarely enough silence to matter,
# but for anything >= 30 s the net is almost always a win.
_VAD_AUTO_MIN_DURATION_S = 30.0


def choose_vad(duration_s: float) -> bool:
    """Auto-enable VAD for audio long enough to contain meaningful silence."""
    return duration_s >= _VAD_AUTO_MIN_DURATION_S


def choose_processors(duration_s: float, cpu_count: int) -> tuple[int, int]:
    """Pick (`-p N`, `-t M`) given clip duration and available cores.

    For audio under 30 s, chunk-parallel adds overhead without payoff —
    use a single processor with all threads. For longer audio, 2 parallel
    decoders each using half the cores typically beat a single decoder
    using all cores by ~25% on 8-core Oryon (measured). The invariant
    `p * t <= cpu_count` is always maintained so we don't oversubscribe.
    """
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if cpu_count < 1:
        cpu_count = 1
    if duration_s < _PARALLEL_MIN_DURATION_S or cpu_count < 4:
        return (1, cpu_count)
    # Longer audio + >=4 cores: split into two parallel decoders.
    threads = max(1, cpu_count // 2)
    return (2, threads)


def choose_audio_ctx(duration_s: float) -> int:
    """Pick whisper-cli -ac for a clip of `duration_s` seconds.

    Caps the encoder's mel attention window to save compute on short clips,
    but always large enough to cover the full audio (plus a small safety pad).
    Returns 0 (full 30 s context) when the clip is long enough that capping
    would save nothing. Returned values are rounded up to a multiple of 64
    for SIMD alignment.
    """
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    needed = math.ceil(duration_s * _MEL_FPS) + _SAFETY_PAD_FRAMES
    if needed >= _FULL_CTX_FRAMES:
        return 0
    aligned = ((needed + 63) // 64) * 64
    return min(_FULL_CTX_FRAMES, aligned)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="asr-local",
        description="Local Taiwanese Mandarin ASR (MediaTek Breeze-ASR-25 via whisper.cpp).",
    )
    parser.add_argument("audio_path", type=Path, help="Input audio file (wav/mp3/ogg/flac/m4a/...)")
    parser.add_argument(
        "--quant",
        choices=sorted(VARIANT_FILES),
        default="q8_0",
        help="GGML quantization variant (default: q8_0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Per-processor thread count (auto-tuned from duration if unset)",
    )
    parser.add_argument(
        "--processors",
        type=int,
        default=None,
        help="Parallel chunk processors -p (auto: 1 for <30s, 2 for >=30s)",
    )
    parser.add_argument("--language", default="zh", help="Source language (default: zh)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Abort if whisper-cli exceeds N seconds (default: no timeout)",
    )
    parser.add_argument(
        "--audio-ctx",
        type=int,
        default=None,
        help="Encoder context in mel frames (auto-tuned from duration if unset; 0=full 30 s)",
    )
    parser.add_argument(
        "--flash-attn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force flash-attention on/off (auto: off for CJK, on otherwise)",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force Silero VAD on/off (auto: on for audio >= 30 s)",
    )
    parser.add_argument(
        "--priority",
        choices=("normal", "high"),
        default="normal",
        help="Set whisper-cli process priority (Windows). 'high' gains 1-3%% under load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Transcript output path (default: <audio>.transcript.txt next to input)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.audio_path.exists():
        print(f"Error: audio file not found: {args.audio_path}", file=sys.stderr)
        return 1

    output_path = args.output or args.audio_path.with_suffix(".transcript.txt")
    wav_path: Path | None = None

    try:
        print(f"[1/4] Converting audio: {args.audio_path.name}")
        wav_path, duration = convert_to_16k_mono_wav(args.audio_path)
        print(f"      -> 16 kHz mono WAV, {duration:.1f}s")

        print(f"[2/4] Preparing model (variant={args.quant})")
        model_path = ensure_ggml(variant=args.quant)
        print(f"      -> {model_path.name} ({model_path.stat().st_size / 1e9:.2f} GB)")

        audio_ctx = args.audio_ctx if args.audio_ctx is not None else choose_audio_ctx(duration)
        flash_attn = (
            args.flash_attn if args.flash_attn is not None else choose_flash_attn(args.language)
        )
        cpu_count = os.cpu_count() or 8
        auto_p, auto_t = choose_processors(duration, cpu_count)
        processors = args.processors if args.processors is not None else auto_p
        threads = args.threads if args.threads is not None else auto_t
        use_vad = args.vad if args.vad is not None else choose_vad(duration)
        vad_model_path = ensure_vad_model() if use_vad else None
        print(
            f"[3/4] Transcribing (-p {processors} -t {threads}, "
            f"language={args.language}, audio_ctx={audio_ctx or 'full'}, "
            f"flash_attn={'on' if flash_attn else 'off'}, "
            f"vad={'on' if use_vad else 'off'})"
        )
        t0 = time.perf_counter()
        segments = run_whisper(
            wav_path=wav_path,
            model_path=model_path,
            language=args.language,
            threads=threads,
            processors=processors,
            audio_ctx=audio_ctx,
            flash_attn=flash_attn,
            vad_model_path=vad_model_path,
            priority=args.priority,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - t0
        rtf = elapsed / duration if duration > 0 else float("inf")
        print(
            f"      -> {len(segments)} segments in {elapsed:.1f}s "
            f"(RTF {rtf:.2f}x realtime)"
        )

        print(f"[4/4] Writing transcript: {output_path}")
        written = save_transcript(segments, output_path)
        if written is None:
            print("Warning: transcription empty; no output file written.", file=sys.stderr)
            return 2
        print("Done.")
        return 0
    except AudioConversionError as e:
        print(f"Error: audio conversion failed - {e}", file=sys.stderr)
        return 3
    except WhisperCliError as e:
        print(f"Error: whisper-cli failed - {e}", file=sys.stderr)
        return 4
    except GgmlValidationError as e:
        print(f"Error: model validation failed - {e}", file=sys.stderr)
        return 5
    except OSError as e:
        # HF download network errors (ConnectionError/TimeoutError) + generic IO.
        print(f"Error: model download or IO failed - {e}", file=sys.stderr)
        return 6
    finally:
        if wav_path is not None and wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
