"""Command-line entry point for asr-local."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from .audio import convert_to_16k_mono_wav
from .model import VARIANT_FILES, ensure_ggml
from .transcriber import run_whisper
from .writer import save_transcript


def choose_audio_ctx(duration_s: float) -> int:
    """Pick whisper-cli -ac value based on clip length.

    Short clips don't need the full 30 s encoder context; capping it gives a
    ~3× speedup with zero accuracy loss (measured 75s→26s on a 5.8 s clip).
    For long audio we must keep the full context so per-chunk decoding
    preserves quality.
    """
    if duration_s < 10.0:
        return 512
    if duration_s < 30.0:
        return 1024
    return 0  # full 30 s


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
    parser.add_argument("--threads", type=int, default=8, help="whisper-cli thread count")
    parser.add_argument("--language", default="zh", help="Source language (default: zh)")
    parser.add_argument(
        "--audio-ctx",
        type=int,
        default=None,
        help="Encoder context in mel frames (auto-tuned from duration if unset; 0=full 30 s)",
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

    print(f"[1/4] Converting audio: {args.audio_path.name}")
    wav_path, duration = convert_to_16k_mono_wav(args.audio_path)
    print(f"      → 16 kHz mono WAV, {duration:.1f}s")

    try:
        print(f"[2/4] Preparing model (variant={args.quant})")
        model_path = ensure_ggml(variant=args.quant)
        print(f"      → {model_path.name} ({model_path.stat().st_size / 1e9:.2f} GB)")

        audio_ctx = args.audio_ctx if args.audio_ctx is not None else choose_audio_ctx(duration)
        print(
            f"[3/4] Transcribing with {args.threads} threads "
            f"(language={args.language}, audio_ctx={audio_ctx or 'full'})"
        )
        t0 = time.perf_counter()
        segments = run_whisper(
            wav_path=wav_path,
            model_path=model_path,
            language=args.language,
            threads=args.threads,
            audio_ctx=audio_ctx,
        )
        elapsed = time.perf_counter() - t0
        rtf = elapsed / duration if duration > 0 else float("inf")
        print(
            f"      → {len(segments)} segments in {elapsed:.1f}s "
            f"(RTF {rtf:.2f}× realtime)"
        )

        print(f"[4/4] Writing transcript: {output_path}")
        written = save_transcript(segments, output_path)
        if written is None:
            print("Warning: transcription empty; no output file written.", file=sys.stderr)
            return 2
        print("Done.")
        return 0
    finally:
        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
