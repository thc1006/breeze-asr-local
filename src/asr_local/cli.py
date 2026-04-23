"""Command-line entry point for asr-local."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from .audio import AudioConversionError, convert_to_16k_mono_wav
from .model import VARIANT_FILES, ensure_ggml
from .transcriber import WhisperCliError, run_whisper
from .writer import save_transcript

# whisper mel spectrogram runs at hop_length=160 over 16 kHz input => 100 fps.
# Full 30 s context = 3000 mel frames. `-ac N` caps the encoder's attention
# window at N frames; any audio past N/100 seconds is silently truncated, so
# the chosen value must cover the entire clip.
_MEL_FPS = 100
_FULL_CTX_FRAMES = 3000
_SAFETY_PAD_FRAMES = 64


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
    needed = int(duration_s * _MEL_FPS) + _SAFETY_PAD_FRAMES
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
    wav_path: Path | None = None

    try:
        print(f"[1/4] Converting audio: {args.audio_path.name}")
        wav_path, duration = convert_to_16k_mono_wav(args.audio_path)
        print(f"      -> 16 kHz mono WAV, {duration:.1f}s")

        print(f"[2/4] Preparing model (variant={args.quant})")
        model_path = ensure_ggml(variant=args.quant)
        print(f"      -> {model_path.name} ({model_path.stat().st_size / 1e9:.2f} GB)")

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
    finally:
        if wav_path is not None and wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
