# AGENTS.md

This file is an instruction manual for AI coding agents (Claude Code, Codex,
Cursor, Aider, Windsurf) working in this repository. It follows the
[AGENTS.md standard](https://agents.md/) maintained by the Linux Foundation
Agentic AI Foundation.

## Project summary

`breeze-asr-local` is a Python CLI that runs MediaTek's Breeze-ASR-25 (a
Whisper-large-v2 fine-tune for Taiwanese Mandarin + code-switching English)
locally on Windows ARM64 Snapdragon X Copilot+ PCs, using a native ARM64
whisper.cpp build with NEON + dotprod + i8mm kernels. No NVIDIA GPU, no
cloud. Measured RTF 0.47x on 57.6 s Mandarin audio (faster than realtime).

## Platform constraints

- **Windows 11 ARM64 only.** The build assumes `aarch64-pc-windows-msvc`
  target. Do not add Linux / macOS paths; do not add x86_64 or WASM targets.
- **Python ≥ 3.12.** Managed via `uv`; the venv at `.venv/` is authoritative.
- **whisper.cpp is a submodule-like dependency**, cloned by `scripts/setup.ps1`
  into `./whisper.cpp/` and built with clang (LLVM 22+) into
  `./bin/native-arm64/`. Never build whisper.cpp with MSVC — the ggml-cpu
  CMake explicitly rejects MSVC for ARM (`"MSVC is not supported for ARM,
  use clang"`).
- **OpenMP must stay ON.** Building with `-DGGML_OPENMP=OFF` regresses
  perf by ~35x (measured). The resulting `libomp140.aarch64.dll` dependency
  is documented in `scripts/setup.ps1` with its license caveat.
- **No AI attribution in commits.** Never add `Co-Authored-By: Claude …`
  or similar trailers. Commit messages list only the human author.

## Build

```powershell
# First time on a Snapdragon X laptop:
pwsh -File scripts\setup.ps1
# Subsequent rebuilds:
pwsh -File scripts\build.ps1
```

## Install Python deps + run tests

```powershell
uv sync --all-extras
uv run pytest tests               # 143 unit tests + 1 skipped slow integration (<15 s)
uv run pytest tests --run-slow    # +1 real-model integration test (~30 s, downloads 1.6 GB on first run)
```

## Run the CLI

```powershell
# Single file
uv run python -m asr_local.cli path\to\audio.m4a

# Batch (model loaded once, amortized across files)
uv run python -m asr_local.cli a.m4a b.m4a c.wav

# Explicit quant choice
uv run python -m asr_local.cli --quant q4_0 long_lecture.m4a     # 1.4-1.7x faster, same content on Mandarin
uv run python -m asr_local.cli --quant fp16 high_precision.m4a   # maximum precision (3 GB model)
```

## Architecture (do not change lightly)

- `src/asr_local/segment.py`: immutable `TimestampedSegment` dataclass
- `src/asr_local/audio.py`: ffmpeg transcoding with 16 kHz mono PCM_S16LE fast-path passthrough
- `src/asr_local/model.py`: Hugging Face download + GGML magic/size validation (multi-repo routing: Q4_0 from `lsheep/Breeze-ASR-25-ggml`, others from `alan314159/Breeze-ASR-25-whispercpp`)
- `src/asr_local/vad.py`: Silero VAD download from `ggml-org/whisper-vad`
- `src/asr_local/transcriber.py`: whisper-cli subprocess wrapper (auto-chooses `-p / -t / -ac / -nfa / --vad / priority`)
- `src/asr_local/writer.py`: UTF-8 TXT output with `[HH:MM:SS - HH:MM:SS] text` lines
- `src/asr_local/cli.py`: argparse + auto-tuners + multi-file orchestration

## Testing philosophy

- TDD is mandatory. Write a failing test before code.
- Do NOT mock heavy tests as slow and skip them in defaults; `--run-slow`
  exists for the real integration path.
- Error paths are tested: `AudioConversionError` → exit 3,
  `WhisperCliError` → exit 4, `GgmlValidationError` → exit 5, `OSError` → 6.

## What NOT to do

- Do not add `Co-Authored-By: Claude …` to commits.
- Do not commit any file under `bin/native-arm64/` (gitignored; regenerated
  by `setup.ps1` from the user's own VS Redist — `libomp140.aarch64.dll`
  specifically is under the VS `debug_nonredist` tree and not freely
  redistributable).
- Do not default `--flash-attn` to on for `zh`/`ja`/`ko`; it degrades CJK
  transcription quality (whisper.cpp issue #3020). `choose_flash_attn`
  keeps it off for CJK; do not change this.
- Do not default to Q4_0 even though it is faster; `choose_quant` does not
  exist, and `q8_0` is the project-stated precision baseline.
- Do not add speculative decoding, batched decoding, or QNN NPU support —
  all three have been researched and are either not available in
  whisper.cpp as of 2026-04 or incompatible with the project's
  fp16-equivalent accuracy bar.

## Performance baseline (Snapdragon X Plus X1P42100, Balanced power plan)

- 5.8 s clip, Q8_0 + `-nfa` + ac=640: 9.6 s wallclock (RTF 1.66×)
- 5.8 s clip, Q4_0 + `-nfa` + ac=640: 6.2 s wallclock (RTF 1.07×, 1.69× faster)
- 57.6 s clip, Q8_0 + `-p 2 -t 4 --vad -nfa`: 27.1 s wallclock (RTF 0.47×)
- 57.6 s clip, Q4_0 + `-p 2 -t 4 --vad -nfa`: 21.9 s wallclock (RTF 0.38×, 1.38× faster)

Power plan "Best Performance" (via `scripts/tune-power.ps1`) adds a further
~30% on top per llama.cpp benchmark #8273.
