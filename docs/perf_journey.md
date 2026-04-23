# Perf journey: RTF 13.7× → 1.4× on Snapdragon X Plus

Full story of why running MediaTek Breeze-ASR-25 on a Copilot+ PC needs more than `pip install`, and the nine-fold speed-up you get once you stop fighting the platform.

## 0. Hardware under test

Acer Swift SFG14-01
- Snapdragon X Plus X1P42100 (entry-tier X Plus, 8 Oryon cores @ 3.24 GHz, NEON + dotprod + i8mm, no SVE, no SME)
- 16 GB LPDDR5x, 135 GB/s shared across CPU / Adreno X1-45 GPU / Hexagon NPU
- Windows 11 ARM64

## 1. First question: can the NPU or GPU help?

Short answer: no, not for MediaTek Breeze-ASR-25 today.

### Hexagon NPU

QNN HTP backend mandates (a) static shapes, (b) full integer quantization (typically a16w8). That fights Whisper-large-v2's 32-layer autoregressive decoder with dynamic KV-cache. Qualcomm AI Hub ships:

- Whisper-Tiny / Base / Small / Medium (float or w8a16)
- Whisper-Large-V3-**Turbo** (pruned 32→4 decoder layers, MHA→SHA, ~809M)
- **No Whisper-Large-V2 anywhere.** [quic/ai-hub-models issue #115](https://github.com/quic/ai-hub-models/issues/115) requesting large-v2 remains open since 2024.

Breeze-ASR-25 is fine-tuned from Whisper-large-v2 (1.55B, 32 encoder + 32 decoder). Porting it onto turbo architecture would mean re-fine-tuning without MediaTek's training data — not feasible. Shoe-horning vanilla large-v2 through QNN with heavy a16w8 quantization violates the "same accuracy as reference" constraint.

### Adreno X1-45 GPU

Three technical paths, all dead ends as of 2026-04:

- **QNN EP GPU backend**: still preview, Whisper-critical ops (MHA, LayerNorm, KV-cache Gather) flagged as HTP/GPU weak spots, no public large-v2 demo.
- **WebGPU via transformers.js**: 128 MiB `maxStorageBufferBindingSize` cap, Chromium flag-gated on WoA, known Whisper-large OOM bugs ([transformers.js #860](https://github.com/huggingface/transformers.js/issues/860)).
- **DirectML**: Microsoft-declared [maintenance mode](https://github.com/microsoft/DirectML).

Plus the architectural reality: Adreno X1-45 shares the **same 135 GB/s DRAM bus as the CPU**. Whisper decode is memory-bandwidth-bound. [llama.cpp benchmarks on X1-**85**](https://github.com/ggml-org/llama.cpp/discussions/8336) (the *stronger* GPU tier) already lose to the CPU on decode (18 vs 20 tok/s). X1-45 has half the GPU compute. The GPU cannot beat CPU on memory-bound transformer decode on this SoC.

**Conclusion**: stay on CPU, focus on maximizing NEON throughput.

## 2. The first attempt and the shock

Official whisper.cpp release v1.8.4 only ships:
- `whisper-bin-Win32.zip` (x86)
- `whisper-bin-x64.zip` (x64)
- `whisper-cublas-*-x64.zip` (NVIDIA)
- `xcframework` (Apple)
- JAR

**Zero Windows ARM64 binary.** Issue [#2132](https://github.com/ggml-org/whisper.cpp/issues/2132) "Version for Windows on Arm?" — open since May 2024.

Pragmatic first try: download `whisper-bin-x64.zip`, run under Windows Prism x64 emulator.

```
Audio: 5.8s Taiwanese Mandarin clip (1woman.m4a)
Model: Breeze-ASR-25 Q8_0 (1.66 GB)
Threads: 8
Result: 79.0 s → RTF 13.72×
```

**A 60-minute meeting would take 14 hours to transcribe.** Unusable.

## 3. Diagnosing the 13.7× with flag tweaks

Warm-cache bench matrix on the same clip:

| Flags | Time | RTF | Interpretation |
|---|---|---|---|
| defaults (beam=5, flash-attn on, ac=0) | 75.4 s | 13.0× | baseline |
| `-bs 1 -bo 1 -nfa` (greedy, no FA) | 77.7 s | 13.4× | **beam & FA are not the bottleneck** |
| `-t 4` | 103.9 s | 17.9× | more threads help — it's **compute-bound**, not memory-bound |
| **`-ac 512`** | **25.8 s** | **4.45×** | **encoder was doing full 30s context on 5.8s of real audio — 3× waste** |

Big takeaway: whisper.cpp's encoder pads short clips to a 3000-frame mel tensor and runs full-context attention over it. For a 5.8 s clip, ~80% of encoder compute is operating on silence. `-ac 512` caps the attention window at ~10 s, killing that waste.

Root cause of the remaining 4.5× (not the 3× padding waste): Prism translates AVX2+FMA loops in ggml's quantized matmul kernels into 128-bit NEON sequences with register-spill overhead. Microsoft's 2024 Prism update supports AVX2 emulation, but each 256-bit AVX2 op becomes multiple 128-bit NEON ops — structurally slower.

## 4. The real fix: native ARM64 whisper.cpp with NEON

Ranking of options (from research across PyPI, HF, Qualcomm AI Hub, ExecuTorch, faster-whisper, Microsoft AI Toolkit):

| Path | RTF (est.) | Effort |
|---|---|---|
| whisper.cpp x64 prebuilt + Prism | 13.7× (measured) | 0 |
| whisper.cpp x64 blas prebuilt + Prism | 11–13× | 0 |
| faster-whisper x64 wheel + Prism | 8–12× | 1 hr |
| pywhispercpp from source (needs ARM64 VS workload) | ~1× | 3–6 hr |
| **whisper.cpp built natively with LLVM + MSVC ARM64 CRT** | **~1×** | **hours, not days** |

No native-ARM64 Python wheel for whisper anywhere on PyPI. The real shortcut is `clang.exe` + `cmake -G Ninja -DGGML_NATIVE=ON`.

## 5. Build chain

The tooling cascade that has to click together:

1. **LLVM ≥ 20 with aarch64-pc-windows-msvc target**
   - `winget install LLVM.LLVM` pulls the right ARM64 build automatically on a WoA host
   - Gives you `clang.exe`, `clang++.exe`, `llvm-rc.exe`, `lld-link.exe`

2. **Visual Studio 2022 BuildTools with `VC.Tools.ARM64` workload**
   - The default BuildTools install on ARM64 installs only `Hostarm64\x64` and `Hostarm64\x86` compilers (cross to x86 targets) — **not ARM64 target**.
   - Must explicitly add `Microsoft.VisualStudio.Component.VC.Tools.ARM64`.
   - This also brings in the ARM64 CRT static libs (`msvcrtd.lib` etc. under `VC\Tools\MSVC\*\lib\arm64\`) and `libomp140.aarch64.dll` (Microsoft's LLVM OpenMP build).
   - Requires admin — `setup.exe modify --add ... --quiet` will fail non-elevated (exit 5007).

3. **CMake generator gotcha**
   - whisper.cpp's `ggml-cpu` explicitly rejects MSVC for ARM: `MSVC is not supported for ARM, use clang`. Trying `cmake -A ARM64` with cl.exe will fail at the GGML CMakeLists level.
   - Must invoke through **vcvarsall arm64** (to set up LIB, INCLUDE env vars for MSVC ARM64 CRT + Windows SDK ARM64 libs) and then drive cmake with `CMAKE_C_COMPILER=clang`.

4. **Runtime DLL set** to ship alongside `whisper-cli.exe`:
   - From build output: `whisper.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`
   - From VC Redist ARM64 (redistributable): `vcruntime140.dll`, `vcruntime140_1.dll`, `vcruntime140_threads.dll`, `msvcp140.dll`, `msvcp140_1.dll`, `msvcp140_2.dll`, `msvcp140_atomic_wait.dll`, `msvcp140_codecvt_ids.dll`, `vccorlib140.dll`, `concrt140.dll`
   - From VC Redist `debug_nonredist/arm64/Microsoft.VC143.OpenMP.LLVM/`: **`libomp140.aarch64.dll`**

Missing any of these → immediate `-1073741515` (STATUS_DLL_NOT_FOUND) on launch.

**The libomp140 trade-off**: `GGML_OPENMP=ON` pulls a hard dependency on `libomp140.aarch64.dll`, which ships only under VS Redist's `debug_nonredist` tree. Microsoft's license marks that tree as *not redistributable* — you may use it on the development machine but not bundle and ship it. We measured what happens with `GGML_OPENMP=OFF` (ggml falling back to its own pthread threadpool): RTF goes from 1.4× to ~55× on the 5.8 s golden clip — **35× slower**, not "negligible". OpenMP is load-bearing for whisper.cpp on this CPU. The setup script therefore copies libomp140.aarch64.dll into `bin/native-arm64/` and prints a warning that `bin/` must not be externally redistributed.

## 6. Configure-time SIMD detection (good news)

cmake's `GGML_NATIVE=ON` auto-probes the host CPU. On Snapdragon X Plus Oryon:

```
-- Performing Test HAVE_DOTPROD                - Success  (NEON UDOT/SDOT)
-- Performing Test HAVE_SVE                    - Failed
-- Performing Test HAVE_MATMUL_INT8            - Success  (Armv8.6 I8MM)
-- Performing Test HAVE_FMA                    - Success
-- Performing Test HAVE_FP16_VECTOR_ARITHMETIC - Success
-- Performing Test HAVE_SME                    - Failed
-- Adding CPU backend variant ggml-cpu:
     -mcpu=native+dotprod+i8mm+nosve+nosme
```

Dotprod and I8MM are the important ones for Q8_0 — they are exactly the instructions that make int8 matmul fly on ARMv8.6+.

## 7. Result

Same 5.8s clip, same Q8_0 weights, same transcript, after switching to the native binary:

| Stage | Time | RTF | vs. baseline |
|---|---|---|---|
| x64 prebuilt + Prism (defaults) | 79.0 s | 13.72× | 1× |
| x64 prebuilt + Prism + `-ac 512` | 25.8 s | 4.45× | 3.1× |
| **native ARM64 + `-ac 512`** | **8.2 s** | **1.41×** | **9.6×** |
| native ARM64 + full ctx (60 min use-case extrapolation) | 15.1 s | 2.60× | 5.3× |

**Q8_0 transcript output is byte-identical to fp16 on the golden clip.** No accuracy drift.

## 8. Extrapolated real-world numbers

For one hour of Mandarin audio on a Snapdragon X Plus X1P42100 (8 cores):

| Binary | Quant | Wall time |
|---|---|---|
| x64 + Prism | Q8_0 | ~14 h |
| native ARM64 | Q8_0 | ~2.6 h |
| native ARM64 | fp16 | ~4.8 h |

On Snapdragon X **Elite** (12 cores, faster cache): shave another ~30–40% off.

## 9. What's still expensive

- No AVX-512 to help on x64 emulated path — doesn't matter since we left that path.
- No SVE on Oryon — no extra gain there.
- Q4_K is smaller (890 MB) but same compute intensity per token — won't go faster, will lose accuracy.
- fp16 is 2× slower than Q8_0 because 2× the weight bytes touched per decode step. For long audio, Q8_0 is the right default.

## 10. One-command setup

All of the above is automated in [`scripts/setup.ps1`](../scripts/setup.ps1). It checks for LLVM, VS ARM64 workload, CMake, Ninja, Git; prints manual install instructions if elevation is needed; clones whisper.cpp at the pinned tag; drives cmake with clang; copies the binary and the eleven-DLL runtime set into `bin/native-arm64/`; and smoke-tests the result.

---

## Sources

- [whisper.cpp Windows-on-Arm tracking issue #2132](https://github.com/ggml-org/whisper.cpp/issues/2132)
- [Qualcomm AI Hub Whisper model catalog (no large-v2)](https://aihub.qualcomm.com/models)
- [quic/ai-hub-models whisper-large-v3 request #115](https://github.com/quic/ai-hub-models/issues/115)
- [MediaTek-Research/Breeze-ASR-25 model card](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)
- [alan314159/Breeze-ASR-25-whispercpp (GGML quantizations)](https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp)
- [llama.cpp Snapdragon X bandwidth discussion #8336](https://github.com/ggml-org/llama.cpp/discussions/8336)
- [Microsoft Prism AVX2 support announcement (2024)](https://techcommunity.microsoft.com/blog/windowsosplatform/windows-on-arm-runs-more-apps-and-games-with-new-prism-update/4475631)
- [ARM64 build flags in GGML (ggml-cpu CMakeLists)](https://github.com/ggml-org/whisper.cpp/blob/master/ggml/src/ggml-cpu/CMakeLists.txt)
