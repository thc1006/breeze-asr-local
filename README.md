# breeze-asr-local

**在 Windows ARM64 (Snapdragon X Copilot+ PC) 本地跑 MediaTek Breeze-ASR-25**,保留 fp16 精準度、用 native NEON 把 RTF 從 13.7× 壓到 1.5×。

Local Taiwanese Mandarin speech-to-text on Windows-on-ARM. MediaTek's fine-tuned Whisper-large-v2 (Breeze-ASR-25), wired through whisper.cpp built natively for aarch64 with NEON + dotprod + i8mm. No NVIDIA, no cloud, no quantization beyond Q8_0.

---

## 為什麼存在 / Why this exists

Snapdragon X Copilot+ PC (例如 Acer Swift 14 AI, ASUS Vivobook S, Lenovo Yoga Slim 7x) 出貨前,沒人把台灣華語 ASR 部署好在 Windows ARM64 上:

| 平台 | Breeze-ASR-25 port 存在嗎? |
|---|---|
| Apple Silicon (CoreML + ANE) | ✓ 社群有 9 個變體,含 4-bit palette |
| NVIDIA GPU (CTranslate2) | ✓ 多個 int8 版本 |
| MediaTek Dimensity NPU | ✓ MediaTek 自家的 BreezeApp |
| **Windows ARM64 (Snapdragon X)** | **❌ 全網零個** |

(調研細節 → [`docs/perf_journey.md`](docs/perf_journey.md))

這個 repo 補上最後一塊。主要成果:

- ✅ **全網第一個** native ARM64 Windows 的 Breeze-ASR-25 部署腳本
- ✅ **9× 實測加速**:從 x64-Prism-模擬的 RTF 13.72× 壓到 native ARM64 的 **RTF 1.54×**
- ✅ **零精度漂移**:Q8_0 與 fp16 在黃金樣本上輸出**完全相同** transcript
- ✅ **74 個 TDD 測試** + 真實 golden integration test

---

## 效能 / Performance

**測試檔案**:5.8 秒台灣華語 m4a (內容:「那個女人是誰 / 那個女人是我的老師」),`C:\Users\PC` Acer Swift SFG14-01 (Snapdragon X Plus X1P42100,8 核 Oryon,16 GB RAM)。

| 版本 | 量化 | audio-ctx | 時間 | RTF | 備註 |
|---|---|---|---|---|---|
| whisper.cpp **x64 prebuilt (Prism)** | Q8_0 | full | 75–79 s | **13.7×** | ❌ 60 min 音檔要跑 14 小時 |
| whisper.cpp x64 prebuilt (Prism) | Q8_0 | 512 | 25.8 s | 4.5× | `-ac 512` auto-tune 幫了 3× |
| **whisper.cpp native ARM64 (本 repo)** | Q8_0 | 512 | **8.2 s** | **1.4×** | ⭐ 建議預設 |
| whisper.cpp native ARM64 | Q8_0 | full | 15.1 s | 2.6× | 長音檔用 |
| whisper.cpp native ARM64 | fp16 | 512 | 13.0 s | 2.2× | 短音檔極限精度 |
| whisper.cpp native ARM64 | fp16 | full | 27.7 s | 4.8× | 長音檔極限精度 |

**全部輸出內容完全相同** — Q8_0 在 5.8s 樣本上 **零 CER 漂移**。

60 分鐘長音檔預估:
- x64 Prism: ~14 小時
- Native ARM64 Q8_0: **~2.6 小時**
- Native ARM64 fp16: ~4.8 小時

---

## Quick start

```powershell
# 1. 裝 Python 環境管理器 uv (native ARM64)
winget install astral-sh.uv

# 2. clone
git clone https://github.com/thc1006/breeze-asr-local
cd breeze-asr-local

# 3. 建 Python venv + 裝依賴
uv sync --all-extras

# 4. 自動 build whisper.cpp native ARM64 (首次約 10 分鐘)
pwsh -File scripts\setup.ps1

# 5. 轉錄
.venv\Scripts\python.exe -m asr_local.cli path\to\audio.m4a
```

首次執行 CLI 會從 Hugging Face 下載 Breeze-ASR-25 Q8_0 GGML (~1.6 GB),之後快取複用。

### CLI 選項

```
asr-local <audio> [--quant q8_0|fp16|q5_k|q4_k]
                  [--threads 8]
                  [--language zh]
                  [--audio-ctx 512]    # 自動依音檔長度選
                  [--output path.txt]
```

---

## Requirements

**硬體**:
- Windows 11 ARM64 (Snapdragon X Plus / Elite / X2 — 任何 Copilot+ PC)
- 16 GB+ RAM (Q8_0 需要 ~3 GB,fp16 需要 ~5 GB 峰值)
- 5 GB 磁碟空間 (模型 + 依賴)

**軟體** (`scripts\setup.ps1` 會檢查並指引):
- [LLVM 20+ for Windows ARM64](https://github.com/llvm/llvm-project/releases) — clang 原生 aarch64 編譯器
- Visual Studio 2022 Build Tools 含 **`Microsoft.VisualStudio.Component.VC.Tools.ARM64`** workload (提供 MSVC ARM64 CRT + libomp)
- CMake 3.14+ (VS BuildTools 附帶即可)
- Git
- Python 3.12+ (建議透過 uv 管理)

---

## 從原始碼 build whisper.cpp / Build whisper.cpp from source

`scripts\setup.ps1` 做完所有事,底層就是:

```powershell
# 1. Clone whisper.cpp 到固定版本
git clone --depth 1 --branch v1.8.4 https://github.com/ggml-org/whisper.cpp

# 2. 啟用 VS ARM64 developer command prompt (提供 MSVC CRT + Windows SDK 環境變數)
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat' arm64

# 3. cmake 設定:clang + Ninja + GGML_NATIVE (自動偵測 NEON/dotprod/i8mm)
cmake -S whisper.cpp -B whisper.cpp\build-arm64 `
  -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ `
  -DCMAKE_RC_COMPILER=llvm-rc `
  -DGGML_NATIVE=ON -DWHISPER_BUILD_TESTS=OFF

# 4. Build (30 秒左右)
cmake --build whisper.cpp\build-arm64 --config Release -j 8

# 5. 連同 libomp140.aarch64.dll 等 VC Redist DLL 一起複製到 bin/native-arm64/
```

whisper.cpp CMake 明確拒絕 MSVC for ARM (`MSVC is not supported for ARM, use clang`),一定要走 clang 路線。

---

## Architecture

```
src/asr_local/
  segment.py      — TimestampedSegment dataclass + HH:MM:SS 格式化
  audio.py        — 任意格式 → 16 kHz mono PCM_S16LE (via imageio-ffmpeg)
  model.py        — HF GGML 下載 + magic/size 驗證 + 變體選擇
  transcriber.py  — whisper-cli subprocess 封裝 + JSON → segments
  writer.py       — timestamped TXT 輸出
  cli.py          — argparse + orchestration + 音檔長度自動調 audio_ctx

tests/
  test_segment.py        15 tests
  test_audio.py           6 tests
  test_model.py          16 tests
  test_transcriber.py    13 tests
  test_writer.py          7 tests
  test_cli.py            17 tests
  test_integration.py     1 test (--run-slow, 真實 1.6 GB 模型)
                        ─────────
                        75 total
```

---

## Testing

```powershell
uv run pytest                  # unit + 快速整合 (< 10 秒)
uv run pytest --run-slow       # + 真實 Breeze 模型下載 + 轉錄 (需 ~2 分鐘)
```

---

## Repo layout reference

- `src/asr_local/` — the real pipeline (what you want to run)
- `tests/` — 75 pytest tests (74 unit + 1 slow integration)
- `scripts/setup.ps1` — one-shot toolchain install + whisper.cpp native build
- `docs/perf_journey.md` — full write-up of the 13.7× → 1.4× optimization
- `notebooks/colab_original.py` — original Colab-GPU pipeline, kept for reference only
- `bin/native-arm64/` — *generated* by setup.ps1, gitignored; `whisper-cli.exe` lives here

---

## Credits

上游 / Upstream:
- [MediaTek Research / Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) — 台灣華語 + 中英混用 ASR 模型
- [ggml-org / whisper.cpp](https://github.com/ggml-org/whisper.cpp) — 推論引擎
- [alan314159 / Breeze-ASR-25-whispercpp](https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp) — GGML 格式轉換

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
