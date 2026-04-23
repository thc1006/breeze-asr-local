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
- ✅ **長音檔 RTF 0.47×** (比 realtime 快兩倍以上),短音檔 RTF ~2×
- ✅ **零精度漂移**:Q8_0 與 fp16 在黃金樣本上輸出**完全相同** transcript
- ✅ **自動調校**:依音檔長度決定 `-p` / `-t` / `--audio-ctx` / `--vad` / `--flash-attn`
- ✅ **130 個 TDD 測試** (129 unit + 1 integration,含 VAD / flash-attn / parallel / priority)

---

## 效能 / Performance

**測試檔案**:5.8 秒台灣華語 m4a (內容:「那個女人是誰 / 那個女人是我的老師」),`C:\Users\PC` Acer Swift SFG14-01 (Snapdragon X Plus X1P42100,8 核 Oryon,16 GB RAM)。

**短音檔** (5.8 秒台灣華語):

| 版本 | 設定 | 時間 | RTF |
|---|---|---|---|
| whisper.cpp x64 prebuilt (Prism) | Q8_0 default | 75.4 s | **13.0×** ❌ |
| whisper.cpp x64 prebuilt (Prism) | Q8_0 `-ac 512` | 25.8 s | 4.45× |
| whisper.cpp native ARM64 | Q8_0 default (舊有 tail-truncation bug) | 8.2 s | 1.41× |
| **whisper.cpp native ARM64 (現行預設)** | Q8_0 `-ac 640 -nfa` | **9.6 s** | **1.66×** ⭐ |

**長音檔** (57.6 秒同內容串 10 次):

| 設定 | 時間 | RTF |
|---|---|---|
| native ARM64 default | 36.0 s | 0.625× |
| native ARM64 `-p 2 -t 4` | 27.3 s | 0.474× |
| **native ARM64 `-p 2 -t 4 -nfa --vad` (現行預設)** | **27.1 s** | **0.47×** ⭐ |

**內容精度**:Q8_0 與 fp16 在所有設定下輸出**完全相同**,`-nfa` (CJK) 與 `--vad` 都不損精度。

60 分鐘真實語音 (有靜音) 預估:
- x64 Prism: ~14 小時 ❌
- **Native ARM64 現行預設 + VAD: ~20-30 分鐘** (VAD 跳過靜音部分)
- Native ARM64 無 VAD + 長音檔: ~30-40 分鐘

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
.venv\Scripts\asr-local.exe path\to\audio.m4a
# 或 python -m asr_local.cli path\to\audio.m4a
```

首次執行 CLI 會從 Hugging Face 下載 Breeze-ASR-25 Q8_0 GGML (~1.6 GB),之後快取複用。

### CLI 選項

```
asr-local <audio> [--quant q8_0|fp16|q5_k|q4_k]
                  [--processors N]              # -p, 自動: 1 (<30s) / 2 (>=30s)
                  [--threads N]                 # -t, 自動: cpu/processors
                  [--audio-ctx N]               # 自動依音檔長度計算
                  [--flash-attn/--no-flash-attn] # 自動: off for CJK
                  [--vad/--no-vad]              # 自動: on for audio >= 30s
                  [--priority normal|high]      # Windows process priority
                  [--timeout SECONDS]
                  [--language zh]
                  [--output path.txt]
```

## 🚀 進階效能調校 / Performance tuning

短音檔 RTF 已經很低了,下列技巧主要針對**長音檔批次處理**。

### 一鍵切換 Windows 電源計畫 (單次即可,影響系統)

```powershell
# 啟用 Best Performance + 禁用 core parking + PerfBoostMode=Aggressive
pwsh -File scripts\tune-power.ps1

# 想要更激進 (Ultimate Performance scheme,可能掉電池壽命):
pwsh -File scripts\tune-power.ps1 -Ultimate

# 還原:
pwsh -File scripts\tune-power.ps1 -Revert
```

**為何這麼大影響**:llama.cpp discussion #8273 實測 Snapdragon X 在「平衡」模式下**比最佳效能慢 ~40%** — 系統會 park 部分核心。sustained 100% CPU 型工作 (如 whisper 轉錄) 受害最深。

### Windows Defender 排除 (節省 3-8%)

```powershell
# 管理員 PowerShell:
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.cache\huggingface"
Add-MpPreference -ExclusionProcess whisper-cli.exe
```

### 其他手動調校

- **接 AC 電源** — 電池供電時 Windows 會強制節流 (約 15-25% perf loss)
- **65W 以上 USB-C PD 充電器** — 過低 wattage 會限制 boost clock
- **`--priority high` 旗標** — 讓 whisper-cli 用 HIGH_PRIORITY_CLASS (1-3% 在 Defender / Search 並發時)

### 精度保證

以下組合在 1woman.m4a 黃金樣本上產出**完全相同** transcript:
- Q8_0 vs fp16 (quant)
- `-ac 640` vs `-ac 0` (audio context)
- `-p 1 -t 8` vs `-p 2 -t 4` (parallelism)
- `--flash-attn` on vs off (CJK: off 更快且相同輸出)
- `--vad` on vs off (短音檔)

所以**任何調校都不會掉台灣華語辨識準確度**。

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
  vad.py          — Silero VAD 下載 (ggml-org/whisper-vad) + 驗證
  transcriber.py  — whisper-cli subprocess 封裝 + JSON → segments
  writer.py       — timestamped TXT 輸出
  cli.py          — argparse + orchestration + 音檔長度自動調校

tests/
  test_segment.py        15 tests
  test_audio.py           7 tests
  test_model.py          16 tests
  test_vad.py             8 tests
  test_transcriber.py    25 tests
  test_writer.py          7 tests
  test_cli.py            51 tests
  test_integration.py     1 test (--run-slow, 真實 1.6 GB 模型)
                        ─────────
                        130 total
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
- `tests/` — 130 pytest tests (129 unit + 1 slow integration)
- `scripts/tune-power.ps1` — Windows power-plan helper (~40% perf on Snapdragon X)
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
