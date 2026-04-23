# setup.ps1 — one-shot environment bootstrap + whisper.cpp native ARM64 build.
#
# Checks and installs (where possible) every prerequisite needed to produce
# bin/native-arm64/whisper-cli.exe with NEON + dotprod + i8mm kernels. Prints
# clear manual instructions for anything that requires UAC elevation.
#
# Usage:
#   pwsh -File scripts\setup.ps1
#   pwsh -File scripts\setup.ps1 -SkipBuild     # just check prereqs
#   pwsh -File scripts\setup.ps1 -WhisperTag v1.8.4
#
[CmdletBinding()]
param(
    [string]$WhisperTag = "v1.8.4",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn2([string]$msg){ Write-Host "    ! $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)  { Write-Host "    X $msg" -ForegroundColor Red }

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# ---------------------------------------------------------------------------
# Step 1 — architecture sanity check
# ---------------------------------------------------------------------------
Write-Step "Checking platform"
if ($env:PROCESSOR_ARCHITECTURE -ne "ARM64") {
    Write-Err "This script targets Windows ARM64 (PROCESSOR_ARCHITECTURE=ARM64). Current: $env:PROCESSOR_ARCHITECTURE"
    Write-Err "The native build only makes sense on Snapdragon X / ARM64 hardware."
    exit 1
}
Write-Ok "Windows ARM64"

# ---------------------------------------------------------------------------
# Step 2 — LLVM (clang targeting aarch64-pc-windows-msvc)
# ---------------------------------------------------------------------------
Write-Step "Checking LLVM (clang for ARM64)"
$llvmBin = "C:\Program Files\LLVM\bin"
$clang = Join-Path $llvmBin "clang.exe"
if (-not (Test-Path $clang)) {
    Write-Warn2 "LLVM not found at $clang"
    Write-Host "    Installing LLVM via winget (no admin needed)..."
    winget install --id LLVM.LLVM --silent --accept-source-agreements --accept-package-agreements
    if (-not (Test-Path $clang)) {
        Write-Err "winget install did not place clang.exe at $clang"
        Write-Err "Install manually: https://github.com/llvm/llvm-project/releases (pick the WoA / aarch64 .exe)"
        exit 1
    }
}
$clangVer = (& $clang --version | Select-Object -First 1)
Write-Ok $clangVer

# ---------------------------------------------------------------------------
# Step 3 — Visual Studio BuildTools with ARM64 target workload
# ---------------------------------------------------------------------------
Write-Step "Checking Visual Studio 2022 BuildTools + ARM64 workload"
$vsInst = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
$vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvarsall.bat"
$arm64CL = Get-ChildItem "$vsRoot\VC\Tools\MSVC\*\bin\Hostarm64\arm64\cl.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
$arm64Lib = Get-ChildItem "$vsRoot\VC\Tools\MSVC\*\lib\arm64\msvcrtd.lib" -ErrorAction SilentlyContinue | Select-Object -First 1
$libomp  = Get-ChildItem "$vsRoot\VC\Redist\MSVC\*\debug_nonredist\arm64\Microsoft.VC143.OpenMP.LLVM\libomp140.aarch64.dll" -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $arm64CL -or -not $arm64Lib -or -not $libomp) {
    Write-Err "MSVC ARM64 target workload is not installed."
    Write-Host ""
    Write-Host "    Run this ONCE in an elevated PowerShell (Run as Administrator):"
    Write-Host ""
    Write-Host "      & '$vsInst\setup.exe' modify ``" -ForegroundColor Yellow
    Write-Host "        --installPath '$vsRoot' ``" -ForegroundColor Yellow
    Write-Host "        --add Microsoft.VisualStudio.Component.VC.Tools.ARM64 ``" -ForegroundColor Yellow
    Write-Host "        --add Microsoft.VisualStudio.Component.VC.Tools.ARM64EC ``" -ForegroundColor Yellow
    Write-Host "        --quiet --norestart --nocache" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    Or: open Visual Studio Installer -> Modify -> Individual Components,"
    Write-Host "        check 'MSVC v143 - VS 2022 C++ ARM64/ARM64EC build tools'."
    Write-Host ""
    Write-Host "    After it finishes, re-run this script."
    exit 1
}
Write-Ok "MSVC ARM64 cl.exe: $($arm64CL.FullName)"
Write-Ok "MSVC ARM64 libs: $($arm64Lib.Directory.FullName)"
Write-Ok "libomp140.aarch64.dll: $($libomp.FullName)"

# ---------------------------------------------------------------------------
# Step 4 — CMake + Ninja (bundled with VS BuildTools)
# ---------------------------------------------------------------------------
Write-Step "Checking CMake + Ninja"
$cmake = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
foreach ($tool in @($cmake, $ninja)) {
    if (-not (Test-Path $tool)) {
        Write-Err "Missing: $tool (comes with VS BuildTools default install)"
        exit 1
    }
}
Write-Ok "CMake + Ninja present"

# ---------------------------------------------------------------------------
# Step 5 — Git
# ---------------------------------------------------------------------------
Write-Step "Checking git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Err "git not on PATH. Install from https://git-scm.com/ or: winget install Git.Git"
    exit 1
}
Write-Ok (git --version)

if ($SkipBuild) {
    Write-Step "Prereqs complete (-SkipBuild given, stopping here)."
    exit 0
}

# ---------------------------------------------------------------------------
# Step 6 — Clone whisper.cpp at pinned tag
# ---------------------------------------------------------------------------
Write-Step "Cloning whisper.cpp @ $WhisperTag"
$src = Join-Path $ProjectRoot "whisper.cpp"
if (Test-Path $src) {
    Write-Warn2 "$src already exists, leaving as-is"
} else {
    git clone --depth 1 --branch $WhisperTag https://github.com/ggml-org/whisper.cpp.git $src | Out-Null
    Write-Ok "Cloned"
}

# ---------------------------------------------------------------------------
# Step 7 — cmake configure + build
# ---------------------------------------------------------------------------
Write-Step "Configuring + building whisper.cpp (native ARM64 / clang / NEON)"
$build = Join-Path $src "build-arm64"
Remove-Item -Recurse -Force $build -ErrorAction SilentlyContinue

# Prepend toolchain dirs so cmake + vcvarsall can find vswhere + cmake + ninja + clang.
$env:Path = "$vsInst;$llvmBin;$(Split-Path $cmake);$(Split-Path $ninja);$env:Path"

$configureCmd = "`"$vcvars`" arm64 > nul && cmake -S `"$src`" -B `"$build`" -G Ninja " +
                "-DCMAKE_BUILD_TYPE=Release " +
                "-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ " +
                "-DCMAKE_RC_COMPILER=llvm-rc " +
                "-DGGML_NATIVE=ON -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON"
$exit = (Start-Process -NoNewWindow -Wait -FilePath cmd.exe -ArgumentList "/c", $configureCmd -PassThru).ExitCode
if ($exit -ne 0) { Write-Err "cmake configure failed (exit $exit)"; exit $exit }

$buildCmd = "`"$vcvars`" arm64 > nul && cmake --build `"$build`" --config Release -j 8"
$exit = (Start-Process -NoNewWindow -Wait -FilePath cmd.exe -ArgumentList "/c", $buildCmd -PassThru).ExitCode
if ($exit -ne 0) { Write-Err "cmake build failed (exit $exit)"; exit $exit }
Write-Ok "Build complete"

# ---------------------------------------------------------------------------
# Step 8 — Stage binary + DLLs into bin/native-arm64/
# ---------------------------------------------------------------------------
Write-Step "Staging binary + DLLs"
$dest = Join-Path $ProjectRoot "bin\native-arm64"
New-Item -ItemType Directory -Path $dest -Force | Out-Null

# Build outputs
Copy-Item "$build\bin\whisper-cli.exe" -Destination $dest -Force
Copy-Item "$build\bin\*.dll"           -Destination $dest -Force

# VC Redist ARM64 runtime (required by clang-built binaries on ARM64)
$redist = "$vsRoot\VC\Redist\MSVC"
$redistArm64 = Get-ChildItem "$redist\*\arm64\Microsoft.VC143.CRT" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($redistArm64) {
    Copy-Item "$($redistArm64.FullName)\*.dll" -Destination $dest -Force
}
Copy-Item $libomp.FullName -Destination $dest -Force

Write-Ok "Staged at: $dest"
Get-ChildItem $dest | Select-Object Name, Length | Format-Table -AutoSize

# ---------------------------------------------------------------------------
# Step 9 — Smoke test
# ---------------------------------------------------------------------------
Write-Step "Smoke-testing whisper-cli"
$w = Join-Path $dest "whisper-cli.exe"
& $w --help > $null 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "whisper-cli --help exit 0"
    Write-Host ""
    Write-Host "Next:" -ForegroundColor Cyan
    Write-Host "  uv sync --all-extras"
    Write-Host "  uv run python -m asr_local.cli path\to\audio.m4a"
} else {
    Write-Err "whisper-cli failed to start (exit $LASTEXITCODE). Check for missing DLLs."
    exit 1
}
