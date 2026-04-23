# build.ps1 — rebuild whisper.cpp only (assumes prereqs already installed).
# Use setup.ps1 for first-time setup.
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

# Arch check (same fallback as setup.ps1 for x64 pwsh on ARM64 host).
$hostArch = $env:PROCESSOR_ARCHITECTURE
if ($hostArch -ne "ARM64" -and $env:PROCESSOR_ARCHITEW6432 -eq "ARM64") {
    $hostArch = "ARM64"
}
if ($hostArch -ne "ARM64") {
    Write-Host "build.ps1 targets Windows ARM64. Detected host: $hostArch" -ForegroundColor Red
    exit 1
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$src = Join-Path $ProjectRoot "whisper.cpp"
if (-not (Test-Path $src)) {
    Write-Host "whisper.cpp source missing — run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vsInst = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
$vcvars = "$vsRoot\VC\Auxiliary\Build\vcvarsall.bat"
$llvmBin = "C:\Program Files\LLVM\bin"
$cmakeBin = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$ninjaBin = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
$env:Path = "$vsInst;$llvmBin;$cmakeBin;$ninjaBin;$env:Path"

$build = Join-Path $src "build-arm64"
$cmd = "`"$vcvars`" arm64 > nul && cmake --build `"$build`" --config Release -j 8"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$dest = Join-Path $ProjectRoot "bin\native-arm64"
Copy-Item "$build\bin\whisper-cli.exe" -Destination $dest -Force
Copy-Item "$build\bin\*.dll"           -Destination $dest -Force

# Re-copy libomp140.aarch64.dll in case bin/ was wiped. Same license caveat
# as setup.ps1: do not redistribute externally.
$libomp = Get-ChildItem "$vsRoot\VC\Redist\MSVC\*\debug_nonredist\arm64\Microsoft.VC143.OpenMP.LLVM\libomp140.aarch64.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($libomp) {
    Copy-Item $libomp.FullName -Destination $dest -Force
}
Write-Host "Rebuilt." -ForegroundColor Green
