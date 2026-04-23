# tune-power.ps1 - unlock sustained-CPU perf on Snapdragon X laptops.
#
# Switches the active Windows power scheme to "Best Performance" (or unlocks
# and activates "Ultimate Performance" if -Ultimate is passed), disables
# processor core parking, and sets PerfBoostMode to Aggressive.
#
# Benchmarks on Snapdragon X (llama.cpp discussion #8273) show the default
# "Balanced" scheme parks cores even under load and gives ~40% less sustained
# CPU throughput than "Best Performance". Run once before a batch of
# transcriptions; safe to revert with:
#     powercfg /setactive SCHEME_BALANCED
#
# Requires admin. The script self-elevates via UAC if not already elevated.
#
# Usage:
#   pwsh -File scripts\tune-power.ps1
#   pwsh -File scripts\tune-power.ps1 -Ultimate
#   pwsh -File scripts\tune-power.ps1 -Revert
[CmdletBinding()]
param(
    [switch]$Ultimate,
    [switch]$Revert
)

$ErrorActionPreference = "Stop"

# Self-elevate if not admin.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Re-launching elevated via UAC..." -ForegroundColor Yellow
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath)
    if ($Ultimate) { $args += "-Ultimate" }
    if ($Revert)   { $args += "-Revert" }
    Start-Process -FilePath pwsh.exe -ArgumentList $args -Verb RunAs -Wait
    exit $LASTEXITCODE
}

Write-Host "Current power scheme:" -ForegroundColor Cyan
powercfg /getactivescheme

if ($Revert) {
    Write-Host "Reverting to Balanced..." -ForegroundColor Yellow
    powercfg /setactive SCHEME_BALANCED
    Write-Host "Done." -ForegroundColor Green
    powercfg /getactivescheme
    exit 0
}

if ($Ultimate) {
    # Unlock Ultimate Performance if not already visible, then activate it.
    $ultimateGuid = "e9a42b02-d5df-448d-aa00-03f14749eb61"
    Write-Host "Unlocking Ultimate Performance scheme..." -ForegroundColor Cyan
    powercfg -duplicatescheme $ultimateGuid 2>&1 | Out-Host
    powercfg /setactive $ultimateGuid
} else {
    # Best Performance (the "Power mode" slider's top position on ARM laptops).
    # Scheme GUID for High Performance (closest powercfg equivalent).
    Write-Host "Activating High Performance scheme..." -ForegroundColor Cyan
    powercfg /setactive SCHEME_MIN
}

# Regardless of scheme, tighten the knobs that actually matter on Oryon:
#   - No core parking (CPMINCORES = 100 %).
#   - 100 % CPU minimum state on AC.
#   - Aggressive perf-boost mode.
# These apply to the currently active scheme.
Write-Host "Applying per-scheme tuning (CPMINCORES=100, MinState=100, PerfBoost=Aggressive)..." -ForegroundColor Cyan

# 614b38e8-9a86-4e16-ad9d-4e9bf1b4e8f3 = CPMINCORES (hidden by default)
powercfg -attributes sub_processor 0cc5b647-c1df-4637-891a-dec35c318583 -ATTRIB_HIDE 2>&1 | Out-Null
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR 0cc5b647-c1df-4637-891a-dec35c318583 100
# PROCTHROTTLEMIN (893dee8e-2bef-41e0-89c6-b55d0929964c) = min processor state
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR 893dee8e-2bef-41e0-89c6-b55d0929964c 100
# PERFBOOSTMODE = Aggressive (2)
powercfg -attributes sub_processor be337238-0d82-4146-a960-4f3749d470c7 -ATTRIB_HIDE 2>&1 | Out-Null
powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR be337238-0d82-4146-a960-4f3749d470c7 2

powercfg /setactive SCHEME_CURRENT

Write-Host "`nNew active scheme:" -ForegroundColor Green
powercfg /getactivescheme

Write-Host "`nAlso recommended (manual):" -ForegroundColor Yellow
Write-Host "  1. Plug in AC. Battery-backed Windows throttles regardless of plan."
Write-Host "  2. Exclude the model dir + whisper-cli.exe from Windows Defender scan:"
Write-Host "     Add-MpPreference -ExclusionPath `"$env:USERPROFILE\.cache\huggingface`""
Write-Host "     Add-MpPreference -ExclusionProcess whisper-cli.exe"
Write-Host "`nTo revert: pwsh -File scripts\tune-power.ps1 -Revert"
