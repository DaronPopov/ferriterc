<#
.SYNOPSIS
    Build Ferrite-OS on Windows using the platform manifest seam.

.DESCRIPTION
    1. Runs generate_manifest.py to detect toolchains and write the manifest.
    2. Sets FERRITE_PLATFORM_MANIFEST and invokes cargo build/check.

.PARAMETER Mode
    "check" (default) or "build"

.PARAMETER Release
    If set, builds in release mode.

.PARAMETER Target
    Rust target triple. Defaults to "x86_64-pc-windows-msvc".
#>

param(
    [ValidateSet("check", "build")]
    [string]$Mode = "check",
    [switch]$Release,
    [string]$Target = "x86_64-pc-windows-msvc"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$ManifestFile = Join-Path $ScriptDir "ferrite_platform_manifest.toml"

Write-Host "=== Ferrite-OS Windows Builder ===" -ForegroundColor Cyan

# Step 1: Generate manifest
Write-Host "`n--- Generating platform manifest ---" -ForegroundColor Yellow
python (Join-Path $ScriptDir "generate_manifest.py")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Manifest generation failed"
    exit 1
}

if (-not (Test-Path $ManifestFile)) {
    Write-Error "Manifest file not found at $ManifestFile"
    exit 1
}

# Step 2: Build
Write-Host "`n--- Running cargo $Mode ---" -ForegroundColor Yellow

$env:FERRITE_PLATFORM_MANIFEST = $ManifestFile

$CargoArgs = @($Mode, "--target", $Target)
if ($Release) {
    $CargoArgs += "--release"
}

# Build the platform crate + daemon + app client
$Packages = @("ferrite-platform", "ferrite-daemon", "ptx-app")
foreach ($pkg in $Packages) {
    $CargoArgs += @("-p", $pkg)
}

Push-Location (Join-Path $RepoRoot "ferrite-os")
try {
    Write-Host "cargo $($CargoArgs -join ' ')" -ForegroundColor DarkGray
    & cargo @CargoArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Cargo $Mode failed"
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host "`n=== Build complete ===" -ForegroundColor Green
