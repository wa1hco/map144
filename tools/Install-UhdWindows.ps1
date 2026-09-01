#requires -Version 5.1
# Install-UhdWindows.ps1 -- install Ettus UHD (WinUSB driver + B210 firmware images).
#
# Used by install.ps1 so the B210 router/MAP144 path does not depend on the
# offline B210 kit (tools/install-b210.ps1) having already been run.
#
# What this installs (into C:\Program Files\UHD\ by default):
#   - uhd_find_devices.exe and other host tools
#   - WinUSB driver for the B210
#   - FPGA/FX3 firmware images under share\uhd\images\
#     (usrp_b210_fpga.bin, usrp_b210_fw.bin, ...)
#
# This does NOT install the Python `uhd` package into .venv (no Windows pip
# wheels).  For `import uhd`, use the B210 kit env\ or conda-forge uhd.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\Install-UhdWindows.ps1
#   powershell -ExecutionPolicy Bypass -File tools\Install-UhdWindows.ps1 -Force
#
# Exit codes: 0 = images present (installed or already there), 1 = failed.

[CmdletBinding()]
param(
    [string]$EttusVersion = '4.10.0.0',
    [ValidateSet('VS2022', 'VS2019')]
    [string]$VsBuild = 'VS2022',
    [string]$CacheDir = '',
    [switch]$Force,
    [switch]$NoElevate
)

$ErrorActionPreference = 'Stop'

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Warn2([string]$msg){ Write-Host "    WARN: $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)  { Write-Host "    ERROR: $msg" -ForegroundColor Red }

$UhdRootCandidates = @(
    'C:\Program Files\UHD',
    'C:\Program Files (x86)\UHD'
)

function Get-UhdRoot {
    foreach ($r in $UhdRootCandidates) {
        if (Test-Path (Join-Path $r 'bin\uhd_find_devices.exe')) { return $r }
    }
    return $null
}

function Get-B210ImagePath([string]$UhdRoot) {
    return (Join-Path $UhdRoot 'share\uhd\images\usrp_b210_fpga.bin')
}

function Test-B210Firmware([string]$UhdRoot) {
    if (-not $UhdRoot) { return $false }
    return (Test-Path (Get-B210ImagePath $UhdRoot))
}

# -- Already present? ----
$uhdRoot = Get-UhdRoot
if ($uhdRoot -and (Test-B210Firmware $uhdRoot) -and -not $Force) {
    Write-Ok "Ettus UHD already installed at $uhdRoot"
    Write-Ok "B210 firmware image present: $(Get-B210ImagePath $uhdRoot)"
    $env:UHD_IMAGES_DIR = Join-Path $uhdRoot 'share\uhd\images'
    exit 0
}

# -- Elevate for Program Files install ----
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin -and -not $NoElevate) {
    Write-Host "Requesting administrator privileges for Ettus UHD install..." -ForegroundColor Yellow
    $argList = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSCommandPath`"",
        '-EttusVersion', $EttusVersion,
        '-VsBuild', $VsBuild
    )
    if ($CacheDir) { $argList += @('-CacheDir', "`"$CacheDir`"") }
    if ($Force) { $argList += '-Force' }
    $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $argList -Verb RunAs -Wait -PassThru
    exit $p.ExitCode
}
if (-not $isAdmin) {
    Write-Err "Administrator rights required to install Ettus UHD into Program Files."
    Write-Host "  Re-run from an elevated PowerShell, or run tools\Install-UhdWindows.ps1" -ForegroundColor Yellow
    exit 1
}

# -- Cache / download ----
if (-not $CacheDir) {
    $CacheDir = Join-Path $PSScriptRoot 'cache'
}
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$EttusFile = "uhd_${EttusVersion}-release_Win64_${VsBuild}.exe"
$EttusUrl  = "https://files.ettus.com/binaries/uhd/latest_release/Windows11/$VsBuild/$EttusFile"
$EttusCache = Join-Path $CacheDir $EttusFile

Write-Step "Ettus UHD Windows installer ($EttusFile)"
if (Test-Path $EttusCache) {
    $sizeMB = [math]::Round((Get-Item $EttusCache).Length / 1MB, 1)
    Write-Ok "Using cached installer ($sizeMB MB): $EttusCache"
} else {
    Write-Host "    Downloading $EttusUrl" -ForegroundColor Cyan
    Write-Host "    (~233 MB; needs internet; may take several minutes)" -ForegroundColor Yellow
    try {
        # TLS 1.2 for older Windows PowerShell defaults
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $EttusUrl -OutFile $EttusCache
    } catch {
        Write-Err "Download failed: $_"
        Write-Host "  Manual fallback: download the installer from" -ForegroundColor Yellow
        Write-Host "    $EttusUrl" -ForegroundColor Yellow
        Write-Host "  save as $EttusCache, then re-run." -ForegroundColor Yellow
        exit 1
    }
    if (-not (Test-Path $EttusCache)) {
        Write-Err "Download reported success but file missing: $EttusCache"
        exit 1
    }
    Write-Ok "Downloaded $([math]::Round((Get-Item $EttusCache).Length / 1MB, 1)) MB"
}

# -- Silent install ----
Write-Step "Installing Ettus UHD silently (driver + firmware images)"
$p = Start-Process -FilePath $EttusCache -ArgumentList '/S' -Wait -PassThru
if ($p.ExitCode -ne 0) {
    Write-Err "Ettus installer exit code $($p.ExitCode)"
    Write-Host "  Try running manually (Run as administrator): $EttusCache" -ForegroundColor Yellow
    exit 1
}

$uhdRoot = Get-UhdRoot
if (-not $uhdRoot) {
    Write-Err "Installer finished but uhd_find_devices.exe not found under Program Files\UHD"
    exit 1
}
Write-Ok "UHD tools at $uhdRoot\bin"

# -- Ensure B210 images (bundled; download if missing) ----
$imagesDir = Join-Path $uhdRoot 'share\uhd\images'
$env:UHD_IMAGES_DIR = $imagesDir
if (-not (Test-B210Firmware $uhdRoot)) {
    Write-Warn2 "usrp_b210_fpga.bin missing after install; running uhd_images_downloader"
    $dl = Join-Path $uhdRoot 'bin\uhd_images_downloader.exe'
    if (Test-Path $dl) {
        & $dl
        if ($LASTEXITCODE -ne 0) {
            Write-Err "uhd_images_downloader failed (exit $LASTEXITCODE)"
            exit 1
        }
    } else {
        Write-Err "uhd_images_downloader.exe not found at $dl"
        exit 1
    }
}

if (-not (Test-B210Firmware $uhdRoot)) {
    Write-Err "B210 firmware still missing at $(Get-B210ImagePath $uhdRoot)"
    exit 1
}

Write-Ok "B210 firmware image: $(Get-B210ImagePath $uhdRoot)"
Write-Ok "UHD_IMAGES_DIR=$imagesDir"

# Soft probe (warn only -- B210 may be unplugged during install)
$find = Join-Path $uhdRoot 'bin\uhd_find_devices.exe'
if (Test-Path $find) {
    Write-Host "    Probing for B210 (optional)..." -ForegroundColor Cyan
    $probe = & $find 2>&1 | Out-String
    if ($probe -match 'B210') {
        Write-Ok "B210 visible to UHD"
    } else {
        Write-Warn2 "B210 not detected yet. Plug into a USB 3.0 port before run-router.bat."
    }
}

exit 0
