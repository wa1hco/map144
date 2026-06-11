#requires -Version 5.1
# build-b210-kit.ps1 -- assemble a self-contained MAP144 B210 install kit.
#
# Run on a HOME machine that already has Miniconda + git installed.
# Produces dist/map144-b210-kit-<gitDescribe>/ ready to copy to a USB
# stick or upload as a release artifact.
#
# Kit contents (~1.5 GB):
#   install-b210.bat               # double-click entry point
#   install-b210.ps1               # the actual installer
#   README-INSTALL.txt             # operator instructions
#   ettus/<ettus installer>.exe    # ~233 MB -- WinUSB driver + firmware
#   map144-env.zip                 # ~500 MB -- conda-packed portable Python env
#   map144-src.zip                 # ~10 MB -- git archive of repo at HEAD
#   VERSION.txt                    # git SHA, build timestamp
#
# Re-runs reuse the Ettus installer in tools/cache/ to avoid re-downloading.

[CmdletBinding()]
param(
    [string]$EttusVersion = '4.10.0.0',
    [string]$VsBuild      = 'VS2022',
    [string]$PythonVersion = '3.11',
    [switch]$ForceFreshEnv,
    [switch]$KeepBuildEnv
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = Resolve-Path (Join-Path $PSScriptRoot '..')
$CacheDir   = Join-Path $RepoRoot 'tools\cache'
$DistRoot   = Join-Path $RepoRoot 'dist'

# -- -1. Start transcript so a failed build is investigable post-mortem -------
# (When the .bat wrapper's window closes, all on-screen output is gone.  This
#  preserves a full log on disk for diagnosis.)
$logTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$BuildLog = Join-Path $CacheDir "build-$logTimestamp.log"
Start-Transcript -Path $BuildLog -Force | Out-Null
Write-Host "Build log: $BuildLog`n" -ForegroundColor Yellow

# Wrap the whole body in try/finally so Stop-Transcript runs even on throw.
# Without this, a mid-script failure leaves the transcript without its tail
# marker (PowerShell still closes the file at process exit, but cleanly
# stopping is nicer for re-reads).
try {

# -- 0. Sanity checks ----------------------------------------------------------
Write-Host "==> Sanity checks" -ForegroundColor Cyan

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not found on PATH.  Open an 'Anaconda Prompt' and re-run, " +
          "or run 'conda init powershell' once to make conda visible in PS."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found on PATH.  Install Git for Windows and retry."
}
Push-Location $RepoRoot
try {
    $GitSha     = (git rev-parse --short HEAD).Trim()
    $GitVersion = (git describe --always --dirty 2>$null).Trim()
    if (-not $GitVersion) { $GitVersion = $GitSha }
} finally {
    Pop-Location
}
$KitName  = "map144-b210-kit-$GitVersion"
$KitDir   = Join-Path $DistRoot $KitName
$BuildEnv = "map144-build-$GitSha"

Write-Host "  Repo:        $RepoRoot"
Write-Host "  Version:     $GitVersion"
Write-Host "  Build env:   $BuildEnv"
Write-Host "  Kit dir:     $KitDir"

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
if (Test-Path $KitDir) {
    Write-Host "  (existing kit dir will be overwritten)" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $KitDir
}
New-Item -ItemType Directory -Force -Path $KitDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $KitDir 'ettus') | Out-Null

# -- 1. Download Ettus UHD Windows installer (cached) -------------------------
Write-Host "`n==> Ettus UHD installer" -ForegroundColor Cyan

$EttusFile = "uhd_${EttusVersion}-release_Win64_${VsBuild}.exe"
$EttusUrl  = "https://files.ettus.com/binaries/uhd/latest_release/Windows11/$VsBuild/$EttusFile"
$EttusCache = Join-Path $CacheDir $EttusFile

if (Test-Path $EttusCache) {
    $sizeMB = [math]::Round((Get-Item $EttusCache).Length / 1MB, 1)
    Write-Host "  Using cached: $EttusCache  ($sizeMB MB)"
} else {
    Write-Host "  Downloading: $EttusUrl"
    Write-Host "  (this is ~233 MB; may take a few minutes)"
    Invoke-WebRequest -Uri $EttusUrl -OutFile $EttusCache
}
Copy-Item $EttusCache "$KitDir\ettus\$EttusFile"

# -- 2. Build conda env -------------------------------------------------------
Write-Host "`n==> Conda env $BuildEnv" -ForegroundColor Cyan

$existingEnv = & conda env list | Select-String -Pattern "^\s*$BuildEnv\s"
if ($existingEnv -and $ForceFreshEnv) {
    Write-Host "  -ForceFreshEnv: removing existing $BuildEnv"
    & conda env remove -n $BuildEnv -y | Out-Null
    $existingEnv = $null
}

if (-not $existingEnv) {
    Write-Host "  Creating $BuildEnv with python=$PythonVersion"
    & conda create -n $BuildEnv "python=$PythonVersion" -y
    if ($LASTEXITCODE -ne 0) { throw "conda create failed" }

    # NumPy must stay <2 (UHD's Python bindings + MAP144 channelizer's numba
    # kernels are ABI-bound to NumPy 1.26.x).  Pin it at conda install time so
    # pip's subsequent `numpy<2` requirement is already satisfied -- otherwise
    # conda installs NumPy 2.x with uhd, pip downgrades it, and the resulting
    # file-overwrite makes conda-pack refuse to pack the env as inconsistent.
    Write-Host "  Installing conda-forge uhd + conda-pack + numpy<2"
    & conda install -n $BuildEnv -c conda-forge uhd "numpy<2" conda-pack -y
    if ($LASTEXITCODE -ne 0) { throw "conda install failed" }

    Write-Host "  Installing pip requirements"
    & conda run -n $BuildEnv --no-capture-output pip install -r (Join-Path $RepoRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
} else {
    Write-Host "  Reusing existing $BuildEnv (use -ForceFreshEnv to rebuild)"
}

# Verify import uhd succeeds in the env
Write-Host "  Verifying 'import uhd' in build env..."
$probe = & conda run -n $BuildEnv --no-capture-output python -c "import uhd; print(uhd.__version__)"
if ($LASTEXITCODE -ne 0) { throw "import uhd failed in build env: $probe" }
Write-Host "  import uhd OK ($probe)"

# -- 3. Pack the env into kit -------------------------------------------------
Write-Host "`n==> Packing env" -ForegroundColor Cyan

$EnvZip = Join-Path $KitDir 'map144-env.zip'
Write-Host "  Running conda-pack (this can take 5-10 minutes)..."
& conda run -n $BuildEnv --no-capture-output conda-pack -n $BuildEnv -o $EnvZip --format zip --force
if ($LASTEXITCODE -ne 0) { throw "conda-pack failed" }
$envSizeMB = [math]::Round((Get-Item $EnvZip).Length / 1MB, 1)
Write-Host "  Wrote $EnvZip ($envSizeMB MB)"

# -- 4. Snapshot the map144 source --------------------------------------------
Write-Host "`n==> Snapshotting source" -ForegroundColor Cyan

$SrcZip = Join-Path $KitDir 'map144-src.zip'
Push-Location $RepoRoot
try {
    & git archive --format=zip --output=$SrcZip HEAD
    if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
} finally {
    Pop-Location
}
$srcSizeMB = [math]::Round((Get-Item $SrcZip).Length / 1MB, 1)
Write-Host "  Wrote $SrcZip ($srcSizeMB MB)"

# -- 5. Copy installer scripts + README into kit ------------------------------
Write-Host "`n==> Assembling kit" -ForegroundColor Cyan

Copy-Item (Join-Path $PSScriptRoot 'install-b210.ps1') (Join-Path $KitDir 'install-b210.ps1')
Copy-Item (Join-Path $PSScriptRoot 'install-b210.bat') (Join-Path $KitDir 'install-b210.bat')
Copy-Item (Join-Path $PSScriptRoot 'kit-README.txt')   (Join-Path $KitDir 'README-INSTALL.txt')

# VERSION.txt for the installer to consult
$buildTime = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
$versionContent = @"
map144 version : $GitVersion
git sha        : $GitSha
ettus installer: $EttusFile
python         : $PythonVersion
built          : $buildTime
"@
Set-Content -Path (Join-Path $KitDir 'VERSION.txt') -Value $versionContent -Encoding utf8

# -- 6. Clean up build env (unless asked to keep) -----------------------------
if (-not $KeepBuildEnv) {
    Write-Host "`n==> Cleaning up build env $BuildEnv"
    & conda env remove -n $BuildEnv -y | Out-Null
}

# -- 7. Summary ---------------------------------------------------------------
$totalMB = [math]::Round((Get-ChildItem -Recurse $KitDir | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "`nKIT READY" -ForegroundColor Green
Write-Host "  Location: $KitDir"
Write-Host "  Size:     $totalMB MB"
Write-Host "  Copy this directory to a USB stick or upload as a release."

} catch {
    Write-Host "`nBUILD FAILED" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Transcript: $BuildLog" -ForegroundColor Yellow
    throw
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}
