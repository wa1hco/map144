# install.ps1 — MAP144 install / update for Windows.
#
# Run from inside the cloned repo (no auto-clone — gives you full
# control of where the working tree lives):
#
#     git clone https://github.com/wa1hco/map144.git C:\WSJT\map144
#     cd C:\WSJT\map144
#     .\install.ps1
#
# Re-runnable: existing venv is reused, requirements re-resolved.
# Uses the venv's python.exe directly so there are no PowerShell
# execution-policy concerns — no activate step required.

$ErrorActionPreference = 'Stop'

# Always operate relative to the script's own directory so PowerShell's
# (frequently-resetting) cwd never confuses us.
$RepoDir = (Resolve-Path $PSScriptRoot).Path
Set-Location $RepoDir

# ── 1. Find a usable Python ─────────────────────────────────────────────
# Refuse the Microsoft Store stub.  On a fresh Win11 install ``python``
# resolves to ``...\WindowsApps\python.exe`` which launches the Store
# instead of running anything.
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "ERROR: 'python' not found on PATH." -ForegroundColor Red
    Write-Host "  Install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  (the python.org installer, NOT the Microsoft Store one)." -ForegroundColor Red
    exit 1
}
if ($pythonCmd.Source -like '*WindowsApps*') {
    Write-Host "ERROR: 'python' resolves to the Microsoft Store stub:" -ForegroundColor Red
    Write-Host "  $($pythonCmd.Source)" -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "and make sure that install comes BEFORE WindowsApps on PATH." -ForegroundColor Red
    exit 1
}

# Version check — bail early so we don't surface a confusing pip error.
$pyVersionLine = & python -c "import sys; v=sys.version_info; print(f'{v[0]}.{v[1]}.{v[2]}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to query Python version." -ForegroundColor Red
    exit 1
}
$pyMajor, $pyMinor, $pyPatch = $pyVersionLine.Split('.')
if ([int]$pyMajor -lt 3 -or ([int]$pyMajor -eq 3 -and [int]$pyMinor -lt 10)) {
    Write-Host "ERROR: Python >= 3.10 required (found $pyVersionLine)." -ForegroundColor Red
    exit 1
}
Write-Host "Python $pyVersionLine at $($pythonCmd.Source)" -ForegroundColor Green

# ── 2. Create venv if missing (idempotent) ──────────────────────────────
$VenvDir = Join-Path $RepoDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating venv at $VenvDir ..." -ForegroundColor Cyan
    & python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: venv creation failed." -ForegroundColor Red
        exit 1
    }
}

# ── 3. Upgrade pip ──────────────────────────────────────────────────────
# Mandatory: older pip on a stale Python install can miss the ABI tag
# for newer Python builds and fall back to source compilation.
Write-Host "Upgrading pip ..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip upgrade failed." -ForegroundColor Red
    exit 1
}

# ── 4. Install requirements ─────────────────────────────────────────────
Write-Host "Installing requirements ..." -ForegroundColor Cyan
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install -r requirements.txt failed." -ForegroundColor Red
    exit 1
}

# ── 5. Verify jt9 discovery ─────────────────────────────────────────────
Write-Host "Verifying jt9 discovery ..." -ForegroundColor Cyan
$jt9Path = & $VenvPython -c "from map144_app.detection import find_jt9; p = find_jt9(); print(p if p else '')"
if (-not $jt9Path) {
    Write-Host "WARNING: jt9 not found." -ForegroundColor Yellow
    Write-Host "  Install WSJT-X from https://wsjt.sourceforge.io/ (provides jt9.exe)," -ForegroundColor Yellow
    Write-Host "  or set MAP144_JT9 to the full jt9.exe path before launching." -ForegroundColor Yellow
} else {
    Write-Host "jt9: $jt9Path" -ForegroundColor Green
}

# ── 6. Resolve installed version (single source of truth) ──────────────
$Map144Version = & $VenvPython -c "from map144_app import __version__; print(__version__)" 2>$null
if (-not $Map144Version) { $Map144Version = "???" }

# ── 7. Done ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "MAP144 v$Map144Version ready." -ForegroundColor Green
Write-Host "To run:"
Write-Host "    cd $RepoDir"
Write-Host "    .\run.bat"
Write-Host ""
Write-Host "The first launch may take ~10 s while numba JIT-compiles its hot paths."
