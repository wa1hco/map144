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
# Prefer 3.12 / 3.13.  requirements.txt pins numpy<2 (UHD ABI on Linux);
# NumPy 1.26 has no cp314 wheels, so Python 3.14 makes pip try to compile
# numpy from source and fails without a full MSVC build stack.
#
# Refuse the Microsoft Store stub.  On a fresh Win11 install ``python``
# resolves to ``...\WindowsApps\python.exe`` which launches the Store
# instead of running anything.

function Test-PythonCandidate([string] $Exe) {
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) { return $null }
    if ($Exe -like '*WindowsApps*') { return $null }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $line = & $Exe -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2])); print(sys.executable)" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $line) { return $null }
        $parts = @($line | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
        if ($parts.Count -lt 1) { return $null }
        $ver = $parts[0]
        $maj, $min, $pat = $ver.Split('.')
        if ([int]$maj -ne 3) { return $null }
        if ([int]$min -lt 10 -or [int]$min -ge 14) { return $null }
        $resolved = $Exe
        if ($parts.Count -ge 2 -and (Test-Path -LiteralPath $parts[1])) { $resolved = $parts[1] }
        if ($resolved -like '*WindowsApps*') { return $null }
        return @{ Exe = $resolved; Version = $ver; Minor = [int]$min }
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $prev
    }
}

$pythonInfo = $null
# Prefer the py launcher pins (3.12, then 3.13, then 3.11, then 3.10).
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    foreach ($tag in @('-3.12', '-3.13', '-3.11', '-3.10')) {
        try {
            $out = & py $tag -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $pythonInfo = Test-PythonCandidate $out.Trim()
                if ($pythonInfo) { break }
            }
        } catch { }
    }
}
if (-not $pythonInfo) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $pythonInfo = Test-PythonCandidate $cmd.Source }
}
if (-not $pythonInfo) {
    $cmd3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($cmd3) { $pythonInfo = Test-PythonCandidate $cmd3.Source }
}

if (-not $pythonInfo) {
    # Explain the common 3.14 failure explicitly.
    $anyPy = Get-Command python -ErrorAction SilentlyContinue
    if ($anyPy -and $anyPy.Source -notlike '*WindowsApps*') {
        $anyVer = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($anyVer -match '^3\.14') {
            Write-Host "ERROR: Python $anyVer found, but MAP144 needs Python 3.10-3.13 on Windows." -ForegroundColor Red
            Write-Host "  requirements pin numpy<2 (no binary wheels for 3.14), so pip tries to" -ForegroundColor Red
            Write-Host "  compile numpy from source and fails." -ForegroundColor Red
            Write-Host "" -ForegroundColor Red
            Write-Host "  Fix: install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Yellow
            Write-Host "       (check 'Add python.exe to PATH'), then:" -ForegroundColor Yellow
            Write-Host "         rmdir /s /q .venv" -ForegroundColor Yellow
            Write-Host "         .\install.ps1" -ForegroundColor Yellow
            exit 1
        }
    }
    Write-Host "ERROR: no suitable Python 3.10-3.13 found on PATH." -ForegroundColor Red
    Write-Host "  Install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  (the python.org installer, NOT the Microsoft Store one)." -ForegroundColor Red
    exit 1
}

$PythonExe = $pythonInfo.Exe
Write-Host "Python $($pythonInfo.Version) at $PythonExe" -ForegroundColor Green

# ── 2. Create venv if missing (idempotent) ──────────────────────────────
$VenvDir = Join-Path $RepoDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $VenvPython) {
    # If an existing venv was built with 3.14+, recreate (no numpy<2 wheels).
    $venvVer = & $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($venvVer) {
        $vm, $vn = $venvVer.Split('.')
        if ([int]$vm -eq 3 -and [int]$vn -ge 14) {
            Write-Host "Existing .venv is Python $venvVer (unsupported for numpy<2). Recreating ..." -ForegroundColor Yellow
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
    }
}
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating venv at $VenvDir ..." -ForegroundColor Cyan
    & $PythonExe -m venv $VenvDir
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
