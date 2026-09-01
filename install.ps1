# install.ps1 - MAP144 install / update for Windows.
#
# Run from inside the cloned repo (no auto-clone - gives you full
# control of where the working tree lives):
#
#     git clone https://github.com/wa1hco/map144.git C:\WSJT\map144
#     cd C:\WSJT\map144
#     .\install.ps1
#
# Re-runnable: existing venv is reused when compatible, requirements
# re-resolved.  Uses the venv's python.exe directly so there are no
# PowerShell execution-policy concerns - no activate step required.
#
# Python / NumPy policy
# --------------------
# Prefer Python 3.14, then 3.13 -> 3.12 -> 3.11 -> 3.10.
# requirements.txt pins numpy==1.26.4 (UHD ABI).  Official PyPI builds of
# 1.26.4 have no cp314 wheels, so a 3.14 venv often fails at pip install.
# In that case we tear down .venv and retry with the next older Python.
#
# Also installs Ettus UHD (WinUSB + B210 FPGA/FX3 firmware images) via
# tools\Install-UhdWindows.ps1 unless -SkipUhd is passed.  That step needs
# admin + internet the first time (~233 MB).  Python `import uhd` still
# requires the B210 kit env\ or conda-forge uhd (no Windows pip wheels).

param(
    [switch]$SkipUhd
)

$ErrorActionPreference = 'Stop'

# Always operate relative to the script's own directory so PowerShell's
# (frequently-resetting) cwd never confuses us.
$RepoDir = (Resolve-Path $PSScriptRoot).Path
Set-Location $RepoDir

# -- 1. Discover usable Python interpreters --------------------
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
        # Accept 3.10 .. 3.14 inclusive.
        if ([int]$min -lt 10 -or [int]$min -gt 14) { return $null }
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

function Get-PythonCandidates {
    $found = @()
    $seen = @{}
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    # Prefer newest first: 3.14 -> 3.10
    foreach ($tag in @('-3.14', '-3.13', '-3.12', '-3.11', '-3.10')) {
        if (-not $pyLauncher) { break }
        try {
            $out = & py $tag -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $info = Test-PythonCandidate $out.Trim()
                if ($info -and -not $seen.ContainsKey($info.Exe)) {
                    $seen[$info.Exe] = $true
                    $found += $info
                }
            }
        } catch { }
    }
    foreach ($name in @('python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $info = Test-PythonCandidate $cmd.Source
        if ($info -and -not $seen.ContainsKey($info.Exe)) {
            $seen[$info.Exe] = $true
            $found += $info
        }
    }
    # Stable preference: higher minor first, then keep discovery order.
    return @($found | Sort-Object -Property Minor -Descending)
}

$candidates = @(Get-PythonCandidates)
if ($candidates.Count -eq 0) {
    Write-Host 'ERROR: no suitable Python 3.10-3.14 found on PATH.' -ForegroundColor Red
    Write-Host '  Install Python 3.14 or 3.12 from https://www.python.org/downloads/' -ForegroundColor Red
    Write-Host '  (the python.org installer, NOT the Microsoft Store one).' -ForegroundColor Red
    Write-Host '  Note: numpy==1.26.4 has no official 3.14 wheels - if 3.14 fails,' -ForegroundColor Yellow
    Write-Host '  install.ps1 will automatically retry with 3.13/3.12.' -ForegroundColor Yellow
    exit 1
}

Write-Host "Python candidates (preferred first):" -ForegroundColor Cyan
foreach ($c in $candidates) {
    Write-Host ("  {0}  {1}" -f $c.Version, $c.Exe)
}

# -- 2-4. Create venv + install requirements (with Python fallback) ------
$VenvDir = Join-Path $RepoDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ReqFile = Join-Path $RepoDir "requirements.txt"
$installed = $false
$lastError = ""

foreach ($pythonInfo in $candidates) {
    $PythonExe = $pythonInfo.Exe
    Write-Host ""
    Write-Host "Trying Python $($pythonInfo.Version) at $PythonExe ..." -ForegroundColor Cyan

    # Drop an incompatible existing venv (different major.minor).
    if (Test-Path $VenvPython) {
        $venvVer = & $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        $wantVer = '{0}.{1}' -f 3, $pythonInfo.Minor
        if ($venvVer -and $venvVer -ne $wantVer) {
            Write-Host "Existing .venv is Python $venvVer; recreating for $wantVer ..." -ForegroundColor Yellow
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating venv at $VenvDir ..." -ForegroundColor Cyan
        & $PythonExe -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            $lastError = "venv creation failed with Python $($pythonInfo.Version)"
            Write-Host "ERROR: $lastError" -ForegroundColor Red
            if (Test-Path $VenvDir) {
                Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
            }
            continue
        }
    }

    Write-Host "Upgrading pip ..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        $lastError = "pip upgrade failed with Python $($pythonInfo.Version)"
        Write-Host "ERROR: $lastError" -ForegroundColor Red
        Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
        continue
    }

    Write-Host "Installing requirements (numpy==1.26.4) ..." -ForegroundColor Cyan
    & $VenvPython -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) {
        $lastError = "pip install failed with Python $($pythonInfo.Version)"
        Write-Host "ERROR: $lastError" -ForegroundColor Red
        if ($pythonInfo.Minor -ge 14) {
            Write-Host "  numpy==1.26.4 typically has no cp314 wheels; falling back to an older Python ..." -ForegroundColor Yellow
        }
        Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
        continue
    }

    # Confirm numpy pin landed.
    $npVer = & $VenvPython -c "import numpy; print(numpy.__version__)" 2>$null
    if (-not $npVer) {
        $lastError = "numpy import failed after install (Python $($pythonInfo.Version))"
        Write-Host "ERROR: $lastError" -ForegroundColor Red
        Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
        continue
    }
    if ($npVer -ne '1.26.4') {
        Write-Host "WARNING: expected numpy 1.26.4, got $npVer" -ForegroundColor Yellow
    } else {
        Write-Host "numpy $npVer OK" -ForegroundColor Green
    }

    $installed = $true
    Write-Host "Using Python $($pythonInfo.Version) + numpy $npVer" -ForegroundColor Green
    break
}

if (-not $installed) {
    Write-Host ""
    Write-Host "ERROR: could not create a working venv." -ForegroundColor Red
    if ($lastError) { Write-Host "  Last failure: $lastError" -ForegroundColor Red }
    Write-Host "  Install Python 3.12 from https://www.python.org/downloads/ and re-run." -ForegroundColor Yellow
    Write-Host "  (3.14 is tried first, but numpy==1.26.4 usually needs 3.10-3.13.)" -ForegroundColor Yellow
    exit 1
}

# -- 5. Ettus UHD + B210 firmware (for router / USRP) -------------------
if (-not $SkipUhd) {
    Write-Host ""
    Write-Host "Installing / verifying Ettus UHD + B210 firmware ..." -ForegroundColor Cyan
    $uhdHelper = Join-Path $RepoDir 'tools\Install-UhdWindows.ps1'
    if (-not (Test-Path -LiteralPath $uhdHelper)) {
        Write-Host "WARNING: $uhdHelper missing; skipping UHD install." -ForegroundColor Yellow
    } else {
        $uhdProc = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$uhdHelper`"") `
            -Wait -PassThru
        if ($uhdProc.ExitCode -ne 0) {
            Write-Host "WARNING: UHD / B210 firmware install failed (exit $($uhdProc.ExitCode))." -ForegroundColor Yellow
            Write-Host "  Router/B210 will not work until UHD is installed." -ForegroundColor Yellow
            Write-Host "  Retry: powershell -ExecutionPolicy Bypass -File tools\Install-UhdWindows.ps1" -ForegroundColor Yellow
            Write-Host "  Or use the offline kit: tools\install-b210.bat" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "Skipping Ettus UHD install (-SkipUhd)." -ForegroundColor Yellow
}

# -- 6. Verify jt9 discovery --------------------------------------------
Write-Host "Verifying jt9 discovery ..." -ForegroundColor Cyan
$jt9Path = & $VenvPython -c "from map144_app.detection import find_jt9; p = find_jt9(); print(p if p else '')"
if (-not $jt9Path) {
    Write-Host "WARNING: jt9 not found." -ForegroundColor Yellow
    Write-Host "  Install WSJT-X from https://wsjt.sourceforge.io/ (provides jt9.exe)," -ForegroundColor Yellow
    Write-Host "  or set MAP144_JT9 to the full jt9.exe path before launching." -ForegroundColor Yellow
} else {
    Write-Host "jt9: $jt9Path" -ForegroundColor Green
}

# -- 7. Resolve installed version (single source of truth) --------------
$Map144Version = & $VenvPython -c "from map144_app import __version__; print(__version__)" 2>$null
if (-not $Map144Version) { $Map144Version = "???" }

# -- 8. Done ------------------------------------------------------------
Write-Host ""
Write-Host "MAP144 v$Map144Version ready." -ForegroundColor Green
Write-Host "To run:"
Write-Host "    cd $RepoDir"
Write-Host "    .\run.bat"
Write-Host "Router (needs UHD firmware above + a Python with 'import uhd'):"
Write-Host "    .\run-router.bat"
Write-Host "  B210 Python bindings: use kit env\ or conda install -c conda-forge uhd"
Write-Host "  (pip .venv alone cannot import uhd on Windows)."
Write-Host ""
Write-Host "The first launch may take ~10 s while numba JIT-compiles its hot paths."
