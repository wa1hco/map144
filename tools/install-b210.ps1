#requires -Version 5.1
# install-b210.ps1 -- MAP144 B210 install from kit on a clean Windows PC.
#
# Operator workflow:
#   1. Plug in the USB stick (or unzip the downloaded kit).
#   2. Double-click install-b210.bat.
#   3. Approve UAC.  Wait ~5 minutes.  Read READY / FAILED line at end.
#   4. Double-click C:\map144\run-b210.bat to launch.
#
# What this script does (all from local files in the kit, no internet):
#   1. Self-elevate to admin via UAC if needed
#   2. Pre-flight (Windows version, disk space, not under OneDrive)
#   3. Install Ettus UHD silently (skipped if already installed)
#   4. Extract portable Python env to <InstallPath>\env\
#   5. Run conda-unpack.exe to fix env paths
#   6. Extract map144 source to <InstallPath>\
#   7. Smoke tests: uhd_find_devices sees the B210; import uhd works
#   8. Print READY or FAILED:<reason>
#
# Idempotent: re-running with the same kit version is a no-op.  A different
# kit version prompts for upgrade (or pass -Force).

[CmdletBinding()]
param(
    [string]$InstallPath = 'C:\map144',
    [switch]$Force,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = 'Stop'
$KitDir = $PSScriptRoot
$Script:LogFile = $null      # set after install path resolves

function Write-Step([string]$msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
    if ($Script:LogFile) { Add-Content $Script:LogFile "==> $msg" }
}
function Write-Ok([string]$msg) {
    Write-Host "    OK: $msg" -ForegroundColor Green
    if ($Script:LogFile) { Add-Content $Script:LogFile "    OK: $msg" }
}
function Write-Warn2([string]$msg) {
    Write-Host "    WARN: $msg" -ForegroundColor Yellow
    if ($Script:LogFile) { Add-Content $Script:LogFile "    WARN: $msg" }
}
function Fail([string]$reason) {
    Write-Host ""
    Write-Host "FAILED: $reason" -ForegroundColor Red
    if ($Script:LogFile) { Add-Content $Script:LogFile "FAILED: $reason" }
    Write-Host ""
    Write-Host "Press any key to close..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
    exit 1
}

# -- 0. Self-elevate ----------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
                 '-InstallPath', "`"$InstallPath`"")
    if ($Force) { $argList += '-Force' }
    if ($SkipSmokeTest) { $argList += '-SkipSmokeTest' }
    Start-Process powershell -ArgumentList $argList -Verb RunAs
    exit
}

# -- 1. Header & VERSION.txt --------------------------------------------------
$VersionFile = Join-Path $KitDir 'VERSION.txt'
if (-not (Test-Path $VersionFile)) {
    Fail "VERSION.txt missing from kit at $KitDir -- kit is incomplete."
}
$kitInfo = Get-Content $VersionFile -Raw
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  MAP144 B210 Kit installer" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host $kitInfo
Write-Host ""

# -- 2. Pre-flight ------------------------------------------------------------
Write-Step "Pre-flight checks"

# Windows version (need Win10 1809+ or Win11 for tar.exe + modern WinUSB)
$winVer = [System.Environment]::OSVersion.Version
if ($winVer.Build -lt 17763) {
    Fail "Windows build $($winVer.Build) is too old (need Win10 1809+ or Win11)."
}
Write-Ok "Windows build $($winVer.Build)"

# Disk space (need at least 3 GB free on the install drive)
$installDrive = ([System.IO.Path]::GetPathRoot($InstallPath)).TrimEnd('\')
$freeGB = [math]::Round((Get-PSDrive ($installDrive.TrimEnd(':'))).Free / 1GB, 1)
if ($freeGB -lt 3) {
    Fail "Only $freeGB GB free on ${installDrive} -- need 3 GB."
}
Write-Ok "$freeGB GB free on ${installDrive}"

# Not under OneDrive
if ($InstallPath -match 'OneDrive') {
    Fail "InstallPath $InstallPath is under OneDrive -- choose a non-synced path " +
         "(e.g. C:\map144) because MAP144 writes WAVs/logs continuously."
}
Write-Ok "InstallPath $InstallPath is not under OneDrive"

# -- 3. Check existing install ------------------------------------------------
$existingVersionFile = Join-Path $InstallPath 'VERSION.txt'
if (Test-Path $existingVersionFile) {
    $existing = (Get-Content $existingVersionFile -Raw).Trim()
    $kit      = $kitInfo.Trim()
    if ($existing -eq $kit -and -not $Force) {
        Write-Host ""
        Write-Host "ALREADY INSTALLED at this version. Nothing to do." -ForegroundColor Green
        Write-Host "(Pass -Force to reinstall.)" -ForegroundColor Yellow
        Write-Host "Launch with: $InstallPath\run-b210.bat" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Press any key to close..." -ForegroundColor Yellow
        $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
        exit 0
    }
    if (-not $Force) {
        Write-Warn2 "Existing install found:"
        Write-Host $existing
        Write-Host ""
        $resp = Read-Host "Overwrite with kit version above? [y/N]"
        if ($resp -ne 'y' -and $resp -ne 'Y') {
            Fail "Cancelled by operator."
        }
    }
    Write-Step "Removing existing install at $InstallPath"
    Remove-Item -Recurse -Force $InstallPath
}

# -- 4. Create install dir + start log ----------------------------------------
New-Item -ItemType Directory -Force -Path $InstallPath | Out-Null
$Script:LogFile = Join-Path $InstallPath 'install.log'
"--- install-b210.ps1 run at $(Get-Date -Format 'o') ---" | Out-File $Script:LogFile -Encoding utf8
Add-Content $Script:LogFile "Kit info:`n$kitInfo"

# -- 5. Install Ettus UHD -----------------------------------------------------
Write-Step "Ettus UHD"

$uhdFindExe = 'C:\Program Files\UHD\bin\uhd_find_devices.exe'
if (Test-Path $uhdFindExe) {
    Write-Ok "Already installed at C:\Program Files\UHD"
} else {
    $ettusInstaller = Get-ChildItem (Join-Path $KitDir 'ettus') -Filter '*.exe' |
        Select-Object -First 1
    if (-not $ettusInstaller) {
        Fail "No Ettus installer found in $KitDir\ettus\"
    }
    Write-Host "    Running $($ettusInstaller.Name) silently (~30 s)..."
    $p = Start-Process -FilePath $ettusInstaller.FullName -ArgumentList '/S' -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        Fail "Ettus installer exit code $($p.ExitCode).  Try running it manually: $($ettusInstaller.FullName)"
    }
    if (-not (Test-Path $uhdFindExe)) {
        Fail "Ettus install completed but uhd_find_devices.exe not at $uhdFindExe"
    }
    Write-Ok "Installed to C:\Program Files\UHD"
}

# -- 6. Extract portable Python env -------------------------------------------
Write-Step "Extracting Python env"

$envZip = Join-Path $KitDir 'map144-env.zip'
$envDir = Join-Path $InstallPath 'env'
if (-not (Test-Path $envZip)) { Fail "map144-env.zip missing from kit" }

New-Item -ItemType Directory -Force -Path $envDir | Out-Null
# tar.exe (bundled with Win10 1809+) handles zip and is faster than Expand-Archive
& tar -xf $envZip -C $envDir
if ($LASTEXITCODE -ne 0) { Fail "tar extract failed (env zip)" }
Write-Ok "Extracted to $envDir"

# -- 7. conda-unpack ----------------------------------------------------------
Write-Step "Fixing env paths (conda-unpack -- can take 5-10 min, no progress output)"

$unpacker = Join-Path $envDir 'Scripts\conda-unpack.exe'
if (-not (Test-Path $unpacker)) { Fail "conda-unpack.exe missing from env at $unpacker" }
& $unpacker
if ($LASTEXITCODE -ne 0) { Fail "conda-unpack failed (exit $LASTEXITCODE)" }
Write-Ok "Env paths rewritten"

# -- 8. Extract map144 source -------------------------------------------------
Write-Step "Extracting map144 source"

$srcZip = Join-Path $KitDir 'map144-src.zip'
if (-not (Test-Path $srcZip)) { Fail "map144-src.zip missing from kit" }
& tar -xf $srcZip -C $InstallPath
if ($LASTEXITCODE -ne 0) { Fail "tar extract failed (src zip)" }
Write-Ok "Extracted to $InstallPath"

# -- 9. Write VERSION.txt at install root -------------------------------------
Copy-Item $VersionFile (Join-Path $InstallPath 'VERSION.txt')

# -- 10. Smoke tests ----------------------------------------------------------
if (-not $SkipSmokeTest) {
    Write-Step "Smoke tests"

    # 10a -- uhd_find_devices should see the B210 (warn-only -- operator may not have
    # plugged it in yet).
    Write-Host "    Probing for B210 via uhd_find_devices..."
    $probe = & $uhdFindExe 2>&1 | Out-String
    Add-Content $Script:LogFile "uhd_find_devices output:`n$probe"
    if ($probe -match 'B210') {
        Write-Ok "B210 visible to UHD"
    } else {
        Write-Warn2 "B210 not detected.  Plug in B210 + USB 3.0 cable before launching map144."
        Write-Host "    (Full output logged to $Script:LogFile)"
    }

    # 10b -- env's python can import uhd
    Write-Host "    Verifying env python + import uhd..."
    $env:UHD_IMAGES_DIR = 'C:\Program Files\UHD\share\uhd\images'
    $envPython = Join-Path $envDir 'python.exe'
    if (-not (Test-Path $envPython)) { Fail "Env python missing at $envPython" }
    $uhdVer = & $envPython -c "import uhd; print(uhd.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "env python -c 'import uhd' failed: $uhdVer" }
    Write-Ok "uhd $uhdVer importable from env"
}

# -- 10.7 Start Menu shortcut (all users) -------------------------------------
# Created here in the installer's already-elevated context, so no extra UAC.
# Named "map144" (no -b210 suffix) so it launches by typing map144 in search.
Write-Step "Start Menu shortcut"
try {
    $programs = [Environment]::GetFolderPath('CommonPrograms')
    $lnkPath  = Join-Path $programs 'map144.lnk'
    $target   = Join-Path $InstallPath 'run-b210.bat'

    $wsh = New-Object -ComObject WScript.Shell
    # Remove any stale shortcut here that points at this install's launcher
    # (e.g. an older map144-b210.lnk), so we end up with one canonical entry.
    Get-ChildItem -LiteralPath $programs -Filter '*.lnk' -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.FullName -ne $lnkPath) {
            try { if ($wsh.CreateShortcut($_.FullName).TargetPath -eq $target) { Remove-Item -LiteralPath $_.FullName -Force } } catch { }
        }
    }
    $sc = $wsh.CreateShortcut($lnkPath)
    $sc.TargetPath       = $target
    $sc.WorkingDirectory = $InstallPath
    $sc.Description       = 'Launch MAP144 (USRP B210)'
    $sc.WindowStyle      = 1
    $ico = Join-Path $InstallPath 'map144_app\data\map144.ico'
    if (Test-Path -LiteralPath $ico) { $sc.IconLocation = $ico }
    $sc.Save()
    Write-Ok "Created '$lnkPath' (type 'map144' in Start search to launch)"
} catch {
    Write-Warn2 "Could not create Start Menu shortcut: $($_.Exception.Message)"
}

# -- 11. Done -----------------------------------------------------------------
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Green
Write-Host "  READY" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Launch MAP144 with the B210:" -ForegroundColor Cyan
Write-Host "  - Type 'map144' in the Windows search bar, or" -ForegroundColor White
Write-Host "  - Run $InstallPath\run-b210.bat" -ForegroundColor White
Write-Host ""
Write-Host "Install log:  $Script:LogFile"
Write-Host ""
Write-Host "Press any key to close..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
