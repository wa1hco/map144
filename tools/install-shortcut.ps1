<#
install-shortcut.ps1 — create a Start Menu shortcut so MAP144 can be launched
by typing its name into the Windows search bar.

Defaults to an ALL-USERS shortcut (visible to every account on the PC,
including a passwordless contest user), which requires admin — the script
self-elevates via UAC.  Pass -CurrentUserOnly for a per-user shortcut (no
elevation).

The shortcut targets run-b210.bat, which auto-selects the portable env\ in the
install folder (no conda, no user-profile dependency).

Usage:
    # all-users shortcut to the C:\map144 install (default):
    powershell -ExecutionPolicy Bypass -File tools\install-shortcut.ps1 -Launcher C:\map144\run-b210.bat

    # per-user, targeting whatever repo this script lives in:
    powershell -ExecutionPolicy Bypass -File tools\install-shortcut.ps1 -CurrentUserOnly
#>

[CmdletBinding()]
param(
    # Launcher the shortcut points at.  Default: run-b210.bat one level up from
    # this script (i.e. the root of the install/repo this script ships in).
    [string]$Launcher = (Join-Path (Split-Path -Parent $PSScriptRoot) 'run-b210.bat'),
    [string]$Name = 'map144',
    [switch]$CurrentUserOnly
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Launcher)) {
    Write-Error "Launcher not found: $Launcher"
}
$Launcher = (Resolve-Path -LiteralPath $Launcher).Path
$workDir  = Split-Path -Parent $Launcher

# All-users shortcut writes to the common Start Menu -> needs admin.  Self-elevate.
if (-not $CurrentUserOnly) {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Host "Requesting administrator privileges (all-users shortcut)..." -ForegroundColor Yellow
        $argList = @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSCommandPath`"",
                     '-Launcher',"`"$Launcher`"",'-Name',"`"$Name`"")
        Start-Process powershell -ArgumentList $argList -Verb RunAs
        exit
    }
}

$folderKey = if ($CurrentUserOnly) { 'Programs' } else { 'CommonPrograms' }
$programs  = [Environment]::GetFolderPath($folderKey)
$lnkPath   = Join-Path $programs "$Name.lnk"

$ws = New-Object -ComObject WScript.Shell

# Clean up any other shortcut in this folder that points at the same launcher
# (e.g. a previously-named map144-b210.lnk), so a rename leaves one canonical entry.
Get-ChildItem -LiteralPath $programs -Filter '*.lnk' -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.FullName -ne $lnkPath) {
        try {
            if ($ws.CreateShortcut($_.FullName).TargetPath -eq $Launcher) {
                Remove-Item -LiteralPath $_.FullName -Force
                Write-Host "  removed old shortcut: $($_.Name)" -ForegroundColor DarkYellow
            }
        } catch { }
    }
}

$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath       = $Launcher
$lnk.WorkingDirectory = $workDir
$lnk.Description       = 'Launch MAP144 (USRP B210)'
$lnk.WindowStyle      = 1

$ico = Join-Path $workDir 'map144_app\data\map144.ico'
if (Test-Path -LiteralPath $ico) { $lnk.IconLocation = $ico }

$lnk.Save()

$scope = if ($CurrentUserOnly) { 'current user' } else { 'all users' }
Write-Host ""
Write-Host "Created Start Menu shortcut ($scope):" -ForegroundColor Green
Write-Host "  $lnkPath"
Write-Host "  -> $Launcher"
Write-Host ""
Write-Host "Type `"$Name`" in the Windows search bar to launch." -ForegroundColor Cyan
Write-Host "(Search may take a few seconds to index the new shortcut.)"
