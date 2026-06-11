@echo off
REM install-b210.bat — double-click entry point for the MAP144 B210 kit.
REM Runs install-b210.ps1 with execution-policy bypass; the .ps1 self-elevates
REM to admin via UAC.  Always pauses at the end so the operator can read the
REM READY/FAILED line before the window closes.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-b210.ps1" %*
set "INSTALL_RC=%ERRORLEVEL%"

echo.
if "%INSTALL_RC%"=="0" (
    echo Installer exited successfully.
) else (
    echo Installer exited with code %INSTALL_RC%.
)
echo.
pause
exit /b %INSTALL_RC%
