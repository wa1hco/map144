@echo off
REM run.bat — launch MAP144 from its venv (Windows).
REM Re-runnable; passes any args straight through to map144.py.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found.  Run install.ps1 first. 1>&2
    exit /b 1
)
".venv\Scripts\python.exe" map144.py %*
