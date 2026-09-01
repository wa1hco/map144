@echo off
REM run-router.bat — launch the B210 audio/IQ router on Windows.
REM
REM Auto-detects the Python env in this order (same as run-b210.bat):
REM   1. <script-dir>\env\python.exe   <- kit-installed portable env
REM   2. <script-dir>\.venv\Scripts\python.exe
REM   3. %USERPROFILE%\miniconda3\envs\map144\python.exe
REM
REM WSJT-X audio on Windows needs VB-CABLE (or VoiceMeeter) + sounddevice.
REM See docs\router.md.

setlocal
cd /d "%~dp0"

set "ENV_PY="
if exist "%~dp0env\python.exe" (
    set "ENV_PY=%~dp0env\python.exe"
    set "ENV_DIR=%~dp0env"
    goto :env_found
)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "ENV_PY=%~dp0.venv\Scripts\python.exe"
    set "ENV_DIR=%~dp0.venv"
    goto :env_found
)
if exist "%USERPROFILE%\miniconda3\envs\map144\python.exe" (
    set "ENV_PY=%USERPROFILE%\miniconda3\envs\map144\python.exe"
    set "ENV_DIR=%USERPROFILE%\miniconda3\envs\map144"
    goto :env_found
)
echo ERROR: no B210-capable Python env found. 1>&2
echo   Looked for env\, .venv\, and conda env map144. 1>&2
echo   Install via tools\install-b210.ps1 or follow docs\ALPHA_NOTES.md. 1>&2
exit /b 1

:env_found
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%PATH%"

if exist "C:\Program Files\UHD\share\uhd\images\usrp_b210_fpga.bin" (
    set "UHD_IMAGES_DIR=C:\Program Files\UHD\share\uhd\images"
) else if exist "C:\Program Files (x86)\UHD\share\uhd\images\usrp_b210_fpga.bin" (
    set "UHD_IMAGES_DIR=C:\Program Files (x86)\UHD\share\uhd\images"
)

"%ENV_PY%" router_app.py %*
