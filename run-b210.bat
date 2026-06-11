@echo off
REM run-b210.bat — launch MAP144 with USRP B210 support on Windows.
REM
REM Auto-detects the Python env in this order:
REM   1. <script-dir>\env\python.exe   <- kit-installed portable env (preferred)
REM   2. %USERPROFILE%\miniconda3\envs\map144\python.exe   <- dev-machine conda env
REM Set UHD_IMAGES_DIR by looking for the Ettus install dir.
REM
REM See docs/ALPHA_NOTES.md for setup; tools/install-b210.ps1 for kit installer.

setlocal
cd /d "%~dp0"

REM --- Pick a Python env -------------------------------------------------------
set "ENV_PY="
if exist "%~dp0env\python.exe" (
    set "ENV_PY=%~dp0env\python.exe"
    set "ENV_DIR=%~dp0env"
    goto :env_found
)
if exist "%USERPROFILE%\miniconda3\envs\map144\python.exe" (
    set "ENV_PY=%USERPROFILE%\miniconda3\envs\map144\python.exe"
    set "ENV_DIR=%USERPROFILE%\miniconda3\envs\map144"
    goto :env_found
)
echo ERROR: no B210-capable Python env found. 1>&2
echo   Looked for: 1>&2
echo     %~dp0env\python.exe 1>&2
echo     %USERPROFILE%\miniconda3\envs\map144\python.exe 1>&2
echo   Install via tools\install-b210.ps1 or follow docs\ALPHA_NOTES.md. 1>&2
exit /b 1

:env_found

REM --- Put the env's DLL dirs on PATH (what 'conda activate' does) --------------
REM Running python.exe directly does NOT add these, so MKL/OpenBLAS and other
REM delay-loaded native DLLs in Library\bin are not found -> numpy/scipy LAPACK
REM (e.g. the audio-BPF filtfilt) crashes the process with 0xC06D007F
REM (ERROR_MOD_NOT_FOUND).  Prepend the same dirs conda activation would.
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%PATH%"

REM --- Locate Ettus UHD images dir --------------------------------------------
set "UHD_IMAGES_DIR="
if exist "C:\Program Files\UHD\share\uhd\images\usrp_b210_fpga.bin" (
    set "UHD_IMAGES_DIR=C:\Program Files\UHD\share\uhd\images"
) else if exist "C:\Program Files (x86)\UHD\share\uhd\images\usrp_b210_fpga.bin" (
    set "UHD_IMAGES_DIR=C:\Program Files (x86)\UHD\share\uhd\images"
)
if "%UHD_IMAGES_DIR%"=="" (
    echo WARNING: Ettus UHD images directory not found at standard locations. 1>&2
    echo   UHD may fail to load FX3 firmware into the B210. 1>&2
)

REM --- Pre-flight: B210 visible? (warn only) ----------------------------------
if exist "C:\Program Files\UHD\bin\uhd_find_devices.exe" (
    "C:\Program Files\UHD\bin\uhd_find_devices.exe" 2>nul | findstr /C:"B210" >nul
    if errorlevel 1 (
        echo WARNING: uhd_find_devices does not see a B210. 1>&2
        echo   Check USB cable (USB 3.0 port, blue^) and Device Manager. 1>&2
    )
)

REM --- Launch -----------------------------------------------------------------
"%ENV_PY%" map144.py %*
