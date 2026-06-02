@echo off
REM run-b210.bat — launch MAP144 with USRP B210 support on Windows.
REM
REM Uses the 'map144' conda env (NOT the .venv pip venv), because the UHD
REM Python bindings are not pip-installable on Windows.  See
REM docs/ALPHA_NOTES.md for the one-time setup.
REM
REM If your install differs from the defaults, edit the two paths below:
REM   - Miniconda installed somewhere other than %USERPROFILE%\miniconda3
REM   - Ettus UHD installed somewhere other than "C:\Program Files\UHD"

set "UHD_IMAGES_DIR=C:\Program Files\UHD\share\uhd\images"

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" map144
if errorlevel 1 (
    echo ERROR: failed to activate conda env 'map144'. 1>&2
    echo   Expected Miniconda at %USERPROFILE%\miniconda3 1>&2
    echo   and conda env 'map144' created per docs/ALPHA_NOTES.md. 1>&2
    exit /b 1
)

cd /d "%~dp0"
python map144.py %*
