@echo off
REM build-b210-kit.bat — double-click entry point for the kit builder.
REM
REM Auto-finds Miniconda/Anaconda, prepends it to PATH so the .ps1 can
REM call 'conda' bare, runs the .ps1 with execution-policy bypass, and
REM pauses at the end so any errors stay on screen.
REM
REM Run this on a HOME machine with Miniconda + git installed.

setlocal

REM --- Locate Miniconda / Anaconda --------------------------------------------
set "CONDA_DIR="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_DIR=%USERPROFILE%\miniconda3"
if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" set "CONDA_DIR=%USERPROFILE%\Miniconda3"
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe"  set "CONDA_DIR=%USERPROFILE%\anaconda3"
if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_DIR=C:\ProgramData\miniconda3"
if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe"  set "CONDA_DIR=C:\ProgramData\Anaconda3"

if "%CONDA_DIR%"=="" (
    echo ERROR: Miniconda/Anaconda not found in standard locations. 1>&2
    echo   Searched: 1>&2
    echo     %%USERPROFILE%%\miniconda3 1>&2
    echo     %%USERPROFILE%%\Miniconda3 1>&2
    echo     %%USERPROFILE%%\anaconda3 1>&2
    echo     C:\ProgramData\miniconda3 1>&2
    echo     C:\ProgramData\Anaconda3 1>&2
    echo   Install Miniconda from https://docs.conda.io/projects/miniconda/ 1>&2
    echo   then re-run this script. 1>&2
    pause
    exit /b 1
)

set "PATH=%CONDA_DIR%;%CONDA_DIR%\Scripts;%CONDA_DIR%\Library\bin;%PATH%"
echo Using conda from: %CONDA_DIR%
echo.

REM --- Run the builder --------------------------------------------------------
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-b210-kit.ps1" %*
set "BUILD_RC=%ERRORLEVEL%"

echo.
if "%BUILD_RC%"=="0" (
    echo Build finished successfully.
) else (
    echo Build failed with exit code %BUILD_RC%.
)
echo.
pause
exit /b %BUILD_RC%
