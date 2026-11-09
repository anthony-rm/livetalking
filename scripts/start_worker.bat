@echo off
REM Start the LiveTalking remote GPU worker on this Windows PC.
REM See docs/remote-gpu-pc-setup.md for the one-time setup.
REM
REM Double-click to run. Keep this window open while streaming from the Mac.
REM
REM This deliberately does NOT call "conda activate": invoking the env's
REM python.exe directly is more reliable than activation, and it avoids the
REM classic "|| after call" pitfall where errorlevel does not reflect whether
REM activation actually succeeded.

setlocal

REM --- adjust these if your paths differ -----------------------------------
set REPO=C:\work\livetalking
set CONDA_ENV=livetalking
set WORKER_PORT=50051
set WORKER_BACKEND=musetalk
REM Optional: set PY to skip auto-detection, e.g.
REM   set PY=C:\Users\you\miniconda3\envs\livetalking\python.exe
REM ------------------------------------------------------------------------

REM Worker MUST run from the repo root: avatars/musetalk/utils/__init__.py
REM appends to sys.path and can shadow the project's `utils` package otherwise.
cd /d "%REPO%"
if errorlevel 1 (
    echo ERROR: repo not found at %REPO%
    echo Edit this file and fix the REPO variable.
    pause
    exit /b 1
)

if not defined PY (
    for /f "delims=" %%i in ('conda info --base 2^>nul') do set CONDA_BASE=%%i
)
if not defined PY if defined CONDA_BASE set PY=%CONDA_BASE%\envs\%CONDA_ENV%\python.exe
if not defined PY (
    echo ERROR: could not locate conda. Install Miniconda, or set PY at the
    echo top of this file to the full path of the env's python.exe.
    pause
    exit /b 1
)
if not exist "%PY%" (
    echo ERROR: python not found at %PY%
    echo Run "conda env list" and fix CONDA_ENV / PY in this file.
    pause
    exit /b 1
)

echo Using: %PY%

"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo ERROR: torch.cuda.is_available^(^) is False for this interpreter.
    echo Reinstall the CUDA build:
    echo   "%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pause
    exit /b 1
)

echo Starting GPU worker on port %WORKER_PORT% (backend=%WORKER_BACKEND%)...
"%PY%" -m remote.worker --host 0.0.0.0 --port %WORKER_PORT% --backend %WORKER_BACKEND%

echo.
echo Worker stopped. Press any key to close.
pause >nul
