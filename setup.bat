@echo off
REM Earshot setup — models + caches live on D: (C: is nearly full).
setlocal
set MODEL_DIR=D:\earshot\models
set HF_HUB_CACHE=D:\earshot\models\hf-hub

mkdir "%MODEL_DIR%" 2>nul
mkdir "%HF_HUB_CACHE%" 2>nul

where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo [earshot] WARNING: nvidia-smi not found — GPU inference will fail.
) else (
  nvidia-smi --query-gpu=name,driver_version --format=csv
)

if not exist ".venv\Scripts\python.exe" (
  echo [earshot] creating venv…
  py -3.13 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

echo.
echo [earshot] CUDA DLLs (cuBLAS + cuDNN 9 for CUDA 12) must be on PATH.
echo [earshot] If model load fails, install the whisper-standalone-win pack.
echo [earshot] Model dir: %MODEL_DIR%  (override: --model-dir / EARSHOT_MODEL_DIR)
echo [earshot] Probe audio: python -u backend\captions.py --list-devices
endlocal
