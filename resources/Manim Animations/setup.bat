@echo off
REM Setup script for Manim animations (Windows)
REM Run once before rendering any animation:  setup.bat
REM
REM FIRST install these manually if not already present:
REM   ffmpeg : https://ffmpeg.org/download.html  (add to PATH)
REM   LaTeX  : https://miktex.org/download       (add to PATH)
cd /d "%~dp0"

REM ── Virtual environment ────────────────────────────────────────────────────
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM ── Python packages ────────────────────────────────────────────────────────
pip install ^
    manim ^
    numpy Pillow scipy colour tqdm requests ^
    rich click cloup ^
    pycairo manimpango ^
    pygments isosurfaces moderngl screeninfo ^
    gtts mutagen

echo.
echo Setup complete. Activate with:
echo   call .venv\Scripts\activate.bat
echo.
echo Render an animation:
echo   manim -ql lora_rank_decomposition.py LoRARankDecomposition
echo.
