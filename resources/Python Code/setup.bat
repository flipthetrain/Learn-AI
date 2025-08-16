@echo off
REM Setup script for Python virtual environment and dependencies (Windows)
REM Run once before using any example scripts:  setup.bat
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
    numpy pandas scipy matplotlib seaborn plotly ^
    scikit-learn xgboost lightgbm catboost ^
    torch torchvision ^
    tensorflow keras tf-keras ^
    transformers sentence-transformers ^
    openai anthropic ^
    boto3 ^
    azure-ai-textanalytics azure-identity ^
    google-cloud-aiplatform ^
    requests ^
    Pillow opencv-python ^
    librosa soundfile moviepy ^
    graphviz ipywidgets ^
    faiss-cpu ^
    peft trl datasets accelerate bitsandbytes ^
    manim

REM NOTE: manim also requires ffmpeg and LaTeX installed on your system PATH.
REM   ffmpeg : https://ffmpeg.org/download.html
REM   LaTeX  : https://miktex.org/download

echo.
echo Setup complete. Activate with:
echo   call .venv\Scripts\activate.bat
echo.
