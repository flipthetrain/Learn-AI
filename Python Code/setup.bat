@echo off
REM Setup script for Python virtual environment and dependencies
cd /d "%~dp0"

REM Create venv if it doesn't exist
if not exist .venv (
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install all required packages
REM (from all scripts in this and child folders)
pip install numpy matplotlib requests openai scikit-learn torch torchvision tensorflow keras tf-keras transformers sentence-transformers xgboost lightgbm catboost pandas scipy seaborn boto3 azure-ai-textanalytics azure-identity anthropic google-cloud-aiplatform plotly graphviz pillow opencv-python librosa soundfile moviepy ipywidgets

echo.
echo Environment setup complete. Activate with:
echo   call .venv\Scripts\activate.bat
echo.
