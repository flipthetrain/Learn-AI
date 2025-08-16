#!/bin/bash
# Setup script for Python virtual environment and dependencies (Linux/macOS)
# Run once before using any example scripts:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# ── System dependencies (Linux/apt only) ──────────────────────────────────────
if command -v apt &>/dev/null; then
    echo "Checking system dependencies..."
    sudo apt install -y \
        python3-dev python3-venv \
        libgl1 \
        libsndfile1-dev \
        graphviz \
        ffmpeg \
        pkg-config \
        libpango1.0-dev \
        libcairo2-dev
fi

# ── Virtual environment ────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

# ── Python packages ────────────────────────────────────────────────────────────
pip install \
    numpy pandas scipy matplotlib seaborn plotly \
    scikit-learn xgboost lightgbm catboost \
    torch torchvision \
    tensorflow keras tf-keras \
    transformers sentence-transformers \
    openai anthropic \
    boto3 \
    azure-ai-textanalytics azure-identity \
    google-cloud-aiplatform \
    requests \
    Pillow opencv-python \
    librosa soundfile moviepy \
    graphviz ipywidgets \
    faiss-cpu \
    peft trl datasets accelerate bitsandbytes \
    manim

echo
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo
