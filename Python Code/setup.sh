#!/bin/bash
# Setup script for Python virtual environment and dependencies (Linux/Mac)
cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install numpy matplotlib requests openai scikit-learn torch torchvision tensorflow keras tf-keras transformers sentence-transformers xgboost lightgbm catboost pandas scipy seaborn boto3 azure-ai-textanalytics azure-identity anthropic google-cloud-aiplatform plotly graphviz pillow opencv-python librosa soundfile moviepy ipywidgets

echo
echo "Environment setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo
