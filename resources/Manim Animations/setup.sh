#!/bin/bash
# Setup script for Manim animations (Linux/macOS)
# Run once before rendering any animation:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# ── System dependencies (Linux/apt only) ──────────────────────────────────────
if command -v apt &>/dev/null; then
    echo "Installing system dependencies..."
    sudo apt install -y \
        python3-dev python3-venv \
        pkg-config \
        libpango1.0-dev \
        libcairo2-dev \
        ffmpeg \
        texlive texlive-latex-extra texlive-fonts-extra
elif command -v brew &>/dev/null; then
    echo "Installing system dependencies via Homebrew..."
    brew install ffmpeg pango cairo pkg-config
    brew install --cask mactex-no-gui
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
    manim \
    numpy Pillow scipy colour tqdm requests \
    rich click cloup \
    pycairo manimpango \
    pygments isosurfaces moderngl screeninfo \
    gtts mutagen

echo
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Render an animation:"
echo "  manim -ql lora_rank_decomposition.py LoRARankDecomposition"
echo
