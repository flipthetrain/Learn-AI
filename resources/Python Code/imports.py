# This script imports all libraries required by any module in the Python Code folder and subfolders.
# Run this to check/install dependencies for all example scripts and future AI code.

import sys
import os
import subprocess
import shutil
import platform

# ── Venv check ────────────────────────────────────────────────────────────────

in_venv = sys.prefix != sys.base_prefix
if not in_venv:
    print(
        "WARNING: no virtual environment is active.\n"
        "It is strongly recommended to use a venv:\n\n"
        "    python3 -m venv .venv\n"
        "    source .venv/bin/activate   # Linux/macOS\n"
        "    .venv\\Scripts\\activate      # Windows\n"
        "    python imports.py\n\n"
        "Continuing anyway...\n"
    )

# ── System dependency installer ───────────────────────────────────────────────

# Maps apt package → what to check to know if it's already present
SYSTEM_DEPS = {
    "libgl1":          ("ldconfig", "-p", "libGL"),     # opencv-python
    "libsndfile1-dev": ("pkg-config", "--exists", "sndfile"),  # librosa / soundfile
    "graphviz":        ("which", "dot"),                # graphviz binary
    "ffmpeg":          ("which", "ffmpeg"),             # moviepy, manim
    "pkg-config":      ("which", "pkg-config"),
    "libpango1.0-dev": ("pkg-config", "--exists", "pangocairo"),  # manim
    "libcairo2-dev":   ("pkg-config", "--exists", "cairo"),       # manim
    "python3-dev":     None,   # always include if anything else is missing
}

def _present(check):
    if check is None:
        return False
    cmd, *args = check
    full_cmd = [shutil.which(cmd) or cmd] + args
    return subprocess.run(full_cmd, capture_output=True).returncode == 0

def ensure_system_deps():
    if platform.system() != "Linux" or not shutil.which("apt"):
        return   # only auto-install on Debian/Ubuntu
    missing = [pkg for pkg, check in SYSTEM_DEPS.items() if not _present(check)]
    if not missing:
        print("System dependencies: OK")
        return
    print(f"Installing system packages: {' '.join(missing)}")
    result = subprocess.run(["sudo", "apt", "install", "-y"] + missing, text=True)
    if result.returncode != 0:
        print(f"ERROR: apt install failed. Try manually:\n\n    sudo apt install -y {' '.join(missing)}\n")
        sys.exit(1)
    print("System dependencies: installed")

ensure_system_deps()

# ── Pip installer ─────────────────────────────────────────────────────────────

def install(package):
    """Install a package, handling Debian's externally-managed-environment."""
    cmd = [sys.executable, "-m", "pip", "install", package]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    if "externally-managed-environment" in result.stderr:
        print(f"  (retrying with --break-system-packages)")
        subprocess.check_call(cmd + ["--break-system-packages"])
    else:
        print(result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, cmd)

required_packages = [
    "numpy",              # Core numerical computing
    "pandas",             # Data manipulation and analysis
    "scipy",              # Scientific computing
    "matplotlib",         # Plotting and visualization
    "seaborn",            # Statistical data visualization
    "scikit-learn",       # Machine learning algorithms
    "xgboost",            # Gradient boosting (ML)
    "lightgbm",           # Fast gradient boosting (ML)
    "catboost",           # Categorical boosting (ML)
    "torch",              # PyTorch deep learning
    "torchvision",        # PyTorch vision utilities
    "tensorflow",         # TensorFlow deep learning
    "keras",              # Keras deep learning (v3+)
    "tf-keras",           # Backwards-compatible Keras for Transformers
    "transformers",       # HuggingFace Transformers (NLP)
    "sentence-transformers", # Sentence embeddings (NLP)
    "openai",             # OpenAI API client
    "requests",           # HTTP requests
    "boto3",              # AWS SDK for Python
    "azure-ai-textanalytics", # Azure Text Analytics (NLP)
    "azure-identity",     # Azure authentication
    "anthropic",          # Anthropic API client
    "google-cloud-aiplatform", # Google Vertex AI (Gemini, etc)
    "plotly",             # Interactive visualization
    "graphviz",           # Graph visualization
    "Pillow",             # Image processing
    "opencv-python",      # Computer vision
    "librosa",            # Audio analysis
    "soundfile",          # Audio file I/O
    "moviepy",            # Video editing
    "ipywidgets",         # Jupyter widgets
    # --- New dependencies for RAG, Agents, Structured Output, LoRA ---
    "faiss-cpu",          # Facebook AI Similarity Search (vector retrieval)
    "peft",               # HuggingFace Parameter-Efficient Fine-Tuning (LoRA)
    "trl",                # Transformer Reinforcement Learning (SFT, DPO trainers)
    "datasets",           # HuggingFace Datasets
    "accelerate",         # HuggingFace multi-GPU / mixed-precision training helper
    "bitsandbytes",       # 4-bit / 8-bit quantization (QLoRA)
    # --- Manim Animations ---
    "manim",              # Mathematical animation engine (also requires ffmpeg + LaTeX)
]

# Some packages install under a different name than their PyPI package name
IMPORT_NAME_OVERRIDES = {
    "faiss-cpu":              "faiss",
    "scikit-learn":           "sklearn",
    "Pillow":                 "PIL",
    "opencv-python":          "cv2",
    "tf-keras":               "tf_keras",
    "azure-ai-textanalytics": "azure.ai.textanalytics",
    "azure-identity":         "azure.identity",
    "google-cloud-aiplatform": "google.cloud.aiplatform",
}

for pkg in required_packages:
    import_name = IMPORT_NAME_OVERRIDES.get(pkg, pkg.replace('-', '_'))
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

# Now import all libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
import lightgbm
import catboost
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
import keras
import tf_keras      
import transformers
import sentence_transformers
import openai
import requests
import boto3
import azure.ai.textanalytics
import azure.identity
import anthropic
import google.cloud.aiplatform
import os
import json
import random
import re
import time
import logging
import plotly
import plotly.express as px
import graphviz
import PIL
import cv2
import librosa
import soundfile as sf
import moviepy.editor as mpy

# RAG, Agents, Fine-tuning libraries
import faiss
import peft
import trl
import datasets
import accelerate

# bitsandbytes may not be available on all platforms (requires CUDA on some systems)
try:
    import bitsandbytes
except Exception:
    pass

# manim: system deps (ffmpeg, pango, cairo) are handled by ensure_system_deps() above.
# LaTeX is still required for MathTex scenes — see resources/Manim Animations/readme.md.
try:
    import manim
except Exception:
    pass

# Jupyter and notebook tools
try:
    get_ipython()
    import ipywidgets as widgets
    from IPython.display import display, HTML, Image
except Exception:
    pass

print("All required libraries are installed and importable.")
