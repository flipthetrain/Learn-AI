# This script imports all libraries required by any module in the Python Code folder and subfolders.
# Run this to check/install dependencies for all example scripts and future AI code.

import sys
import subprocess

# Install any missing packages first

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

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
    "ipywidgets"          # Jupyter widgets
]

for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_'))
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

# Jupyter and notebook tools
try:
    get_ipython()
    import ipywidgets as widgets
    from IPython.display import display, HTML, Image
except Exception:
    pass

print("All required libraries are installed and importable.")
