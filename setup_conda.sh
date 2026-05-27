#!/bin/bash

# Exit on error
set -e

ENV_NAME="minerl"

echo "Creating Conda environment: $ENV_NAME"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda could not be found. Please install Conda (or Miniconda/Miniforge) first."
    exit 1
fi

# Create the conda environment with required packages
conda create -y -n "$ENV_NAME" -c conda-forge \
    python=3.10 \
    pip \
    openjdk=8 \
    "setuptools<81" \
    wheel

echo "Activating environment..."
# Make conda activate available in this script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing MineRL..."
python -m pip install --no-cache-dir git+https://github.com/minerllabs/minerl

echo "Installing additional dependencies..."
python -m pip install --no-cache-dir stable-baselines3 opencv-python shimmy

echo "============================================================"
echo "Environment setup complete!"
echo "To activate this environment, run: conda activate $ENV_NAME"
echo ""
echo "Note: You may need to install the following system dependencies depending on your OS."
echo "For Debian/Ubuntu:"
echo "  sudo apt-get update && sudo apt-get install -y build-essential ca-certificates git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 xvfb"
echo "============================================================"
