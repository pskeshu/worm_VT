#!/bin/bash
# ============================================================================
# WormVT Setup Script for Linux/Mac
# ============================================================================
# This script will:
#   1. Create a Python virtual environment
#   2. Activate it
#   3. Upgrade pip
#   4. Install all dependencies
#   5. Run tests to verify installation
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "WormVT Setup - Starting Installation"
echo "============================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10+ from your package manager or https://www.python.org/"
    exit 1
fi

echo "[1/6] Checking Python version..."
python3 --version
echo ""

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to delete and recreate it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing virtual environment..."
        rm -rf venv
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[2/6] Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created successfully."
else
    echo "[2/6] Using existing virtual environment..."
fi
echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated."
echo ""

# Upgrade pip
echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install dependencies
echo "[5/6] Installing dependencies..."
echo "This may take several minutes (especially PyTorch)..."
echo ""
pip install -r requirements.txt
echo ""
echo "Dependencies installed successfully!"
echo ""

# Run tests
echo "[6/6] Running tests to verify installation..."
echo ""
if python test_model.py; then
    echo ""
    echo "============================================================================"
    echo "SUCCESS! WormVT is ready to use!"
    echo "============================================================================"
else
    echo ""
    echo "WARNING: Tests failed. Installation may be incomplete."
    echo "You can still try running training, but there may be issues."
fi
echo ""

# Show next steps
echo "============================================================================"
echo "Next Steps:"
echo "============================================================================"
echo ""
echo "To use WormVT, activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can:"
echo "  1. Test training:  python train.py data.batch_size=1 training.max_epochs=2 wandb.mode=disabled"
echo "  2. Start pretraining:  bash scripts/train_pretrain.sh"
echo "  3. Monitor training:  bash scripts/monitor_training.sh outputs/vt_former_pretrain 5"
echo ""
echo "For more information, see:"
echo "  - RUN_TRAINING.md"
echo "  - TRAINING_GUIDE.md"
echo "  - GETTING_STARTED.md"
echo ""
echo "============================================================================"
echo ""
