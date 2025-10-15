# Setup Guide

## 🚀 Quick Setup (Easiest)

### Windows:
```bash
# Run the setup script
setup.bat
```

### Linux/Mac:
```bash
# Make script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

**This will:**
1. ✅ Create virtual environment
2. ✅ Install all dependencies (~2-5 min)
3. ✅ Run tests to verify installation
4. ✅ Show you what to do next

---

## 📋 What Gets Installed

The setup script installs:

- **PyTorch 2.5+** with CUDA support
- **PyTorch Lightning** for training
- **Hydra** for configuration
- **MONAI** for 3D medical imaging
- **Weights & Biases** for logging
- **All other dependencies** from `requirements.txt`

**Total size:** ~5-8 GB (mostly PyTorch)

---

## 🎯 Step-by-Step Manual Setup

If you prefer to do it manually:

### 1. Create Virtual Environment

**Windows:**
```bash
# Try python3 first, fall back to python
python3 -m venv venv
# OR
python -m venv venv

# Activate
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This takes ~5-10 minutes (PyTorch is large).

### 4. Verify Installation

```bash
python test_model.py
```

Should show: `✓ ALL TESTS PASSED`

---

## ⚡ Super Quick Setup (No Tests)

If you just want to get started immediately:

**Windows:**
```bash
setup_quick.bat
```

This skips tests and just installs dependencies.

---

## 🔧 Activating the Environment Later

You need to activate the virtual environment each time you open a new terminal.

### Windows:
```bash
venv\Scripts\activate.bat

# Or use the helper script:
activate.bat
```

### Linux/Mac:
```bash
source venv/bin/activate
```

You'll see `(venv)` in your prompt when active.

---

## ✅ Verify Installation

After setup, verify everything works:

```bash
# 1. Check Python
python --version
# Should show Python 3.10+

# 2. Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# Should show: PyTorch 2.5.0 or higher

# 3. Check CUDA (if you have GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should show: CUDA available: True (if you have NVIDIA GPU)

# 4. Run full test suite
python test_model.py
# Should show: ✓ ALL TESTS PASSED
```

---

## 🐛 Troubleshooting

### Issue 1: Python not found

**Error:** `'python' is not recognized as an internal or external command`

**Solution:**
- Install Python 3.10+ from https://www.python.org/
- During installation, check "Add Python to PATH"
- Or use `python3` instead of `python`
- On Linux/Mac: `sudo apt install python3` or `brew install python3`

### Issue 2: "python3" vs "python"

**Different systems use different commands:**
- **Linux/Mac:** Usually `python3` (python might be Python 2.x)
- **Windows:** Usually `python` (python3 may not exist)

**Our scripts detect this automatically**, but if you're doing manual setup:
- Try `python3 --version` first
- If that fails, try `python --version`
- Use whichever one shows Python 3.10+

### Issue 3: pip install fails

**Error:** `ERROR: Could not install packages due to an OSError`

**Solutions:**

1. **Run as administrator** (Windows):
   - Right-click Command Prompt → "Run as administrator"
   - Run `setup.bat` again

2. **Clear pip cache:**
   ```bash
   pip cache purge
   pip install -r requirements.txt --no-cache-dir
   ```

3. **Install dependencies one by one:**
   ```bash
   pip install torch torchvision
   pip install pytorch-lightning
   pip install -r requirements.txt
   ```

### Issue 4: PyTorch CUDA version mismatch

**Error:** `RuntimeError: CUDA out of memory` or CUDA version warnings

**Solution:** Install PyTorch with correct CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Check your CUDA version:
```bash
nvidia-smi
```

### Issue 5: Virtual environment not activating

**Windows:**
```bash
# Try with full path
C:\path\to\worm_VT\venv\Scripts\activate.bat

# Or use PowerShell
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
# Try with full path
source /path/to/worm_VT/venv/bin/activate

# Make sure you're in the right directory
cd /path/to/worm_VT
```

### Issue 6: ModuleNotFoundError after installation

**Error:** `ModuleNotFoundError: No module named 'xyz'`

**Solution:**
1. Make sure virtual environment is activated (you should see `(venv)`)
2. Reinstall the missing package:
   ```bash
   pip install package_name
   ```
3. Verify you're using the venv python:
   ```bash
   which python  # Linux/Mac
   where python  # Windows
   # Should show path to venv/
   ```

### Issue 7: Tests fail

If `test_model.py` fails:

1. **Check error message** - it will tell you what's wrong
2. **Verify imports:**
   ```bash
   python -c "from src.models import vt_former_small"
   ```
3. **Check dependencies:**
   ```bash
   pip list | grep torch
   pip list | grep lightning
   ```

---

## 📦 Alternative: Conda Setup

If you prefer Conda:

```bash
# Create environment
conda create -n wormvt python=3.10
conda activate wormvt

# Install PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## 🔄 Updating Dependencies

To update to latest versions:

```bash
# Activate environment
venv\Scripts\activate.bat  # Windows
source venv/bin/activate   # Linux/Mac

# Update all packages
pip install --upgrade -r requirements.txt

# Or update specific package
pip install --upgrade pytorch-lightning
```

---

## 🗑️ Uninstalling

To completely remove WormVT:

```bash
# Deactivate environment first
deactivate

# Delete virtual environment
rmdir /s /q venv  # Windows
rm -rf venv       # Linux/Mac

# Delete Python cache
# Windows PowerShell:
Get-ChildItem -Path . -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force

# Linux/Mac:
find . -type d -name "__pycache__" -exec rm -rf {} +
```

---

## 💾 Disk Space Requirements

- **Virtual Environment:** ~5-8 GB
  - PyTorch: ~3-4 GB
  - Other dependencies: ~2-4 GB
- **Training outputs:** Variable
  - Checkpoints: ~2-5 GB per model
  - Logs: ~10-100 MB

**Total:** Plan for at least **10-15 GB** free space.

---

## ⚙️ System Requirements

### Minimum:
- **OS:** Windows 10+, Linux, macOS
- **Python:** 3.10+ (3.11 recommended)
- **RAM:** 16 GB
- **GPU:** NVIDIA GPU with 8 GB VRAM (GTX 1080, RTX 2060, etc.)
- **Disk:** 15 GB free space

### Recommended:
- **RAM:** 32 GB+
- **GPU:** NVIDIA GPU with 24 GB VRAM (RTX 3090, A5000, A6000)
- **Disk:** 50 GB+ free space (for data + checkpoints)

### For CPU-Only:
You can train on CPU, but it will be **very slow** (~100x slower than GPU).

To force CPU:
```bash
python train.py device=cpu
```

---

## 🎓 Next Steps After Setup

Once setup is complete:

1. **Verify data access:**
   ```bash
   # Windows
   dir ..\vlm_worm_test\nih-ls\

   # Linux/Mac
   ls ../vlm_worm_test/nih-ls/
   ```

2. **Run quick test:**
   ```bash
   python train.py data.batch_size=1 training.max_epochs=2 wandb.mode=disabled
   ```

3. **Start real training:**
   ```bash
   # Windows
   scripts\train_pretrain.bat

   # Linux/Mac
   bash scripts/train_pretrain.sh
   ```

4. **Monitor progress:**
   ```bash
   # Windows
   scripts\monitor_training.bat outputs\vt_former_pretrain 5

   # Linux/Mac
   bash scripts/monitor_training.sh outputs/vt_former_pretrain 5
   ```

---

## 📚 More Help

- **Training:** See `RUN_TRAINING.md`
- **Monitoring:** See `MONITORING_GUIDE.md`
- **Architecture:** See `README.md`
- **Quick commands:** See `RUN_TRAINING.md`

---

## ✨ You're Ready!

Setup complete! Start training with:

```bash
python train.py training=pretrain model=vt_former_small
```

Good luck! 🪱🔬🤖
