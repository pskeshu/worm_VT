@echo off
REM ============================================================================
REM WormVT GPU Setup - Forces Python 3.11 for CUDA support
REM ============================================================================
REM Python 3.13 is too new - PyTorch doesn't have CUDA wheels for it yet
REM This script uses Python 3.11 which has full CUDA support
REM ============================================================================

echo ============================================================================
echo WormVT GPU Setup - Using Python 3.11 for CUDA Support
echo ============================================================================
echo.

REM Use Python 3.11 specifically (has CUDA wheels available)
set PYTHON311=C:\Users\christensenr\AppData\Local\Programs\Python\Python311\python.exe

if not exist "%PYTHON311%" (
    echo ERROR: Python 3.11 not found at %PYTHON311%
    echo Please install Python 3.11 from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Using Python 3.11 for CUDA support...
%PYTHON311% --version
echo.

REM Delete existing venv if it exists
if exist "venv" (
    echo Deleting existing virtual environment (Python 3.13)...
    rmdir /s /q venv
    echo.
)

REM Create new venv with Python 3.11
echo [2/6] Creating virtual environment with Python 3.11...
%PYTHON311% -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

REM Activate venv
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch with CUDA first
echo [5/6] Installing PyTorch with CUDA 12.1 support...
echo This may take several minutes...
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch with CUDA
    echo Falling back to CPU version...
    pip install torch torchvision torchaudio
)
echo.

REM Verify CUDA is available
echo Verifying CUDA support...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

REM Install other dependencies
echo Installing other dependencies...
pip install -r requirements.txt
echo.

REM Run tests
echo [6/6] Running tests to verify installation...
echo.
python test_model.py
if errorlevel 1 (
    echo.
    echo WARNING: Tests failed. Installation may be incomplete.
    pause
) else (
    echo.
    echo ============================================================================
    echo SUCCESS! WormVT is ready with GPU support!
    echo ============================================================================
)
echo.

REM Show GPU info
echo ============================================================================
echo GPU Information:
echo ============================================================================
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); cuda_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0; print('GPU memory:', str(round(cuda_mem, 1)) + ' GB' if cuda_mem > 0 else 'N/A')"
echo.

echo ============================================================================
echo Next Steps:
echo ============================================================================
echo.
echo To use WormVT, activate the virtual environment:
echo   venv\Scripts\activate.bat
echo   or just run: activate.bat
echo.
echo Then start training:
echo   python train.py training=pretrain model=vt_former_small
echo.
echo ============================================================================
echo.

pause
