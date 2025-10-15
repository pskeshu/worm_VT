@echo off
REM ============================================================================
REM WormVT Setup Script for Windows
REM ============================================================================
REM This script will:
REM   1. Create a Python virtual environment
REM   2. Activate it
REM   3. Upgrade pip
REM   4. Install all dependencies
REM   5. Run tests to verify installation
REM ============================================================================

echo ============================================================================
echo WormVT Setup - Starting Installation
echo ============================================================================
echo.

REM Find Python executable (try python3 first, then python)
set PYTHON_CMD=
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
) else (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
    ) else (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.10+ from https://www.python.org/
        pause
        exit /b 1
    )
)

echo [1/6] Checking Python version...
%PYTHON_CMD% --version
echo Using: %PYTHON_CMD%
echo.

REM Check if virtual environment already exists
if exist "venv" (
    echo Virtual environment already exists.
    choice /C YN /M "Do you want to delete and recreate it"
    if errorlevel 2 goto skip_venv_creation
    if errorlevel 1 (
        echo Deleting existing virtual environment...
        rmdir /s /q venv
    )
)

:skip_venv_creation

REM Create virtual environment
if not exist "venv" (
    echo [2/6] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo [2/6] Using existing virtual environment...
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [5/6] Installing dependencies...
echo This may take several minutes (especially PyTorch)...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Try running: pip install -r requirements.txt --no-cache-dir
    pause
    exit /b 1
)
echo.
echo Dependencies installed successfully!
echo.

REM Run tests
echo [6/6] Running tests to verify installation...
echo.
python test_model.py
if errorlevel 1 (
    echo.
    echo WARNING: Tests failed. Installation may be incomplete.
    echo You can still try running training, but there may be issues.
    pause
) else (
    echo.
    echo ============================================================================
    echo SUCCESS! WormVT is ready to use!
    echo ============================================================================
)
echo.

REM Show next steps
echo ============================================================================
echo Next Steps:
echo ============================================================================
echo.
echo To use WormVT, activate the virtual environment:
echo   venv\Scripts\activate.bat
echo   or just run: activate.bat
echo.
echo Then you can:
echo   1. Test training:  python train.py data.batch_size=1 training.max_epochs=2 wandb.mode=disabled
echo   2. Start pretraining:  scripts\train_pretrain.bat
echo   3. Monitor training:  scripts\monitor_training.bat outputs\vt_former_pretrain 5
echo.
echo For more information, see:
echo   - RUN_TRAINING.md
echo   - TRAINING_GUIDE.md
echo   - GETTING_STARTED.md
echo.
echo ============================================================================
echo.

pause
