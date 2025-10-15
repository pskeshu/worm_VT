@echo off
REM ============================================================================
REM Quick Setup - Minimal installation without tests
REM ============================================================================
REM Use this if you just want to get started quickly
REM ============================================================================

echo ============================================================================
echo WormVT Quick Setup
echo ============================================================================
echo.

REM Find Python
set PYTHON_CMD=
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
) else (
    set PYTHON_CMD=python
)

REM Create and activate venv
if not exist "venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
)

call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

echo.
echo ============================================================================
echo Done! Virtual environment is activated.
echo ============================================================================
echo.
echo Run training with: python train.py training=pretrain model=vt_former_small
echo.
