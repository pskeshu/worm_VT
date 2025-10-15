@echo off
REM ============================================================================
REM Activate WormVT Virtual Environment
REM ============================================================================
REM Simple script to activate the virtual environment
REM ============================================================================

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please run setup.bat first to create the environment:
    echo   setup.bat
    echo.
    pause
    exit /b 1
)

echo Activating WormVT virtual environment...
call venv\Scripts\activate.bat

echo.
echo ============================================================================
echo WormVT Virtual Environment Activated
echo ============================================================================
echo.
echo You can now:
echo   - Run training: python train.py training=pretrain
echo   - Run tests: python test_model.py
echo   - Monitor: scripts\monitor_training.bat outputs\vt_former_pretrain 5
echo.
echo To deactivate, type: deactivate
echo ============================================================================
echo.
