@echo off
REM Monitor training progress (Windows)

REM Usage:
REM   scripts\monitor_training.bat outputs\vt_former_pretrain
REM   scripts\monitor_training.bat outputs\vt_former_pretrain 5

if "%1"=="" (
    echo Usage: scripts\monitor_training.bat output_dir [refresh_interval]
    echo Example: scripts\monitor_training.bat outputs\vt_former_pretrain 5
    exit /b 1
)

if "%2"=="" (
    python scripts/monitor_training.py %1
) else (
    python scripts/monitor_training.py %1 --refresh %2
)
