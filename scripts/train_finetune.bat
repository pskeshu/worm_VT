@echo off
REM Fine-tuning script for stage classification (Windows)

REM Usage:
REM   scripts\train_finetune.bat

python train.py ^
    training=finetune ^
    model=vt_former_small ^
    data=nih_ls_classified ^
    experiment_name=vt_former_finetune ^
    %*
