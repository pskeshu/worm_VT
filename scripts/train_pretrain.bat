@echo off
REM Pretraining script for VT-Former with VideoMAE-3D (Windows)

REM Usage:
REM   scripts\train_pretrain.bat

python train.py ^
    training=pretrain ^
    model=vt_former_small ^
    data=nih_ls ^
    experiment_name=vt_former_pretrain ^
    %*
