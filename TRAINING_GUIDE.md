# Training Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Installation

```bash
python test_model.py
```

Should show: `✓ ALL TESTS PASSED`

---

## Running Training

### Option 1: Pretraining (VideoMAE-3D)

Self-supervised pretraining on unlabeled 3D+time data.

**Windows:**
```bash
scripts\train_pretrain.bat
```

**Linux/Mac:**
```bash
bash scripts/train_pretrain.sh
```

**Or directly:**
```bash
python train.py training=pretrain model=vt_former_small data=nih_ls
```

### Option 2: Fine-tuning (Stage Classification)

Supervised fine-tuning on VLM-labeled stages.

**Windows:**
```bash
scripts\train_finetune.bat
```

**Linux/Mac:**
```bash
bash scripts/train_finetune.sh
```

**Or directly:**
```bash
python train.py training=finetune model=vt_former_small data=nih_ls_classified
```

---

## Configuration Options

### Change Model Size

```bash
# Small model (~50M params) - for testing
python train.py model=vt_former_small

# Base model (~300M params) - for full training
python train.py model=vt_former_base
```

### Adjust Batch Size

```bash
# Reduce if out of memory
python train.py data.batch_size=1

# Increase with gradient accumulation
python train.py data.batch_size=2 training.accumulate_grad_batches=8
```

### Change Spatial Size

```bash
# Smaller volumes (less memory)
python train.py data.spatial_crop_size=[128,128,32]

# Larger volumes (more detail)
python train.py data.spatial_crop_size=[512,512,100] model=vt_former_base
```

### Disable W&B Logging

```bash
python train.py wandb.mode=disabled
```

### Override Multiple Settings

```bash
python train.py \
    model=vt_former_small \
    data.batch_size=2 \
    training.max_epochs=50 \
    data.spatial_crop_size=[128,128,32] \
    wandb.mode=offline
```

---

## Expected Resource Requirements

### VT-Former Small
- **GPU**: Single RTX 3090/4090 (24GB)
- **Batch size**: 2-4
- **Spatial size**: 256×256×64
- **Training time**: ~3-5 days for 100 epochs

### VT-Former Base
- **GPU**: A100 40GB or 2× RTX 4090
- **Batch size**: 1-2
- **Spatial size**: 512×512×100
- **Training time**: ~1-2 weeks for 100 epochs

### Memory Optimization Tips

If you run out of memory:

1. **Reduce batch size**: `data.batch_size=1`
2. **Reduce spatial size**: `data.spatial_crop_size=[128,128,32]`
3. **Reduce temporal frames**: `data.num_frames=8`
4. **Use gradient accumulation**: `training.accumulate_grad_batches=16`
5. **Use smaller model**: `model=vt_former_small`

---

## Training Workflow

### Step 1: Pretraining (Recommended)

First, pretrain on unlabeled data with VideoMAE-3D:

```bash
python train.py \
    training=pretrain \
    model=vt_former_small \
    training.max_epochs=100
```

**Output**: `outputs/vt_former_pretrain/checkpoints/best.ckpt`

### Step 2: Fine-tuning

Then, fine-tune on VLM-labeled stages:

```bash
python train.py \
    training=finetune \
    model=vt_former_small \
    training.checkpoint=outputs/vt_former_pretrain/checkpoints/best.ckpt \
    training.max_epochs=50
```

**Output**: `outputs/vt_former_finetune/checkpoints/best.ckpt`

### Step 3: Evaluation (Coming Soon)

```bash
python evaluate.py checkpoint=outputs/vt_former_finetune/checkpoints/best.ckpt
```

---

## Monitoring Training

### Weights & Biases (W&B)

If W&B is enabled (default), you can monitor:
- Loss curves (train/val)
- Learning rate schedule
- Accuracy metrics
- System metrics (GPU usage, etc.)

Access at: https://wandb.ai/your-entity/wormvt

### TensorBoard (Alternative)

```bash
tensorboard --logdir outputs/
```

### Console Output

Training progress is logged to console:
```
Epoch 5/100: 100%|████████| 25/25 [02:15<00:00]
train_loss: 0.342, train_acc: 0.85, val_loss: 0.398, val_acc: 0.82
```

---

## Checkpointing

Checkpoints are automatically saved to:
```
outputs/
└── experiment_name/
    └── checkpoints/
        ├── epoch=10-val_loss=0.234.ckpt
        ├── epoch=25-val_loss=0.198.ckpt
        └── epoch=42-val_loss=0.187.ckpt  (best)
```

### Resume Training

To resume from a checkpoint:

```bash
python train.py \
    ckpt_path=outputs/vt_former_pretrain/checkpoints/last.ckpt
```

---

## Common Issues

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `data.batch_size=1`
2. Reduce spatial size: `data.spatial_crop_size=[128,128,32]`
3. Use gradient accumulation: `training.accumulate_grad_batches=8`
4. Reduce temporal frames: `data.num_frames=8`

### Issue 2: Data Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: Check data paths in config:
```bash
# Verify data exists
ls ../vlm_worm_test/nih-ls/

# Update config if needed
python train.py data.data_root=/path/to/your/data
```

### Issue 3: Slow Data Loading

**Symptom**: Training pauses frequently

**Solutions**:
1. Increase workers: `num_workers=8`
2. Use SSD for data storage
3. Pre-load data to RAM if possible

### Issue 4: Loss Not Decreasing

**Symptom**: Training loss stays constant

**Solutions**:
1. Check learning rate: `training.optimizer.lr=1e-3`
2. Verify data loading: Check one batch manually
3. Increase model capacity: `model=vt_former_base`
4. Increase training epochs: `training.max_epochs=200`

---

## Example Training Runs

### Quick Test (Small Data, Fast Iteration)

```bash
python train.py \
    model=vt_former_small \
    data.batch_size=2 \
    data.spatial_crop_size=[128,128,32] \
    data.num_frames=8 \
    training.max_epochs=10 \
    wandb.mode=disabled
```

**Expected time**: ~30 minutes
**Purpose**: Test pipeline, verify no errors

### Full Pretraining

```bash
python train.py \
    training=pretrain \
    model=vt_former_base \
    data.batch_size=2 \
    data.spatial_crop_size=[512,512,100] \
    data.num_frames=32 \
    training.max_epochs=100 \
    training.accumulate_grad_batches=4
```

**Expected time**: 1-2 weeks on 8× A100
**Purpose**: Full self-supervised pretraining

### Fine-tuning on Classifications

```bash
python train.py \
    training=finetune \
    model=vt_former_base \
    data.batch_size=8 \
    training.checkpoint=outputs/pretrain/best.ckpt \
    training.max_epochs=50 \
    training.freeze_encoder=true \
    training.freeze_epochs=5
```

**Expected time**: 1-2 days on single A100
**Purpose**: Supervised classification

---

## Next Steps

After training completes:

1. **Evaluate**: Test accuracy on held-out data
2. **Visualize**: t-SNE of learned features
3. **Analyze**: Confusion matrix, per-stage accuracy
4. **Deploy**: Export to ONNX for inference

See `EVALUATION_GUIDE.md` (coming soon) for details.

---

## Getting Help

- Check `IMPLEMENTATION_STATUS.md` for current capabilities
- Review `GETTING_STARTED.md` for setup
- See `README.md` for architecture details
- Report issues at: [GitHub Issues](https://github.com/...)

Happy training! 🪱🔬🤖
