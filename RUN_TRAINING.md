# Run Training - Quick Commands

## 🚀 Ready to Train!

All components are implemented. Here's how to start training:

---

## Setup (One-Time)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python test_model.py

# 3. Check data access
ls ../vlm_worm_test/nih-ls/
```

---

## Option 1: Quick Test Run (Recommended First)

Test the entire pipeline with small settings:

```bash
python train.py \
    model=vt_former_small \
    data.batch_size=1 \
    data.spatial_crop_size=[128,128,32] \
    data.num_frames=8 \
    training.max_epochs=2 \
    wandb.mode=disabled
```

**Time**: ~5-10 minutes
**Purpose**: Verify everything works before full training

---

## Option 2: Pretraining (VideoMAE-3D)

Self-supervised pretraining on unlabeled data:

### Windows:
```bash
scripts\train_pretrain.bat
```

### Linux/Mac:
```bash
bash scripts/train_pretrain.sh
```

### Or directly:
```bash
python train.py training=pretrain model=vt_former_small
```

**Expected Output**:
```
Train samples: 960
Val samples: 240
Epoch 1/100: train_loss=0.523, val_loss=0.487
Epoch 2/100: train_loss=0.412, val_loss=0.398
...
```

---

## Option 3: Fine-tuning (Stage Classification)

Supervised fine-tuning on VLM labels:

### Windows:
```bash
scripts\train_finetune.bat
```

### Linux/Mac:
```bash
bash scripts/train_finetune.sh
```

### Or directly:
```bash
python train.py training=finetune model=vt_former_small data=nih_ls_classified
```

**Expected Output**:
```
Train samples: 280
Val samples: 60
Epoch 1/50: train_loss=2.134, train_acc=0.32, val_acc=0.35
Epoch 2/50: train_loss=1.876, train_acc=0.48, val_acc=0.52
...
```

---

## Common Adjustments

### If Out of Memory:

```bash
python train.py \
    data.batch_size=1 \
    data.spatial_crop_size=[128,128,32] \
    data.num_frames=8 \
    training.accumulate_grad_batches=8
```

### Use Different Model Size:

```bash
# Small model (~50M params)
python train.py model=vt_former_small

# Base model (~300M params)
python train.py model=vt_former_base
```

### Disable W&B Logging:

```bash
python train.py wandb.mode=disabled
```

### Change Output Directory:

```bash
python train.py output_dir=my_experiments/run1
```

---

## What to Expect

### Pretraining (VideoMAE-3D):
- **Loss starts**: ~0.5-0.6
- **Loss converges**: ~0.1-0.2 (after 100 epochs)
- **Time per epoch**: ~10-30 min (depends on GPU)

### Fine-tuning (Classification):
- **Accuracy starts**: ~10-20% (random)
- **Accuracy converges**: >90% (with good pretrain)
- **Time per epoch**: ~2-5 min

---

## Monitoring

### Console Output:
```
Epoch 5/100: 100%|████████| 25/25 [02:15<00:00]
train_loss: 0.342, val_loss: 0.398
```

### Weights & Biases:
- Go to: https://wandb.ai/your-entity/wormvt
- See: Loss curves, metrics, system stats

### Checkpoints:
Saved automatically to:
```
outputs/
└── vt_former_pretrain/
    └── checkpoints/
        ├── epoch=10-val_loss=0.234.ckpt
        ├── epoch=25-val_loss=0.198.ckpt
        └── epoch=42-val_loss=0.187.ckpt  ← best
```

---

## Typical Workflow

### Day 1: Test
```bash
# Quick test run (5 min)
python train.py \
    data.batch_size=1 \
    data.spatial_crop_size=[128,128,32] \
    training.max_epochs=2 \
    wandb.mode=disabled
```

### Day 2-7: Pretrain
```bash
# Start pretraining (runs for several days)
scripts\train_pretrain.bat
```

### Day 8: Fine-tune
```bash
# Fine-tune on classifications
python train.py \
    training=finetune \
    training.checkpoint=outputs/vt_former_pretrain/checkpoints/best.ckpt
```

---

## Troubleshooting

### Issue: "No module named 'src'"

**Solution**: Run from project root
```bash
cd /path/to/worm_VT
python train.py ...
```

### Issue: "CUDA out of memory"

**Solution**: Reduce memory usage
```bash
python train.py data.batch_size=1 data.spatial_crop_size=[128,128,32]
```

### Issue: "Data not found"

**Solution**: Check data path
```bash
ls ../vlm_worm_test/nih-ls/
# If not found, update config:
python train.py data.data_root=/correct/path/to/nih-ls
```

### Issue: Training is slow

**Solution**: Increase workers
```bash
python train.py num_workers=8
```

---

## Next Steps After Training

1. **Evaluate**: Test on held-out data
2. **Visualize**: t-SNE, attention maps
3. **Analyze**: Confusion matrix, per-stage metrics
4. **Iterate**: Adjust hyperparameters based on results

See `TRAINING_GUIDE.md` for detailed information.

---

## 🎯 Start Training Now!

Just run:
```bash
python train.py training=pretrain model=vt_former_small
```

Or for a quick test first:
```bash
python train.py data.batch_size=1 training.max_epochs=2 wandb.mode=disabled
```

Good luck! 🪱🔬🤖
