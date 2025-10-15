# Training Monitoring Guide

## 📊 Log Files Created

When you run training, several log files are automatically created:

```
outputs/
└── vt_former_pretrain/
    ├── progress.txt                    # ⭐ Quick status (always current)
    ├── logs/
    │   └── training_20251015_143022.log  # 📝 Detailed log
    ├── csv_logs/
    │   ├── version_0/
    │   │   └── metrics.csv             # 📈 Metrics CSV
    └── checkpoints/
        ├── epoch=10-val_loss=0.234.ckpt
        └── epoch=25-val_loss=0.198.ckpt
```

---

## 🚀 Quick Monitoring

### Option 1: Check Progress File (Fastest)

Just open `progress.txt` - it's always up-to-date:

```bash
# Windows
type outputs\vt_former_pretrain\progress.txt

# Linux/Mac
cat outputs/vt_former_pretrain/progress.txt
```

**Output:**
```
Last Updated: 2025-10-15 14:35:12
Epoch: 15/100
Progress: 15.0%

Metrics:
  train_loss: 0.342
  train_loss_epoch: 0.345
  val_loss: 0.398
  val_loss_epoch: 0.401

Checkpoint: outputs/vt_former_pretrain/checkpoints/epoch=12-val_loss=0.387.ckpt
```

### Option 2: Use Monitor Script (Auto-Refresh)

Watch training in real-time with auto-refresh:

```bash
# Windows - refresh every 5 seconds
scripts\monitor_training.bat outputs\vt_former_pretrain 5

# Linux/Mac - refresh every 5 seconds
bash scripts/monitor_training.sh outputs/vt_former_pretrain 5
```

**Shows:**
- Current epoch and progress
- Latest metrics
- Recent log entries
- Auto-refreshes every N seconds

### Option 3: Tail the Detailed Log

Follow the detailed log in real-time:

```bash
# Windows (using PowerShell)
Get-Content outputs\vt_former_pretrain\logs\training_*.log -Wait -Tail 20

# Linux/Mac
tail -f outputs/vt_former_pretrain/logs/training_*.log

# Or use the monitor script
python scripts/monitor_training.py outputs/vt_former_pretrain --tail
```

---

## 📝 What Each Log Contains

### 1. `progress.txt` (Quick Status)

**Purpose**: Quick check of current status
**Updates**: After each epoch
**Best for**: "Is it still running? What epoch?"

**Contains:**
- Last update time
- Current epoch / total epochs
- Progress percentage
- Latest metrics (loss, accuracy)
- Best checkpoint path

**Example:**
```
Last Updated: 2025-10-15 14:35:12
Epoch: 15/100
Progress: 15.0%

Metrics:
  train_loss_epoch: 0.345
  val_loss_epoch: 0.401
  val_acc: 0.867
```

### 2. `logs/training_TIMESTAMP.log` (Detailed)

**Purpose**: Complete training history
**Updates**: Continuously (line-buffered)
**Best for**: Debugging, detailed progress

**Contains:**
- Epoch start/end times
- Batch-level progress (every 50 steps)
- Loss values per batch
- Learning rate
- Validation results
- Epoch summaries with all metrics
- Error messages

**Example:**
```
2025-10-15 14:30:15 - wormvt - INFO - =====================================
2025-10-15 14:30:15 - wormvt - INFO - Epoch 15/100 - Started
2025-10-15 14:30:15 - wormvt - INFO - Time: 2025-10-15 14:30:15
2025-10-15 14:30:15 - wormvt - INFO - =====================================
2025-10-15 14:30:45 - wormvt - INFO -   Batch 0/125 | Loss: 0.3421 | LR: 1.45e-04
2025-10-15 14:31:15 - wormvt - INFO -   Batch 50/125 | Loss: 0.3387 | LR: 1.45e-04
...
2025-10-15 14:35:10 - wormvt - INFO - Epoch 15 Summary:
2025-10-15 14:35:10 - wormvt - INFO -   Duration: 295.23s (4.92min)
2025-10-15 14:35:10 - wormvt - INFO -   train_loss_epoch: 0.3451
2025-10-15 14:35:10 - wormvt - INFO -   val_loss_epoch: 0.4012
2025-10-15 14:35:10 - wormvt - INFO -   val_acc: 0.8670
```

### 3. `csv_logs/metrics.csv` (Metrics Only)

**Purpose**: Machine-readable metrics for plotting
**Updates**: After each epoch
**Best for**: Plotting, analysis, spreadsheets

**Contains:**
- Epoch number
- Step number
- All logged metrics (train_loss, val_loss, accuracy, etc.)

**Example:**
```csv
epoch,step,train_loss,train_loss_epoch,val_loss,val_loss_epoch,val_acc
0,125,0.523,,0.487,0.487,0.324
1,250,0.412,0.412,0.398,0.398,0.456
2,375,0.357,0.357,0.365,0.365,0.523
...
```

---

## 🎯 Common Monitoring Tasks

### Check if Training is Running

```bash
# Quick check
type outputs\vt_former_pretrain\progress.txt

# Check last modified time (Windows PowerShell)
(Get-Item outputs\vt_former_pretrain\progress.txt).LastWriteTime

# Check last modified time (Linux/Mac)
ls -lh outputs/vt_former_pretrain/progress.txt
```

If `progress.txt` was updated recently, training is running.

### See Current Loss

```bash
# From progress file
type outputs\vt_former_pretrain\progress.txt | findstr loss

# From detailed log (last 5 lines)
Get-Content outputs\vt_former_pretrain\logs\training_*.log -Tail 5
```

### Check Training Speed

Look in the detailed log for epoch summaries:
```
Epoch 15 Summary:
  Duration: 295.23s (4.92min)
```

Calculate total time:
- Time per epoch × remaining epochs = estimated time remaining

### Plot Training Curves

Use the CSV file:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('outputs/vt_former_pretrain/csv_logs/version_0/metrics.csv')

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_loss_epoch'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss_epoch'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curves.png')
```

### Find Best Checkpoint

Check `progress.txt` or look at checkpoint names:

```bash
# List checkpoints sorted by validation loss
dir outputs\vt_former_pretrain\checkpoints /O:N

# Best checkpoint is typically the one with lowest val_loss
```

---

## 🔍 Monitoring During Training

### Real-Time Progress

**Option 1: Auto-refresh monitor**
```bash
scripts\monitor_training.bat outputs\vt_former_pretrain 10
```

**Option 2: Watch progress file**
```bash
# Windows (PowerShell) - refresh every 5 seconds
while ($true) {
    Clear-Host
    Get-Content outputs\vt_former_pretrain\progress.txt
    Start-Sleep -Seconds 5
}
```

**Option 3: Tail detailed log**
```bash
python scripts/monitor_training.py outputs/vt_former_pretrain --tail
```

### Remote Monitoring

If training on a remote server:

```bash
# SSH and monitor
ssh user@server
cd /path/to/worm_VT
python scripts/monitor_training.py outputs/vt_former_pretrain --refresh 10

# Or periodically download progress file
scp user@server:/path/to/worm_VT/outputs/vt_former_pretrain/progress.txt ./
```

### Weights & Biases (Web Dashboard)

If W&B is enabled:
1. Go to https://wandb.ai/your-entity/wormvt
2. See real-time plots of all metrics
3. Compare multiple runs
4. View system metrics (GPU, CPU, memory)

---

## 📈 What to Watch For

### Good Signs:
- ✅ Training loss decreasing steadily
- ✅ Validation loss decreasing (may plateau)
- ✅ No divergence between train and val loss
- ✅ Accuracy increasing (for classification)
- ✅ Epoch time consistent

### Warning Signs:
- ⚠️ Training loss stuck or increasing
- ⚠️ Validation loss increasing while train loss decreases (overfitting)
- ⚠️ Loss becomes NaN
- ⚠️ Epoch time increasing (memory leak?)
- ⚠️ No updates to progress.txt for >1 hour

### Troubleshooting:

**Loss is NaN:**
```
Check: outputs/.../logs/training_*.log
Look for: Error messages before NaN
Solution: Reduce learning rate or check data
```

**Training stopped:**
```
Check: Progress file last modified time
Check: Detailed log for errors at end
Check: System logs for OOM errors
```

**Too slow:**
```
Check: Epoch duration in logs
Solution: Reduce batch size, spatial size, or num_frames
```

---

## 🛠️ Advanced Monitoring

### Custom Metrics Extraction

```python
import re

def extract_metrics_from_log(log_file):
    """Extract all epoch summaries from log."""
    metrics = []
    with open(log_file, 'r') as f:
        current_epoch = {}
        for line in f:
            if 'Epoch' in line and 'Summary' in line:
                if current_epoch:
                    metrics.append(current_epoch)
                current_epoch = {}
            elif ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip().split()[-1]
                    value = parts[1].strip()
                    try:
                        current_epoch[key] = float(value)
                    except:
                        current_epoch[key] = value
        if current_epoch:
            metrics.append(current_epoch)
    return metrics

# Use it
metrics = extract_metrics_from_log('outputs/.../logs/training_*.log')
print(f"Completed {len(metrics)} epochs")
```

### Compare Multiple Runs

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple CSV files
run1 = pd.read_csv('outputs/run1/csv_logs/version_0/metrics.csv')
run2 = pd.read_csv('outputs/run2/csv_logs/version_0/metrics.csv')

# Plot comparison
plt.plot(run1['epoch'], run1['val_loss_epoch'], label='Run 1')
plt.plot(run2['epoch'], run2['val_loss_epoch'], label='Run 2')
plt.legend()
plt.show()
```

---

## 💡 Pro Tips

1. **Keep progress.txt open** in a text editor with auto-reload
2. **Use monitor script** with auto-refresh during active development
3. **Check logs regularly** for warnings or errors
4. **Save CSV metrics** for later analysis
5. **Use W&B** for complex experiments with many runs

---

## 🎓 Example Workflow

```bash
# 1. Start training
python train.py training=pretrain model=vt_former_small

# 2. In another terminal, monitor progress
scripts\monitor_training.bat outputs\vt_former_pretrain 10

# 3. Check detailed status occasionally
type outputs\vt_former_pretrain\progress.txt

# 4. After training, analyze metrics
python scripts/analyze_training.py outputs/vt_former_pretrain
```

---

## 📚 Log File Locations Summary

| File | Purpose | Update Frequency | Best For |
|------|---------|------------------|----------|
| `progress.txt` | Current status | Every epoch | Quick checks |
| `logs/training_*.log` | Detailed history | Every batch | Debugging |
| `csv_logs/metrics.csv` | Machine-readable | Every epoch | Plotting |
| Checkpoints | Model weights | Every N epochs | Best model |

---

Happy monitoring! 📊🔬🪱
