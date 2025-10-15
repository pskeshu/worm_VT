# Getting Started with WormVT

## Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Model

```bash
# Run test suite
python test_model.py
```

Expected output:
```
✓ Small model created: 50,331,136 parameters (~50.3M)
✓ Base model created: 306,847,488 parameters (~306.8M)
✓ Output shape: [2, 8193, 512]
✓ CLS token shape: [2, 512]
✓ Classification logits shape: [2, 10]
✓ ALL TESTS PASSED
```

### 3. Verify Data Access

```bash
# Check that NIH-LS data is accessible
ls ../vlm_worm_test/nih-ls/
```

Should show embryo directories like `nih_diSPIM_deconv_1/`.

### 4. Test Data Loading (when implemented)

```python
from src.data import NIHLSDataset

dataset = NIHLSDataset(
    data_root="../vlm_worm_test/nih-ls",
    embryo_names=["nih_diSPIM_deconv_1"],
    num_frames=16,
    spatial_size=(256, 256, 64),
    split="train"
)

# Load first sample
sample = dataset[0]
print(f"Volume shape: {sample['volume'].shape}")  # [16, 1, 256, 256, 64]
```

---

## Project Structure

```
worm_VT/
├── src/
│   ├── models/               # ✅ VT-Former architecture
│   │   ├── vt_former.py
│   │   ├── patch_embed.py
│   │   ├── attention.py
│   │   └── heads.py
│   └── data/                 # ✅ Data loading
│       ├── nih_ls_dataset.py
│       └── transforms.py
├── configs/                  # ✅ Hydra configs
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── training/
├── test_model.py             # ✅ Test suite
├── requirements.txt          # ✅ Dependencies
├── README.md                 # ✅ Documentation
├── IMPLEMENTATION_STATUS.md  # ✅ Progress tracker
└── GETTING_STARTED.md        # ✅ This file
```

---

## Usage Examples

### Example 1: Instantiate VT-Former

```python
from src.models import vt_former_small, vt_former_base

# Small model (~50M params)
model = vt_former_small()

# Or base model (~300M params)
model = vt_former_base()

# Check parameters
print(f"Parameters: {model.get_num_params():,}")
```

### Example 2: Forward Pass

```python
import torch

# Input: [batch, time, channels, height, width, depth]
x = torch.randn(2, 16, 1, 256, 256, 64)

# Forward pass
tokens = model(x)  # [2, 1 + 16*num_patches, embed_dim]

# Extract CLS token (global representation)
cls = model.get_cls_token(x)  # [2, embed_dim]
```

### Example 3: With Classification Head

```python
from src.models import ClassificationHead, VTFormerWithHead

# Create encoder
encoder = vt_former_small()

# Add classification head for 10 stages
head = ClassificationHead(
    embed_dim=512,
    num_classes=10
)

# Combine
model = VTFormerWithHead(encoder, head, pooling="cls")

# Predict stage
x = torch.randn(2, 16, 1, 256, 256, 64)
logits = model(x)  # [2, 10]
```

### Example 4: Load NIH-LS Data

```python
from src.data import NIHLSDataset, get_train_transforms
from torch.utils.data import DataLoader

# Create dataset
dataset = NIHLSDataset(
    data_root="../vlm_worm_test/nih-ls",
    embryo_names=["nih_diSPIM_deconv_1"],
    num_frames=16,
    frame_stride=1,
    spatial_size=(256, 256, 64),
    transform=get_train_transforms(),
    split="train",
)

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
)

# Iterate
for batch in loader:
    volume = batch["volume"]  # [B, T, C, H, W, D]
    embryo = batch["embryo"]
    print(f"Batch shape: {volume.shape}")
    break
```

### Example 5: Load Classified Data

```python
from src.data import NIHLSClassifiedDataset

# Create dataset with VLM labels
dataset = NIHLSClassifiedDataset(
    data_root="../vlm_worm_test/nih-ls",
    classification_file="../vlm_worm_test/embryo_classifications_all.json",
    spatial_size=(256, 256, 64),
    split="train",
)

# Check stage names
print(f"Stages: {dataset.get_stage_names()}")

# Load sample
sample = dataset[0]
print(f"Volume: {sample['volume'].shape}")  # [1, 1, 256, 256, 64]
print(f"Label: {sample['label']}")          # tensor(stage_idx)
print(f"Stage: {sample['stage']}")          # "2-cell"
```

---

## Configuration with Hydra

Configurations are in `configs/` directory. Use Hydra to override:

```bash
# Use small model
python train.py model=vt_former_small

# Change batch size
python train.py data.batch_size=4

# Use different embryo
python train.py data.embryos=[nih_diSPIM_deconv_2]

# Combine overrides
python train.py model=vt_former_base data.batch_size=2 training.max_epochs=50
```

---

## Model Variants

### VT-Former Small (~50M params)
- For testing and rapid prototyping
- Can run on single GPU (RTX 3090/4090)
- Input: 256×256×64 volumes
- Temporal: 16 frames

```python
model = vt_former_small()
```

### VT-Former Base (~300M params)
- For full pretraining
- Requires A100 (40GB+)
- Input: 512×512×100 volumes
- Temporal: 32 frames

```python
model = vt_former_base()
```

### VT-Former Large (~500M params)
- For offline analysis
- Requires multi-GPU (8× A100)
- Maximum quality

```python
model = vt_former_large()
```

---

## Common Issues

### Issue 1: Out of Memory

**Solution**: Reduce batch size or spatial dimensions

```yaml
# In config
data:
  batch_size: 1  # or 2
  spatial_crop_size: [128, 128, 32]  # Smaller
```

### Issue 2: Data Loading Slow

**Solution**: Increase num_workers

```python
loader = DataLoader(dataset, num_workers=8)
```

### Issue 3: TIFF Files Not Found

**Solution**: Check data_root path

```python
import os
data_root = "../vlm_worm_test/nih-ls"
assert os.path.exists(data_root), f"Data root {data_root} does not exist"
```

---

## Next Steps

1. **Implement VideoMAE-3D Pretraining**
   - Random masking of patches
   - Reconstruction loss
   - PyTorch Lightning module

2. **Create Training Scripts**
   - Pretraining script with W&B logging
   - Fine-tuning script for stage classification
   - Evaluation script

3. **Add Visualization**
   - t-SNE of embeddings
   - Attention maps
   - Reconstruction quality

4. **Benchmarking**
   - Compare divided vs joint attention
   - Ablation studies
   - Performance metrics

---

## Resources

- **Architecture Paper**: TimeSformer (Bertasius et al., ICML 2021)
- **Pretraining Paper**: VideoMAE (Tong et al., NeurIPS 2022)
- **Dataset Paper**: Moyle et al., Nature 2021
- **Project Docs**: `../vlm_worm_test/docs/WORMVLA_COMPREHENSIVE_PLAN.md`

---

## Getting Help

- Check `IMPLEMENTATION_STATUS.md` for current progress
- Review `README.md` for architecture details
- See example notebooks (coming soon)
- Refer to comprehensive plan in `../vlm_worm_test/docs/`

---

Happy coding! 🪱🔬🤖
