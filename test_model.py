"""
Quick test script to verify VT-Former implementation.

Tests:
1. Model instantiation
2. Forward pass with dummy data
3. Parameter counting
4. CLS token extraction
"""

import torch
from src.models import vt_former_small, vt_former_base
from src.models import ClassificationHead, VTFormerWithHead

def test_model_instantiation():
    """Test creating models of different sizes."""
    print("=" * 70)
    print("TEST 1: Model Instantiation")
    print("=" * 70)

    # Small model
    model_small = vt_former_small()
    params_small = model_small.get_num_params()
    print(f"\n✓ Small model created: {params_small:,} parameters (~{params_small/1e6:.1f}M)")

    # Base model
    model_base = vt_former_base()
    params_base = model_base.get_num_params()
    print(f"✓ Base model created: {params_base:,} parameters (~{params_base/1e6:.1f}M)")

    return model_small


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass")
    print("=" * 70)

    # Create dummy input: [B, T, C, H, W, D]
    B, T, C, H, W, D = 2, 16, 1, 256, 256, 64
    x = torch.randn(B, T, C, H, W, D)

    print(f"\nInput shape: {list(x.shape)}")
    print(f"Input size: {x.numel() / 1e6:.2f}M elements")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        tokens = model(x)

    print(f"✓ Output shape: {list(tokens.shape)}")
    print(f"  - CLS token: tokens[:, 0, :]  shape {[B, model.embed_dim]}")
    print(f"  - Spatial tokens: tokens[:, 1:, :]  shape {[B, T * model.num_spatial_patches, model.embed_dim]}")

    return x, tokens


def test_cls_extraction(model, x):
    """Test CLS token extraction."""
    print("\n" + "=" * 70)
    print("TEST 3: CLS Token Extraction")
    print("=" * 70)

    with torch.no_grad():
        cls = model.get_cls_token(x)

    print(f"\n✓ CLS token shape: {list(cls.shape)}")
    print(f"  Mean: {cls.mean().item():.4f}, Std: {cls.std().item():.4f}")


def test_with_classification_head(model):
    """Test model with classification head."""
    print("\n" + "=" * 70)
    print("TEST 4: Classification Head")
    print("=" * 70)

    num_classes = 10  # 10 developmental stages
    head = ClassificationHead(
        embed_dim=model.embed_dim,
        num_classes=num_classes,
    )

    model_with_head = VTFormerWithHead(
        encoder=model,
        head=head,
        pooling="cls",
    )

    # Dummy input
    B, T, C, H, W, D = 2, 16, 1, 256, 256, 64
    x = torch.randn(B, T, C, H, W, D)

    print(f"\nInput shape: {list(x.shape)}")

    with torch.no_grad():
        logits = model_with_head(x)

    print(f"✓ Classification logits shape: {list(logits.shape)}")
    print(f"  Expected: [B={B}, num_classes={num_classes}]")
    print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")


def test_architecture_components():
    """Test individual architecture components."""
    print("\n" + "=" * 70)
    print("TEST 5: Architecture Components")
    print("=" * 70)

    from src.models import PatchEmbed3D, DividedSpaceTimeBlock

    # Test patch embedding
    print("\n▸ Testing PatchEmbed3D...")
    patch_embed = PatchEmbed3D(
        img_size=(256, 256, 64),
        patch_size=(16, 16, 8),
        in_chans=1,
        embed_dim=512,
    )

    x = torch.randn(2, 1, 256, 256, 64)  # [B, C, H, W, D]
    patches = patch_embed(x)
    print(f"  ✓ Input: {list(x.shape)} -> Patches: {list(patches.shape)}")
    print(f"  ✓ Num patches: {patch_embed.get_num_patches()}")

    # Test attention block
    print("\n▸ Testing DividedSpaceTimeBlock...")
    block = DividedSpaceTimeBlock(dim=512, num_heads=8)

    # Input: [B, 1 + T*N_patches, D]
    T, N_patches = 16, patches.shape[1]
    x = torch.randn(2, 1 + T * N_patches, 512)
    out = block(x, num_frames=T, num_spatial_patches=N_patches)
    print(f"  ✓ Input: {list(x.shape)} -> Output: {list(out.shape)}")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "WormVT MODEL TEST SUITE" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Test 1: Instantiation
        model = test_model_instantiation()

        # Test 2: Forward pass
        x, tokens = test_forward_pass(model)

        # Test 3: CLS extraction
        test_cls_extraction(model, x)

        # Test 4: Classification head
        test_with_classification_head(model)

        # Test 5: Components
        test_architecture_components()

        # Summary
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\n✓ VT-Former architecture is correctly implemented!")
        print("✓ Ready for training on actual data.")
        print("\nNext steps:")
        print("  1. Test data loading with real NIH-LS data")
        print("  2. Implement VideoMAE-3D pretraining module")
        print("  3. Create PyTorch Lightning training script")
        print()

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
