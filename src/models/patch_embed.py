"""
3D Patch Embedding for volumetric data

Converts 3D volumes into patch tokens for transformer processing.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding layer for volumetric microscopy data.

    Converts input volume [B, C, H, W, D] into a sequence of patch tokens [B, N_patches, embed_dim]
    using a 3D convolution with stride equal to patch size (non-overlapping patches).

    Args:
        img_size: Tuple of (H, W, D) for input volume dimensions
        patch_size: Tuple of (Ph, Pw, Pd) for patch dimensions
        in_chans: Number of input channels (typically 1 for fluorescence)
        embed_dim: Embedding dimension for output tokens
        norm_layer: Optional normalization layer
        flatten: If True, flatten spatial dimensions into sequence

    Example:
        >>> patch_embed = PatchEmbed3D(
        ...     img_size=(512, 512, 100),
        ...     patch_size=(16, 16, 8),
        ...     in_chans=1,
        ...     embed_dim=768
        ... )
        >>> x = torch.randn(2, 1, 512, 512, 100)  # [B, C, H, W, D]
        >>> tokens = patch_embed(x)  # [B, N_patches, 768]
        >>> print(tokens.shape)
        torch.Size([2, 32*32*12, 768])  # (512/16)*(512/16)*(100/8) = 12288 patches
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (512, 512, 100),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: nn.Module = None,
        flatten: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten

        # Calculate number of patches in each dimension
        self.grid_size = (
            img_size[0] // patch_size[0],  # H patches
            img_size[1] // patch_size[1],  # W patches
            img_size[2] // patch_size[2],  # D patches
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # 3D convolution for patch projection
        # Using kernel_size = patch_size and stride = patch_size creates non-overlapping patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Optional normalization
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input volume [B, C, H, W, D]

        Returns:
            tokens: Patch tokens [B, N_patches, embed_dim] if flatten=True
                    or [B, embed_dim, H', W', D'] if flatten=False
        """
        B, C, H, W, D = x.shape

        # Verify input dimensions
        assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})"
        assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})"
        assert D == self.img_size[2], f"Input depth ({D}) doesn't match model ({self.img_size[2]})"

        # Project patches: [B, C, H, W, D] -> [B, embed_dim, H', W', D']
        x = self.proj(x)

        if self.flatten:
            # Flatten spatial dimensions: [B, embed_dim, H', W', D'] -> [B, embed_dim, N_patches]
            x = x.flatten(2)
            # Transpose: [B, embed_dim, N_patches] -> [B, N_patches, embed_dim]
            x = x.transpose(1, 2)

        # Apply normalization
        x = self.norm(x)

        return x

    def get_num_patches(self) -> int:
        """Return number of patches."""
        return self.num_patches


class PatchEmbed3DOverlapping(nn.Module):
    """
    3D Patch Embedding with overlapping patches.

    Uses smaller stride than kernel size to create overlapping patches,
    which can capture finer spatial details at the cost of more patches.

    Args:
        img_size: Tuple of (H, W, D) for input volume dimensions
        patch_size: Tuple of (Ph, Pw, Pd) for patch dimensions
        stride: Tuple of (Sh, Sw, Sd) for stride (None = non-overlapping)
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (512, 512, 100),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        stride: Tuple[int, int, int] = None,
        in_chans: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        # Calculate number of patches with overlapping
        self.grid_size = (
            (img_size[0] - patch_size[0]) // self.stride[0] + 1,
            (img_size[1] - patch_size[1]) // self.stride[1] + 1,
            (img_size[2] - patch_size[2]) // self.stride[2] + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # 3D convolution with custom stride
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, C, H, W, D = x.shape

        # Project patches
        x = self.proj(x)  # [B, embed_dim, H', W', D']

        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, embed_dim]

        return x
