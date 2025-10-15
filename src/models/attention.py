"""
Divided Space-Time Attention Blocks

Implements factorized attention for efficient 4D (3D+time) processing.
Inspired by TimeSformer (Bertasius et al., ICML 2021).
"""

import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with GELU activation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention module.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N is sequence length, D is dimension

        Returns:
            output: [B, N, D]
        """
        B, N, D = x.shape

        # Generate Q, K, V: [B, N, 3*D] -> [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # Permute to [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, num_heads, N, head_dim]

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DividedSpaceTimeBlock(nn.Module):
    """
    Divided Space-Time Attention Block.

    Factorizes 4D attention into two sequential operations:
    1. Spatial attention: Attend within each timepoint
    2. Temporal attention: Attend across timepoints for each spatial location

    This reduces complexity from O((T*H*W*D)^2) to O(T*H*W*D + T*H*W*D),
    making it tractable for large 3D+time volumes.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: Whether to use bias in QKV projection
        dropout: Dropout probability
        attn_dropout: Attention dropout probability
        drop_path: Stochastic depth probability

    Example:
        >>> block = DividedSpaceTimeBlock(dim=768, num_heads=12)
        >>> # Input: [B, T*N_patches + 1, D] (with CLS token)
        >>> x = torch.randn(2, 16*64 + 1, 768)
        >>> out = block(x, num_frames=16, num_spatial_patches=64)
        >>> print(out.shape)
        torch.Size([2, 1025, 768])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Spatial attention
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            proj_drop=dropout,
        )

        # Temporal attention
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            proj_drop=dropout,
        )

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
        )

        # Stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        num_spatial_patches: int,
    ) -> torch.Tensor:
        """
        Forward pass with divided space-time attention.

        Args:
            x: Input tokens [B, N_total, D] where N_total = 1 + T*N_patches
               (1 for CLS token, T frames, N_patches per frame)
            num_frames: Number of temporal frames (T)
            num_spatial_patches: Number of spatial patches per frame (N_patches)

        Returns:
            output: [B, N_total, D]
        """
        B, N_total, D = x.shape

        # Extract CLS token
        cls_token = x[:, 0:1, :]  # [B, 1, D]
        x_no_cls = x[:, 1:, :]  # [B, T*N_patches, D]

        # --- Spatial Attention ---
        # Reshape to [B, T, N_patches, D]
        x_spatial = rearrange(
            x_no_cls, "b (t n) d -> b t n d", t=num_frames, n=num_spatial_patches
        )

        # Process each timepoint independently
        x_spatial_attended = []
        for t in range(num_frames):
            x_t = x_spatial[:, t, :, :]  # [B, N_patches, D]
            # Apply spatial attention with residual connection
            x_t = x_t + self.drop_path1(self.spatial_attn(self.norm1(x_t)))
            x_spatial_attended.append(x_t)

        # Stack back: [B, T, N_patches, D]
        x_spatial = torch.stack(x_spatial_attended, dim=1)

        # --- Temporal Attention ---
        # Reshape to [B, N_patches, T, D] for temporal attention
        x_temporal = x_spatial.permute(0, 2, 1, 3)  # [B, N_patches, T, D]

        # Process each spatial location independently
        B, N_patches, T, D = x_temporal.shape
        x_temporal = rearrange(x_temporal, "b n t d -> (b n) t d")  # [B*N_patches, T, D]

        # Apply temporal attention
        x_temporal = x_temporal + self.drop_path2(
            self.temporal_attn(self.norm2(x_temporal))
        )

        # Reshape back: [B, N_patches, T, D] -> [B, T, N_patches, D]
        x_temporal = rearrange(
            x_temporal, "(b n) t d -> b t n d", b=B, n=N_patches
        )

        # Flatten back to sequence: [B, T*N_patches, D]
        x_out = rearrange(x_temporal, "b t n d -> b (t n) d")

        # --- MLP ---
        x_out = x_out + self.drop_path3(self.mlp(self.norm3(x_out)))

        # Re-attach CLS token
        x_out = torch.cat([cls_token, x_out], dim=1)  # [B, 1 + T*N_patches, D]

        return x_out


class JointSpaceTimeBlock(nn.Module):
    """
    Joint Space-Time Attention Block (baseline comparison).

    Applies standard attention over all spatial and temporal tokens jointly.
    Much more computationally expensive than divided attention.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Single joint attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            proj_drop=dropout,
        )

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
        )

        # Stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with joint attention over all tokens.

        Args:
            x: Input tokens [B, N_total, D]
            **kwargs: Ignored (for compatibility with DividedSpaceTimeBlock)

        Returns:
            output: [B, N_total, D]
        """
        # Attention with residual
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # MLP with residual
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
