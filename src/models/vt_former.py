"""
Volumetric-Temporal Former (VT-Former)

Main architecture for 3D+time vision encoding of C. elegans embryo development.
Implements factorized space-time attention for efficient 4D processing.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

from .patch_embed import PatchEmbed3D
from .attention import DividedSpaceTimeBlock, JointSpaceTimeBlock


class VTFormer(nn.Module):
    """
    Volumetric-Temporal Transformer for 3D+time microscopy data.

    Processes 4D input (3D volume + time) using factorized space-time attention.
    Inspired by TimeSformer (Bertasius et al., ICML 2021) and ViViT (Arnab et al., ICCV 2021),
    adapted for volumetric microscopy data.

    Architecture:
        1. 3D Patch Embedding: Split each 3D frame into patches
        2. Positional Encoding: Learnable spatial + temporal positions
        3. CLS Token: Global representation token
        4. Divided Space-Time Attention: Factorized attention blocks
        5. Output: Sequence of tokens representing the 4D volume

    Args:
        img_size: Tuple of (H, W, D) for input volume dimensions
        patch_size: Tuple of (Ph, Pw, Pd) for patch dimensions
        in_channels: Number of input channels (1 for fluorescence)
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        temporal_resolution: Maximum number of frames per clip
        attention_type: 'divided' for factorized or 'joint' for full attention
        dropout: Dropout probability
        attn_dropout: Attention dropout probability
        path_dropout: Stochastic depth dropout probability

    Example:
        >>> model = VTFormer(
        ...     img_size=(256, 256, 64),
        ...     patch_size=(16, 16, 8),
        ...     embed_dim=512,
        ...     depth=6,
        ...     num_heads=8,
        ...     temporal_resolution=16
        ... )
        >>> # Input: [B, T, C, H, W, D]
        >>> x = torch.randn(2, 16, 1, 256, 256, 64)
        >>> tokens = model(x)  # [B, N_tokens, 512]
        >>> print(tokens.shape)
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (256, 256, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        temporal_resolution: int = 32,
        attention_type: str = "divided",  # 'divided' or 'joint'
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        path_dropout: float = 0.1,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.temporal_resolution = temporal_resolution
        self.attention_type = attention_type

        # 3D Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        num_spatial_patches = self.patch_embed.get_num_patches()
        self.num_spatial_patches = num_spatial_patches

        # CLS token: learnable global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_spatial_patches, embed_dim)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, temporal_resolution, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, path_dropout, depth)]

        # Transformer blocks
        if attention_type == "divided":
            BlockClass = DividedSpaceTimeBlock
        elif attention_type == "joint":
            BlockClass = JointSpaceTimeBlock
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.blocks = nn.ModuleList(
            [
                BlockClass(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights(init_std)

    def _init_weights(self, std: float = 0.02):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.spatial_pos_embed, std=std)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=std)
        nn.init.trunc_normal_(self.cls_token, std=std)

        # Initialize patch embedding projection
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=std)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # Initialize transformer blocks
        self.apply(self._init_block_weights)

    def _init_block_weights(self, module):
        """Initialize weights for transformer blocks."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through VT-Former.

        Args:
            x: Input tensor [B, T, C, H, W, D] where
               B = batch size
               T = number of temporal frames
               C = channels (typically 1 for fluorescence)
               H, W, D = spatial dimensions
            return_features: If True, return intermediate features

        Returns:
            If return_features=False:
                tokens: Output tokens [B, N_total, embed_dim]
            If return_features=True:
                (tokens, features): Tuple of output tokens and list of intermediate features
        """
        B, T, C, H, W, D = x.shape

        # Verify temporal dimension
        assert (
            T <= self.temporal_resolution
        ), f"Input has {T} frames but max is {self.temporal_resolution}"

        # Process each timepoint through patch embedding
        tokens_per_time = []
        for t in range(T):
            # Extract frame at time t: [B, C, H, W, D]
            frame_t = x[:, t, :, :, :, :]

            # Embed patches: [B, N_patches, embed_dim]
            patches = self.patch_embed(frame_t)

            # Add spatial positional encoding
            patches = patches + self.spatial_pos_embed

            tokens_per_time.append(patches)

        # Stack temporal dimension: [B, T, N_patches, embed_dim]
        tokens = torch.stack(tokens_per_time, dim=1)

        # Add temporal positional encoding
        # Broadcast: [1, T, 1, embed_dim]
        temporal_pos = self.temporal_pos_embed[:, :T, :].unsqueeze(2)
        tokens = tokens + temporal_pos

        # Flatten to sequence: [B, T*N_patches, embed_dim]
        tokens = rearrange(tokens, "b t n d -> b (t n) d")

        # Add CLS token: [B, 1 + T*N_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Dropout
        tokens = self.pos_drop(tokens)

        # Apply transformer blocks
        features = []
        for block in self.blocks:
            if self.attention_type == "divided":
                tokens = block(
                    tokens,
                    num_frames=T,
                    num_spatial_patches=self.num_spatial_patches,
                )
            else:
                tokens = block(tokens)

            if return_features:
                features.append(tokens)

        # Final layer norm
        tokens = self.norm(tokens)

        if return_features:
            return tokens, features
        else:
            return tokens

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token representation.

        Args:
            x: Input tensor [B, T, C, H, W, D]

        Returns:
            cls: CLS token [B, embed_dim]
        """
        tokens = self.forward(x)
        cls = tokens[:, 0, :]  # Extract first token (CLS)
        return cls

    def get_spatial_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial tokens (excluding CLS token).

        Args:
            x: Input tensor [B, T, C, H, W, D]

        Returns:
            spatial_tokens: [B, T*N_patches, embed_dim]
        """
        tokens = self.forward(x)
        spatial_tokens = tokens[:, 1:, :]  # Exclude CLS token
        return spatial_tokens


def build_vt_former(config: dict) -> VTFormer:
    """
    Build VT-Former from configuration dictionary.

    Args:
        config: Configuration dict with model parameters

    Returns:
        model: VTFormer instance
    """
    return VTFormer(
        img_size=tuple(config["img_size"]),
        patch_size=tuple(config["patch_size"]),
        in_channels=config["in_channels"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config.get("mlp_ratio", 4.0),
        temporal_resolution=config["temporal_resolution"],
        attention_type=config.get("attention_type", "divided"),
        dropout=config.get("dropout", 0.1),
        attn_dropout=config.get("attention_dropout", 0.1),
        path_dropout=config.get("path_dropout", 0.1),
        init_std=config.get("init_std", 0.02),
    )


# Model configurations
VT_FORMER_CONFIGS = {
    "small": {
        "img_size": [256, 256, 64],
        "patch_size": [16, 16, 8],
        "in_channels": 1,
        "embed_dim": 512,
        "depth": 6,
        "num_heads": 8,
        "temporal_resolution": 16,
    },
    "base": {
        "img_size": [512, 512, 100],
        "patch_size": [16, 16, 8],
        "in_channels": 1,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "temporal_resolution": 32,
    },
    "large": {
        "img_size": [512, 512, 100],
        "patch_size": [16, 16, 8],
        "in_channels": 1,
        "embed_dim": 1024,
        "depth": 12,
        "num_heads": 16,
        "temporal_resolution": 32,
    },
}


def vt_former_small(**kwargs) -> VTFormer:
    """Build small VT-Former (~50M params)."""
    config = VT_FORMER_CONFIGS["small"].copy()
    config.update(kwargs)
    return build_vt_former(config)


def vt_former_base(**kwargs) -> VTFormer:
    """Build base VT-Former (~300M params)."""
    config = VT_FORMER_CONFIGS["base"].copy()
    config.update(kwargs)
    return build_vt_former(config)


def vt_former_large(**kwargs) -> VTFormer:
    """Build large VT-Former (~500M params)."""
    config = VT_FORMER_CONFIGS["large"].copy()
    config.update(kwargs)
    return build_vt_former(config)
