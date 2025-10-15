"""
Task-specific heads for VT-Former

Includes classification heads, reconstruction decoders, and other output layers.
"""

import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange


class ClassificationHead(nn.Module):
    """
    Simple classification head for developmental stage prediction.

    Args:
        embed_dim: Input embedding dimension
        num_classes: Number of output classes
        hidden_dim: Optional hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CLS token or pooled features [B, embed_dim]

        Returns:
            logits: Class logits [B, num_classes]
        """
        return self.head(x)


class VideoMAEDecoder(nn.Module):
    """
    Decoder for VideoMAE-3D reconstruction.

    Reconstructs masked patches from transformer features.
    Lightweight decoder following VideoMAE design.

    Args:
        embed_dim: Input embedding dimension from encoder
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder transformer blocks
        decoder_num_heads: Number of attention heads in decoder
        patch_size: Spatial patch size
        in_channels: Number of channels to reconstruct
    """

    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        in_channels: int = 1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size

        # Projection from encoder to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Mask token: learnable token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embeddings (learned)
        # Note: These will be initialized based on sequence length at runtime
        self.decoder_pos_embed = None

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_embed_dim,
                    nhead=decoder_num_heads,
                    dim_feedforward=decoder_embed_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Prediction head: map to pixel space
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_volume * in_channels, bias=True
        )

        # Initialize weights
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x: Encoder output [B, N_visible + 1, embed_dim] (CLS + visible patches)
            mask: Binary mask [B, N_total] where 1 = masked, 0 = visible

        Returns:
            pred: Reconstructed patches [B, N_masked, patch_volume]
        """
        B, N_visible_plus_cls, _ = x.shape
        N_total = mask.shape[1]

        # Remove CLS token
        x = x[:, 1:, :]  # [B, N_visible, embed_dim]

        # Project to decoder dimension
        x = self.decoder_embed(x)  # [B, N_visible, decoder_embed_dim]

        # Create mask tokens for masked positions
        N_masked = N_total - x.shape[1]
        mask_tokens = self.mask_token.expand(B, N_masked, -1)

        # Interleave visible and masked tokens
        # For simplicity, we'll concatenate and let the model learn positions
        x_full = torch.cat([x, mask_tokens], dim=1)  # [B, N_total, decoder_embed_dim]

        # Add positional embeddings
        if self.decoder_pos_embed is None or self.decoder_pos_embed.shape[1] != N_total:
            # Initialize decoder positional embeddings
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, N_total, self.decoder_embed_dim)
            ).to(x.device)
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        x_full = x_full + self.decoder_pos_embed

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)

        x_full = self.decoder_norm(x_full)

        # Predict pixel values
        pred = self.decoder_pred(x_full)  # [B, N_total, patch_volume]

        # Return only masked patches for loss computation
        # Extract masked positions
        mask_indices = mask.bool()
        pred_masked = pred[mask_indices]  # [B*N_masked, patch_volume]

        return pred_masked


class TemporalConsistencyHead(nn.Module):
    """
    Head for enforcing temporal consistency between adjacent frames.

    Predicts whether two frames are temporally adjacent and their order.
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 3),  # 3 classes: before, same, after
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Features from frame 1 [B, embed_dim]
            x2: Features from frame 2 [B, embed_dim]

        Returns:
            logits: Temporal relationship logits [B, 3]
        """
        x = torch.cat([x1, x2], dim=-1)
        return self.head(x)


class SpatialPooling(nn.Module):
    """
    Spatial pooling strategies for extracting frame-level features.
    """

    def __init__(self, strategy: str = "cls"):
        """
        Args:
            strategy: Pooling strategy
                - 'cls': Use CLS token
                - 'mean': Average all tokens
                - 'max': Max pooling over tokens
        """
        super().__init__()
        self.strategy = strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence [B, N, D] with CLS token at position 0

        Returns:
            pooled: Pooled features [B, D]
        """
        if self.strategy == "cls":
            return x[:, 0, :]
        elif self.strategy == "mean":
            return x[:, 1:, :].mean(dim=1)  # Exclude CLS token
        elif self.strategy == "max":
            return x[:, 1:, :].max(dim=1)[0]  # Exclude CLS token
        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")


class VTFormerWithHead(nn.Module):
    """
    VT-Former with task-specific head.

    Combines VT-Former encoder with a head for specific tasks.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module, pooling: str = "cls"):
        super().__init__()
        self.encoder = encoder
        self.pooling = SpatialPooling(pooling)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, C, H, W, D]

        Returns:
            output: Task-specific output
        """
        # Encode
        tokens = self.encoder(x)

        # Pool
        features = self.pooling(tokens)

        # Head
        output = self.head(features)

        return output
