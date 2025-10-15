"""
PyTorch Lightning module for VideoMAE-3D self-supervised pretraining

Implements masked autoencoding for 3D+time volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from typing import Dict, Any, Tuple

from ..models import VTFormer, VideoMAEDecoder


class VideoMAEModule(pl.LightningModule):
    """
    PyTorch Lightning module for VideoMAE-3D pretraining.

    Trains VT-Former using masked autoencoding on unlabeled 3D+time data.

    Args:
        encoder: VT-Former encoder
        decoder_config: Config dict for VideoMAE decoder
        mask_ratio: Ratio of patches to mask (0.75 recommended)
        learning_rate: Peak learning rate
        warmup_epochs: Number of warmup epochs
        max_epochs: Total training epochs
        weight_decay: AdamW weight decay
    """

    def __init__(
        self,
        encoder: VTFormer,
        decoder_config: Dict[str, Any] = None,
        mask_ratio: float = 0.75,
        learning_rate: float = 1.5e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        weight_decay: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        # Build decoder
        if decoder_config is None:
            decoder_config = {
                "embed_dim": encoder.embed_dim,
                "decoder_embed_dim": 384,
                "decoder_depth": 4,
                "decoder_num_heads": 6,
                "patch_size": encoder.patch_size,
                "in_channels": 1,
            }

        self.decoder = VideoMAEDecoder(**decoder_config)

        # For logging
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with masking.

        Args:
            x: Input [B, T, C, H, W, D]

        Returns:
            pred: Reconstructed patches [B*N_masked, patch_volume]
            mask: Binary mask [B, N_total]
        """
        B, T, C, H, W, D = x.shape

        # Generate random mask
        mask = self._generate_mask(B, T)  # [B, N_total]

        # Encode visible patches
        # For simplicity, we'll encode all patches and then mask
        # (More efficient implementation would skip masked patches)
        encoded = self.encoder(x)  # [B, 1 + N_total, embed_dim]

        # Decode with mask tokens
        pred = self.decoder(encoded, mask)

        return pred, mask

    def _generate_mask(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """
        Generate random binary mask for patches.

        Args:
            batch_size: Batch size
            num_frames: Number of temporal frames

        Returns:
            mask: Binary mask [B, N_total] where 1 = masked, 0 = visible
        """
        N_total = num_frames * self.encoder.num_spatial_patches

        # Random masking
        mask = torch.rand(batch_size, N_total, device=self.device) < self.mask_ratio

        return mask

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert volume to patches.

        Args:
            x: Input [B, T, C, H, W, D]

        Returns:
            patches: [B, T*N_patches, patch_volume]
        """
        B, T, C, H, W, D = x.shape
        ph, pw, pd = self.encoder.patch_size

        # Ensure dimensions are divisible
        assert H % ph == 0 and W % pw == 0 and D % pd == 0

        # Number of patches per dimension
        nh, nw, nd = H // ph, W // pw, D // pd

        # Reshape to patches
        patches = x.reshape(B, T, C, nh, ph, nw, pw, nd, pd)
        patches = patches.permute(0, 1, 3, 5, 7, 2, 4, 6, 8)  # [B, T, nh, nw, nd, C, ph, pw, pd]
        patches = patches.reshape(B, T * nh * nw * nd, C * ph * pw * pd)

        return patches

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["volume"]  # [B, T, C, H, W, D]

        # Convert to patches (ground truth)
        target_patches = self._patchify(x)  # [B, N_total, patch_volume]

        # Forward pass with masking
        pred_patches, mask = self(x)

        # Compute loss only on masked patches
        # Extract masked patches from target
        mask_expanded = mask.unsqueeze(-1).expand_as(target_patches)
        target_masked = target_patches[mask_expanded].reshape(-1, target_patches.shape[-1])

        # MSE loss
        loss = F.mse_loss(pred_patches, target_masked)

        # Log
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_losses.append(loss.item())

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch["volume"]

        # Convert to patches
        target_patches = self._patchify(x)

        # Forward pass with masking
        pred_patches, mask = self(x)

        # Compute loss on masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(target_patches)
        target_masked = target_patches[mask_expanded].reshape(-1, target_patches.shape[-1])

        loss = F.mse_loss(pred_patches, target_masked)

        # Log
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_losses.append(loss.item())

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine annealing with warmup
        warmup_steps = self.warmup_epochs * self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if len(self.train_losses) > 0:
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("train_loss_epoch", avg_loss)
            self.train_losses = []

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if len(self.val_losses) > 0:
            avg_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("val_loss_epoch", avg_loss)
            self.val_losses = []
