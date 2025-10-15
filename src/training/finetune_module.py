"""
PyTorch Lightning module for stage classification fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl
from typing import Dict, Any
from torchmetrics import Accuracy, F1Score

from ..models import VTFormer, ClassificationHead, VTFormerWithHead


class StageClassificationModule(pl.LightningModule):
    """
    PyTorch Lightning module for developmental stage classification.

    Fine-tunes pretrained VT-Former on VLM-labeled stages.

    Args:
        encoder: Pretrained VT-Former encoder
        num_classes: Number of developmental stages
        learning_rate: Learning rate
        weight_decay: Weight decay
        freeze_encoder: Whether to freeze encoder initially
        freeze_epochs: Number of epochs to keep encoder frozen
    """

    def __init__(
        self,
        encoder: VTFormer,
        num_classes: int = 10,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        freeze_encoder: bool = False,
        freeze_epochs: int = 0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        # Build model with classification head
        head = ClassificationHead(
            embed_dim=encoder.embed_dim,
            num_classes=num_classes,
        )

        self.model = VTFormerWithHead(
            encoder=encoder,
            head=head,
            pooling="cls",
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        self.freeze_epochs = freeze_epochs
        self.label_smoothing = label_smoothing

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["volume"]  # [B, 1, C, H, W, D]
        labels = batch["label"]  # [B]

        # Forward pass
        logits = self(x)

        # Loss with label smoothing
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
        )

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        # Log
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        x = batch["volume"]
        labels = batch["label"]

        # Forward pass
        logits = self(x)

        # Loss
        loss = F.cross_entropy(logits, labels)

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)

        # Log
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step."""
        x = batch["volume"]
        labels = batch["label"]

        # Forward pass
        logits = self(x)

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)

        # Log
        self.log("test_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7,
        )

        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        """Called at start of training epoch."""
        # Unfreeze encoder after specified epochs
        if self.freeze_encoder and self.current_epoch >= self.freeze_epochs:
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            self.freeze_encoder = False
            print(f"Unfreezing encoder at epoch {self.current_epoch}")
