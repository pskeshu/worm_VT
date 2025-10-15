"""
Main training script for WormVT

Supports both pretraining (VideoMAE-3D) and fine-tuning (stage classification).

Usage:
    # Pretraining
    python train.py training=pretrain model=vt_former_small

    # Fine-tuning
    python train.py training=finetune model=vt_former_small checkpoint=path/to/pretrained.ckpt
"""

import os
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import torch
from torch.utils.data import DataLoader

from src.models import build_vt_former
from src.data import NIHLSDataset, NIHLSClassifiedDataset, get_train_transforms, get_val_transforms
from src.training import VideoMAEModule, StageClassificationModule
from src.training.logging_callback import DetailedFileLogger, ProgressFileLogger


def setup_logging(output_dir: str):
    """Setup Python logging to file."""
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("wormvt")
    logger.setLevel(logging.INFO)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Setup logging
    logger = setup_logging(cfg.output_dir)
    logger.info("=" * 80)
    logger.info("WormVT Training")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Set seed
    pl.seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed}")

    # Optimize for Tensor Cores (RTX GPUs)
    if cfg.device == "cuda":
        torch.set_float32_matmul_precision('medium')  # Trade precision for speed on Tensor Cores
        logger.info("Enabled Tensor Core optimization (float32 matmul precision: medium)")

    # Build model
    logger.info("\nBuilding model...")
    encoder = build_vt_former(cfg.model)
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Parameters: {encoder.get_num_params():,}")

    # Create data loaders
    logger.info("\nCreating data loaders...")
    if cfg.training.name == "videomae_3d_pretrain":
        train_dataset = NIHLSDataset(
            data_root=cfg.data.data_root,
            embryo_names=cfg.data.embryos,
            num_frames=cfg.data.num_frames,
            frame_stride=cfg.data.frame_stride,
            spatial_size=tuple(cfg.data.spatial_crop_size),
            transform=get_train_transforms(cfg.data.augmentation),
            split="train",
            train_ratio=cfg.data.train_ratio,
        )

        val_dataset = NIHLSDataset(
            data_root=cfg.data.data_root,
            embryo_names=cfg.data.embryos,
            num_frames=cfg.data.num_frames,
            frame_stride=cfg.data.frame_stride,
            spatial_size=tuple(cfg.data.spatial_crop_size),
            transform=get_val_transforms(),
            split="val",
            train_ratio=cfg.data.train_ratio,
        )

    elif cfg.training.name == "stage_classification":
        train_dataset = NIHLSClassifiedDataset(
            data_root=cfg.data.data_root,
            classification_file=cfg.data.classification_file,
            spatial_size=tuple(cfg.data.spatial_crop_size),
            transform=get_train_transforms(cfg.data.augmentation),
            split="train",
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
        )

        val_dataset = NIHLSClassifiedDataset(
            data_root=cfg.data.data_root,
            classification_file=cfg.data.classification_file,
            spatial_size=tuple(cfg.data.spatial_crop_size),
            transform=get_val_transforms(),
            split="val",
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
        )

    else:
        raise ValueError(f"Unknown training type: {cfg.training.name}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {cfg.data.batch_size}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Build Lightning module
    logger.info("\nBuilding Lightning module...")
    if cfg.training.name == "videomae_3d_pretrain":
        model = VideoMAEModule(
            encoder=encoder,
            mask_ratio=cfg.training.masking.mask_ratio,
            learning_rate=cfg.training.optimizer.lr,
            warmup_epochs=cfg.training.lr_scheduler.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    elif cfg.training.name == "stage_classification":
        # Load pretrained checkpoint if specified
        if hasattr(cfg.training, "checkpoint") and cfg.training.checkpoint:
            logger.info(f"Loading pretrained checkpoint: {cfg.training.checkpoint}")
            # Load encoder weights from checkpoint
            checkpoint = torch.load(cfg.training.checkpoint)
            encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)

        model = StageClassificationModule(
            encoder=encoder,
            num_classes=cfg.training.num_classes,
            learning_rate=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            freeze_encoder=cfg.training.freeze_encoder,
            freeze_epochs=cfg.training.freeze_epochs,
            label_smoothing=cfg.training.loss.label_smoothing,
        )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=cfg.training.save_top_k,
            monitor=cfg.training.monitor,
            mode=cfg.training.mode,
        ),
        LearningRateMonitor(logging_interval="step"),
        DetailedFileLogger(
            log_dir=os.path.join(cfg.output_dir, "logs"),
            log_every_n_steps=cfg.log_every_n_steps,
        ),
        ProgressFileLogger(
            log_dir=cfg.output_dir,
        ),
    ]

    if cfg.training.early_stopping.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                patience=cfg.training.early_stopping.patience,
                mode=cfg.training.early_stopping.mode,
            )
        )

    # Loggers
    loggers = [
        CSVLogger(
            save_dir=cfg.output_dir,
            name="csv_logs",
        )
    ]

    if cfg.wandb.mode != "disabled":
        loggers.append(
            WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.experiment_name,
                save_dir=cfg.output_dir,
                offline=(cfg.wandb.mode == "offline"),
            )
        )

    # Trainer
    logger.info("\nInitializing trainer...")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Precision: {cfg.precision}")
    logger.info(f"Max epochs: {cfg.training.max_epochs}")
    logger.info(f"Gradient accumulation: {cfg.training.accumulate_grad_batches}")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.device,
        devices=1,
        precision=cfg.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=cfg.output_dir,
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Log files:")
    logger.info(f"  - Detailed: {cfg.output_dir}/logs/training_*.log")
    logger.info(f"  - Progress: {cfg.output_dir}/progress.txt")
    logger.info(f"  - CSV: {cfg.output_dir}/csv_logs/")
    logger.info("=" * 80 + "\n")

    trainer.fit(model, train_loader, val_loader)

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"Logs saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
