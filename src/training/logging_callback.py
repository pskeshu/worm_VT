"""
Custom logging callback for detailed file-based logging
"""

import os
import time
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class DetailedFileLogger(Callback):
    """
    Callback that writes detailed training progress to a log file.

    Logs:
    - Epoch start/end
    - Batch progress
    - Loss values
    - Metrics
    - GPU memory usage
    - Training speed
    """

    def __init__(self, log_dir: str, log_every_n_steps: int = 50):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.log_file = None
        self.epoch_start_time = None
        self.batch_start_time = None

    def setup(self, trainer, pl_module, stage):
        """Setup log file."""
        os.makedirs(self.log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir, f"training_{timestamp}.log")
        self.log_file = open(log_path, "w", buffering=1)  # Line buffering

        self._write_log("=" * 80)
        self._write_log(f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log("=" * 80)
        self._write_log(f"Stage: {stage}")
        self._write_log(f"Model: {pl_module.__class__.__name__}")
        self._write_log(f"Max Epochs: {trainer.max_epochs}")
        self._write_log(f"Device: {trainer.strategy.root_device}")
        self._write_log("=" * 80)
        self._write_log("")

    def on_train_epoch_start(self, trainer, pl_module):
        """Log epoch start."""
        self.epoch_start_time = time.time()
        self._write_log(f"\n{'='*80}")
        self._write_log(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} - Started")
        self._write_log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"{'='*80}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log batch progress periodically."""
        if batch_idx % self.log_every_n_steps == 0:
            # Get current loss
            loss = outputs.get("loss", None)
            loss_str = f"{loss.item():.4f}" if loss is not None else "N/A"

            # Get learning rate
            lr = trainer.optimizers[0].param_groups[0]["lr"]

            # Log progress
            self._write_log(
                f"  Batch {batch_idx}/{trainer.num_training_batches} | "
                f"Loss: {loss_str} | "
                f"LR: {lr:.2e}"
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch end with summary."""
        epoch_time = time.time() - self.epoch_start_time

        # Get metrics from logger
        metrics = trainer.callback_metrics

        self._write_log(f"\nEpoch {trainer.current_epoch + 1} Summary:")
        self._write_log(f"  Duration: {epoch_time:.2f}s ({epoch_time/60:.2f}min)")

        # Log all metrics
        for key, value in metrics.items():
            if "train" in key or "val" in key:
                self._write_log(f"  {key}: {value:.4f}")

        self._write_log(f"{'='*80}\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation results."""
        metrics = trainer.callback_metrics

        self._write_log(f"\nValidation Results (Epoch {trainer.current_epoch + 1}):")
        for key, value in metrics.items():
            if "val" in key:
                self._write_log(f"  {key}: {value:.4f}")
        self._write_log("")

    def on_train_end(self, trainer, pl_module):
        """Log training completion."""
        self._write_log("\n" + "=" * 80)
        self._write_log(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"Total Epochs: {trainer.current_epoch + 1}")
        self._write_log(f"Best Model: {trainer.checkpoint_callback.best_model_path}")
        self._write_log("=" * 80)

        if self.log_file:
            self.log_file.close()

    def _write_log(self, message: str):
        """Write message to log file."""
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()  # Ensure immediate write


class ProgressFileLogger(Callback):
    """
    Simple callback that writes a progress file with current status.

    Creates a 'progress.txt' file that always shows the latest status.
    Useful for quick checks without reading the full log.
    """

    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self.progress_file = None

    def setup(self, trainer, pl_module, stage):
        """Setup progress file."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.progress_file = os.path.join(self.log_dir, "progress.txt")

    def on_train_epoch_end(self, trainer, pl_module):
        """Update progress file."""
        metrics = trainer.callback_metrics

        with open(self.progress_file, "w") as f:
            f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epoch: {trainer.current_epoch + 1}/{trainer.max_epochs}\n")
            f.write(f"Progress: {(trainer.current_epoch + 1) / trainer.max_epochs * 100:.1f}%\n")
            f.write("\nMetrics:\n")

            for key, value in sorted(metrics.items()):
                f.write(f"  {key}: {value:.4f}\n")

            f.write(f"\nCheckpoint: {trainer.checkpoint_callback.best_model_path}\n")
