"""
Dataset loader for NIH-LS C. elegans embryo data

Loads 3D+time lightsheet microscopy data with cell tracking information.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import tifffile
from pathlib import Path


class NIHLSDataset(Dataset):
    """
    NIH-LS dataset for self-supervised pretraining.

    Loads 3D+time volumes from the NIH-LS dataset with cell tracking data.
    Used for VideoMAE-3D pretraining (no labels needed).

    Args:
        data_root: Root directory containing NIH-LS data
        embryo_names: List of embryo directory names to include
        num_frames: Number of consecutive frames per clip
        frame_stride: Stride for temporal sampling
        spatial_size: Target spatial size (H, W, D) for cropping
        transform: Optional transform to apply
        split: 'train' or 'val'
        train_ratio: Fraction of data for training

    Directory structure expected:
        data_root/
        └── embryo_name/
            ├── images/
            │   ├── embryo_name_t000.tif
            │   ├── embryo_name_t001.tif
            │   └── ...
            └── tracks/
                └── tracks.txt
    """

    def __init__(
        self,
        data_root: str,
        embryo_names: List[str] = None,
        num_frames: int = 16,
        frame_stride: int = 1,
        spatial_size: Tuple[int, int, int] = (256, 256, 64),
        transform: Optional[callable] = None,
        split: str = "train",
        train_ratio: float = 0.8,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.spatial_size = spatial_size
        self.transform = transform
        self.split = split

        # Default to all embryos in directory if not specified
        if embryo_names is None:
            embryo_names = [
                d.name
                for d in self.data_root.iterdir()
                if d.is_dir() and (d / "images").exists()
            ]

        # Find all valid frame sequences
        self.samples = []
        for embryo_name in embryo_names:
            embryo_dir = self.data_root / embryo_name
            frames_dir = embryo_dir / "images"

            if not frames_dir.exists():
                print(f"Warning: {frames_dir} does not exist, skipping...")
                continue

            # Get all frame files (sorted)
            frame_files = sorted(frames_dir.glob("*.tif"))

            if len(frame_files) == 0:
                print(f"Warning: No TIFF files in {frames_dir}, skipping...")
                continue

            # Create clips: sliding window with stride
            max_start_idx = len(frame_files) - (num_frames * frame_stride)
            if max_start_idx < 0:
                print(
                    f"Warning: Not enough frames in {embryo_name} "
                    f"(need {num_frames * frame_stride}, have {len(frame_files)})"
                )
                continue

            for start_idx in range(0, max_start_idx + 1, num_frames // 2):
                # Extract frame indices for this clip
                frame_indices = list(
                    range(start_idx, start_idx + num_frames * frame_stride, frame_stride)
                )

                clip_frames = [frame_files[i] for i in frame_indices]

                self.samples.append(
                    {
                        "embryo": embryo_name,
                        "frame_paths": clip_frames,
                        "start_idx": start_idx,
                    }
                )

        # Train/val split
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * train_ratio)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]

        print(f"NIHLSDataset ({split}): {len(self.samples)} clips from {len(embryo_names)} embryos")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a 3D+time clip.

        Returns:
            sample: Dict with keys:
                - 'volume': [T, C, H, W, D] tensor
                - 'embryo': embryo name
                - 'start_idx': starting frame index
        """
        sample_info = self.samples[idx]

        # Load all frames in clip
        frames = []
        for frame_path in sample_info["frame_paths"]:
            # Load 3D TIFF stack
            volume = tifffile.imread(str(frame_path))  # [H, W, D] or [D, H, W]

            # Ensure consistent shape: [H, W, D]
            if volume.ndim == 3:
                if volume.shape[0] < volume.shape[2]:
                    # Likely [D, H, W], transpose to [H, W, D]
                    volume = np.transpose(volume, (1, 2, 0))
            else:
                raise ValueError(f"Unexpected volume shape: {volume.shape}")

            # Add channel dimension: [1, H, W, D]
            volume = volume[np.newaxis, ...].astype(np.float32)

            # Normalize to [0, 1]
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

            frames.append(volume)

        # Stack into [T, C, H, W, D]
        clip = np.stack(frames, axis=0)

        # Spatial crop/resize to target size
        clip = self._spatial_crop(clip, self.spatial_size)

        # Convert to tensor
        clip = torch.from_numpy(clip).float()

        # Apply transforms
        if self.transform is not None:
            clip = self.transform(clip)

        sample = {
            "volume": clip,  # [T, C, H, W, D]
            "embryo": sample_info["embryo"],
            "start_idx": sample_info["start_idx"],
        }

        return sample

    def _spatial_crop(
        self, volume: np.ndarray, target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Randomly crop or resize volume to target spatial size.

        Args:
            volume: [T, C, H, W, D]
            target_size: (target_H, target_W, target_D)

        Returns:
            cropped: [T, C, target_H, target_W, target_D]
        """
        T, C, H, W, D = volume.shape
        target_H, target_W, target_D = target_size

        # Random crop
        if H >= target_H and W >= target_W and D >= target_D:
            # Random offsets
            h_start = np.random.randint(0, H - target_H + 1)
            w_start = np.random.randint(0, W - target_W + 1)
            d_start = np.random.randint(0, D - target_D + 1)

            volume = volume[
                :,
                :,
                h_start : h_start + target_H,
                w_start : w_start + target_W,
                d_start : d_start + target_D,
            ]
        else:
            # Center crop and pad if necessary
            # For simplicity, just use center crop
            h_start = max(0, (H - target_H) // 2)
            w_start = max(0, (W - target_W) // 2)
            d_start = max(0, (D - target_D) // 2)

            h_end = min(H, h_start + target_H)
            w_end = min(W, w_start + target_W)
            d_end = min(D, d_start + target_D)

            cropped = volume[:, :, h_start:h_end, w_start:w_end, d_start:d_end]

            # Pad if needed
            if cropped.shape[2:] != target_size:
                pad_h = target_H - cropped.shape[2]
                pad_w = target_W - cropped.shape[3]
                pad_d = target_D - cropped.shape[4]

                padding = (
                    (0, 0),  # T
                    (0, 0),  # C
                    (pad_h // 2, pad_h - pad_h // 2),  # H
                    (pad_w // 2, pad_w - pad_w // 2),  # W
                    (pad_d // 2, pad_d - pad_d // 2),  # D
                )
                volume = np.pad(cropped, padding, mode="constant", constant_values=0)
            else:
                volume = cropped

        return volume


class NIHLSClassifiedDataset(Dataset):
    """
    NIH-LS dataset with VLM classifications for supervised fine-tuning.

    Loads frames that have been classified by Claude VLM.

    Args:
        data_root: Root directory containing NIH-LS data
        classification_file: Path to embryo_classifications_all.json
        spatial_size: Target spatial size (H, W, D)
        transform: Optional transform
        split: 'train', 'val', or 'test'
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    """

    def __init__(
        self,
        data_root: str,
        classification_file: str,
        spatial_size: Tuple[int, int, int] = (256, 256, 64),
        transform: Optional[callable] = None,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.spatial_size = spatial_size
        self.transform = transform
        self.split = split

        # Load classifications
        with open(classification_file, "r") as f:
            classifications = json.load(f)

        # Parse stage labels
        self.stage_to_idx = {}
        self.idx_to_stage = {}
        stage_idx = 0

        self.samples = []
        for entry in classifications:
            # Extract stage
            stage = self._extract_stage(entry)
            if stage not in self.stage_to_idx:
                self.stage_to_idx[stage] = stage_idx
                self.idx_to_stage[stage_idx] = stage
                stage_idx += 1

            # Build sample
            self.samples.append(
                {
                    "frame_path": entry["frame_path"],
                    "stage": stage,
                    "stage_idx": self.stage_to_idx[stage],
                    "confidence": entry.get("confidence", 1.0),
                    "timepoint": entry.get("timepoint_minutes", 0),
                }
            )

        # Train/val/test split
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        train_end = int(len(self.samples) * train_ratio)
        val_end = train_end + int(len(self.samples) * val_ratio)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:train_end]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[train_end:val_end]]
        else:  # test
            self.samples = [self.samples[i] for i in indices[val_end:]]

        print(
            f"NIHLSClassifiedDataset ({split}): {len(self.samples)} frames, "
            f"{len(self.stage_to_idx)} stages"
        )

    def _extract_stage(self, entry: dict) -> str:
        """Extract stage label from classification entry."""
        # Try different possible keys
        if "stage" in entry:
            return entry["stage"]
        elif "classification" in entry:
            return entry["classification"]
        else:
            # Parse from response text
            response = entry.get("response", "")
            # Simple heuristic: look for common stage names
            response_lower = response.lower()
            if "1-cell" in response_lower:
                return "1-cell"
            elif "2-cell" in response_lower:
                return "2-cell"
            elif "4-cell" in response_lower:
                return "4-cell"
            elif "8-cell" in response_lower:
                return "8-cell"
            elif "gastrulation" in response_lower:
                if "early" in response_lower:
                    return "early gastrulation"
                elif "late" in response_lower:
                    return "late gastrulation"
                else:
                    return "gastrulation"
            elif "comma" in response_lower:
                return "comma"
            elif "fold" in response_lower:
                if "1.5" in response_lower:
                    return "1.5-fold"
                elif "2" in response_lower:
                    return "2-fold"
                elif "3" in response_lower:
                    return "3-fold"
            return "unknown"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single frame with classification label.

        Returns:
            sample: Dict with keys:
                - 'volume': [1, C, H, W, D] tensor (single frame)
                - 'label': stage index
                - 'stage': stage name
        """
        sample_info = self.samples[idx]

        # Load 3D volume
        frame_path = sample_info["frame_path"]
        if not os.path.isabs(frame_path):
            # Make path absolute relative to data_root
            frame_path = self.data_root / ".." / frame_path

        volume = tifffile.imread(str(frame_path))

        # Ensure [H, W, D]
        if volume.ndim == 3:
            if volume.shape[0] < volume.shape[2]:
                volume = np.transpose(volume, (1, 2, 0))

        # Add batch and channel dimensions: [1, 1, H, W, D]
        volume = volume[np.newaxis, np.newaxis, ...].astype(np.float32)

        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Spatial crop/resize
        volume = self._spatial_crop(volume, self.spatial_size)

        # Convert to tensor
        volume = torch.from_numpy(volume).float()
        volume = volume.squeeze(0)  # Remove batch dim: [1, H, W, D]

        # Apply transforms
        if self.transform is not None:
            volume = self.transform(volume)

        sample = {
            "volume": volume,  # [1, C, H, W, D]
            "label": torch.tensor(sample_info["stage_idx"], dtype=torch.long),
            "stage": sample_info["stage"],
        }

        return sample

    def _spatial_crop(
        self, volume: np.ndarray, target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Center crop to target size."""
        T, C, H, W, D = volume.shape
        target_H, target_W, target_D = target_size

        h_start = max(0, (H - target_H) // 2)
        w_start = max(0, (W - target_W) // 2)
        d_start = max(0, (D - target_D) // 2)

        h_end = min(H, h_start + target_H)
        w_end = min(W, w_start + target_W)
        d_end = min(D, d_start + target_D)

        return volume[:, :, h_start:h_end, w_start:w_end, d_start:d_end]

    def get_stage_names(self) -> List[str]:
        """Return list of stage names in order."""
        return [self.idx_to_stage[i] for i in range(len(self.idx_to_stage))]
