"""
3D augmentation transforms for volumetric microscopy data

Uses MONAI and custom transforms for 3D+time data.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class RandomFlip3D:
    """Random flip along spatial axes."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, C, H, W, D]
        Returns:
            flipped: [T, C, H, W, D]
        """
        if np.random.rand() < self.prob:
            # Flip H
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[2])
            # Flip W
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[3])
            # Flip D
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[4])
        return x


class RandomIntensityShift:
    """Random brightness and contrast adjustments."""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        prob: float = 0.5,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, C, H, W, D]
        Returns:
            adjusted: [T, C, H, W, D]
        """
        if np.random.rand() < self.prob:
            # Brightness
            brightness = np.random.uniform(*self.brightness_range)
            x = x * brightness

            # Contrast
            contrast = np.random.uniform(*self.contrast_range)
            mean = x.mean()
            x = (x - mean) * contrast + mean

            # Clip to valid range
            x = torch.clamp(x, 0.0, 1.0)

        return x


class RandomGaussianNoise:
    """Add random Gaussian noise."""

    def __init__(self, std: float = 0.01, prob: float = 0.3):
        self.std = std
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, C, H, W, D]
        Returns:
            noisy: [T, C, H, W, D]
        """
        if np.random.rand() < self.prob:
            noise = torch.randn_like(x) * self.std
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)
        return x


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transforms(config: dict = None) -> Compose:
    """
    Get training augmentation pipeline.

    Args:
        config: Augmentation configuration dict

    Returns:
        transform: Composed transform
    """
    if config is None:
        config = {
            "random_flip_prob": 0.5,
            "brightness_range": [0.8, 1.2],
            "contrast_range": [0.8, 1.2],
            "gaussian_noise_std": 0.01,
        }

    transforms = [
        RandomFlip3D(prob=config.get("random_flip_prob", 0.5)),
        RandomIntensityShift(
            brightness_range=config.get("brightness_range", [0.8, 1.2]),
            contrast_range=config.get("contrast_range", [0.8, 1.2]),
            prob=0.5,
        ),
        RandomGaussianNoise(
            std=config.get("gaussian_noise_std", 0.01),
            prob=0.3,
        ),
    ]

    return Compose(transforms)


def get_val_transforms() -> Compose:
    """
    Get validation transforms (no augmentation).

    Returns:
        transform: Identity transform
    """
    return Compose([])
