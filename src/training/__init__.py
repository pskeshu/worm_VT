"""
Training modules for WormVT
"""

from .pretrain_module import VideoMAEModule
from .finetune_module import StageClassificationModule

__all__ = [
    "VideoMAEModule",
    "StageClassificationModule",
]
