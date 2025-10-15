"""
Model architectures for WormVT
"""

from .vt_former import (
    VTFormer,
    build_vt_former,
    vt_former_small,
    vt_former_base,
    vt_former_large,
)
from .patch_embed import PatchEmbed3D, PatchEmbed3DOverlapping
from .attention import DividedSpaceTimeBlock, JointSpaceTimeBlock, Attention, MLP
from .heads import (
    ClassificationHead,
    VideoMAEDecoder,
    TemporalConsistencyHead,
    SpatialPooling,
    VTFormerWithHead,
)

__all__ = [
    # Main model
    "VTFormer",
    "build_vt_former",
    "vt_former_small",
    "vt_former_base",
    "vt_former_large",
    # Components
    "PatchEmbed3D",
    "PatchEmbed3DOverlapping",
    "DividedSpaceTimeBlock",
    "JointSpaceTimeBlock",
    "Attention",
    "MLP",
    # Heads
    "ClassificationHead",
    "VideoMAEDecoder",
    "TemporalConsistencyHead",
    "SpatialPooling",
    "VTFormerWithHead",
]
