"""
Data utilities for AR semantic segmentation.

Exposes the dataset, transforms, scene-aware sampler and split checker
used across training and evaluation scripts.
"""
from .dataset import ARSegDataset, CITYSCAPES_TO_UNIFIED
from .transforms import build_train_transform, MultiViewTransform
from .sampler import SceneAwareSampler
from .scene_check import check_scene_split

__all__ = [
    "ARSegDataset",
    "CITYSCAPES_TO_UNIFIED",
    "build_train_transform",
    "MultiViewTransform",
    "SceneAwareSampler",
    "check_scene_split",
]