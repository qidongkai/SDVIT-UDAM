"""
Utility package.

Exposes the deterministic seed setter used by the ten-seed protocol in
Section 4.1.4 of the thesis, the logger used across training and
evaluation scripts, and the feature-heatmap visualiser used for the
qualitative ablation in Section 4.2.4.
"""
from .seed import set_seed
from .logger import get_logger
from .vis import feature_heatmap, overlay_heatmap

__all__ = [
    "set_seed",
    "get_logger",
    "feature_heatmap",
    "overlay_heatmap",
]