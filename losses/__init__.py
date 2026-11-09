"""
Unified entry point for the loss modules.

This package exposes all training losses, reliability control utilities,
and UDAM-related functions so that external code can import them in a
single statement, e.g.:

    from losses import udam_total_loss, self_distill_loss
"""
from .sd_loss import (
    self_distill_loss,
    total_sdvit_loss,
    compute_reliability_mask,
)
from .reliability import (
    pixel_filter,
    sample_filter,
    augmentation_consistency,
    temporal_stability,
    adaptive_fallback,
    multi_view_consistency,
)
from .udam_loss import (
    mmd_rbf,
    adaptive_gamma,
    kl_shared,
    udam_total_loss,
)

__all__ = [
    "self_distill_loss",
    "total_sdvit_loss",
    "compute_reliability_mask",
    "pixel_filter",
    "sample_filter",
    "augmentation_consistency",
    "temporal_stability",
    "adaptive_fallback",
    "multi_view_consistency",
    "mmd_rbf",
    "adaptive_gamma",
    "kl_shared",
    "udam_total_loss",
]