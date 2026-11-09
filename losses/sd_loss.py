"""
SDViT self-distillation loss module.

Corresponds to Section 3.2.1 of the thesis:
    - Eq 3.4:  L_sd = || F_deep - P(F_mid) ||^2
    - Eq 3.5:  L_SDViT = L_seg + lambda_sd * L_sd
    - Eq 3.18: reliability-weighted variant

Core idea:
    The deepest (unpruned) layer provides the complete feature F_deep;
    the intermediate (pruned) layer is aligned via P(·) to produce
    F_mid_proj. The L2 distance serves as the self-distillation
    compensation signal. A feature-consistency-based token mask is used
    to avoid anomalous positions polluting the gradient.
"""
import torch
import torch.nn.functional as F


def self_distill_loss(f_deep, f_mid_proj, reliability_mask=None):
    """
    Self-distillation L2 loss (Eq 3.18).

    Args:
        f_deep:            [B, N, D] unpruned deep features.
        f_mid_proj:        [B, N, D] intermediate features after
                           projection alignment.
        reliability_mask:  [B, N] optional; 1 means the token is reliable.

    Returns:
        Scalar loss.
    """
    diff = (f_deep - f_mid_proj) ** 2                     # [B, N, D]
    if reliability_mask is not None:
        m = reliability_mask.float().unsqueeze(-1)        # [B, N, 1]
        num = (diff * m).sum()
        den = m.sum() * f_deep.shape[-1] + 1e-8
        return num / den
    return diff.mean()


def total_sdvit_loss(logits, gt, l_sd, lambda_sd=0.3):
    """
    Total SDViT training loss (Eq 3.5).

    Args:
        logits:    [B, C, H, W] student segmentation output.
        gt:        [B, H, W] ground-truth labels, 255 = ignore.
        l_sd:      scalar self-distillation loss.
        lambda_sd: self-distillation weight; the thesis reports 0.3 as the
                   grid-search optimum.

    Returns:
        (total_loss, l_seg, l_sd) tuple.
    """
    l_seg = F.cross_entropy(logits, gt, ignore_index=255)
    total = l_seg + lambda_sd * l_sd
    return total, l_seg, l_sd


def compute_reliability_mask(f_mid, f_deep, thr=0.8):
    """
    Token-level reliability mask based on feature consistency.

    Defined on page 100 of the thesis: if the cosine similarity between
    the projected intermediate features and the deep features exceeds
    `thr`, the token is deemed reliable and contributes to the
    self-distillation loss; otherwise it is treated as anomalous and its
    gradient is masked out.

    Args:
        f_mid:  [B, N, D] intermediate features after projection.
        f_deep: [B, N, D] deep features.
        thr:    similarity threshold.

    Returns:
        [B, N] float mask.
    """
    sim = F.cosine_similarity(f_mid, f_deep, dim=-1)      # [B, N]
    return (sim > thr).float()