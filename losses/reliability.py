"""
Pseudo-label reliability control module.

Corresponds to Sections 3.2.2 and 4.3.3 of the thesis:
    - Eq 3.12  pixel-level confidence + entropy filter
    - Eq 3.13  augmentation consistency check (K views)
    - Eq 3.15  sample-level mean entropy filter
    - temporal stability check across checkpoints
    - adaptive fallback: linearly relax the threshold if too few pixels
      survive the strict filter

Together these four filters form a closed-loop reliability pipeline
covering output-level, consistency-level and temporal-level control.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T


def pixel_filter(prob, tau_p=0.9, tau_H=1.5):
    """
    Stage 1: pixel-level confidence and entropy filter (Eq 3.12).

    Args:
        prob:  [B, C, H, W] softmax probabilities.
        tau_p: maximum-class-probability threshold; pixels below this
               value are considered unreliable.
        tau_H: entropy threshold; pixels above this value are considered
               highly uncertain.

    Returns:
        [B, H, W] float mask where 1 means keep and 0 means discard.
    """
    max_p, _ = prob.max(dim=1)                            # [B, H, W]
    H = -(prob * torch.log(prob + 1e-8)).sum(dim=1)       # [B, H, W]
    return ((max_p >= tau_p) & (H <= tau_H)).float()


def sample_filter(prob, tau_H=1.5):
    """
    Stage 2: sample-level mean entropy filter (Eq 3.15).

    If the mean entropy over a whole image is too high, the entire image
    is considered unreliable and should be discarded or down-weighted.

    Args:
        prob:  [B, C, H, W] softmax probabilities.
        tau_H: sample-level mean entropy threshold.

    Returns:
        [B] float mask where 1 means keep the whole sample.
    """
    H = -(prob * torch.log(prob + 1e-8)).sum(dim=1)       # [B, H, W]
    sample_H = H.mean(dim=[1, 2])                         # [B]
    return (sample_H <= tau_H).float()


def augmentation_consistency(teacher, x, K=4, thr=0.8):
    """
    Stage 3: K-view augmentation consistency check (Eq 3.13).

    The same input is passed through K-1 augmented versions. The
    per-pixel agreement of the argmax predictions is compared against
    the original view. Pixels with agreement below `thr` are marked
    unreliable.

    Args:
        teacher: frozen teacher model, callable, returns (logits, feats).
        x:       [B, 3, H, W] original input.
        K:       total number of views (1 original + K-1 augmented).
        thr:     agreement threshold.

    Returns:
        [B, H, W] float mask.
    """
    aug = T.Compose([
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])
    with torch.no_grad():
        preds = [teacher(x)[0].argmax(dim=1)]             # [B, H, W]
        for _ in range(K - 1):
            preds.append(teacher(aug(x))[0].argmax(dim=1))
    preds = torch.stack(preds, dim=0)                     # [K, B, H, W]
    agree = (preds == preds[0:1]).float().mean(dim=0)     # [B, H, W]
    return (agree >= thr).float()


def temporal_stability(teacher, x, prev_pred, thr=0.9):
    """
    Stage 4: cross-checkpoint temporal stability check.

    If the prediction for the same input flips excessively between the
    current checkpoint and a previous one, the model is unstable on this
    sample and the sample should be excluded.

    Args:
        teacher:   current teacher model.
        x:         [B, 3, H, W] input.
        prev_pred: [B, H, W] argmax predictions from a previous checkpoint.
        thr:       agreement threshold.

    Returns:
        [B] float mask.
    """
    with torch.no_grad():
        cur = teacher(x)[0].argmax(dim=1)                 # [B, H, W]
    agree = (cur == prev_pred).float().mean(dim=[1, 2])   # [B]
    return (agree >= thr).float()


def adaptive_fallback(prob, tau_p, tau_H, rho_min, floor):
    """
    Adaptive fallback mechanism (thesis Section 4.3.3, Eq 3.20).

    When the retained pixel ratio `rho` under the strict threshold falls
    below `rho_min`, `tau_p` is linearly relaxed toward `floor` until the
    minimum retained ratio is satisfied. This prevents the model from
    collapsing on hard subsets such as night + rain, where strict
    filtering retains fewer than 5% of pixels.

    Args:
        prob:    [B, C, H, W] softmax probabilities.
        tau_p:   initial confidence threshold (typically 0.9).
        tau_H:   entropy threshold.
        rho_min: minimum retained pixel ratio (typically 0.1).
        floor:   lower bound for the relaxed confidence threshold.

    Returns:
        mask: [B, H, W] float.
        rho:  actual retained pixel ratio.
        thr:  actual confidence threshold used.
    """
    mask = pixel_filter(prob, tau_p, tau_H)
    rho = mask.mean().item()
    if rho >= rho_min:
        return mask, rho, tau_p

    max_p, _ = prob.max(dim=1)
    H = -(prob * torch.log(prob + 1e-8)).sum(dim=1)
    lo, hi = floor, tau_p
    for _ in range(12):
        mid = (lo + hi) / 2.0
        m = ((max_p >= mid) & (H <= tau_H)).float()
        if m.mean().item() >= rho_min:
            hi = mid
        else:
            lo = mid
    final_mask = ((max_p >= hi) & (H <= tau_H)).float()
    return final_mask, final_mask.mean().item(), hi


def multi_view_consistency(f_orig, f_aug):
    """
    Multi-view feature cosine consistency (Eq 3.8).

    The student should produce consistent features on the original view
    and the augmented view. This is the self-supervised core of UDAM,
    i.e. the L_ss term.

    Args:
        f_orig: [B, D] original-view features.
        f_aug:  [B, D] augmented-view features.

    Returns:
        Scalar loss, equal to 1 - mean cosine similarity.
    """
    sim = F.cosine_similarity(f_orig, f_aug, dim=-1).mean()
    return 1.0 - sim