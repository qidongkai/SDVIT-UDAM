"""
Visualisation utilities.

Implements the feature-heatmap visualisation used in the ablation study
of Section 4.2.4 (Figure 4-7) and a convenience helper that overlays the
heatmap on the input image for qualitative inspection.

The heatmap is the activation magnitude in the final encoder layer,
averaged over the validation set and normalised per image, following the
description on page 179 of the thesis.
"""
import torch
import torch.nn.functional as F


def feature_heatmap(feat, size=512):
    """
    Convert a token feature map into a normalised activation heatmap.

    Args:
        feat: [B, N, D] token features from the final encoder layer.
        size: spatial resolution of the output heatmap.

    Returns:
        [B, 1, size, size] heatmap normalised to [0, 1] per image.
    """
    B, N, D = feat.shape
    h = int(round(N ** 0.5))
    assert h * h == N, f"N={N} is not a perfect square"

    x = feat.transpose(1, 2).reshape(B, D, h, h)          # [B, D, h, h]
    x = x.mean(dim=1, keepdim=True)                        # [B, 1, h, h]
    x = F.interpolate(x, size=(size, size),
                      mode="bilinear", align_corners=False)

    # Per-image min-max normalisation.
    flat = x.flatten(1)
    min_v = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    max_v = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    return (x - min_v) / (max_v - min_v + 1e-8)


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay a heatmap on an input image for qualitative inspection.

    Args:
        image:   [B, 3, H, W] input image, expected in [0, 1].
        heatmap: [B, 1, H, W] heatmap in [0, 1].
        alpha:   blending weight of the heatmap.

    Returns:
        [B, 3, H, W] blended image.
    """
    assert image.shape[0] == heatmap.shape[0]
    assert image.shape[-2:] == heatmap.shape[-2:]

    # Simple jet-like colour map: low -> blue, high -> red.
    h = heatmap.squeeze(1)                                  # [B, H, W]
    r = h
    g = 1.0 - (2.0 * h - 1.0).abs()
    b = 1.0 - h
    colour = torch.stack([r, g, b], dim=1)                  # [B, 3, H, W]
    colour = colour.clamp(0.0, 1.0)

    return (1.0 - alpha) * image + alpha * colour